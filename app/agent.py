"""
agent.py — Conversation engine, built as a LangGraph StateGraph.

Why a graph instead of a hand-rolled while-loop: the agent <-> tools
exchange (LLM proposes a tool call -> we execute it over MCP -> result
goes back to the LLM -> repeat until it has a final answer) is exactly
the conditional-loop shape LangGraph is for. Modeling it explicitly as
a graph also makes the one piece of real branching logic in this app —
"does the model want to call a tool, or is it done talking?" — a named,
inspectable edge instead of an if/else buried in a loop body.

Design choice worth calling out: this graph is built and invoked fresh
per turn, reading prior turns from Postgres (db.get_recent_messages) the
same way reconnect/resume already works. We do NOT use LangGraph's own
checkpointer/persistence layer — db.py's conversations/conversation_messages
tables are already the single source of truth for conversation state, and
wiring up a second persistence mechanism on top would just be two sources
of truth fighting each other for no benefit at this scope.

Streaming note: LangGraph nodes normally return a finished state update,
but main.py needs token-by-token text deltas *as they happen* to forward
over the WebSocket (`ai_text_delta`) and sentence-by-sentence audio
(`audio_output`) with low latency. So the agent node takes an `on_text_delta`
callback through the graph's RunnableConfig (`config["configurable"]`)
and calls it directly while consuming llm.stream_chat() — the graph still
returns one clean final state update, but the caller gets live deltas too.
"""

import json
import operator
from typing import Annotated, Optional, TypedDict
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.runnables.config import RunnableConfig

from . import llm,tools
from .lang import system_prompt_for_locale

MAX_TOOL_LOOPS = 4 # hard cap — stops a misbehaving model from looping forever

#----------GRAPH STATE------------

class ConversationState(TypedDict):
    messages: Annotated[list[dict], operator.add]
    locale: str
    slots: dict
    tool_loop_count : int


#----------Nodes------------
async def agent_node(state: ConversationState, config: RunnableConfig)-> dict:
    """Call the LLM with the current message history + tool schemas.
    Streams text deltas out via config['configurable']['on_text_delta']
    as they arrive, and returns the finished assistant message (with any
    tool_calls) as the state update."""
    configurable = config.get("configurable", {})
    on_text_delta = configurable.get("on_text_delta")
    cancel_event = configurable.get("cancel_event")
    
    system_prompt = system_prompt_for_locale(state["locale"],state["slots"])
    messages = [{
        "role": "system",
        "content": system_prompt
    }] + state["messages"]
    
    full_text = ""
    tool_calls: list[llm.ToolCall] = []
    
    async for event in llm.stream_chat(messages,tools=tools.TOOL_SCHEMAS, cancel_event=cancel_event):
        if event.type == "text_delta":
            if event.text:
                full_text += event.text
            if on_text_delta is not None:
                await on_text_delta(event.text)
        elif event.type == "tool_call":
            if event.tool_call is not None:
                tool_calls.append(event.tool_call)
        elif event.type == "done":
            break
    assistant_message: dict = {"role": "assistant", "content": full_text}
    if tool_calls:
        assistant_message["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name , "arguments": tc.arguments},
            }
            for tc in tool_calls
        ]
    return {"messages": [assistant_message]}

async def tools_node(state: ConversationState) -> dict:
    """Execute every tool call from the last assistant message over MCP,
    and fold create_appointment's result into `slots` so the system prompt
    (and the eventual DB-persisted state) reflects what's actually booked."""
    last_message = state["messages"][-1]
    tool_calls = last_message.get("tool_calls", [])
    
    tool_messages = []
    slot_updates ={}
    
    for tc in tool_calls:
        name = tc["function"]["name"]
        try:
            arguments = json.loads(tc["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            arguments = {}
        result_text = await tools.call_tool(name,arguments)
        
        tool_messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "name": name,
            "content": result_text,
        })
        try:
            result_json = json.loads(result_text)
        except json.JSONDecodeError:
            result_json ={}
        if name == "create_appointment" and result_json.get("success"):
            slot_updates["booking_complete"] = True
            slot_updates["appointment_id"] = result_json.get("appointment_id")
            
    return {
        "messages": tool_messages,
        "slots": {**state["slots"] ,**slot_updates},
        "tool_loop_count": state["tool_loop_count"] + 1,
    }


def should_continue(state: ConversationState) -> str:
    """The one real conditional edge: does the latest assistant message
    want to call a tool, or are we done for this turn?"""
    last_message = state["messages"][-1]
    if last_message.get("role") != "assistant":
        return "end"
    if last_message.get("tool_calls") and state["tool_loop_count"] < MAX_TOOL_LOOPS :
        return "tools"
    return "end"


#--------Graph Assembly----------

def build_graph():
    graph = StateGraph(ConversationState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools","agent") # loop back after executing tool calls
    
    return graph.compile()

_compiled_graph = build_graph()

async def run_turn(
    conversation_id: str,
    locale: str,
    slots: dict,
    history: list[dict],
    user_text: str,
    on_text_delta,
    cancel_event,
) -> dict:
    """
    Run one conversational turn: append the user's utterance, drive the
    agent/tools graph until the model produces a final (non-tool-call)
    reply, and return the new state for the caller (main.py) to persist
    via db.add_message / db.update_conversation_state.
    """
    initial_state: ConversationState = {
        "messages": history + [{"role": "user", "content": user_text}],
        "locale": locale,
        "slots": slots,
        "tool_loop_count": 0,
    }
    config: RunnableConfig = {"configurable": {"on_text_delta": on_text_delta, "cancel_event": cancel_event}}
    final_state = await _compiled_graph.ainvoke(initial_state,config=config)
    
    final_reply = final_state["messages"][-1]["content"]
    
    return {
        "reply_text": final_reply,
        "slots": final_state["slots"],
        "messages": final_state["messages"],
    }