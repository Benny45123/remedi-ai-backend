"""
llm.py — Groq chat completions wrapper (streaming).

Groq's chat endpoint is OpenAI-compatible, so this uses the same
streaming SSE protocol and the same `tools` schema OpenAI uses for
function calling. agent.py owns the tool-call loop (deciding when to
execute a tool and feed its result back); this module just exposes a
single streaming call that yields text deltas and completed tool calls.
"""

import asyncio
import json
import os
from dataclasses import dataclass
from typing import AsyncIterator,Optional

import httpx
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
LLM_MODEL = "llama-3.3-70b-versatile"


class LLMError(Exception):
    """Raised when the chat completion fails after the retry attempt."""
    
@dataclass
class ToolCall:
    id: str
    name: str 
    arguments: str # raw JSON string; agent.py parses it once it's complete
    

@dataclass
class StreamEvent:
    """One unit yielded from stream_chat(): a text delta, a completed
    tool call, or the terminal 'done' signal with the finish reason."""
    type: str # "text_delta" | "tool_call" | "done"
    text: Optional[str] = None
    tool_call: Optional[ToolCall] = None
    finish_reason: Optional[str] = None
    

async def stream_chat(
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    cancel_event: Optional[asyncio.Event] = None,
) -> AsyncIterator[StreamEvent] :
    """
    Stream a chat completion from Groq.

    `cancel_event`, if provided, is checked between chunks — main.py sets
    this when an `interruption` event arrives, so an in-flight generation
    stops producing further text/audio immediately instead of finishing
    a reply nobody's listening to anymore.

    Retries once (~300ms backoff) only on the connection attempt itself,
    per the reliability rules in §8. Once tokens have started streaming we
    don't restart mid-reply — we stop cleanly and let the caller (agent.py)
    decide whether to retry the turn.
    """
    last_exec: Optional[Exception] = None
    for attempt in range(2):
        try:
            async for event in _stream_once(messages,tools,cancel_event):
                yield event
            return
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            last_exec = exc
            if attempt==0:
                await asyncio.sleep(0.3)
                continue
        raise LLMError(f"Groq chat completion failed: {last_exec}")
    

async def _stream_once(
    messages: list[dict],
    tools: Optional[list[dict]],
    cancel_event: Optional[asyncio.Event],
) -> AsyncIterator[StreamEvent] :
    body ={
        "model": LLM_MODEL,
        "messages": messages,
        "stream": True,
        "temperature": 0.5, 
    }
    if tools:
        body["tools"]= tools
        body["tools_choice"] = "auto"
    
    pending_tool_calls: dict[int, dict] = {}
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream(
            "POST",
            GROQ_CHAT_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json=body,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if cancel_event is not None and cancel_event.is_set():
                    return # interruption — stop consuming the stream entirely

                if not line or not line.startswith("data: "):
                    continue
                raw = line[len("data: "):]
                if raw.strip() == "[DONE]":
                    break
                
                chunk = json.loads(raw)
                choice = chunk["choices"][0]
                delta = choice.get("delta", {})
                
                if delta.get("content"):
                    yield StreamEvent(type = "text/delta", text=delta["content"])
                
                for tc in delta.get("tool_calls", []) or []:
                    idx = tc["index"]
                    entry = pending_tool_calls.setdefault(
                        idx,{"id": None, "name": None, "arguments": ""}
                    )
                    if tc.get("id"):
                        entry["id"] = tc["id"]
                    fn = tc.get("function", {})
                    if fn.get("name"):
                        entry["name"] = fn["name"]
                    if fn.get("arguments"):
                        entry["arguments"] += fn["arguments"]
                        
                finish_reason = choice.get("finish_reason")
                if finish_reason:
                    for entry in pending_tool_calls.values():
                        yield StreamEvent(
                            type="tool_call",
                            tool_call=ToolCall(
                                id=entry["id"],
                                name = entry["name"],
                                arguments = entry["arguments"],
                            ),
                        )
                    yield StreamEvent(type="done", finish_reason=finish_reason)
                    return
