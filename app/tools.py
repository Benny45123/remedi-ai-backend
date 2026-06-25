"""
tools.py — MCP tool server + client helper for MediAssist's two booking tools.

Architecture note: tools are exposed over MCP (Model Context Protocol)
rather than called as plain Python functions, per the project's standing
preference for LangGraph + MCP over ad-hoc tool-calling glue. This file
does double duty:

  1. Defines the MCP server itself (`mcp_server`, built with FastMCP) and
     its two tools, check_availability and create_appointment. Running
     `python -m app.tools` starts this server on stdio. It's run with -m
     (not as a bare script) specifically so relative imports (`from . import
     db`) resolve correctly when this file executes standalone as a
     subprocess.
  2. Exposes a persistent MCP *client* (start_mcp_client / call_tool /
     stop_mcp_client) that the main backend process uses to talk to that
     server. main.py starts the client once at FastAPI startup (spawning
     this file as a stdio subprocess) and agent.py's LangGraph tool node
     calls through it for every tool invocation, for every conversation —
     one long-lived MCP session shared across the app, not one subprocess
     spun up per call.

Tool *handlers* still just call db.py underneath — MCP is the calling
convention between the LLM-driven agent and the booking logic, not a
reimplementation of the booking logic itself.
"""
import json
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Optional
from uuid import UUID

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.fastmcp import FastMCP

from . import db

# ---------- MCP server: tool definitions ----------

mcp_server = FastMCP("remidiai-tools")

@mcp_server.tool()
async def check_availability(specialization: str) -> str:
    """Get open appointment slots for a specialization.

    Returns a JSON string (MCP tool results are text) with up to 5
    upcoming open slots: [{slot_id, doctor_id, doctor_name, slot_start}, ...].
    """
    slots = await db.get_open_slots(specialization,limit=5)
    return json.dumps([
        {
            "slot_id": str(s["slot_id"]),
            "doctor_id": str(s["doctor_id"]),
            "doctor_name": s["doctor_name"],
            "slot_start": s["slot_start"].isoformat(),
        }
        for s in slots
    ])

@mcp_server.tool()
async def create_appointment(
    patient_name:str, patient_phone: str, doctor_id: str, slot_id: str
) -> str:
    """Book a confirmed appointment slot for a patient.

    Runs the atomic conditional UPDATE from db.book_slot_atomic() first —
    if the slot was already taken by the time this executes, returns a
    structured {"error": "slot_taken", "alternatives": [...]} JSON string
    so the LLM can naturally offer the next slot instead of failing the turn.
    """
    slot_uuid = UUID(slot_id)
    doctor_uuid = UUID(doctor_id)
    
    won = await db.book_slot_atomic(slot_uuid)
    if not won:
        slot = await db.get_slot(slot_uuid)
        specialization_hint = slot["doctor_name"] if slot else ""
        alternatives = await db.get_open_slots(specialization_hint,limit=3)
        return json.dumps({
            "error": "slot_taken",
            "alternatives": [
                {
                    "slot_id": str(s["slot_id"]),
                    "doctor_name": s["doctor_name"],
                    "slot_start": s["slot_start"].isoformat(),
                }
                for s in alternatives
            ],
        })
    try:
        patient_id = await db.create_patient(patient_name,patient_phone)
        appointment_id = await db.create_appointment(patient_id,doctor_uuid,slot_uuid)
    except Exception as exc:
        await db.release_slot(slot_uuid)
        return json.dumps({
            "error": "booking_failed", "detail": str(exc)
        })
    return json.dumps({
        "success": True,
        "appointment_id": str(appointment_id)
    })

# ---------- MCP client: used by the main backend process ----------

_client_session: Optional[ClientSession] = None
_exit_stack: Optional[AsyncExitStack] = None

# Populated from the MCP server's own tool listing in start_mcp_client() —
# deliberately NOT hand-duplicated here. Deriving the Groq/OpenAI-format
# function-calling schemas from the server's real `inputSchema` means
# there's exactly one place (the @mcp_server.tool() defs above) that
# defines a tool's shape; agent.py just reads this list after startup.
TOOL_SCHEMAS: list[dict] = []

def _sanitize_schema(schema: dict) -> dict:
    """
    Strip JSON Schema fields that Groq's API doesn't accept.
    FastMCP generates title, $schema, additionalProperties — Groq rejects
    all of these with a 400. Strip to type/properties/required only.
    """
    allowed = {"type", "properties", "required", "description"}
    clean = {k: v for k, v in schema.items() if k in allowed}

    if "properties" in clean:
        clean["properties"] = {
            prop_name: {k: v for k, v in prop_schema.items() if k in {"type", "description", "enum"}}
            for prop_name, prop_schema in clean["properties"].items()
        }

    clean.setdefault("type", "object")
    return clean
        
async def start_mcp_client() -> None:
    """Spawn this file as a stdio MCP server subprocess and open one
    long-lived ClientSession against it. Call once on FastAPI startup,
    before the app starts accepting WebSocket connections."""
    global _client_session, _exit_stack, TOOL_SCHEMAS
    
    _exit_stack = AsyncExitStack()
    backend_dir = Path(__file__).resolve().parent.parent # .../backend
    server_params = StdioServerParameters(
        command=sys.executable,
        args= ["-m","app.tools"],
        cwd=str(backend_dir),
    )
    read, write = await _exit_stack.enter_async_context(stdio_client(server_params))
    _client_session = await _exit_stack.enter_async_context(ClientSession(read,write))
    await _client_session.initialize()
    
    listing = await _client_session.list_tools()
    TOOL_SCHEMAS =[
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema,
            },
        }
        for t in listing.tools
    ]

async def stop_mcp_client() -> None:
    """Tear down the client session and terminate the server subprocess.
    Call on FastAPI shutdown."""
    global _client_session, _exit_stack
    if _exit_stack is not None:
        await _exit_stack.aclose()
    _client_session = None
    _exit_stack = None
    

async def call_tool(name:str, arguments: dict) -> str:
    """Invoke a tool by name over the persistent MCP session. Returns the
    tool's text result (a JSON string — see the tool docstrings above).
    This is what agent.py's LangGraph tool node calls for every tool_call
    the LLM produces."""
    if _client_session is None:
        raise RuntimeError("MCP client not started - call start_mcp_client() on startup")
    result = await _client_session.call_tool(name,arguments)
    for block in result.content:
        if block.type == "text":
            return block.text
    return ""

if __name__ == "__main__":
    mcp_server.run()
    