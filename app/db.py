"""
db.py — Database access layer for MediAssist backend.

Direct asyncpg against Supabase Postgres (no ORM, per spec — this is a
small enough schema that an ORM would just add setup time for no benefit).

Responsibilities:
  - connection pool lifecycle (init on FastAPI startup, close on shutdown)
  - doctor/slot/patient/appointment queries
  - the conditional-UPDATE booking pattern that prevents double-booking
  - conversation + message persistence, used for reconnect/resume
"""


import json
import os
from typing import Optional
from uuid import UUID

import asyncpg

DATABASE_URL = os.environ["DATABASE_URL"]  #SUPABASE Conn string

_pool: Optional[asyncpg.Pool] = None

async def init_db() ->None :
    """Create Global Connection Pool Once on FastAPI Start"""
    global _pool
    _pool = await asyncpg.create_pool(DATABASE_URL,min_size=2,max_size=10)
    

async def close_db() -> None:
    """Close the pool. Call on FastAPI shutdown."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None

def get_pool() -> asyncpg.Pool :
    if _pool is None:
        raise RuntimeError("DB pool not initialized — call init_db() on startup")
    return _pool


# ---------- Doctors / slots ----------

async def get_open_slots(specialization: str,limit: int = 5) ->list[dict]:
    """Open slots for a specialization, soonest first."""
    query = """
        SELECT s.id AS slot_id , s.doctor_id , s.slot_start ,d.name as doctor_name 
        FROM doctor_slots s
        JOIN doctors d ON d.id = s.doctor_id 
        WHERE s.status = 'open' AND d.specialization ILIKE $1
        ORDER BY s.slot_start ASC
        LIMIT $2
    """
    async with get_pool().acquire() as conn:
        rows = await conn.fetch(query,specialization,limit)
    return [dict(r) for r in rows]


async def book_slot_atomic(slot_id: UUID) -> bool :
    """
    The core double-booking guard. A single conditional UPDATE — no
    separate read-then-write, so there's no window for a race.
    Returns True if this call won the race and booked the slot,
    False if it was already taken (0 rows affected).
    """
    query = """
        UPDATE doctor_slots
        SET status= 'booked'
        WHERE id = $1 AND status = 'open'
    """
    async with get_pool().acquire() as conn:
        result = await conn.execute(query,slot_id)
    return result.endsWith(" 1")

async def release_slot(slot_id: UUID) ->None :
    """Roll a slot back to 'open' if appointment creation fails after it was booked."""
    query = "UPDATE doctor_slots SET status = 'open' WHERE id = $1"
    async with get_pool().acquire() as conn:
        await conn.execute(query,slot_id)
        
async def get_slot(slot_id: UUID) -> Optional(dict):
    query = """
        SELECT s.id AS slot_id , s.doctor_id , s.slot_start , s.status , d.name AS doctor_name
        FROM doctor_slots s JOIN doctors d ON d.id = s.doctor_id
    """
    async with get_pool().acquire() as conn :
        row = await conn.fetchrow(query, slot_id)
    return dict(row) if row else None 



# ---------- Patients / appointments ----------


async def create_patient(name: str, phone : str) -> UUID:
    query = "INSERT INTO patients (name , phone) VALUES ($1,$2) RETURNING id"
    async with get_pool().acquire() as conn:
        row = conn.fetchrow(query,name,phone)
    return row["id"]


async def list_today_appointments() -> list[dict] :
    """Backs the no-auth /admin page — today's bookings, plain and simple."""
    query = """
        SELECT a.id , p.name AS patient_name , p.phone , d.name AS doctor_name,
            s.slot_start , a.created_at
        FROM appointments a
        JOIN patients p ON p.id = a.patient_id
        JOIN doctors d ON d.id = a.doctor_id
        JOIN doctor_slots s ON s.id = a.slot_id
        WHERE s.slot_start :: date = now()::date
        ORDER BY s.slot_start ASC
    """
    async with get_pool().acquire() as conn:
        rows = await conn.fetch(query)
    return [dict(r) for r in rows]


# ---------- Conversations (reconnect/resume) ----------

async def create_conversation(locale:str = "en") ->UUID :
    query = "INSERT INTO conversations (locale) VALUES ($1) RETURNING id"
    async with get_pool().acquire() as conn:
        row = conn.fetchrow(query,locale)
    return row["id"]

async def get_conversation(conversation_id: UUID) ->Optional(dict):
    query = "SELECT id,locale,state,started_at,ended_at FROM conversations WHERE id = $1"
    async with get_pool().acquire() as conn:
        row = await conn.fetchrow(query,conversation_id)
    if not row:
        return None
    d=dict(row)
    d["state"] = json.loads(d["state"]) if isinstance(d["state"], str) else d["state"]
    return d

async def update_conversation_state(conversation_id: UUID, state: dict) -> None:
    query = "UPDATE conversations SET state = $2 WHERE id = $1"
    async with get_pool().acquire() as conn:
        await conn.execute(query,conversation_id,json.dumps(state))


async def update_conversation_locale(conversation_id: UUID, locale: str) ->None:
    query = "UPDATE conversations SET locale = $2 WHERE id = $1"
    async with get_pool().acquire() as conn:
        await conn.execute(query,conversation_id,locale)


async def end_conversation(conversation_id: UUID) -> None :
    query = "UPDATE conversations SET ended_at = now() WHERE id = $1"
    async with get_pool().acquire() as conn:
        await conn.execute(query, conversation_id)
        

async def add_message(
    conversation_id: UUID,
    role: str,
    text: str,
    locale: Optional[str] = None
) ->None :
    query = """
        INSERT INTO conversation_messages (conversation_id, role, text, locale)
        VALUES ($1,$2,$3,$4)
    """
    async with get_pool().acquire() as conn:
        await conn.execute(query, conversation_id, role, text, locale)
        

async def get_recent_messages(conversation_id: UUID, limit: int = 20) -> list[dict]:
    """Most recent messages, returned oldest-first so they replay cleanly into LLM context."""
    query = """
        SELECT role,text,locale,ts FROM conversation_messages
        WHERE conversation_id = $1
        ORDER BY ts DESC LIMIT $2
    """
    async with get_pool().acquire() as conn:
        rows = await conn.fetch(query,conversation_id,limit)
    return [dict(r) for r in reversed(rows)]


