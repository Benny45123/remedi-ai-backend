"""
main.py — FastAPI app entrypoint.

Owns:
  - app lifespan: open the asyncpg pool and start the MCP tool-server
    subprocess on startup; close both cleanly on shutdown.
  - the single WebSocket endpoint, ws/voice, implementing the protocol
    table from the build spec (session_start, audio_chunk, transcript,
    ai_response_start, ai_text_delta, audio_output, turn_complete,
    interruption, session_end).
  - reconnect/resume: a client-supplied session_id (= conversations.id)
    reloads conversation state + recent messages instead of starting over.
  - interruption: cancels the in-flight LLM stream and stops sending
    audio_output the moment the client signals the user started talking
    over the AI.
  - a plain, no-auth /admin page listing today's bookings (per §9 step 9
    — explicitly no auth needed for the demo).

One FastAPI service, one WebSocket route, no message queue, no
microservices — matches the "explicitly do NOT build" list in the spec.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional
from uuid import UUID

# Must run before any of this app's own modules are imported below — db.py,
# stt.py, llm.py, and tts.py all read required keys from os.environ at
# import time (e.g. `GROQ_API_KEY = os.environ["GROQ_API_KEY"]`), so for
# local dev (where secrets live in .env rather than real env vars) this has
# to populate os.environ first or those imports raise KeyError immediately.
# In production (Railway/Fly), real env vars are already set and this is a
# harmless no-op since there's no .env file present.
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from app import db, stt, tts, tools, agent
from app.lang import detect_locale, DEFAULT_LOCALE
from app.translate import translate

logging.basicConfig(
    level=logging.DEBUG,  # set to INFO in prod; DEBUG shows Sarvam response bodies
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("mediassist.main")

# Generic, locale-agnostic line spoken when a provider call fails twice in
# a row (§8: "speak a graceful fallback line instead of letting the
# session hang or crash"). Kept in English deliberately — if TTS itself is
# what's broken, we still want *some* audio to come back, and we don't
# want the fallback path to depend on locale-specific synthesis succeeding
# when the whole point is that synthesis is unreliable right now.
FALLBACK_LINE = "Sorry, one moment — let me try that again."


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    await tools.start_mcp_client()
    logger.info("MediAssist backend ready: DB pool + MCP tool client started.")
    yield
    await tools.stop_mcp_client()
    await db.close_db()
    logger.info("MediAssist backend shut down cleanly.")


app = FastAPI(title="MediAssist", lifespan=lifespan)


# ---------- WebSocket session state ----------

class VoiceSession:
    """Per-connection state for one /ws/voice WebSocket. Not shared across
    connections — each browser tab gets its own instance."""

    def __init__(self, websocket: WebSocket):
        self.ws = websocket
        self.conversation_id: Optional[UUID] = None
        self.locale: str = DEFAULT_LOCALE
        self.slots: dict = {}
        self.history: list[dict] = []  # OpenAI-style role/content dicts for the LLM
        self.cancel_event: asyncio.Event = asyncio.Event()
        self.ai_speaking: bool = False

    async def send_json(self, event: str, payload: dict) -> None:
        await self.ws.send_text(json.dumps({"event": event, **payload}))

    async def send_audio(self, audio_bytes: bytes) -> None:
        await self.ws.send_bytes(audio_bytes)


# ---------- Session lifecycle: start / resume ----------

async def handle_session_start(session: VoiceSession, payload: dict) -> None:
    """Per §8 reconnect/resume: if the client sends a session_id we
    recognize, reload conversation state + recent messages and continue;
    otherwise start a fresh conversation row.

    Optional `locale` field in the payload lets the client set the
    starting language explicitly: {"event":"session_start","locale":"te"}
    defaults to Telugu; "en"/"hi"/"te" are accepted.
    """
    # Client can pin the starting locale — useful for a Telugu-first demo
    # without changing the server default.
    requested_locale = payload.get("locale")
    from app.lang import SUPPORTED_LOCALES
    if requested_locale in SUPPORTED_LOCALES:
        session.locale = requested_locale

    incoming_id = payload.get("session_id")

    if incoming_id:
        try:
            existing = await db.get_conversation(UUID(incoming_id))
        except (ValueError, TypeError):
            existing = None

        if existing is not None:
            session.conversation_id = existing["id"]
            session.locale = existing["locale"] or DEFAULT_LOCALE
            session.slots = existing["state"] or {}
            recent = await db.get_recent_messages(session.conversation_id, limit=20)
            session.history = [
                {"role": m["role"], "content": m["text"]} for m in recent
            ]
            await session.send_json(
                "session_start",
                {"session_id": str(session.conversation_id), "resumed": True},
            )
            return

    session.conversation_id = await db.create_conversation(locale=DEFAULT_LOCALE)
    await session.send_json(
        "session_start", {"session_id": str(session.conversation_id), "resumed": False}
    )


# ---------- Turn handling: the STT -> agent -> TTS pipeline ----------

async def handle_audio_turn(session: VoiceSession, audio_bytes: bytes) -> None:
    """One full turn: transcribe the buffered utterance, run it through the
    LangGraph agent (streaming text deltas live), synthesize the reply,
    and persist everything. Bails out cleanly if interrupted mid-flight."""
    session.cancel_event.clear()

    # --- STT ---
    try:
        stt_result = await stt.transcribe(audio_bytes, language_hint=session.locale)
    except stt.STTError as exc:
        logger.warning("STT failed for conversation %s: %s", session.conversation_id, exc)
        await _speak_fallback(session)
        return

    transcript_text = stt_result["text"]
    if not transcript_text:
        return  # empty utterance (likely a VAD false trigger) — nothing to do

    session.locale = detect_locale(transcript_text, stt_result.get("language"), session.locale)
    await session.send_json("transcript", {"text": transcript_text, "locale": session.locale})
    await db.add_message(session.conversation_id, "user", transcript_text, session.locale)

    # --- Agent (LangGraph: LLM + MCP tool loop), streamed ---
    await session.send_json("ai_response_start", {})
    session.ai_speaking = True

    sentence_buffer = ""

    async def on_text_delta(delta: str) -> None:
        nonlocal sentence_buffer
        if session.cancel_event.is_set():
            return
        await session.send_json("ai_text_delta", {"delta": delta})
        sentence_buffer += delta
        # Speak in sentence-sized chunks rather than waiting for the whole
        # reply — keeps latency down without needing token-level TTS.
        if delta.strip().endswith((".", "?", "!", "।")):  # "।" = Hindi/Telugu sentence end
            chunk, sentence_buffer = sentence_buffer, ""
            await _speak_chunk(session, chunk)

    try:
        result = await agent.run_turn(
            conversation_id=str(session.conversation_id),
            locale=session.locale,
            slots=session.slots,
            history=session.history,
            user_text=transcript_text,
            on_text_delta=on_text_delta,
            cancel_event=session.cancel_event,
        )
    except Exception as exc:  # noqa: BLE001 — agent/LLM failure, not a crash-the-session case
        logger.warning("Agent turn failed for conversation %s: %s", session.conversation_id, exc)
        session.ai_speaking = False
        await _speak_fallback(session)
        return

    if session.cancel_event.is_set():
        # Interrupted mid-turn — don't speak whatever's left in the buffer,
        # don't persist a reply that was never fully delivered.
        session.ai_speaking = False
        return

    # Translate the complete English reply ONCE — reuse for both DB storage
    # and any trailing sentence not yet spoken. This replaces per-sentence
    # translation in _speak_chunk, saving one Groq call per sentence chunk.
    reply_english = result["reply_text"]
    reply_translated = await translate(reply_english, session.locale)

    # Speak any trailing partial sentence that didn't end on punctuation.
    if sentence_buffer.strip():
        tail_translated = await translate(sentence_buffer, session.locale)
        await _speak_translated_chunk(session, tail_translated)

    session.ai_speaking = False
    session.slots = result["slots"]
    session.history = result["messages"]

    # Store English (what LLM said) + translated (what patient heard) separately
    # so both are queryable from Supabase. translated_text is NULL for English sessions.
    await db.add_message(
        session.conversation_id,
        "assistant",
        reply_english,
        session.locale,
        translated_text=reply_translated if reply_translated != reply_english else None,
    )
    await db.update_conversation_state(session.conversation_id, session.slots)
    await db.update_conversation_locale(session.conversation_id, session.locale)

    await session.send_json(
        "turn_complete", {"state": "ok", "slots": session.slots}
    )


async def _speak_chunk(session: VoiceSession, text: str) -> None:
    """Translate English text then speak. Used for per-sentence chunks
    during streaming (on_text_delta sentence splits)."""
    if session.cancel_event.is_set() or not text.strip():
        return
    try:
        tts_text = await translate(text, session.locale)
        audio_bytes = await tts.synthesize(tts_text, session.locale)
    except tts.TTSError as exc:
        logger.warning("TTS failed for conversation %s: %s", session.conversation_id, exc)
        return
    if not session.cancel_event.is_set():
        await session.send_audio(audio_bytes)


async def _speak_translated_chunk(session: VoiceSession, translated_text: str) -> None:
    """Speak already-translated text — skips translation step.
    Used for the trailing sentence buffer at end of turn where full-reply
    translation has already been done once above."""
    if session.cancel_event.is_set() or not translated_text.strip():
        return
    try:
        audio_bytes = await tts.synthesize(translated_text, session.locale)
    except tts.TTSError as exc:
        logger.warning("TTS failed for conversation %s: %s", session.conversation_id, exc)
        return
    if not session.cancel_event.is_set():
        await session.send_audio(audio_bytes)


async def _speak_fallback(session: VoiceSession) -> None:
    """STT/agent failed outright for this turn — say something rather than
    going silent, per §8's provider-failure rule."""
    await session.send_json("ai_response_start", {})
    await session.send_json("ai_text_delta", {"delta": FALLBACK_LINE})
    try:
        audio_bytes = await tts.synthesize(FALLBACK_LINE, session.locale)
        await session.send_audio(audio_bytes)
    except tts.TTSError:
        pass  # even the fallback line failed to synthesize — caption alone will have to do
    await session.send_json("turn_complete", {"state": "error", "slots": session.slots})


# ---------- WebSocket route ----------

@app.websocket("/ws/voice")
async def ws_voice(websocket: WebSocket):
    await websocket.accept()
    session = VoiceSession(websocket)
    audio_buffer = bytearray()

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            if "text" in message and message["text"] is not None:
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                event = data.get("event")

                if event == "session_start":
                    await handle_session_start(session, data)

                elif event == "interruption":
                    # User started talking over the AI: stop the in-flight
                    # LLM stream and any further audio_output immediately.
                    session.cancel_event.set()
                    session.ai_speaking = False

                elif event == "session_end":
                    if session.conversation_id:
                        await db.end_conversation(session.conversation_id)
                    break

            elif "bytes" in message and message["bytes"] is not None:
                # Per the spec's VAD model: the client buffers one full
                # utterance client-side and sends it as a single binary
                # frame once VAD detects end-of-speech — so each binary
                # frame here IS one complete turn's audio, not a stream
                # of small chunks needing reassembly.
                if session.conversation_id is None:
                    # Defensive: client should always send session_start
                    # first, but don't crash if audio arrives before it.
                    session.conversation_id = await db.create_conversation(
                        locale=DEFAULT_LOCALE
                    )
                await session.send_json("speech_started", {})
                await handle_audio_turn(session, bytes(message["bytes"]))

    except WebSocketDisconnect:
        pass
    finally:
        # A dropped connection is not a session_end — conversation state is
        # already persisted turn-by-turn, so a later reconnect with the
        # same session_id picks back up cleanly per §8.
        logger.info("WebSocket closed for conversation %s", session.conversation_id)


# ---------- No-auth /admin page (per §9 step 9 — explicitly no auth for the demo) ----------

@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    appointments = await db.list_todays_appointments()

    rows = "\n".join(
        f"<tr><td>{a['patient_name']}</td><td>{a['phone']}</td>"
        f"<td>{a['doctor_name']}</td><td>{a['slot_start']}</td>"
        f"<td>{a['created_at']}</td></tr>"
        for a in appointments
    )
    if not rows:
        rows = "<tr><td colspan='5' style='text-align:center;color:#888'>No bookings yet today</td></tr>"

    html = f"""
    <html>
      <head>
        <title>MediAssist — Today's Bookings</title>
        <style>
          body {{ font-family: -apple-system, sans-serif; max-width: 800px; margin: 40px auto; }}
          table {{ width: 100%; border-collapse: collapse; }}
          th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #eee; }}
          th {{ color: #555; font-weight: 600; }}
        </style>
      </head>
      <body>
        <h2>Today's Bookings</h2>
        <table>
          <tr><th>Patient</th><th>Phone</th><th>Doctor</th><th>Time</th><th>Booked at</th></tr>
          {rows}
        </table>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/health")
async def health():
    return {"status": "ok"}



# """
# main.py — FastAPI app entrypoint.

# Owns:
#   - app lifespan: open the asyncpg pool and start the MCP tool-server
#     subprocess on startup; close both cleanly on shutdown.
#   - the single WebSocket endpoint, ws/voice, implementing the protocol
#     table from the build spec (session_start, audio_chunk, transcript,
#     ai_response_start, ai_text_delta, audio_output, turn_complete,
#     interruption, session_end).
#   - reconnect/resume: a client-supplied session_id (= conversations.id)
#     reloads conversation state + recent messages instead of starting over.
#   - interruption: cancels the in-flight LLM stream and stops sending
#     audio_output the moment the client signals the user started talking
#     over the AI.
#   - a plain, no-auth /admin page listing today's bookings (per §9 step 9
#     — explicitly no auth needed for the demo).

# One FastAPI service, one WebSocket route, no message queue, no
# microservices — matches the "explicitly do NOT build" list in the spec.
# """

# import asyncio
# import json
# import logging
# from contextlib import asynccontextmanager
# from typing import Optional
# from uuid import UUID


# from dotenv import load_dotenv

# load_dotenv()

# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# from fastapi.responses import HTMLResponse

# from app import db, stt, tts, tools, agent
# from app.lang import detect_locale, DEFAULT_LOCALE

# logging.basicConfig(
#     level=logging.DEBUG,  # set to INFO in prod; DEBUG shows Sarvam response bodies
#     format="%(levelname)s %(name)s: %(message)s",
# )
# from app.translate import translate

# logger = logging.getLogger("mediassist.main")

# FALLBACK_LINE = "Sorry, one moment — let me try that again."


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     await db.init_db()
#     await tools.start_mcp_client()
#     logger.info("MediAssist backend ready: DB pool + MCP tool client started.")
#     yield
#     await tools.stop_mcp_client()
#     await db.close_db()
#     logger.info("MediAssist backend shut down cleanly.")


# app = FastAPI(title="MediAssist", lifespan=lifespan)


# class VoiceSession:
#     """Per-connection state for one /ws/voice WebSocket."""

#     def __init__(self, websocket: WebSocket):
#         self.ws = websocket
#         self.conversation_id: Optional[UUID] = None
#         self.locale: str = DEFAULT_LOCALE
#         self.slots: dict = {}
#         self.history: list[dict] = []
#         self.cancel_event: asyncio.Event = asyncio.Event()
#         self.ai_speaking: bool = False

#     async def send_json(self, event: str, payload: dict) -> None:
#         await self.ws.send_text(json.dumps({"event": event, **payload}))

#     async def send_audio(self, audio_bytes: bytes) -> None:
#         await self.ws.send_bytes(audio_bytes)


# async def handle_session_start(session: VoiceSession, payload: dict) -> None:
#     incoming_id = payload.get("session_id")

#     if incoming_id:
#         try:
#             existing = await db.get_conversation(UUID(incoming_id))
#         except (ValueError, TypeError):
#             existing = None

#         if existing is not None:
#             session.conversation_id = existing["id"]
#             session.locale = existing["locale"] or DEFAULT_LOCALE
#             session.slots = existing["state"] or {}
#             recent = await db.get_recent_messages(session.conversation_id, limit=20)
#             session.history = [
#                 {"role": m["role"], "content": m["text"]} for m in recent
#             ]
#             await session.send_json(
#                 "session_start",
#                 {"session_id": str(session.conversation_id), "resumed": True},
#             )
#             return

#     session.conversation_id = await db.create_conversation(locale=DEFAULT_LOCALE)
#     await session.send_json(
#         "session_start", {"session_id": str(session.conversation_id), "resumed": False}
#     )


# async def handle_audio_turn(session: VoiceSession, audio_bytes: bytes) -> None:
#     session.cancel_event.clear()

#     try:
#         stt_result = await stt.transcribe(audio_bytes, language_hint=session.locale)
#     except stt.STTError as exc:
#         logger.warning("STT failed for conversation %s: %s", session.conversation_id, exc)
#         await _speak_fallback(session)
#         return

#     transcript_text = stt_result["text"]
#     if not transcript_text:
#         return

#     session.locale = detect_locale(transcript_text, stt_result.get("language"), session.locale)
#     await session.send_json("transcript", {"text": transcript_text, "locale": session.locale})
#     await db.add_message(session.conversation_id, "user", transcript_text, session.locale)

#     await session.send_json("ai_response_start", {})
#     session.ai_speaking = True

#     sentence_buffer = ""

#     async def on_text_delta(delta: str) -> None:
#         nonlocal sentence_buffer
#         if session.cancel_event.is_set():
#             return
#         await session.send_json("ai_text_delta", {"delta": delta})
#         sentence_buffer += delta
#         if delta.strip().endswith((".", "?", "!", "।")):
#             chunk, sentence_buffer = sentence_buffer, ""
#             await _speak_chunk(session, chunk)

#     try:
#         result = await agent.run_turn(
#             conversation_id=str(session.conversation_id),
#             locale=session.locale,
#             slots=session.slots,
#             history=session.history,
#             user_text=transcript_text,
#             on_text_delta=on_text_delta,
#             cancel_event=session.cancel_event,
#         )
#     except Exception as exc:  # noqa: BLE001
#         logger.warning("Agent turn failed for conversation %s: %s", session.conversation_id, exc)
#         session.ai_speaking = False
#         await _speak_fallback(session)
#         return

#     if session.cancel_event.is_set():
#         session.ai_speaking = False
#         return

#     if sentence_buffer.strip():
#         await _speak_chunk(session, sentence_buffer)

#     session.ai_speaking = False
#     session.slots = result["slots"]
#     session.history = result["messages"]

#     await db.add_message(
#         session.conversation_id, "assistant", result["reply_text"], session.locale
#     )
#     await db.update_conversation_state(session.conversation_id, session.slots)
#     await db.update_conversation_locale(session.conversation_id, session.locale)

#     await session.send_json("turn_complete", {"state": "ok", "slots": session.slots})


# async def _speak_chunk(session: VoiceSession, text: str) -> None:
#     if session.cancel_event.is_set() or not text.strip():
#         return
#     try:
#         tts_text = await translate(text, session.locale)
#         audio_bytes = await tts.synthesize(tts_text, session.locale)
#     except tts.TTSError as exc:
#         logger.warning("TTS failed for conversation %s: %s", session.conversation_id, exc)
#         return
#     if not session.cancel_event.is_set():
#         await session.send_audio(audio_bytes)


# async def _speak_fallback(session: VoiceSession) -> None:
#     await session.send_json("ai_response_start", {})
#     await session.send_json("ai_text_delta", {"delta": FALLBACK_LINE})
#     try:
#         audio_bytes = await tts.synthesize(FALLBACK_LINE, session.locale)
#         await session.send_audio(audio_bytes)
#     except tts.TTSError:
#         pass
#     await session.send_json("turn_complete", {"state": "error", "slots": session.slots})


# @app.websocket("/ws/voice")
# async def ws_voice(websocket: WebSocket):
#     await websocket.accept()
#     session = VoiceSession(websocket)

#     try:
#         while True:
#             message = await websocket.receive()

#             if message["type"] == "websocket.disconnect":
#                 break

#             if "text" in message and message["text"] is not None:
#                 try:
#                     data = json.loads(message["text"])
#                 except json.JSONDecodeError:
#                     continue
#                 event = data.get("event")

#                 if event == "session_start":
#                     await handle_session_start(session, data)

#                 elif event == "interruption":
#                     session.cancel_event.set()
#                     session.ai_speaking = False

#                 elif event == "session_end":
#                     if session.conversation_id:
#                         await db.end_conversation(session.conversation_id)
#                     break

#             elif "bytes" in message and message["bytes"] is not None:
#                 if session.conversation_id is None:
#                     session.conversation_id = await db.create_conversation(
#                         locale=DEFAULT_LOCALE
#                     )
#                 await session.send_json("speech_started", {})
#                 await handle_audio_turn(session, bytes(message["bytes"]))

#     except WebSocketDisconnect:
#         pass
#     finally:
#         logger.info("WebSocket closed for conversation %s", session.conversation_id)


# @app.get("/admin", response_class=HTMLResponse)
# async def admin_page():
#     appointments = await db.list_todays_appointments()

#     rows = "\n".join(
#         f"<tr><td>{a['patient_name']}</td><td>{a['phone']}</td>"
#         f"<td>{a['doctor_name']}</td><td>{a['slot_start']}</td>"
#         f"<td>{a['created_at']}</td></tr>"
#         for a in appointments
#     )
#     if not rows:
#         rows = "<tr><td colspan='5' style='text-align:center;color:#888'>No bookings yet today</td></tr>"

#     html = f"""
#     <html>
#       <head>
#         <title>MediAssist — Today's Bookings</title>
#         <style>
#           body {{ font-family: -apple-system, sans-serif; max-width: 800px; margin: 40px auto; }}
#           table {{ width: 100%; border-collapse: collapse; }}
#           th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #eee; }}
#           th {{ color: #555; font-weight: 600; }}
#         </style>
#       </head>
#       <body>
#         <h2>Today's Bookings</h2>
#         <table>
#           <tr><th>Patient</th><th>Phone</th><th>Doctor</th><th>Time</th><th>Booked at</th></tr>
#           {rows}
#         </table>
#       </body>
#     </html>
#     """
#     return HTMLResponse(content=html)


# @app.get("/health")
# async def health():
#     return {"status": "ok"}