"""
stt.py — Speech-to-text via Groq's Whisper endpoint.

Important: Groq's Whisper API is a fast *batch* endpoint — you POST one
complete audio clip and get a transcript back. It is NOT a token-streaming
API. So there's no chunked partial-transcript handling here on purpose:
the client does VAD to detect end-of-utterance, buffers that one utterance,
and main.py sends the whole clip to transcribe() in one call. Groq's
inference is fast enough (often well under 500ms) that this still feels
real-time without the complexity of a streaming STT integration.
"""

import asyncio
import io
import os
from typing import Optional

import httpx

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
STT_MODEL = "whisper-large-v3-turbo"

SUPPORTED_LANGUAGE_HINTS = {"en", "hi", "te"}


class STTError(Exception):
    """Raised when transcription fails after the retry attempt."""


async def transcribe(audio_bytes: bytes, language_hint: Optional[str] = None) -> dict:
    """
    Transcribe one buffered utterance (raw audio bytes — e.g. a WAV container
    produced client-side once VAD detects the user stopped talking).

    Returns: {"text": str, "language": str} — `language` is Whisper's own
    detection (via verbose_json), which we use to drive locale
    auto-switching even when no hint was supplied.

    Retries once on failure (~300ms backoff) per the reliability spec in §8;
    raises STTError if both attempts fail so the caller can speak a fallback
    line instead of hanging the session.
    """
    for attempt in range(2):
        try:
            return await _call_groq_whisper(audio_bytes, language_hint)
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            if attempt == 0:
                await asyncio.sleep(0.3)
                continue
            raise STTError(f"Groq Whisper transcription failed: {exc}") from exc
    raise STTError("Groq Whisper transcription failed: exhausted retries")


async def _call_groq_whisper(audio_bytes: bytes, language_hint: Optional[str]) -> dict:
    files = {"file": ("utterance.wav", io.BytesIO(audio_bytes), "audio/wav")}
    data = {
        "model": STT_MODEL,
        "response_format": "verbose_json",  # gives us back Whisper's detected `language`
    }
    if language_hint and language_hint in SUPPORTED_LANGUAGE_HINTS:
        data["language"] = language_hint

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(
            GROQ_STT_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files=files,
            data=data,
        )
        response.raise_for_status()
        payload = response.json()

    return {
        "text": payload.get("text", "").strip(),
        "language": payload.get("language", language_hint or "te"),
    }


# """
# stt.py — Speech-to-text via Groq's Whisper endpoint.

# Important: Groq's Whisper API is a fast *batch* endpoint — you POST one
# complete audio clip and get a transcript back. It is NOT a token-streaming
# API. So there's no chunked partial-transcript handling here on purpose:
# the client does VAD to detect end-of-utterance, buffers that one utterance,
# and main.py sends the whole clip to transcribe() in one call. Groq's
# inference is fast enough (often well under 500ms) that this still feels
# real-time without the complexity of a streaming STT integration.
# """
# import asyncio
# import io
# import os
# from typing import Optional

# import httpx

# GROQ_API_KEY = os.environ["GROQ_API_KEY"]
# GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
# STT_MODEL = "whisper-large-v3-turbo"

# SUPPORTED_LANGUAGE_HINTS = {"en", "hi", "te"}

# class STTError(Exception):
#     """Raised when transcription fails after the retry attempt."""
#     async def transcribe(audio_bytes: bytes, language_hint: Optional[str] = None) -> dict:
#         """
#         Transcribe one buffered utterance (raw audio bytes — e.g. a WAV container
#         produced client-side once VAD detects the user stopped talking).

#         Returns: {"text": str, "language": str} — `language` is Whisper's own
#         detection (via verbose_json), which we use to drive locale
#         auto-switching even when no hint was supplied.

#         Retries once on failure (~300ms backoff) per the reliability spec in §8;
#         raises STTError if both attempts fail so the caller can speak a fallback
#         line instead of hanging the session.
#         """
#         for attempt in range(2):
#             try:
#                 return await _call_groq_whisper(audio_bytes, language_hint)
#             except (httpx.HTTPError,httpx.TimeoutException) as exc:
#                 if attempt ==0:
#                     await asyncio.sleep(0.3)
#                     continue
#                 raise STTError(f"Groq Whisper transcription failed: {exc}") from exc
#         raise STTError("Groq Whisper transcription failed: exhausted retries")


# async def _call_groq_whisper(audio_bytes: bytes, language_hint: Optional[str]) -> dict:
#     files = {"file": ("utterance.wav", io.BytesIO(audio_bytes), "audio/wav")}
#     data ={
#         "model": STT_MODEL,
#         "response_format": "verbose_json" #gives back whisper's detected language
#     }
#     if language_hint and language_hint in SUPPORTED_LANGUAGE_HINTS:
#         data["language"] = language_hint
    
#     async with httpx.AsyncClient(timeout=15.0) as client:
#         response = await client.post(
#             GROQ_STT_URL,
#             headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
#             files=files,
#             data=data
#         )
#         response.raise_for_status()
#         payload = response.json()
#     return {
#         "text": payload.get("text", "").strip(),
#         "language": payload.get("language",language_hint or "te"),
#     }