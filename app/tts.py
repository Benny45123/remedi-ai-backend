"""
tts.py — Text-to-speech wrapper. Single public entrypoint: synthesize().

Three-tier fallback chain, in order:
  1. Sarvam AI (Bulbul) — primary. Purpose-built for Indian languages;
     this is the one that actually sounds right for Hindi/Telugu, which
     matters more for this demo than raw latency.
  2. Azure Cognitive Services Speech — secondary. Used if Sarvam errors,
     rate-limits, or its credits run out. Has reliable en-IN/hi-IN/te-IN
     neural voices.
  3. Piper (local, free, offline) — last-resort tier for when BOTH paid
     providers are unavailable (e.g. both out of credits during a demo
     with no time to top up). Runs entirely on-box via the `piper`
     binary, no API key, no per-call cost.

     Important limitation, documented here so it isn't a surprise live:
     Piper has no Telugu voice. Its supported Indic coverage is
     effectively Hindi (hi_IN) and English. So for locale="te", this
     tier transparently falls back to the English Piper voice rather
     than failing outright — degraded quality beats dead silence, but
     it will NOT actually speak Telugu. If Sarvam+Azure are both down
     during a Telugu demo, the honest expectation is "English audio with
     a warning logged", not "Telugu audio for free."

Despite three tiers, this stays a single wrapper function per the
"no generic multi-provider abstraction layer" guidance — there's no
plugin registry, just a fixed try/except cascade. Swapping or
reordering providers later is a small edit to synthesize(), not an
architecture change.
"""


import asyncio
import base64
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


# ---------- Provider config ----------

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY")
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
SARVAM_MODEL = os.environ.get("SARVAM_TTS_MODEL", "bulbul:v3")
SARVAM_LOCALE_MAP = {"te": "te-IN", "hi": "hi-IN", "en": "en-IN"}
SARVAM_SPEAKER_MAP = {"te": "priya", "en": "ritu", "hi": "roopa"}

AZURE_SPEECH_KEY = os.environ.get("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION", "centralindia")
AZURE_VOICE_MAP = {
    "en": "en-IN-NeerjaNeural",
    "hi": "hi-IN-SwaraNeural",
    "te": "te-IN-ShrutiNeural",
}
PIPER_BINARY = os.environ.get("PIPER_BINARY", "piper")
PIPER_VOICE_DIR = Path(os.environ.get("PIPER_VOICE_DIR", "/opt/piper-voices"))
PIPER_VOICE_MAP = {
    # Telugu (Priority)
    "te": "padmavathi-medium.onnx", 
    # Hindi
    "hi": "pratham-medium.onnx",
    # English (US)
    "en": "lessac-medium.onnx",
}
class TTSError(Exception):
    """Raised only if Sarvam, Azure, AND Piper all fail — total TTS outage.
    Per §8, the caller should speak (or play a pre-cached) fallback line
    rather than let the turn hang; since TTS itself is what's down, that
    fallback audio should be pre-synthesized and cached at startup, not
    generated on the fly here."""

async def synthesize(text: str, locale: str) -> bytes:
    """
    Convert `text` to speech in `locale` ("en" | "hi" | "te").
    Returns WAV-encoded audio bytes. Tries Sarvam, then Azure, then Piper.
    Raises TTSError only if all three fail.
    """
    errors: list[str] = []
    
    if SARVAM_API_KEY:
        try:
            return await _with_retry(lambda: _synthesize_sarvam(text,locale))
        except Exception as exc: #noqa : BLE001 - deliberately broad, this is a fallback chain
            logger.warning("Sarvam TTS failes, falling back to Azure: %s", exc)
            errors.append(f"sarvam: {exc}")
    else:
        errors.append("sarvam: no SARVAM_API_KEY configured")
    if AZURE_SPEECH_KEY:
        try:
            return await _with_retry(lambda: _synthesize_azure(text, locale))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Azure TTS failed, falling back to Piper: %s", exc)
            errors.append(f"azure: {exc}")
    else:
        errors.append("azure: no AZURE_SPEECH_KEY configured")

    try:
        return await _synthesize_piper(text, locale)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"piper: {exc}")

    raise TTSError(f"All TTS providers failed for locale={locale}: {'; '.join(errors)}")

async def _with_retry(call):
    """Shared single-retry-with-backoff wrapper, matching the ~300ms
    backoff pattern used in stt.py/llm.py for transient provider errors."""
    try:
        return await call()
    except (httpx.HTTPError, httpx.TimeoutException):
        await asyncio.sleep(0.3)
        return await call()
    
    

# ---------- Sarvam AI (primary) ----------

async def _synthesize_sarvam(text: str, locale: str) -> bytes:
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY is not configured")
    
    target_language_code = SARVAM_LOCALE_MAP.get(locale,"te")
    speaker = SARVAM_SPEAKER_MAP.get(locale,"priya")
    
    body = {
        "inputs": [text],
        "target_language_code": target_language_code,
        "speaker": speaker,
        "model": SARVAM_MODEL,
        "enable_processing": True, #normalizes numbers - matters for phn numbers
    }
    async with httpx.AsyncClient(timeout= 15.0) as Client:
        response = await Client.post(
            SARVAM_TTS_URL,
            headers={
                "api-subscription-key": SARVAM_API_KEY,
                "Content-Type": "application/json"
            },
            json = body,
        )
        response.raise_for_status()
        payload = response.json()
        
    audios = payload.get("audios") or []
    if not audios:
        raise RuntimeError(f"Sarvam returned no audio: {payload}")
    audio_base64 = audios[0]
    return base64.b64decode(audio_base64)



# ---------- Azure Cognitive Services Speech (secondary) ----------

def _escape_ssml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )

async def _synthesize_azure(text:str, locale: str) ->bytes:
    if not AZURE_SPEECH_KEY:
        raise ValueError("AZURE_SPEECH_KEY is not configured")
    
    voice = AZURE_VOICE_MAP.get(locale,AZURE_VOICE_MAP["te"])
    lang_tag = {"en": "en-IN", "hi": "hi-IN", "te": "te-IN"}.get(locale,"te-IN")
    url = f"https://{AZURE_SPEECH_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"
    ssml = (
        f'<speak version="1.0" xml:lang="{lang_tag}">'
        f'<voice xml:lang="{lang_tag}" name="{voice}">{_escape_ssml(text)}</voice>'
        f"</speak>"
    )
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(
            url,
            headers={
                "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": "riff-16khz-16bit-mono-pcm",
            },
            content=ssml.encode("utf-8"),
        )
        response.raise_for_status()
        return response.content

# ---------- Piper (local, free, last resort) ----------

async def _synthesize_piper(text: str, locale: str) -> bytes:
    voice_filename = PIPER_VOICE_MAP.get(locale)
    # Generic fallback: if a locale isn't in the map, default to English
    if voice_filename is None:
        logger.warning(
            "No Piper voice found for locale='%s'; falling back to the English Piper voice instead.",
            locale,
        )
        voice_filename = PIPER_VOICE_MAP["te"]

    model_path = PIPER_VOICE_DIR / voice_filename
    
    if not model_path.exists():
        raise RuntimeError(f"Piper voice model not found: {model_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "out.wav"
        
        proc = await asyncio.create_subprocess_exec(
            PIPER_BINARY,
            "--model", str(model_path),
            "--output_file", str(out_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        
        _, stderr = await proc.communicate(input=text.encode("utf-8"))
        
        if proc.returncode != 0 or not out_path.exists():
            raise RuntimeError(f"Piper synthesis failed: {stderr.decode(errors='ignore')}")
            
        return out_path.read_bytes()