"""
translate.py — English → Telugu/Hindi translation via Groq (llama-3.1-8b-instant).

Why Groq instead of Sarvam Mayura:
  - Same API key already in use — zero new credentials
  - llama-3.1-8b-instant on Groq runs at ~500 tokens/sec — faster than
    any translation REST API round-trip
  - 8B models translate Indian languages accurately enough for voice calls
  - Fraction of the cost vs. Sarvam Mayura per-character pricing
  - No per-character credit consumption — uses the same Groq token budget
    as the main LLM, which has a generous free tier

For English locale, translate() is a no-op (returns immediately, zero API call).
"""

import asyncio
import json
import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger("mediassist.translate")

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
TRANSLATION_MODEL = "llama-3.3-70b-versatile"  # fast + cheap; accurate enough for voice

LANGUAGE_NAMES = {
    "te": "Telugu",
    "hi": "Hindi",
}

# Translation system prompt — terse and explicit to minimise output tokens
# (we only want the translated sentence, nothing else).
_SYSTEM = (
    "You are a translation engine. "
    "Translate the user's English text to {language}. "
    "Output ONLY the translated text — no explanation, no prefix, no quotes. "
    "Preserve medical terms, doctor names, phone numbers, and times as-is."
)


async def translate(text: str, target_locale: str) -> str:
    """
    Translate English LLM output to target_locale ("en" | "hi" | "te").
    Returns original text unchanged for locale="en" (no API call made).
    Falls back to English on any error — silence is worse than English audio.
    """
    language = LANGUAGE_NAMES.get(target_locale)
    if not language:
        return text  # "en" or unknown — no translation needed

    try:
        translated = await _call_groq_translate(text, language)
        logger.info("Translated [en→%s]: '%s' → '%s'", target_locale, text[:60], translated[:60])
        return translated
    except Exception as exc:  # noqa: BLE001
        logger.warning("Translation failed (%s) — speaking English: %s", target_locale, exc)
        return text  # English audio beats silence


async def _call_groq_translate(text: str, language: str) -> str:
    body = {
        "model": TRANSLATION_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM.format(language=language)},
            {"role": "user", "content": text},
        ],
        "temperature": 0.1,   # low — translation is deterministic, not creative
        "max_tokens": 300,    # voice replies are short; 300 is generous
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=8.0) as client:
        response = await client.post(
            GROQ_CHAT_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        response.raise_for_status()
        payload = response.json()

    translated = payload["choices"][0]["message"]["content"].strip()
    if not translated:
        raise ValueError("Groq returned empty translation")
    return translated

# """
# translate.py — English → Hindi/Telugu translation via Sarvam Mayura.

# Why this exists: llama-3.3-70b (and most open LLMs) generate Telugu/Hindi
# slowly and unreliably when forced to write in those scripts. Rather than
# asking the LLM to respond in Telugu and waiting 5+ minutes for garbled
# output, we keep the LLM in English (fast, reliable) and translate the
# finished English reply to the target language here before TTS.

# Sarvam's Mayura model is purpose-built for Indian language translation —
# it's far better at this than a general-purpose LLM.

# For English locale, translate() is a no-op and returns the text unchanged.
# """

# import asyncio
# import logging
# import os
# from typing import Optional

# import httpx

# logger = logging.getLogger("mediassist.translate")

# SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY")
# SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"

# # Sarvam language codes for translation (different format from TTS)
# TRANSLATE_LOCALE_MAP = {
#     "en": None,       # no-op — already English
#     "hi": "hi-IN",
#     "te": "te-IN",
# }


# class TranslateError(Exception):
#     pass


# async def translate(text: str, target_locale: str) -> str:
#     """
#     Translate English LLM output to target_locale ("en" | "hi" | "te").
#     Returns the original text unchanged if target_locale is "en".
#     Falls back to English gracefully if the Sarvam API fails — English
#     audio is better than silence.
#     """
#     target_code = TRANSLATE_LOCALE_MAP.get(target_locale)

#     if target_code is None:
#         return text  # English — no translation needed

#     if not SARVAM_API_KEY:
#         logger.warning("No SARVAM_API_KEY — skipping translation, TTS will speak English")
#         return text

#     for attempt in range(2):
#         try:
#             translated = await _call_sarvam_translate(text, target_code)
#             logger.info(
#                 "Translated [%s→%s]: '%s' → '%s'",
#                 "en", target_locale, text[:60], translated[:60]
#             )
#             return translated
#         except (httpx.HTTPError, httpx.TimeoutException) as exc:
#             if attempt == 0:
#                 await asyncio.sleep(0.3)
#                 continue
#             logger.warning("Translation failed after retry: %s — speaking English", exc)
#             return text
#         except Exception as exc:  # noqa: BLE001
#             logger.warning("Translation error: %s — speaking English", exc)
#             return text

#     return text


# async def _call_sarvam_translate(text: str, target_language_code: str) -> str:
#     body = {
#         "input": text,
#         "source_language_code": "en-IN",
#         "target_language_code": target_language_code,
#         "speaker_gender": "Female",
#         "mode": "formal",
#         "model": "mayura:v1",
#         "enable_preprocessing": False,
#     }

#     async with httpx.AsyncClient(timeout=10.0) as client:
#         response = await client.post(
#             SARVAM_TRANSLATE_URL,
#             headers={
#                 "api-subscription-key": SARVAM_API_KEY,
#                 "Content-Type": "application/json",
#             },
#             json=body,
#         )
#         logger.debug(
#             "Sarvam translate response (status=%d): %s",
#             response.status_code,
#             response.text[:300],
#         )
#         response.raise_for_status()
#         payload = response.json()

#     translated = payload.get("translated_text") or payload.get("translation") or ""
#     if not translated:
#         logger.warning("Sarvam translate returned empty text. payload=%s", payload)
#         return text  # fall back to English rather than sending empty string to TTS

#     return translated