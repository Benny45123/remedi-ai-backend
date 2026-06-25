"""
lang.py — locale handling: system prompts per language + lightweight
language-switch detection.

MediAssist supports three locales: "en", "hi", "te". This module owns
two things:

  1. system_prompt_for_locale() — the locale-specific system prompt the
     agent graph's LLM node uses every turn, including a summary of
     currently filled slots so the model doesn't re-ask for something
     it already has.

  2. detect_locale() — decides which locale is "active" for a given turn.
     Whisper's own language detection (returned alongside the transcript
     in stt.py) is the primary signal, but it's noisy on short utterances
     ("yes", "4 PM", a phone number) which can get misdetected as a
     different language than actually being spoken. So this also does a
     cheap Unicode-script check as a high-confidence override, and falls
     back to *sticking* with the previous locale rather than flapping
     when the signal is ambiguous.
"""

import re
from typing import Optional

SUPPORTED_LOCALES = {"en", "hi", "te"}
DEFAULT_LOCALE = "te"

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")  # Hindi
_TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")  # Telugu

LOCALE_NAMES = {"en": "English", "hi": "Hindi", "te": "Telugu"}


def detect_locale(
    transcript_text: str,
    whisper_language: Optional[str],
    previous_locale: str = DEFAULT_LOCALE,
) -> str:
    """
    Decide the active locale for this turn.

    Priority:
      1. Unicode script in the transcript itself (highest confidence —
         text that rendered as Devanagari or Telugu script isn't a guess).
      2. Whisper's own detected language, if it's one of our three
         supported locales.
      3. Stick with whatever locale was active last turn — short
         utterances like "yes" or a phone number are language-ambiguous,
         and re-detecting from scratch every turn would cause the AI to
         flip languages mid-conversation for no real reason.
    """
    if _DEVANAGARI_RE.search(transcript_text):
        return "hi"
    if _TELUGU_RE.search(transcript_text):
        return "te"

    if whisper_language in SUPPORTED_LOCALES:
        return whisper_language

    return previous_locale if previous_locale in SUPPORTED_LOCALES else DEFAULT_LOCALE


# ---------- System prompts ----------

_BASE_INSTRUCTIONS = """You are MediAssist, a friendly, efficient voice receptionist for a clinic. \
You are speaking with a patient over a live voice call — keep replies short, natural, and \
conversational, the way a helpful human receptionist would talk, not like a chatbot writing \
paragraphs. One or two sentences per turn is usually right.

Your job, in order:
1. Find out what kind of doctor/specialization the patient needs.
2. Call check_availability to find open slots for that specialization, then offer 1-2 options \
(e.g. "tomorrow at 11 AM or 4 PM").
3. Once the patient picks a time, collect their full name and phone number if you don't already \
have them.
4. Read back a clear confirmation — doctor, day/time, patient name — and ask "Shall I book it?" \
before calling create_appointment. Never book without an explicit yes.
5. After calling create_appointment: if it succeeds, confirm warmly and ask if there's anything \
else. If it returns a "slot_taken" error, apologize briefly and naturally offer one of the \
alternatives it gives you instead of restarting the conversation.

Stay strictly in character as a clinic receptionist. Don't mention tools, function calls, JSON, \
or anything technical — just talk naturally."""

_LOCALE_INSTRUCTIONS = {
    "en": "Respond in natural, conversational English.",
    "hi": (
        "Respond entirely in natural, conversational Hindi (Devanagari script in your text, "
        "but remember this is a spoken call — keep sentences simple and easy to say aloud). "
        "Patient names and phone numbers may stay in their original form."
    ),
    "te": (
        "Respond entirely in natural, conversational Telugu (Telugu script in your text, but "
        "remember this is a spoken call — keep sentences simple and easy to say aloud). "
        "Patient names and phone numbers may stay in their original form."
    ),
}


def system_prompt_for_locale(locale: str, slots: dict) -> str:
    """Build the per-turn system prompt: base receptionist instructions +
    a locale directive + a summary of slots already filled, so the model
    doesn't re-ask for information it already has."""
    locale_instruction = _LOCALE_INSTRUCTIONS.get(locale, _LOCALE_INSTRUCTIONS[DEFAULT_LOCALE])

    filled = []
    if slots.get("specialization"):
        filled.append(f"specialization requested: {slots['specialization']}")
    if slots.get("doctor_id"):
        filled.append("a doctor/slot has been selected but not yet confirmed booked")
    if slots.get("patient_name"):
        filled.append(f"patient name: {slots['patient_name']}")
    if slots.get("patient_phone"):
        filled.append(f"patient phone: {slots['patient_phone']}")
    if slots.get("booking_complete"):
        filled.append(
            "an appointment has ALREADY been booked this call — don't book another "
            "unless the patient explicitly asks for a second appointment"
        )

    slots_summary = (
        "Known so far this call: " + "; ".join(filled) + "."
        if filled
        else "Nothing has been collected yet this call."
    )

    return f"{_BASE_INSTRUCTIONS}\n\n{locale_instruction}\n\n{slots_summary}"