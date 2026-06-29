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
     different language than the one actually being spoken. So this also
     does a cheap Unicode-script check as a high-confidence override, and
     falls back to *sticking* with the previous locale rather than
     flapping when the signal is ambiguous.
"""

import re
from typing import Optional

SUPPORTED_LOCALES = {"en", "hi", "te"}
# Telugu is the default — change to "en" or "hi" if you need a different
# starting language. The client can also override this per-session by
# passing {"event": "session_start", "locale": "te"} on connect.
DEFAULT_LOCALE = "te"

# Unicode block ranges — if any character in the transcript falls in one
# of these, that's a near-certain signal of the script being spoken,
# regardless of what Whisper's own `language` field guessed.
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
paragraphs. CRITICAL: Keep every reply to ONE sentence maximum, two only if absolutely necessary.

Your job, in order:
1. Find out what kind of doctor/specialization the patient needs.
2. Call check_availability with the English specialization name to find open slots, \
then offer 1-2 options (e.g. "tomorrow at 11 AM or 4 PM").
3. Once the patient picks a time, collect their full name and phone number if you don't \
already have them.
4. Read back a clear confirmation — doctor, day/time, patient name — and ask "Shall I book it?" \
before calling create_appointment. Never book without an explicit yes.
5. After calling create_appointment: if it succeeds, confirm warmly and ask if there's \
anything else. If it returns slot_taken, apologize briefly and offer one of the alternatives.

TOOL-CALLING RULES — follow these exactly or the booking will fail:
- check_availability: the `specialization` argument MUST be in English. \
Use one of: 'Dermatologist', 'General Physician', 'Pediatrician'. \
NEVER pass a doctor's name (like 'Dr. Mehta'). \
NEVER pass Telugu or Hindi text as the specialization. \
If the patient names a doctor, infer their specialty and use that English word instead.
- create_appointment: ONLY use slot_id and doctor_id values that came directly from \
a check_availability result. NEVER invent, guess, or reuse UUIDs from memory. \
If you don't have a real slot_id from a recent check_availability call, call \
check_availability again first.

Always respond in English. Your English response will be automatically translated to the \
patient's language before being spoken — you do not need to translate yourself. \
Stay strictly in character as a clinic receptionist. Never mention tools, UUIDs, JSON, \
translation, or anything technical."""

# Single instruction regardless of locale — LLM always writes English,
# translate.py handles Telugu/Hindi conversion before TTS.
_LOCALE_INSTRUCTIONS = {
    "en": "",
    "hi": "",
    "te": "",
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