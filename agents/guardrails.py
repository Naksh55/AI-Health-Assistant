"""
agents/guardrails.py — Input Validation & Sanitization
========================================================
Validates, sanitizes, and filters user input before it enters
the LangGraph pipeline.

WHAT IT CATCHES:
  - Prompt injection attempts ("ignore previous instructions")
  - HTML/script injection (<script>, <img onerror>)
  - Off-topic non-medical queries ("write me a poem")
  - Inputs that are too short or too long
  - Excessive special characters

USAGE:
  from agents.guardrails import validate_input, sanitize_input

  is_valid, error_msg = validate_input(user_text)
  if not is_valid:
      return error_msg
  clean_text = sanitize_input(user_text)
"""

import re
import unicodedata


# ── Constants ─────────────────────────────────────────────────────────────────
MAX_INPUT_LENGTH = 2000
MIN_INPUT_LENGTH = 2


# ── Prompt injection patterns ────────────────────────────────────────────────
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?above\s+instructions",
    r"disregard\s+(all\s+)?previous",
    r"forget\s+(all\s+)?previous",
    r"you\s+are\s+now\s+a",
    r"act\s+as\s+if\s+you\s+are",
    r"pretend\s+you\s+are",
    r"new\s+instructions?\s*:",
    r"system\s*:\s*you",
    r"system\s+prompt\s*:",
    r"\[system\]",
    r"\[inst\]",
    r"<\s*system\s*>",
    r"override\s+your\s+instructions",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"dan\s+mode",
]

# ── Off-topic patterns ───────────────────────────────────────────────────────
OFF_TOPIC_PATTERNS = [
    r"write\s+(me\s+)?a\s+(poem|story|essay|song|code|script|program)",
    r"tell\s+me\s+a\s+joke",
    r"solve\s+this\s+math",
    r"translate\s+this\s+to",
    r"what\s+is\s+the\s+capital\s+of",
    r"who\s+is\s+the\s+president",
    r"play\s+a\s+game",
    r"write\s+python\s+code",
    r"create\s+a\s+website",
    r"generate\s+(a\s+)?password",
    r"what\s+is\s+\d+\s*[\+\-\*\/]\s*\d+",  # math equations
    r"help\s+me\s+with\s+my\s+homework",
    r"recipe\s+for",
    r"how\s+to\s+cook",
    r"stock\s+market",
    r"crypto\s+price",
    r"weather\s+in",
]

# ── HTML/Script patterns ─────────────────────────────────────────────────────
HTML_PATTERNS = [
    r"<\s*script",
    r"<\s*iframe",
    r"<\s*img\s+[^>]*onerror",
    r"<\s*svg\s+[^>]*onload",
    r"javascript\s*:",
    r"on(click|load|error|mouseover)\s*=",
    r"<\s*object",
    r"<\s*embed",
    r"<\s*form",
    r"<\s*input",
]


def validate_input(text: str) -> tuple[bool, str]:
    """
    Validates user input for safety and relevance.

    Returns:
        (is_valid, error_message) — if is_valid is False, error_message
        explains why the input was rejected.
    """
    if not text or not text.strip():
        return False, "Please enter a message."

    text_stripped = text.strip()

    # ── Length checks ─────────────────────────────────────────────────────
    if len(text_stripped) < MIN_INPUT_LENGTH:
        return False, "Your message is too short. Please describe your symptoms in more detail."

    if len(text_stripped) > MAX_INPUT_LENGTH:
        return False, (
            f"Your message is too long ({len(text_stripped)} characters). "
            f"Please keep it under {MAX_INPUT_LENGTH} characters."
        )

    text_lower = text_stripped.lower()

    # ── Prompt injection detection ────────────────────────────────────────
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            return False, (
                "⚠️ I'm a medical health assistant. "
                "I can only help with health-related questions and symptom analysis."
            )

    # ── HTML/Script injection detection ───────────────────────────────────
    for pattern in HTML_PATTERNS:
        if re.search(pattern, text_lower):
            return False, (
                "⚠️ Your message contains invalid characters. "
                "Please describe your symptoms in plain text."
            )

    # ── Off-topic detection ───────────────────────────────────────────────
    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, text_lower):
            return False, (
                "🏥 I'm a medical health assistant and can only help with "
                "health-related questions. Try describing your symptoms, "
                "asking about a medical condition, or uploading a medical report."
            )

    # ── Excessive special characters ──────────────────────────────────────
    special_char_ratio = sum(1 for c in text_stripped if not c.isalnum() and not c.isspace()) / max(len(text_stripped), 1)
    if special_char_ratio > 0.5:
        return False, (
            "Your message contains too many special characters. "
            "Please describe your symptoms in plain text."
        )

    return True, ""


def sanitize_input(text: str) -> str:
    """
    Cleans and normalizes user input for safe processing.

    Steps:
    1. Strip leading/trailing whitespace
    2. Normalize Unicode characters
    3. Remove HTML tags
    4. Collapse excessive whitespace
    5. Truncate to max length
    """
    if not text:
        return ""

    # Normalize Unicode (e.g., convert accented chars to closest ASCII)
    text = unicodedata.normalize("NFKC", text)

    # Strip HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Remove null bytes and control characters (keep newlines, tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Collapse multiple whitespace into single space
    text = re.sub(r"\s+", " ", text).strip()

    # Truncate to max length
    if len(text) > MAX_INPUT_LENGTH:
        text = text[:MAX_INPUT_LENGTH]

    return text
