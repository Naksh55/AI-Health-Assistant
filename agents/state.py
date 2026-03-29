"""
agents/state.py — Shared State Definition
==========================================
Single shared state object flowing through every LangGraph node.

NEW FIELD: intent — classified by supervisor before routing
  SIMPLE_QUESTION  → "what is hypothyroidism"
  REPORT_QUERY     → "what is my cholesterol", "patient name"
  SYMPTOM_ANALYSIS → "I have fever and headache"
  ACTION_REQUEST   → "schedule appointment"
  REPORT_OVERVIEW  → "check my report", "what's wrong"
"""

from typing import TypedDict, Optional


class HealthAgentState(TypedDict):

    # ── Core input ────────────────────────────────────────────────────────────
    user_input: str

    # ── NEW: Intent classified by supervisor ──────────────────────────────────
    intent: Optional[str]  # SIMPLE_QUESTION / REPORT_QUERY / SYMPTOM_ANALYSIS / ACTION_REQUEST / REPORT_OVERVIEW

    # ── Report fields ─────────────────────────────────────────────────────────
    has_report:      Optional[bool]
    report_data:     Optional[str]   # extracted text (PDF) or base64 (image)
    report_type:     Optional[str]   # "pdf" or "image"
    report_analysis: Optional[dict]  # structured findings from ReportAnalyzer

    # ── Symptom pipeline fields ───────────────────────────────────────────────
    raw_symptoms:         Optional[list[str]]
    normalized_symptoms:  Optional[list[str]]
    predicted_conditions: Optional[list[dict]]
    risk_assessment:      Optional[dict]

    # ── Final output ──────────────────────────────────────────────────────────
    final_response: Optional[str]

    # ── Conversation memory for follow-up context ───────────────────────────
    chat_history: Optional[list[dict]]

    # ── Error handling ────────────────────────────────────────────────────────
    error:         Optional[bool]
    error_message: Optional[str]