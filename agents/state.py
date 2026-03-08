"""
agents/state.py — Shared State Definition
==========================================
In LangGraph, ALL agents (nodes) communicate through a single shared state object.
Think of it as a "baton" passed from one node to the next — each agent reads from it
and writes its results back into it.

TypedDict gives us type safety so we always know what fields exist.
"""

from typing import TypedDict, Optional
from dotenv import load_dotenv
load_dotenv()

class HealthAgentState(TypedDict):
    """
    The single shared state object that flows through every node in our graph.

    Fields are populated progressively as the graph executes:
    - user_input          → Set at the very start (by the user)
    - raw_symptoms        → Filled by SymptomExtractionAgent
    - normalized_symptoms → Filled by SymptomNormalizationAgent
    - predicted_conditions→ Filled by DiseasePredictionAgent
    - risk_assessment     → Filled by RiskAssessmentAgent
    - final_response      → Filled by MedicalAdviceAgent
    - error               → Set if something goes wrong (triggers early exit)
    - error_message       → Human-readable error description
    """

    user_input: str
    raw_symptoms: Optional[list[str]]
    normalized_symptoms: Optional[list[str]]
    predicted_conditions: Optional[list[dict]]
    risk_assessment: Optional[dict]
    final_response: Optional[str]
    error: Optional[bool]
    error_message: Optional[str]
