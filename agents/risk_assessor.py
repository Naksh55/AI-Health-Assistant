"""
agents/risk_assessor.py — Risk Assessment Node
===============================================
KEY FIX:
  Added a HARD-CODED emergency keyword check BEFORE calling the LLM.
  This ensures stroke, heart attack, and other life-threatening symptoms
  are ALWAYS classified as EMERGENCY — no LLM guessing.

  Before fix: "face droopy + arm weakness" → HIGH ❌
  After fix:  "face droopy + arm weakness" → EMERGENCY ✅
"""

import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agents.state import HealthAgentState
from dotenv import load_dotenv
load_dotenv()


# ── Hard-coded emergency symptom patterns ─────────────────────────────────────
# These ALWAYS trigger EMERGENCY regardless of what the LLM thinks.
# Grouped by condition for clarity.

EMERGENCY_PATTERNS = {

    "Stroke / TIA": [
        "face droopy", "face droop", "facial droop", "drooping face",
        "arm weakness", "arm numb", "cannot lift arm", "can't lift arm",
        "sudden weakness", "sudden numbness", "slurred speech", "speech slurred",
        "confusion sudden", "sudden confusion", "vision loss sudden",
        "sudden severe headache", "worst headache", "thunderclap headache",
        "droopy face", "one side weak", "left side weak", "right side weak",
        "face feels droopy", "face is drooping", "cannot speak", "can't speak",
        "paresis", "hemiplegia", "hemiplegic", "facial palsy",
    ],

    "Heart Attack": [
        "chest pain", "chest pressure", "chest tightness", "chest crushing",
        "pain in chest", "heart attack", "left arm pain", "jaw pain",
        "spreading to arm", "radiating to arm", "pain radiating",
        "sweating heavily", "cold sweat", "sudden chest",
        "myocardial", "cardiac arrest",
    ],

    "Severe Breathing": [
        "cannot breathe", "can't breathe", "stopped breathing",
        "throat closing", "throat swelling", "throat feels closed",
        "throat is closing", "not breathing", "stopped breathing",
        "severe breathlessness", "gasping for air",
    ],

    "Anaphylaxis": [
        "anaphylaxis", "anaphylactic", "hives all over",
        "throat swelling after eating", "allergic reaction severe",
        "throat closing up", "swollen throat",
    ],

    "Loss of Consciousness": [
        "unconscious", "loss of consciousness", "passed out",
        "fainted", "unresponsive", "not responding",
        "collapsed", "seizure", "convulsion", "convulsing",
    ],

    "Severe Bleeding": [
        "bleeding heavily", "blood not stopping", "heavy bleeding",
        "massive bleeding", "arterial bleeding", "spurting blood",
    ],

    "Diabetic Emergency": [
        "blood sugar very low", "hypoglycemia severe",
        "diabetic coma", "going into coma",
    ],
}


def check_emergency_keywords(symptoms_text: str, user_input: str) -> dict | None:
    """
    Checks symptom text and original user input against known emergency patterns.
    Returns emergency risk dict if matched, None otherwise.
    This runs BEFORE the LLM to catch obvious emergencies instantly.
    """
    combined = (symptoms_text + " " + user_input).lower()

    for condition, patterns in EMERGENCY_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in combined:
                print(f"  [RiskAssessor] EMERGENCY keyword matched: '{pattern}' → {condition}")
                return {
                    "risk_level": "EMERGENCY",
                    "reason": f"Symptom '{pattern}' is a potential sign of {condition} — a life-threatening emergency.",
                    "action": "Call emergency services (112/911) immediately. Do not wait.",
                    "emergency_signs": [pattern]
                }
    return None


RISK_SYSTEM = """You are an emergency medicine triage specialist.
Assess the risk level for a patient based on their symptoms and possible conditions.

Risk Level Definitions:
- EMERGENCY : Life-threatening. Requires calling emergency services immediately.
              Examples: chest pain + shortness of breath, signs of stroke (face drooping,
              arm weakness, speech difficulty), severe bleeding, anaphylaxis,
              loss of consciousness, sudden severe headache
- HIGH      : Serious. Needs urgent care within hours today.
              Examples: high fever in infant, severe dehydration, difficulty breathing,
              suspected fracture, severe pain
- MEDIUM    : Moderate. Should see a doctor within 1-2 days.
              Examples: flu symptoms, moderate pain, persistent fever, UTI
- LOW       : Mild. Home care and monitoring is appropriate.
              Examples: common cold, mild headache, minor cuts, mild fatigue

IMPORTANT: When in doubt between EMERGENCY and HIGH, always choose EMERGENCY.
Stroke symptoms (face drooping, arm weakness, speech difficulty) = ALWAYS EMERGENCY.
Chest pain = ALWAYS HIGH or EMERGENCY.

Return ONLY a valid JSON object:
{{
  "risk_level": "EMERGENCY|HIGH|MEDIUM|LOW",
  "reason": "One sentence clinical reasoning",
  "action": "Specific instruction for what the patient should do right now",
  "emergency_signs": ["sign1", "sign2"]
}}

No markdown. No extra text."""

prompt = ChatPromptTemplate.from_messages([
    ("system", RISK_SYSTEM),
    ("human", """Symptoms: {symptoms}
Possible conditions: {conditions}
Original patient message: {user_input}

Perform triage assessment. Remember: err on the side of caution.""")
])

llm   = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
chain = prompt | llm | StrOutputParser()


def risk_assessment_node(state: HealthAgentState) -> dict:
    """
    LangGraph Node: Risk Assessment

    Input  (from state): state["normalized_symptoms"], state["predicted_conditions"]
    Output (to state)  : {"risk_assessment": {...}}

    Flow:
    1. First check hard-coded emergency keywords (instant, no API call)
    2. If no keyword match, call LLM for full triage assessment
    """
    print("  [Node] RiskAssessor running...")

    symptoms        = state.get("normalized_symptoms") or []
    conditions      = state.get("predicted_conditions") or []
    user_input      = state.get("user_input", "")
    condition_names = [c.get("name", "") for c in conditions]
    symptoms_text   = ", ".join(symptoms)

    # ── Step 1: Hard-coded emergency check (runs first, no LLM needed) ────────
    emergency = check_emergency_keywords(symptoms_text, user_input)
    if emergency:
        print(f"  [Node] Risk level: EMERGENCY (keyword match)")
        return {"risk_assessment": emergency}

    # ── Step 2: LLM-based triage assessment ───────────────────────────────────
    raw_text = chain.invoke({
        "symptoms":   symptoms_text,
        "conditions": ", ".join(condition_names),
        "user_input": user_input
    })

    try:
        clean = raw_text.strip().replace("```json", "").replace("```", "")
        risk  = json.loads(clean)
    except json.JSONDecodeError:
        risk = {
            "risk_level":      "MEDIUM",
            "reason":          "Could not fully assess — caution recommended.",
            "action":          "Please consult a healthcare provider.",
            "emergency_signs": []
        }

    print(f"  [Node] Risk level: {risk.get('risk_level')}")
    return {"risk_assessment": risk}