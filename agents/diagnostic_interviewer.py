"""
agents/diagnostic_interviewer.py — Multi-Turn Diagnostic Interview Node
========================================================================
WHAT IT DOES:
  When the user gives vague or incomplete symptom descriptions
  (e.g., "I feel bad", "I have pain"), this node generates
  targeted follow-up questions to gather more clinical detail
  before proceeding to diagnosis.

HOW IT WORKS:
  1. After symptom extraction, if symptoms are vague/few → route here
  2. This node generates 2-3 specific follow-up questions
  3. Response is returned to the user (graph ends)
  4. User's reply goes through the normal pipeline with richer context
  5. After 2-3 rounds OR enough info → proceed to diagnosis

ACTIVATION:
  Controlled by a sidebar toggle in app.py. Only activates when:
  - Toggle is ON
  - Extracted symptoms are vague or fewer than 3
  - question_count < 3 (max 3 follow-up rounds)
"""

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from agents.state import HealthAgentState
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)


# ── Vague symptom indicators ─────────────────────────────────────────────────
VAGUE_SYMPTOMS = {
    "pain", "ache", "not feeling well", "feeling bad", "sick",
    "unwell", "discomfort", "not well", "something wrong",
    "feeling off", "not right", "problem", "issue", "trouble",
    "hurt", "hurting", "sore", "bad", "worse", "uncomfortable"
}


def needs_followup(state: HealthAgentState) -> bool:
    """
    Determines if the bot should ask follow-up questions
    before proceeding to diagnosis.

    Returns True if:
    - Toggle is enabled
    - Fewer than 3 symptoms extracted
    - Symptoms are vague
    - Haven't asked 3 follow-ups yet
    """
    enable = state.get("enable_diagnostic_interview", False)
    if not enable:
        return False

    raw_symptoms = state.get("raw_symptoms") or []
    question_count = state.get("question_count") or 0

    # Max 3 rounds of follow-up
    if question_count >= 3:
        return False

    # If we already have 3+ specific symptoms, proceed
    if len(raw_symptoms) >= 3:
        # Check if symptoms are specific enough
        vague_count = sum(1 for s in raw_symptoms if s.lower().strip() in VAGUE_SYMPTOMS)
        if vague_count < len(raw_symptoms) / 2:
            return False  # Most symptoms are specific

    # Fewer than 3 symptoms → ask for more
    if len(raw_symptoms) < 3:
        return True

    # Check if symptoms are too vague
    for symptom in raw_symptoms:
        if symptom.lower().strip() in VAGUE_SYMPTOMS:
            return True

    return False


INTERVIEW_SYSTEM = """You are a medical doctor doing an initial consultation.
The patient described some symptoms but you need more details for an accurate assessment.

RULES:
- Ask exactly 2-3 specific, targeted follow-up questions
- Questions should help narrow down the diagnosis
- Ask about: location, duration, severity, triggers, associated symptoms
- Be warm and professional — like a real doctor
- Number your questions (1, 2, 3)
- Do NOT diagnose yet — just gather information
- Keep it concise — 3-4 sentences introduction + questions
- End with: "Take your time — your answers will help me give you better guidance."

GOOD QUESTIONS:
- "Where exactly do you feel the pain — is it in your chest, abdomen, or head?"
- "How long have you been experiencing this — hours, days, or weeks?"
- "On a scale of 1-10, how severe would you rate this?"
- "Does anything make it better or worse?"
- "Are you experiencing any other symptoms like fever, nausea, or dizziness?"
"""


def diagnostic_interview_node(state: HealthAgentState) -> dict:
    """
    LangGraph Node: Diagnostic Interview

    Generates targeted follow-up questions when symptoms are
    vague or insufficient for accurate diagnosis.

    Input  (from state): state["raw_symptoms"], state["user_input"]
    Output (to state)  : {"final_response": questions, "diagnostic_phase": "COLLECTING"}
    """
    print("  [Node] DiagnosticInterviewer running...")

    raw_symptoms = state.get("raw_symptoms") or []
    user_input = state.get("user_input", "")
    question_count = state.get("question_count") or 0
    chat_history = state.get("chat_history") or []

    # Build context from previous exchanges
    prev_context = ""
    if chat_history:
        recent = chat_history[-4:]
        lines = []
        for msg in recent:
            role = "Patient" if msg.get("role") == "user" else "Doctor"
            content = str(msg.get("content", ""))[:200]
            lines.append(f"{role}: {content}")
        prev_context = "\n".join(lines)

    messages = [
        SystemMessage(content=INTERVIEW_SYSTEM),
        HumanMessage(content=f"""Patient's current message: "{user_input}"

Symptoms detected so far: {', '.join(raw_symptoms) if raw_symptoms else 'Very vague — need clarification'}

Previous conversation:
{prev_context or 'This is the first message.'}

Follow-up round: {question_count + 1} of 3

Generate your follow-up questions now.""")
    ]

    response = llm.invoke(messages)

    print(f"  [Node] Generated follow-up questions (round {question_count + 1})")

    return {
        "final_response": response.content.strip(),
        "diagnostic_phase": "COLLECTING",
        "question_count": question_count + 1
    }
