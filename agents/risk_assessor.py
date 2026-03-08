"""
agents/risk_assessor.py — Risk Assessment Node
===============================================
WHAT IT DOES:
  Evaluates the severity of the patient's condition and assigns a risk level:
  EMERGENCY / HIGH / MEDIUM / LOW

  This is the most safety-critical node — it determines whether the user
  needs to call emergency services immediately.

LANGGRAPH ROLE:
  Fourth specialist node. Reads `normalized_symptoms` + `predicted_conditions`.
  Writes `risk_assessment` to state.

LANGCHAIN CONCEPT — PydanticOutputParser (structured output):
  Instead of parsing raw JSON strings manually, LangChain's PydanticOutputParser
  lets you define a Pydantic model and have LangChain:
    1. Inject formatting instructions into the prompt automatically
    2. Parse and validate the response into a typed Python object
  
  We use a simple dict approach here but show how you'd use Pydantic too.
"""

import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agents.state import HealthAgentState
from dotenv import load_dotenv
load_dotenv()

RISK_SYSTEM = """You are an emergency medicine triage specialist.
Assess the risk level for a patient based on their symptoms and possible conditions.

Risk Level Definitions:
- EMERGENCY : Life-threatening. Requires calling emergency services immediately.
              Examples: chest pain + shortness of breath, signs of stroke, severe bleeding,
              anaphylaxis, loss of consciousness
- HIGH      : Serious. Needs urgent care within hours today.
              Examples: high fever in infant, severe dehydration, difficulty breathing
- MEDIUM    : Moderate. Should see a doctor within 1-2 days.
              Examples: flu symptoms, moderate pain, persistent fever
- LOW       : Mild. Home care and monitoring is appropriate.
              Examples: common cold, mild headache, minor cuts

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

Perform triage assessment.""")
])

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
chain = prompt | llm | StrOutputParser()


def risk_assessment_node(state: HealthAgentState) -> dict:
    """
    LangGraph Node: Risk Assessment

    Input  (from state): state["normalized_symptoms"], state["predicted_conditions"]
    Output (to state)  : {"risk_assessment": {...}}
    """
    print("  [Node] RiskAssessor running...")

    symptoms = state.get("normalized_symptoms", [])
    conditions = state.get("predicted_conditions", [])

    condition_names = [c.get("name", "") for c in conditions]

    raw_text = chain.invoke({
        "symptoms": ", ".join(symptoms),
        "conditions": ", ".join(condition_names)
    })

    try:
        clean = raw_text.strip().replace("```json", "").replace("```", "")
        risk = json.loads(clean)
    except json.JSONDecodeError:
        risk = {
            "risk_level": "MEDIUM",
            "reason": "Could not fully assess — caution recommended.",
            "action": "Please consult a healthcare provider.",
            "emergency_signs": []
        }

    print(f"  [Node] Risk level: {risk.get('risk_level')}")
    return {"risk_assessment": risk}
