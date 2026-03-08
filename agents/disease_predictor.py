"""
agents/disease_predictor.py — Disease Prediction Node
======================================================
WHAT IT DOES:
  Takes the normalized symptom list and predicts the top 3 most likely
  medical conditions, with probability and clinical reasoning for each.

LANGGRAPH ROLE:
  Third specialist node. Reads `normalized_symptoms`, writes `predicted_conditions`.

LANGCHAIN CONCEPT — ChatPromptTemplate:
  Instead of manually building [SystemMessage, HumanMessage] lists every time,
  LangChain's ChatPromptTemplate lets you define a reusable template with
  variables (like {symptoms}) that get filled in at runtime.

  Pattern:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are..."),
        ("human",  "Symptoms: {symptoms}")
    ])
    chain = prompt | llm | parser
    result = chain.invoke({"symptoms": "fever, chills"})

  This is cleaner, reusable, and composable.
"""
from dotenv import load_dotenv
load_dotenv()

import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agents.state import HealthAgentState


# ── ChatPromptTemplate — reusable template with {symptoms} variable ────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a clinical diagnosis assistant with deep medical knowledge.
Given a list of symptoms, identify the top 3 most likely medical conditions.

Respond ONLY with a valid JSON object in this exact format:
{{
  "conditions": [
    {{
      "name": "Influenza",
      "probability": "High",
      "reasoning": "Fever, chills, and body ache are classic flu presentation"
    }},
    {{
      "name": "Malaria",
      "probability": "Medium", 
      "reasoning": "Cyclical fever with chills is characteristic of malaria"
    }},
    {{
      "name": "Viral Fever",
      "probability": "Low",
      "reasoning": "General viral syndrome cannot be ruled out"
    }}
  ]
}}

No markdown. No explanation. Only the JSON object."""),

    ("human", "Patient symptoms: {symptoms}")
])

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
parser = StrOutputParser()

# Full LCEL chain: template → llm → string parser
chain = prompt | llm | parser


def disease_prediction_node(state: HealthAgentState) -> dict:
    """
    LangGraph Node: Disease Prediction

    Input  (from state): state["normalized_symptoms"]
    Output (to state)  : {"predicted_conditions": [...]}
    """
    print("  [Node] DiseasePredictor running...")

    normalized = state.get("normalized_symptoms", [])

    if not normalized:
        return {"predicted_conditions": []}

    # .invoke() fills in the {symptoms} template variable
    symptoms_str = ", ".join(normalized)
    raw_text = chain.invoke({"symptoms": symptoms_str})

    try:
        clean = raw_text.strip().replace("```json", "").replace("```", "")
        result = json.loads(clean)
        conditions = result.get("conditions", [])
    except json.JSONDecodeError:
        conditions = []

    print(f"  [Node] Predicted conditions: {[c.get('name') for c in conditions]}")
    return {"predicted_conditions": conditions}
