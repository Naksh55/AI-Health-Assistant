"""
agents/symptom_extractor.py — Symptom Extraction Node
======================================================
WHAT IT DOES:
  Reads the raw user message (e.g., "I have a bad headache and feel hot")
  and extracts ONLY the symptom keywords from it.

HOW IT FITS INTO LANGGRAPH:
  This is the FIRST specialist node called by the Supervisor.
  It receives the full HealthAgentState, reads `user_input`,
  and writes `raw_symptoms` back into the state.

LANGCHAIN USED HERE:
  - ChatAnthropic: LangChain's wrapper around the Anthropic Claude API
  - SystemMessage / HumanMessage: LangChain's standard message format
    (replaces raw dicts like {"role": "user", "content": "..."})
"""

import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from agents.state import HealthAgentState
from dotenv import load_dotenv
load_dotenv()

# ── LangChain Model Initialization ────────────────────────────────────────────
# ChatGroq is LangChain's wrapper for Groq — it handles auth, retries, and gives
# us a consistent .invoke() interface regardless of which LLM we swap in later.
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


# ── System Prompt ──────────────────────────────────────────────────────────────
SYSTEM = """You are a medical symptom extraction specialist.
Extract ONLY symptom keywords from the user's message.

Return a valid JSON array of strings. Example: ["fever", "headache", "chills"]
Return [] if no symptoms are found.
Return ONLY the JSON array — no explanation, no markdown, no extra text."""


# ── LangGraph Node Function ────────────────────────────────────────────────────
# Every LangGraph node MUST follow this signature:
#   def node_name(state: StateType) -> dict
# The dict you return is MERGED into the current state automatically.
def symptom_extraction_node(state: HealthAgentState) -> dict:
    """
    LangGraph Node: Symptom Extraction
    
    Input  (from state): state["user_input"]
    Output (to state)  : {"raw_symptoms": [...]}
    """
    print("  [Node] SymptomExtractor running...")

    # Build messages using LangChain's message objects
    # SystemMessage = instructions for the AI
    # HumanMessage  = the actual user content to process
    messages = [
        SystemMessage(content=SYSTEM),
        HumanMessage(content=state["user_input"])
    ]

    # .invoke() sends the messages to Claude and returns an AIMessage
    response = llm.invoke(messages)
    raw_text = response.content.strip()

    # Safely parse the JSON list Claude returns
    try:
        symptoms = json.loads(raw_text)
        if not isinstance(symptoms, list):
            symptoms = []
    except json.JSONDecodeError:
        symptoms = []

    print(f"  [Node] Extracted symptoms: {symptoms}")

    # Return only the fields we want to UPDATE in the state
    # LangGraph merges this dict into the existing state
    if not symptoms:
        return {
            "raw_symptoms": [],
            "error": True,
            "error_message": (
                "I couldn't identify any symptoms in your message. "
                "Please describe what you're feeling — for example: "
                "'I have fever, chills, and a headache.'"
            )
        }

    return {"raw_symptoms": symptoms, "error": False}
