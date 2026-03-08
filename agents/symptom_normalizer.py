"""
agents/symptom_normalizer.py — Symptom Normalization Node
==========================================================
WHAT IT DOES:
  Converts informal / colloquial symptom terms into proper medical terminology.
  "tummy ache" → "abdominal pain"  |  "throwing up" → "vomiting"

WHY THIS MATTERS:
  Disease prediction works better with standardized medical terms.
  Real systems match against medical ontologies (ICD-10, SNOMED CT).
  Claude's knowledge also aligns better with formal terminology.

LANGGRAPH ROLE:
  Second specialist node. Reads `raw_symptoms`, writes `normalized_symptoms`.

LANGCHAIN CONCEPT — StrOutputParser:
  LangChain's .invoke() returns an AIMessage object.
  StrOutputParser is a simple parser that extracts just the .content string.
  We can chain it:  llm | StrOutputParser()
  This is the LangChain Expression Language (LCEL) chain pattern.
"""

import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from agents.state import HealthAgentState
from dotenv import load_dotenv
load_dotenv()

# ── Model + Parser Chain (LCEL) ────────────────────────────────────────────────
# The pipe `|` operator creates a chain:
#   llm.invoke(messages) → AIMessage → StrOutputParser → plain string
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
parser = StrOutputParser()
chain = llm | parser   # LCEL chain: LLM output piped directly into the parser


# ── System Prompt ──────────────────────────────────────────────────────────────
SYSTEM = """You are a medical terminology normalization expert.
Convert informal symptom terms to standard medical terminology.

Input: JSON array of informal symptom strings
Output: JSON array of normalized medical symptom strings

Examples:
- "tummy ache"   → "abdominal pain"
- "throwing up"  → "vomiting"
- "high temp"    → "high fever"
- "burning pee"  → "dysuria"
- "can't breathe"→ "dyspnea"
- "shaking"      → "rigors/chills"

Return ONLY a valid JSON array. No explanation. No markdown."""


def symptom_normalization_node(state: HealthAgentState) -> dict:
    """
    LangGraph Node: Symptom Normalization

    Input  (from state): state["raw_symptoms"]
    Output (to state)  : {"normalized_symptoms": [...]}
    """
    print("  [Node] SymptomNormalizer running...")

    raw_symptoms = state.get("raw_symptoms", [])

    if not raw_symptoms:
        return {"normalized_symptoms": []}

    messages = [
        SystemMessage(content=SYSTEM),
        HumanMessage(content=json.dumps(raw_symptoms))
    ]

    # Using the LCEL chain — .invoke() here returns a plain string (not AIMessage)
    # because StrOutputParser automatically extracts .content
    raw_text = chain.invoke(messages)

    try:
        normalized = json.loads(raw_text.strip())
        if not isinstance(normalized, list):
            normalized = raw_symptoms  # fallback to originals
    except json.JSONDecodeError:
        normalized = raw_symptoms  # fallback to originals

    print(f"  [Node] Normalized symptoms: {normalized}")
    return {"normalized_symptoms": normalized}
