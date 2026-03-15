"""
agents/disease_predictor.py — Disease Prediction Node
======================================================
WHAT IT DOES:
  Takes normalized symptoms AND report findings (if available)
  and predicts the top 3 most likely medical conditions.

KEY FIX:
  Now receives report_analysis from state and passes abnormal
  findings directly into the prompt — so the model prioritizes
  report data over generic symptom matching.

  Before fix: "tired + breathless" → Pneumonia, Bronchitis (WRONG)
  After fix:  "tired + breathless" + "Hb=8.4, TSH=8.92" → Anemia + Hypothyroidism (CORRECT)
"""

from dotenv import load_dotenv
load_dotenv()

import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agents.state import HealthAgentState


prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a clinical diagnosis assistant with deep medical knowledge.

IMPORTANT INSTRUCTION:
- If a medical report with abnormal findings is provided, you MUST use those 
  findings as the PRIMARY basis for your diagnosis.
- Symptoms alone can be misleading — lab values are more objective evidence.
- Connect the symptoms to the report abnormalities wherever possible.

Given symptoms and optional report findings, identify the top 3 most likely conditions.

Respond ONLY with a valid JSON object in this exact format:
{{
  "conditions": [
    {{
      "name": "Iron Deficiency Anemia",
      "probability": "High",
      "reasoning": "Hb 8.4 g/dL (low), low ferritin, low MCV directly confirms this"
    }},
    {{
      "name": "Hypothyroidism",
      "probability": "High",
      "reasoning": "TSH 8.92 (elevated) + low Free T4 + high Anti-TPO = autoimmune hypothyroidism"
    }},
    {{
      "name": "Vitamin D Deficiency",
      "probability": "Medium",
      "reasoning": "Vit D 11.4 ng/mL is severely deficient, causing fatigue and weakness"
    }}
  ]
}}

No markdown. No explanation. Only the JSON object."""),

    ("human", """Patient Symptoms: {symptoms}

{report_section}

Based on BOTH the symptoms and report findings above, what are the top 3 most likely conditions?
Remember: if report data is available, it should GUIDE your diagnosis more than symptoms alone.""")
])

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
chain = prompt | llm | StrOutputParser()


def disease_prediction_node(state: HealthAgentState) -> dict:
    """
    LangGraph Node: Disease Prediction

    Input  (from state): state["normalized_symptoms"] + state["report_analysis"]
    Output (to state)  : {"predicted_conditions": [...]}

    KEY CHANGE: now reads report_analysis from state and injects
    abnormal findings into the prompt for accurate diagnosis.
    """
    print("  [Node] DiseasePredictor running...")

    normalized      = state.get("normalized_symptoms", []) or []
    report_analysis = state.get("report_analysis")

    # ── Build report section for the prompt ──────────────────────────────────
    report_section = ""
    if report_analysis:
        abnormal    = report_analysis.get("abnormal_findings", [])
        key_findings = report_analysis.get("key_findings", [])

        if key_findings:
            # Use detailed key findings with actual values
            lines = []
            for kf in key_findings:
                param  = kf.get("parameter", "")
                value  = kf.get("value", "")
                ref    = kf.get("normal_range", "")
                status = kf.get("status", "")
                sig    = kf.get("significance", "")
                lines.append(f"  - {param}: {value} (Ref: {ref}) [{status}] — {sig}")
            report_section = (
                "Medical Report Findings (USE THESE AS PRIMARY DIAGNOSIS BASIS):\n"
                + "\n".join(lines)
            )
        elif abnormal:
            # Fallback to just abnormal list
            report_section = (
                "Medical Report — Abnormal Findings (USE THESE AS PRIMARY DIAGNOSIS BASIS):\n"
                + "\n".join([f"  - {f}" for f in abnormal])
            )

    if not report_section:
        report_section = "No medical report available — diagnose based on symptoms only."

    # ── Handle no symptoms case ───────────────────────────────────────────────
    if not normalized and not report_analysis:
        return {"predicted_conditions": []}

    symptoms_str = ", ".join(normalized) if normalized else "No specific symptoms described"

    raw_text = chain.invoke({
        "symptoms":       symptoms_str,
        "report_section": report_section
    })

    try:
        clean      = raw_text.strip().replace("```json", "").replace("```", "")
        result     = json.loads(clean)
        conditions = result.get("conditions", [])
    except json.JSONDecodeError:
        conditions = []

    print(f"  [Node] Predicted: {[c.get('name') for c in conditions]}")
    return {"predicted_conditions": conditions}