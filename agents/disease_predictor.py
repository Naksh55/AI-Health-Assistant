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
    LangGraph Node: Disease Prediction (HYBRID ML + LLM)

    Input  (from state): state["normalized_symptoms"] + state["report_analysis"]
    Output (to state)  : {"predicted_conditions": [...], "ml_predictions": [...]}

    HYBRID APPROACH:
      1. Run ML model (RandomForest) for statistical prediction
      2. Pass ML predictions + report data as context to LLM
      3. LLM validates/refines ML predictions with clinical reasoning
      4. Final output combines both — ML gives backing, LLM gives reasoning
    """
    print("  [Node] DiseasePredictor running...")

    normalized      = state.get("normalized_symptoms", []) or []
    raw_symptoms    = state.get("raw_symptoms", []) or []
    report_analysis = state.get("report_analysis")

    # ── Step 1: ML Model Prediction ──────────────────────────────────────────
    ml_predictions = []
    ml_context = "No ML model predictions available."
    try:
        from ml.predictor import MLDiseasePredictor
        predictor = MLDiseasePredictor()
        # Combine raw + normalized symptoms for best matching coverage
        # Raw symptoms match Kaggle features directly ("headache", "back_pain")
        # Normalized symptoms get mapped via synonym dictionary ("dysuria" → "burning_micturition")
        combined_symptoms = list(set(raw_symptoms + normalized))
        if predictor.is_ready and combined_symptoms:
            ml_predictions = predictor.predict(combined_symptoms, top_k=3)
            if ml_predictions:
                ml_lines = [
                    f"  - {p['name']} (ML confidence: {p['ml_confidence']}, "
                    f"probability: {p['probability']:.1%})"
                    for p in ml_predictions
                ]
                ml_context = (
                    "ML Model Predictions (RandomForest, use as statistical reference):\n"
                    + "\n".join(ml_lines)
                )
                print(f"  [Node] ML predictions: {[p['name'] for p in ml_predictions]}")
    except Exception as e:
        print(f"  [Node] ML prediction skipped: {e}")

    # ── Step 2: Build report section for the prompt ──────────────────────────
    report_section = ""
    if report_analysis:
        abnormal    = report_analysis.get("abnormal_findings", [])
        key_findings = report_analysis.get("key_findings", [])

        if key_findings:
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
            report_section = (
                "Medical Report — Abnormal Findings (USE THESE AS PRIMARY DIAGNOSIS BASIS):\n"
                + "\n".join([f"  - {f}" for f in abnormal])
            )

    if not report_section:
        report_section = "No medical report available — diagnose based on symptoms only."

    # ── Handle no symptoms case ───────────────────────────────────────────────
    if not normalized and not report_analysis:
        return {"predicted_conditions": [], "ml_predictions": ml_predictions}

    symptoms_str = ", ".join(normalized) if normalized else "No specific symptoms described"

    # ── Step 3: LLM Prediction (with ML context) ─────────────────────────────
    # Inject ML predictions into the report_section so LLM considers them
    combined_section = f"{report_section}\n\n{ml_context}"

    raw_text = chain.invoke({
        "symptoms":       symptoms_str,
        "report_section": combined_section
    })

    try:
        clean      = raw_text.strip().replace("```json", "").replace("```", "")
        result     = json.loads(clean)
        conditions = result.get("conditions", [])
    except json.JSONDecodeError:
        conditions = []

    print(f"  [Node] Predicted (hybrid): {[c.get('name') for c in conditions]}")
    return {"predicted_conditions": conditions, "ml_predictions": ml_predictions}