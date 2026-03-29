"""
agents/medical_advisor.py — Medical Advice Node
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agents.state import HealthAgentState


# ── PROMPT ─────────────────────────────────────────────
ADVICE_SYSTEM = """You are an AI medical assistant.

Respond in a clean, structured, and concise format suitable for a health app UI.

Rules:

* Use SIMPLE and clear language
* Avoid long paragraphs and heavy medical jargon
* Keep explanations informative but not verbose

Structure:

## What Might Be Going On

* Write 1–2 short sentences explaining the likely condition
* Keep it slightly descriptive but easy to understand

## Self-Care Tips

* Provide up to 3 bullet points
* Each bullet must be short (max 12–15 words)
* Make them practical and actionable

## Seek Help If

* Provide up to 3 bullet points
* Keep them clear warning signs (short but meaningful)

## Next Steps

* Provide up to 3 bullet points
* Focus on what the user should do next

Constraints:

* Do NOT exceed 3 bullets per section
* Do NOT write long paragraphs
* Prefer clarity over detail
* If needed, summarize automatically

---
⚠️ *AI-generated · Not a medical diagnosis · Consult a qualified doctor*
---
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", ADVICE_SYSTEM),
    ("human", """Patient said: "{user_input}"

Symptoms: {symptoms}
Risk Level: {risk_level} — {risk_reason}
Recommended Action: {risk_action}

Predicted Conditions:
{conditions_text}

{report_context}

Report Values:
{report_summary}

Generate response.""")
])


llm   = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
chain = prompt | llm | StrOutputParser()


# ── HELPER FUNCTION (🔥 FIX) ────────────────────────────
def safe_join(value):
    if isinstance(value, list):
        return ", ".join(map(str, value)) if value else "None"
    elif isinstance(value, str):
        return value
    else:
        return "None"


# ── MAIN NODE ───────────────────────────────────────────
def medical_advice_node(state: HealthAgentState) -> dict:

    conditions      = state.get("predicted_conditions", [])
    risk            = state.get("risk_assessment", {})
    report_analysis = state.get("report_analysis")

    # CONDITIONS TEXT
    conditions_text = "\n".join([
        f"  • {c.get('name', 'Unknown')} ({c.get('probability', '')}): {c.get('reasoning', '')}"
        for c in conditions
    ]) or "  • No specific conditions identified"


    report_context = ""
    report_summary = "No medical report uploaded."

    # ── REPORT HANDLING ────────────────────────────────
    if report_analysis:

        abnormal  = report_analysis.get("abnormal_findings", [])
        normal    = report_analysis.get("normal_findings", [])
        doc_notes = report_analysis.get("doctor_notes", "")
        urgency   = report_analysis.get("urgency_level", "ROUTINE")
        summary   = report_analysis.get("summary", "")

        # 🔥 SAFE JOIN FIXES
        meds_text      = safe_join(report_analysis.get("medications_mentioned"))
        followup_text  = safe_join(report_analysis.get("recommended_tests"))

        report_context = f"""
━━━ MEDICAL REPORT (PRIORITIZE THIS) ━━━
Type: {report_analysis.get('report_type', 'Unknown')} · Urgency: {urgency}
Summary: {summary}

Abnormal Findings:
{chr(10).join([f"  ⚠ {f}" for f in abnormal]) if abnormal else "  None"}

Normal Findings:
{chr(10).join([f"  ✓ {f}" for f in normal[:5]]) if normal else "  None"}

Doctor Notes: {doc_notes or "None"}
Medications: {meds_text}
Follow-up: {followup_text}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

        key_findings = report_analysis.get("key_findings", [])

        if key_findings:
            report_summary = "\n".join([
                f"  {kf.get('parameter')}: {kf.get('value')} (Normal: {kf.get('normal_range')}) [{kf.get('status')}]"
                for kf in key_findings
            ])
        else:
            report_summary = (
                f"Type: {report_analysis.get('report_type', 'Unknown')}\n"
                f"Summary: {report_analysis.get('patient_friendly_summary', '')}\n"
                f"Abnormal: {safe_join(abnormal)}"
            )


    # ── LLM CALL ───────────────────────────────────────
    response = chain.invoke({
        "user_input":      state.get("user_input", ""),
        "symptoms":        ", ".join(state.get("normalized_symptoms") or []),
        "risk_level":      risk.get("risk_level", "MEDIUM"),
        "risk_reason":     risk.get("reason", ""),
        "risk_action":     risk.get("action", "Consult a healthcare provider"),
        "conditions_text": conditions_text,
        "report_context":  report_context,
        "report_summary":  report_summary,
    })

    print("  [Node] Medical advice generated.")
    return {"final_response": response}