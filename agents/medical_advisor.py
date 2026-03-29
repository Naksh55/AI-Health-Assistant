"""
agents/medical_advisor.py — Medical Advice Node
================================================
KEY FIXES:
  1. Prompt now STRICTLY requires all 4 sections to have content
  2. Added fallback content generator if LLM leaves sections empty
  3. REPORT_OVERVIEW gives natural doctor-style response
  4. Conversation context passed for follow-up awareness
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from agents.state import HealthAgentState


llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


def safe_join(value):
    if isinstance(value, list):
        return ", ".join(map(str, value)) if value else "None"
    elif isinstance(value, str):
        return value
    return "None"


# ── MAIN ADVICE SYSTEM PROMPT ─────────────────────────────────────────────────
ADVICE_SYSTEM = """You are a warm, experienced medical doctor speaking directly to your patient.

CRITICAL RULES — YOU MUST FOLLOW THESE:
1. You MUST fill ALL 4 sections below — never leave any section empty
2. Every section MUST have exactly 3 bullet points (no more, no less)
3. Each bullet point must be a complete, meaningful sentence
4. Reference the patient's SPECIFIC symptoms and report values — not generic advice
5. Speak like a doctor, not a robot

FORMAT — use EXACTLY these section headers, word for word, no changes:

## 🩺 WHAT IS GOING ON
[Write 2-3 sentences explaining the likely condition. Reference actual values if available.
Example: "Based on your symptoms and the report showing Hb of 8.4 g/dL, you have iron deficiency anemia..."]

## 💊 SELF-CARE TIPS
- [Specific self-care tip 1 relevant to their condition]
- [Specific self-care tip 2 relevant to their condition]
- [Specific self-care tip 3 relevant to their condition]

## ⚠️ WARNING SIGNS
- [Warning sign 1 — specific to their condition]
- [Warning sign 2 — specific to their condition]
- [Warning sign 3 — when to seek immediate help]

## 📋 NEXT STEPS
- [Action item 1 — most urgent]
- [Action item 2 — follow-up]
- [Action item 3 — monitoring/prevention]

---
⚠️ *AI-generated · Not a medical diagnosis · Always consult a qualified doctor*
---

REMINDER: All 4 sections are MANDATORY. Never skip any section.
If you cannot think of relevant tips, give general but helpful ones.
Do NOT use the emergency format here — this is for non-emergency cases only."""


prompt = ChatPromptTemplate.from_messages([
    ("system", ADVICE_SYSTEM),
    ("human", """Patient said: "{user_input}"

Symptoms found: {symptoms}
Risk Level: {risk_level} — {risk_reason}
Next action: {risk_action}

Likely conditions:
{conditions_text}

{report_context}

Exact report values for reference:
{report_summary}

Recent conversation context (for follow-up awareness):
{conversation_context}

Now write your response with ALL 4 sections filled completely.""")
])

chain = prompt | llm | StrOutputParser()


def _build_report_context(report_analysis: dict) -> tuple[str, str]:
    """Builds report_context and report_summary strings from report_analysis dict."""
    if not report_analysis:
        return "", "No medical report uploaded."

    abnormal      = report_analysis.get("abnormal_findings") or []
    normal        = report_analysis.get("normal_findings") or []
    doc_notes     = report_analysis.get("doctor_notes", "")
    urgency       = report_analysis.get("urgency_level", "ROUTINE")
    summary       = report_analysis.get("summary", "")
    meds_text     = safe_join(report_analysis.get("medications_mentioned"))
    followup_text = safe_join(report_analysis.get("recommended_tests"))

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
Follow-up Tests: {followup_text}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    key_findings = report_analysis.get("key_findings") or []
    if key_findings:
        report_summary = "\n".join([
            f"  {kf.get('parameter')}: {kf.get('value')} "
            f"(Normal: {kf.get('normal_range')}) [{kf.get('status')}]"
            for kf in key_findings
        ])
    else:
        report_summary = (
            f"Type: {report_analysis.get('report_type', 'Unknown')}\n"
            f"Summary: {report_analysis.get('patient_friendly_summary', '')}\n"
            f"Abnormal: {safe_join(abnormal)}"
        )

    return report_context, report_summary


def _ensure_sections_filled(response: str, conditions: list, symptoms: list) -> str:
    """
    Fallback: if LLM left any section empty, inject default content.
    This prevents the empty card bug seen in the UI.
    """
    import re

    condition_name = conditions[0].get("name", "the condition") if conditions else "your condition"
    symptom_str    = ", ".join(symptoms[:3]) if symptoms else "your symptoms"

    # Default fallbacks for each section
    defaults = {
        "SELF-CARE TIPS": [
            f"Rest adequately and avoid strenuous activity until your symptoms improve.",
            f"Stay well-hydrated by drinking 8-10 glasses of water daily.",
            f"Eat a balanced, nutritious diet to support your recovery.",
        ],
        "WARNING SIGNS": [
            f"Sudden worsening of {symptom_str} — seek medical help immediately.",
            f"High fever above 103°F (39.4°C) that does not come down.",
            f"Difficulty breathing, chest pain, or loss of consciousness.",
        ],
        "NEXT STEPS": [
            f"Consult your doctor to discuss treatment options for {condition_name}.",
            f"Get any recommended follow-up tests done promptly.",
            f"Monitor your symptoms and return for a check-up as advised.",
        ],
    }

    for section_name, fallback_bullets in defaults.items():
        # Check if section exists but has no bullets after it
        pattern = rf"(##[^\n]*{re.escape(section_name.split()[0])}[^\n]*\n)([ \t]*\n|$)"
        if re.search(pattern, response, re.IGNORECASE):
            bullets = "\n".join([f"- {b}" for b in fallback_bullets])
            response = re.sub(
                pattern,
                rf"\1{bullets}\n",
                response,
                flags=re.IGNORECASE
            )

    return response


# ── MAIN NODE ─────────────────────────────────────────────────────────────────
def medical_advice_node(state: HealthAgentState) -> dict:

    intent          = state.get("intent", "SYMPTOM_ANALYSIS")
    conditions      = state.get("predicted_conditions") or []
    risk            = state.get("risk_assessment") or {}
    report_analysis = state.get("report_analysis")
    user_input      = state.get("user_input", "")
    chat_history    = state.get("chat_history") or []

    print(f"  [Node] MedicalAdvisor running for intent: {intent}")

    # ── Conditions text ───────────────────────────────────────────────────────
    conditions_text = "\n".join([
        f"  • {c.get('name', 'Unknown')} ({c.get('probability', '')}): {c.get('reasoning', '')}"
        for c in conditions
    ]) or "  • No specific conditions identified"

    # ── Report context ────────────────────────────────────────────────────────
    report_context, report_summary = _build_report_context(report_analysis)

    # ── Conversation context for follow-up awareness ──────────────────────────
    # Pass last 3 exchanges so the model understands "yes" type follow-ups
    conversation_context = "No previous conversation."
    if chat_history:
        recent = chat_history[-6:]  # last 3 pairs
        lines  = []
        for msg in recent:
            role    = "Patient" if msg.get("role") == "user" else "Doctor"
            content = str(msg.get("content", ""))[:200]  # truncate long messages
            lines.append(f"{role}: {content}")
        conversation_context = "\n".join(lines)

    # ── REPORT OVERVIEW — natural doctor summary ──────────────────────────────
    if intent == "REPORT_OVERVIEW" and report_analysis:
        abnormal_list = report_analysis.get("abnormal_findings") or []
        key_findings  = report_analysis.get("key_findings") or []

        findings_text = ""
        if key_findings:
            findings_text = "\n".join([
                f"{kf.get('parameter')}: {kf.get('value')} "
                f"(Normal: {kf.get('normal_range')}) [{kf.get('status')}] — {kf.get('significance','')}"
                for kf in key_findings
                if kf.get("status") in ("LOW", "HIGH", "ABNORMAL")
            ])

        response = llm.invoke([
            SystemMessage(content="""You are a doctor reviewing a patient's medical report with them in a clinic.
Explain the report findings conversationally — like you are sitting across from them.

Rules:
- Start by acknowledging you have reviewed their report
- Walk through the KEY abnormal findings naturally using actual values
- Explain what each abnormal finding means in plain language
- Group related findings (e.g., anemia group together, thyroid together)
- End with overall urgency and 2-3 clear next steps
- Sound warm and professional, not robotic
- Do NOT use the rigid 4-section format with headers
- Use natural paragraphs — 150 to 200 words total
- Always add: AI-generated disclaimer at the end"""),
            HumanMessage(content=f"""Patient said: "{user_input}"

Report type: {report_analysis.get('report_type', 'Unknown')}
Report summary: {report_analysis.get('summary', '')}
Urgency: {report_analysis.get('urgency_level', 'ROUTINE')}

Abnormal findings with values:
{findings_text or safe_join(abnormal_list)}

Doctor notes: {report_analysis.get('doctor_notes', 'None')}

Give a warm, natural doctor-style explanation.""")
        ])

        final = response.content.strip()
        final += "\n\n---\n⚠️ *AI-generated · Not a medical diagnosis · Always consult a qualified doctor*"
        return {"final_response": final}

    # ── SYMPTOM ANALYSIS — full 4-section structured response ─────────────────
    response = chain.invoke({
        "user_input":             user_input,
        "symptoms":               ", ".join(state.get("normalized_symptoms") or []),
        "risk_level":             risk.get("risk_level", "MEDIUM"),
        "risk_reason":            risk.get("reason", ""),
        "risk_action":            risk.get("action", "Consult a healthcare provider"),
        "conditions_text":        conditions_text,
        "report_context":         report_context,
        "report_summary":         report_summary,
        "conversation_context":   conversation_context,
    })

    # Fallback: ensure no empty sections reach the UI
    response = _ensure_sections_filled(response, conditions, state.get("normalized_symptoms") or [])

    print("  [Node] Medical advice generated.")
    return {"final_response": response}