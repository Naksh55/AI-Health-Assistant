from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from agents.state import HealthAgentState
from agents.symptom_extractor import symptom_extraction_node
from agents.symptom_normalizer import symptom_normalization_node
from agents.disease_predictor import disease_prediction_node
from agents.risk_assessor import risk_assessment_node
from agents.medical_advisor import medical_advice_node
from agents.report_analyzer import report_analysis_node

from dotenv import load_dotenv
load_dotenv()

# ── Shared LLM ────────────────────────────────────────────────────────────────
_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


# ── Intent Classifier ─────────────────────────────────────────────────────────
# This is the KEY addition. Before doing anything, we classify what the user
# actually wants. This determines which path the graph takes.
#
# INTENT TYPES:
#   SIMPLE_QUESTION   → "what is hypothyroidism", "what is TSH", "what is anemia"
#   REPORT_QUERY      → "what is my cholesterol", "patient name", "my TSH value"
#   SYMPTOM_ANALYSIS  → "I have fever", "I feel tired and breathless"
#   ACTION_REQUEST    → "schedule appointment", "book a doctor"
#   REPORT_OVERVIEW   → "check my report", "what's wrong in my report"

def classify_intent(user_input: str, has_report: bool) -> str:
    """
    Classifies user intent using keyword rules first (fast),
    then falls back to LLM classification for ambiguous cases.
    """
    text = user_input.lower().strip()

    # ── Rule-based fast classification ───────────────────────────────────────

    # Action requests — scheduling, booking
    action_keywords = ["schedule", "book", "appointment", "set up appointment",
                       "book a doctor", "reserve"]
    if any(k in text for k in action_keywords):
        return "ACTION_REQUEST"

    # Report overview — user wants full report summary
    report_overview_keywords = ["check my report", "what's wrong", "whats wrong",
                                "analyze my report", "read my report", "review my report",
                                "tell me about my report", "what does my report say",
                                "summarize my report"]
    if any(k in text for k in report_overview_keywords):
        return "REPORT_OVERVIEW"

    # Report-specific value queries — user asks about THEIR specific results
    my_value_keywords = ["my cholesterol", "my hemoglobin", "my tsh", "my t4",
                         "my vitamin", "my sugar", "my creatinine", "my uric",
                         "my wbc", "my rbc", "my platelets", "my ferritin",
                         "my iron", "my hba1c", "my bilirubin", "my sgot", "my sgpt",
                         "patient name", "name of the patient", "who is the patient",
                         "patient age", "age of the patient", "my report shows",
                         "in my report", "from my report", "according to my report",
                         "what is my", "show me my", "what are my levels",
                         "my levels", "my results", "my values"]
    if any(k in text for k in my_value_keywords):
        return "REPORT_QUERY"

    # General medical education questions — no personal data involved
    education_keywords = ["what is ", "what are ", "explain ", "define ",
                          "tell me about ", "how does ", "why is ", "what does ",
                          "meaning of ", "difference between ", "what causes ",
                          "how to treat ", "how to cure ", "is it dangerous",
                          "can it be cured", "is it serious"]
    if any(text.startswith(k) or k in text for k in education_keywords):
        # Make sure they're not asking about THEIR values (already caught above)
        if "my " not in text:
            return "SIMPLE_QUESTION"

    # Symptom descriptions — personal health complaints
    symptom_keywords = ["i have", "i am having", "i feel", "i am feeling",
                        "i've been", "i've had", "since yesterday", "since morning",
                        "since last", "for the past", "for 2 days", "for 3 days",
                        "pain", "ache", "fever", "cough", "vomiting", "nausea",
                        "headache", "dizzy", "tired", "fatigue", "breathless",
                        "burning", "swelling", "rash", "bleeding", "weakness",
                        "shaking", "trembling", "cold", "hot", "chills"]
    if any(k in text for k in symptom_keywords):
        return "SYMPTOM_ANALYSIS"

    # ── LLM fallback for ambiguous cases ─────────────────────────────────────
    try:
        response = _llm.invoke([HumanMessage(content=f"""Classify this health assistant message into exactly one category.

Message: "{user_input}"
Has medical report uploaded: {has_report}

Categories:
- SIMPLE_QUESTION: General medical education (what is diabetes, explain anemia, how does thyroid work)
- REPORT_QUERY: Asking about specific values in their uploaded report (what is my cholesterol, patient name, my TSH level)
- SYMPTOM_ANALYSIS: Describing personal symptoms they are experiencing (I have fever, I feel tired)
- ACTION_REQUEST: Wants to do something like schedule appointment, book doctor
- REPORT_OVERVIEW: Wants full report summary/analysis (check my report, what's wrong)

Reply with ONLY one word: SIMPLE_QUESTION, REPORT_QUERY, SYMPTOM_ANALYSIS, ACTION_REQUEST, or REPORT_OVERVIEW""")])
        intent = response.content.strip().upper()
        if intent in ["SIMPLE_QUESTION", "REPORT_QUERY", "SYMPTOM_ANALYSIS",
                      "ACTION_REQUEST", "REPORT_OVERVIEW"]:
            return intent
    except:
        pass

    # Default fallback
    return "SYMPTOM_ANALYSIS"


# ── Supervisor Node ────────────────────────────────────────────────────────────
def supervisor_node(state: HealthAgentState) -> dict:
    print("\n[Supervisor] Pipeline started.")
    print(f"[Supervisor] User input: {state['user_input']}")

    user_input = state.get("user_input", "").strip()
    has_report = state.get("has_report", False)

    if not user_input and not has_report:
        return {
            "error": True,
            "error_message": "Please describe your symptoms to get started.",
            "intent": "ERROR"
        }

    # Classify intent
    intent = classify_intent(user_input, has_report)
    print(f"[Supervisor] Detected intent: {intent}")

    return {
        "error": False,
        "intent": intent,
        "raw_symptoms": None,
        "normalized_symptoms": None,
        "predicted_conditions": None,
        "risk_assessment": None,
        "final_response": None
    }


# ── Route Entry — based on intent ─────────────────────────────────────────────
def route_entry(state: HealthAgentState) -> str:
    intent     = state.get("intent", "SYMPTOM_ANALYSIS")
    has_report = state.get("has_report", False)

    print(f"[Router] Intent={intent}, has_report={has_report}")

    # Action requests → direct answer, no pipeline needed
    if intent == "ACTION_REQUEST":
        return "answer_query"

    # Simple educational question → direct answer, no pipeline needed
    if intent == "SIMPLE_QUESTION":
        return "answer_query"

    # User asking about their specific report values → need report first
    if intent == "REPORT_QUERY":
        if has_report:
            return "analyze_report"
        return "answer_query"  # no report uploaded, tell them

    # Full report overview → analyze report then full pipeline
    if intent == "REPORT_OVERVIEW":
        if has_report:
            return "analyze_report"
        return "answer_query"

    # Symptom analysis → full pipeline
    if intent == "SYMPTOM_ANALYSIS":
        if has_report:
            return "analyze_report"  # analyze report first, then symptoms
        return "extract_symptoms"

    return "extract_symptoms"


# ── Direct Answer Node ─────────────────────────────────────────────────────────
# Handles: SIMPLE_QUESTION, ACTION_REQUEST, REPORT_QUERY (without going through full pipeline)
def answer_query_node(state: HealthAgentState) -> dict:
    user_input      = state.get("user_input", "").strip()
    intent          = state.get("intent", "SIMPLE_QUESTION")
    report_analysis = state.get("report_analysis") or {}
    report_data     = state.get("report_data")
    has_report      = state.get("has_report", False)

    print(f"  [Node] AnswerQuery running for intent: {intent}")

    # ── FOLLOW-UP DETECTION ───────────────────────────────────────────────────
    # If user says "yes", "ok", "sure", "please" etc — check last bot message
    # and answer THAT question instead of treating it as a new query
    short_confirmations = ["yes", "ok", "sure", "please", "yeah", "yep",
                           "okay", "go ahead", "tell me", "yes please"]
    if user_input.lower().strip() in short_confirmations:
        chat_history = state.get("chat_history") or []
        last_bot_msg = ""
        for msg in reversed(chat_history):
            if msg.get("role") == "assistant":
                last_bot_msg = msg.get("content", "")
                break

        if last_bot_msg and "?" in last_bot_msg:
            # Find the question the bot asked and answer it
            response = _llm.invoke([
                SystemMessage(content="""You are a medical assistant. The patient is saying 'yes' to your previous question.
Look at your previous message, identify the question you asked, and answer it helpfully.
Be concise and direct. Sound like a doctor responding to a patient."""),
                HumanMessage(content=f"""Your previous message: "{last_bot_msg}"

The patient replied: "{user_input}"

They are confirming yes to your question. Please answer the question you asked them.""")
            ])
            return {"final_response": response.content.strip()}

    # ── ACTION REQUEST ────────────────────────────────────────────────────────
    if intent == "ACTION_REQUEST":
        text = user_input.lower()
        if any(k in text for k in ["schedule", "appointment", "book"]):
            return {"final_response": (
                "I'm not able to book appointments directly. "
                "To schedule a visit, you can:\n\n"
                "1. Call your doctor's clinic directly\n"
                "2. Use the hospital's online booking portal\n"
                "3. Try apps like Practo or Apollo 247 for online appointments\n\n"
                "Would you like me to tell you which specialist to see based on your report?"
            )}
        # Generic action
        return {"final_response": "I can't perform that action directly, but I can answer health questions or analyze your medical report."}

    # ── NO REPORT uploaded but user asked about report ────────────────────────
    if intent == "REPORT_QUERY" and not has_report:
        return {"final_response": (
            "It looks like you haven't uploaded a medical report yet. "
            "Please upload your report using the sidebar on the left, "
            "then ask your question again."
        )}

    # ── SIMPLE EDUCATIONAL QUESTION ───────────────────────────────────────────
    if intent == "SIMPLE_QUESTION":
        response = _llm.invoke([
            SystemMessage(content="""You are a knowledgeable medical doctor explaining things to a patient.
Answer the question clearly and conversationally — like a doctor talking to a patient in a clinic.

Rules:
- Give a direct, focused answer to exactly what was asked
- Use simple language, avoid heavy jargon
- Keep it concise — 3 to 6 sentences max
- Do NOT add self-care tips, warning signs, or next steps unless specifically asked
- Do NOT use bullet points for simple explanations — use natural flowing sentences
- End with one short practical note if relevant
- Always add: "If you have concerns, speak with your doctor." at the end"""),
            HumanMessage(content=user_input)
        ])
        return {"final_response": response.content.strip()}

    # ── REPORT QUERY — specific value from report ─────────────────────────────
    if intent == "REPORT_QUERY" and report_analysis:

        # Build rich context from report
        context_parts = []

        # Add key findings with exact values
        key_findings = report_analysis.get("key_findings", [])
        if key_findings:
            findings_text = "\n".join([
                f"{kf.get('parameter')}: {kf.get('value')} (Normal: {kf.get('normal_range')}) "
                f"[{kf.get('status')}] — {kf.get('significance', '')}"
                for kf in key_findings
            ])
            context_parts.append(f"Key findings from report:\n{findings_text}")

        # Add patient info if available
        if report_analysis.get("patient_name"):
            context_parts.append(f"Patient name: {report_analysis.get('patient_name')}")

        # Add raw report text for name/age extraction
        if report_data and isinstance(report_data, str):
            context_parts.append(f"Report text excerpt:\n{report_data[:2000]}")

        # Add summary
        if report_analysis.get("summary"):
            context_parts.append(f"Report summary: {report_analysis.get('summary')}")

        context = "\n\n".join(context_parts)

        response = _llm.invoke([
            SystemMessage(content="""You are a medical doctor reviewing a patient's report with them.
Answer their specific question directly using the report data provided.

Rules:
- Answer ONLY what was asked — do not add unrelated information
- Quote exact values from the report (e.g., "Your LDL is 138 mg/dL, normal is below 100")
- Explain what the value means in 1-2 simple sentences
- If the value is normal, reassure them. If abnormal, note it calmly
- Do NOT give full health advice, self-care tips, or next steps unless asked
- Sound like a doctor talking naturally to their patient
- Keep response to 3-5 sentences maximum
- End with: "Let me know if you have more questions about your report." """),
            HumanMessage(content=f"Patient question: {user_input}\n\nReport data:\n{context}")
        ])
        return {"final_response": response.content.strip()}

    # Fallback
    return {"final_response": "I'm not sure how to answer that. Could you rephrase or upload your medical report?"}


# ── After report analysis — route based on original intent ───────────────────
def route_after_report(state: HealthAgentState) -> str:
    intent = state.get("intent", "SYMPTOM_ANALYSIS")

    # For simple report queries — go to answer_query with report context now loaded
    if intent in ("REPORT_QUERY", "ACTION_REQUEST"):
        return "answer_query"

    # For report overview — generate full advice using report
    if intent == "REPORT_OVERVIEW":
        return "extract_symptoms"

    # For symptom analysis with report — proceed to symptom extraction
    return "extract_symptoms"


# ── Emergency Fast-Path Node ──────────────────────────────────────────────────
def emergency_response_node(state: HealthAgentState) -> dict:
    print("  [Node] EMERGENCY fast-path triggered!")

    risk     = state.get("risk_assessment", {})
    symptoms = state.get("normalized_symptoms", [])

    messages = [
        SystemMessage(content="""You are an emergency medical triage assistant.
The patient has EMERGENCY-level symptoms. Respond like an urgent doctor.
Be direct, clear, and calm but serious.
Tell them exactly what to do right now.
Do NOT use bullet point lists — speak naturally and urgently.
End with the medical disclaimer."""),
        HumanMessage(content=f"""
Symptoms: {', '.join(symptoms)}
Risk reason: {risk.get('reason', '')}
Action needed: {risk.get('action', 'Call emergency services immediately')}
""")
    ]

    response = _llm.invoke(messages)
    return {"final_response": f"🚨 **EMERGENCY — Call for help immediately**\n\n{response.content}"}


# ── Conditional Routing Functions ─────────────────────────────────────────────
def route_after_extraction(state: HealthAgentState) -> str:
    has_report   = state.get("has_report", False)
    has_symptoms = bool(state.get("raw_symptoms"))
    error        = state.get("error", False)
    intent       = state.get("intent", "SYMPTOM_ANALYSIS")

    # Report overview with no symptoms — still run pipeline for report-based advice
    if not has_symptoms and has_report and intent == "REPORT_OVERVIEW":
        return "normalize_symptoms"

    if not has_symptoms and has_report:
        return "normalize_symptoms"

    if error or not has_symptoms:
        return "end_with_error"

    return "normalize_symptoms"


def route_by_risk_level(state: HealthAgentState) -> str:
    risk  = state.get("risk_assessment", {})
    level = risk.get("risk_level", "MEDIUM")

    if level == "EMERGENCY":
        print("[Router] EMERGENCY detected — routing to emergency node")
        return "emergency_response"

    print(f"[Router] Risk={level} — routing to normal advice")
    return "generate_advice"


# ── Error Termination Node ────────────────────────────────────────────────────
def end_with_error_node(state: HealthAgentState) -> dict:
    return {"final_response": state.get("error_message",
        "I couldn't identify any symptoms. Please describe what you're feeling, "
        "for example: 'I have fever and headache since yesterday.'")}


# ── Build the Graph ───────────────────────────────────────────────────────────
def build_health_graph():
    graph = StateGraph(HealthAgentState)

    # Register nodes
    graph.add_node("supervisor",         supervisor_node)
    graph.add_node("analyze_report",     report_analysis_node)
    graph.add_node("answer_query",       answer_query_node)
    graph.add_node("extract_symptoms",   symptom_extraction_node)
    graph.add_node("normalize_symptoms", symptom_normalization_node)
    graph.add_node("predict_disease",    disease_prediction_node)
    graph.add_node("assess_risk",        risk_assessment_node)
    graph.add_node("generate_advice",    medical_advice_node)
    graph.add_node("emergency_response", emergency_response_node)
    graph.add_node("end_with_error",     end_with_error_node)

    # Entry
    graph.add_edge(START, "supervisor")

    # After supervisor — route by intent
    graph.add_conditional_edges(
        "supervisor",
        route_entry,
        {
            "analyze_report":   "analyze_report",
            "extract_symptoms": "extract_symptoms",
            "answer_query":     "answer_query"
        }
    )

    # After report analysis — route based on intent
    graph.add_conditional_edges(
        "analyze_report",
        route_after_report,
        {
            "answer_query":     "answer_query",
            "extract_symptoms": "extract_symptoms"
        }
    )

    # Direct query → END
    graph.add_edge("answer_query", END)

    # Symptom pipeline
    graph.add_conditional_edges(
        "extract_symptoms",
        route_after_extraction,
        {
            "end_with_error":     "end_with_error",
            "normalize_symptoms": "normalize_symptoms"
        }
    )

    graph.add_edge("end_with_error",     END)
    graph.add_edge("normalize_symptoms", "predict_disease")
    graph.add_edge("predict_disease",    "assess_risk")

    graph.add_conditional_edges(
        "assess_risk",
        route_by_risk_level,
        {
            "emergency_response": "emergency_response",
            "generate_advice":    "generate_advice"
        }
    )

    graph.add_edge("generate_advice",    END)
    graph.add_edge("emergency_response", END)

    return graph.compile()


# Singleton
health_graph = build_health_graph()