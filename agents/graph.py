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
def classify_intent(user_input: str, has_report: bool) -> str:
    """
    Classifies user intent using keyword rules first (fast),
    then falls back to LLM classification for ambiguous cases.

    INTENT TYPES:
      SIMPLE_QUESTION  → "what is hypothyroidism"
      REPORT_QUERY     → "what is my cholesterol", "patient name"
      SYMPTOM_ANALYSIS → "I have fever and chills"
      ACTION_REQUEST   → "schedule appointment"
      REPORT_OVERVIEW  → "check my report", "what's wrong"

    FIX: If message contains BOTH symptoms AND report keywords,
    SYMPTOM_ANALYSIS always wins so the user gets the full 4-section
    structured response with cards — not just a plain paragraph.
    """
    text = user_input.lower().strip()
    
 # ── FIX: Catch follow-up confirmations FIRST before anything else ─────────
    # These short words mean the user is replying to a previous bot question.
    # Route to answer_query so the follow-up detection logic can handle them.
    short_confirmations = [
        "yes", "ok", "sure", "please", "yeah", "yep", "okay",
        "go ahead", "tell me", "yes please", "no", "nope", "skip"
    ]
    if text in short_confirmations:
        print(f"[Classifier] Short confirmation '{text}' → ACTION_REQUEST for follow-up handling")
        return "ACTION_REQUEST"  # routes to answer_query_node which has follow-up detection
    # ─────────────────────────────────────────────────────────────────────────

    # ── Action requests ───────────────────────────────────────────────────────
    action_keywords = ["schedule", "book", "appointment", "set up appointment",
                       "book a doctor", "reserve"]
    if any(k in text for k in action_keywords):
        return "ACTION_REQUEST"

    # ── FIX 1: Mixed message detection ───────────────────────────────────────
    # If user describes symptoms AND mentions their report together,
    # treat as SYMPTOM_ANALYSIS so they get the full structured card response.
    # Example: "I feel tired and cold, check my report" → SYMPTOM_ANALYSIS
    symptom_indicators = [
        "i have", "i feel", "i am", "i've", "i am having",
        "tired", "fatigue", "hair", "cold", "pain", "ache",
        "fever", "breathless", "dizzy", "weak", "nausea",
        "vomiting", "cough", "burning", "shaking", "chills",
        "swelling", "rash", "bleeding", "headache", "tingling"
    ]
    report_mention_keywords = [
        "check my report", "my report", "in my report",
        "from my report", "what's wrong", "whats wrong"
    ]
    has_symptoms_in_text  = any(k in text for k in symptom_indicators)
    has_report_mention    = any(k in text for k in report_mention_keywords)

    if has_symptoms_in_text and has_report_mention and has_report:
        print("[Classifier] Mixed symptom+report message → SYMPTOM_ANALYSIS")
        return "SYMPTOM_ANALYSIS"

    # ── Report overview ───────────────────────────────────────────────────────
    report_overview_keywords = ["check my report", "what's wrong", "whats wrong",
                                "analyze my report", "read my report", "review my report",
                                "tell me about my report", "what does my report say",
                                "summarize my report"]
    if any(k in text for k in report_overview_keywords):
        return "REPORT_OVERVIEW"

    # ── Report-specific value queries ─────────────────────────────────────────
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

    # ── General medical education ─────────────────────────────────────────────
    education_keywords = ["what is ", "what are ", "explain ", "define ",
                          "tell me about ", "how does ", "why is ", "what does ",
                          "meaning of ", "difference between ", "what causes ",
                          "how to treat ", "how to cure ", "is it dangerous",
                          "can it be cured", "is it serious"]
    if any(text.startswith(k) or k in text for k in education_keywords):
        if "my " not in text:
            return "SIMPLE_QUESTION"

    # ── Symptom descriptions ──────────────────────────────────────────────────
    symptom_keywords = ["i have", "i am having", "i feel", "i am feeling",
                        "i've been", "i've had", "since yesterday", "since morning",
                        "since last", "for the past", "for 2 days", "for 3 days",
                        "pain", "ache", "fever", "cough", "vomiting", "nausea",
                        "headache", "dizzy", "tired", "fatigue", "breathless",
                        "burning", "swelling", "rash", "bleeding", "weakness",
                        "shaking", "trembling", "cold", "hot", "chills"]
    if any(k in text for k in symptom_keywords):
        return "SYMPTOM_ANALYSIS"

    # ── LLM fallback ──────────────────────────────────────────────────────────
    try:
        response = _llm.invoke([HumanMessage(content=f"""Classify this health assistant message into exactly one category.

Message: "{user_input}"
Has medical report uploaded: {has_report}

Categories:
- SIMPLE_QUESTION: General medical education (what is diabetes, explain anemia)
- REPORT_QUERY: Asking about specific values in their report (what is my cholesterol, patient name)
- SYMPTOM_ANALYSIS: Describing personal symptoms (I have fever, I feel tired)
- ACTION_REQUEST: Wants to book/schedule something
- REPORT_OVERVIEW: Wants full report summary (check my report, what's wrong)

Reply with ONLY one word.""")])
        intent = response.content.strip().upper()
        if intent in ["SIMPLE_QUESTION", "REPORT_QUERY", "SYMPTOM_ANALYSIS",
                      "ACTION_REQUEST", "REPORT_OVERVIEW"]:
            return intent
    except:
        pass

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


# ── Route Entry ────────────────────────────────────────────────────────────────
def route_entry(state: HealthAgentState) -> str:
    intent     = state.get("intent", "SYMPTOM_ANALYSIS")
    has_report = state.get("has_report", False)

    print(f"[Router] Intent={intent}, has_report={has_report}")

    if intent == "ACTION_REQUEST":
        # FIX: If report uploaded, analyze it first so follow-up
        # specialist recommendations have actual report data
        if has_report:
            return "analyze_report"   # ← was "answer_query" before
        return "answer_query"

    if intent == "SIMPLE_QUESTION":
        return "answer_query"

    if intent == "REPORT_QUERY":
        return "analyze_report" if has_report else "answer_query"

    if intent == "REPORT_OVERVIEW":
        return "analyze_report" if has_report else "answer_query"

    if intent == "SYMPTOM_ANALYSIS":
        return "analyze_report" if has_report else "extract_symptoms"

    return "extract_symptoms"


# ── Direct Answer Node ─────────────────────────────────────────────────────────
def answer_query_node(state: HealthAgentState) -> dict:
    user_input      = state.get("user_input", "").strip()
    intent          = state.get("intent", "SIMPLE_QUESTION")
    report_analysis = state.get("report_analysis") or {}
    report_data     = state.get("report_data")
    has_report      = state.get("has_report", False)

    print(f"  [Node] AnswerQuery running for intent: {intent}")

    # ── Follow-up detection ("yes", "ok", "sure") ─────────────────────────────
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
            # Build report context so specialist recommendations are accurate
            report_context = "No report uploaded."
            if report_analysis:
                key_findings = report_analysis.get("key_findings") or []
                abnormal     = report_analysis.get("abnormal_findings") or []
                if key_findings:
                    # Get only abnormal findings with actual values
                    abnormal_lines = [
                        f"{kf.get('parameter')}: {kf.get('value')} "
                        f"(Normal: {kf.get('normal_range')}) [{kf.get('status')}] — {kf.get('significance','')}"
                        for kf in key_findings
                        if kf.get("status") in ("LOW", "HIGH", "ABNORMAL")
                    ]
                    report_context = "".join(abnormal_lines) if abnormal_lines else "All values normal"
                elif abnormal:
                    report_context = ", ".join(abnormal)

            response = _llm.invoke([
                SystemMessage(content="""You are a medical doctor answering a patient's follow-up question.
The patient confirmed 'yes' to your previous question — answer it specifically.

CRITICAL: If recommending specialists, use the REPORT DATA to name the CORRECT ones:
- Low Hemoglobin / Iron / Ferritin → Hematologist
- High TSH / Low T4 / Anti-TPO → Endocrinologist  
- Bacteria in Urine / UTI → Urologist or General Physician
- High Cholesterol / Lipids → Cardiologist or General Physician
- Low Vitamin D / B12 → General Physician or Endocrinologist

Always recommend based on what the report actually shows.
Be specific, warm, and concise. 3-5 sentences maximum."""),
                HumanMessage(content=f"""Your previous question to the patient: "{last_bot_msg}"
Patient replied: "{user_input}"

Patient's report abnormal findings:
{report_context}

Answer the question you asked, recommending the RIGHT specialists based on the report data above.""")
            ])
            return {"final_response": response.content.strip()}

    # ── Action request ────────────────────────────────────────────────────────
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
        return {"final_response": "I can't perform that action directly, but I can answer health questions or analyze your medical report."}

    # ── No report uploaded ────────────────────────────────────────────────────
    if intent == "REPORT_QUERY" and not has_report:
        return {"final_response": (
            "It looks like you haven't uploaded a medical report yet. "
            "Please upload your report using the sidebar on the left, "
            "then ask your question again."
        )}

    # ── Simple educational question ───────────────────────────────────────────
    if intent == "SIMPLE_QUESTION":
        response = _llm.invoke([
            SystemMessage(content="""You are a knowledgeable medical doctor explaining things to a patient.
Answer clearly and conversationally — like a doctor in a clinic.
- Give a direct, focused answer to exactly what was asked
- Use simple language, 3 to 6 sentences max
- No self-care tips, warning signs, or next steps unless asked
- No bullet points — use natural sentences
- End with: "If you have concerns, speak with your doctor." """),
            HumanMessage(content=user_input)
        ])
        return {"final_response": response.content.strip()}

    # ── Report value query ────────────────────────────────────────────────────
    if intent == "REPORT_QUERY" and report_analysis:
        context_parts = []

        key_findings = report_analysis.get("key_findings", [])
        if key_findings:
            findings_text = "\n".join([
                f"{kf.get('parameter')}: {kf.get('value')} (Normal: {kf.get('normal_range')}) "
                f"[{kf.get('status')}] — {kf.get('significance', '')}"
                for kf in key_findings
            ])
            context_parts.append(f"Key findings:\n{findings_text}")

        if report_analysis.get("patient_name"):
            context_parts.append(f"Patient name: {report_analysis.get('patient_name')}")

        if report_data and isinstance(report_data, str):
            context_parts.append(f"Report text:\n{report_data[:2000]}")

        if report_analysis.get("summary"):
            context_parts.append(f"Summary: {report_analysis.get('summary')}")

        context = "\n\n".join(context_parts)

        response = _llm.invoke([
            SystemMessage(content="""You are a medical doctor reviewing a patient's report.
Answer ONLY what was asked. Quote exact values from the report.
Keep to 3-5 sentences. End with: "Let me know if you have more questions about your report." """),
            HumanMessage(content=f"Question: {user_input}\n\nReport data:\n{context}")
        ])
        return {"final_response": response.content.strip()}

    return {"final_response": "I'm not sure how to answer that. Could you rephrase or upload your medical report?"}


# ── After report analysis routing ─────────────────────────────────────────────
def route_after_report(state: HealthAgentState) -> str:
    intent = state.get("intent", "SYMPTOM_ANALYSIS")

    if intent in ("REPORT_QUERY", "ACTION_REQUEST"):
        return "answer_query"

    # REPORT_OVERVIEW and SYMPTOM_ANALYSIS both go through full pipeline
    return "extract_symptoms"


# ── Emergency Fast-Path Node ──────────────────────────────────────────────────
def emergency_response_node(state: HealthAgentState) -> dict:
    print("  [Node] EMERGENCY fast-path triggered!")

    risk     = state.get("risk_assessment", {})
    symptoms = state.get("normalized_symptoms", [])

    messages = [
        SystemMessage(content="""You are an emergency medical triage assistant.
The patient has EMERGENCY-level symptoms. Respond like an urgent doctor.
Be direct, clear, and serious. Tell them exactly what to do right now.
Do NOT use bullet points — speak naturally and urgently.
End with the medical disclaimer."""),
        HumanMessage(content=f"""
Symptoms: {', '.join(symptoms)}
Risk reason: {risk.get('reason', '')}
Action needed: {risk.get('action', 'Call emergency services immediately')}
""")
    ]

    response = _llm.invoke(messages)
    return {"final_response": f"🚨 **EMERGENCY — Call for help immediately**\n\n{response.content}"}


# ── Conditional Routing ───────────────────────────────────────────────────────
def route_after_extraction(state: HealthAgentState) -> str:
    has_report   = state.get("has_report", False)
    has_symptoms = bool(state.get("raw_symptoms"))
    error        = state.get("error", False)
    intent       = state.get("intent", "SYMPTOM_ANALYSIS")

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


# ── Error Node ────────────────────────────────────────────────────────────────
def end_with_error_node(state: HealthAgentState) -> dict:
    return {"final_response": state.get("error_message",
        "I couldn't identify any symptoms. Please describe what you're feeling, "
        "for example: 'I have fever and headache since yesterday.'")}


# ── Build Graph ───────────────────────────────────────────────────────────────
def build_health_graph():
    graph = StateGraph(HealthAgentState)

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

    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges(
        "supervisor", route_entry,
        {
            "analyze_report":   "analyze_report",
            "extract_symptoms": "extract_symptoms",
            "answer_query":     "answer_query"
        }
    )

    graph.add_conditional_edges(
        "analyze_report", route_after_report,
        {
            "answer_query":     "answer_query",
            "extract_symptoms": "extract_symptoms"
        }
    )

    graph.add_edge("answer_query", END)

    graph.add_conditional_edges(
        "extract_symptoms", route_after_extraction,
        {
            "end_with_error":     "end_with_error",
            "normalize_symptoms": "normalize_symptoms"
        }
    )

    graph.add_edge("end_with_error",     END)
    graph.add_edge("normalize_symptoms", "predict_disease")
    graph.add_edge("predict_disease",    "assess_risk")

    graph.add_conditional_edges(
        "assess_risk", route_by_risk_level,
        {
            "emergency_response": "emergency_response",
            "generate_advice":    "generate_advice"
        }
    )

    graph.add_edge("generate_advice",    END)
    graph.add_edge("emergency_response", END)

    return graph.compile()


health_graph = build_health_graph()