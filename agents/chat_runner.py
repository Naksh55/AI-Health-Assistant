import traceback

from agents.graph import health_graph


def run_chat_turn(
    *,
    user_input: str,
    has_report: bool,
    report_data,
    report_type,
    chat_history: list,
    enable_diagnostic_interview: bool,
    question_count: int,
) -> dict:
    """Run the health graph and always return a serializable state dict."""
    try:
        return health_graph.invoke(
            {
                "user_input": user_input,
                "has_report": has_report,
                "report_data": report_data,
                "report_type": report_type,
                "chat_history": chat_history,
                "error": False,
                "enable_diagnostic_interview": enable_diagnostic_interview,
                "question_count": question_count,
            }
        )
    except Exception as exc:
        print(f"[ChatRunner] Graph execution failed: {exc}")
        traceback.print_exc()
        return {
            "error": True,
            "error_message": str(exc),
            "intent": "ERROR",
            "final_response": (
                "I’m sorry — I couldn’t process your request right now. "
                "Please try again in a moment."
            ),
            "raw_symptoms": [],
            "normalized_symptoms": [],
            "predicted_conditions": [],
            "risk_assessment": {},
            "report_analysis": None,
            "ml_predictions": [],
            "diagnostic_phase": None,
            "question_count": 0,
        }
