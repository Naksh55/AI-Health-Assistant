from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from agents.state import HealthAgentState
from agents.symptom_extractor import symptom_extraction_node
from agents.symptom_normalizer import symptom_normalization_node
from agents.disease_predictor import disease_prediction_node
from agents.risk_assessor import risk_assessment_node
from agents.medical_advisor import medical_advice_node

from dotenv import load_dotenv
load_dotenv()
# ── Supervisor Node ─────────────────────────────────────────────────────────
# The supervisor is a lightweight "intent check" node.
# In a more complex system it would dynamically decide which agents to call.
# Here it validates input and sets up context.
def supervisor_node(state: HealthAgentState) -> dict:
    """
    Supervisor / Orchestrator Node
    
    Validates the user input and logs the start of the pipeline.
    In a more complex system, this would dynamically decide WHICH agents
    to call and in what order — true agentic behavior.
    """
    print("\n[Supervisor] Pipeline started.")
    print(f"[Supervisor] User input: {state['user_input']}")

    user_input = state.get("user_input", "").strip()

    if not user_input:
        return {
            "error": True,
            "error_message": "Please describe your symptoms to get started."
        }

    # Reset state for a fresh run
    return {
        "error": False,
        "raw_symptoms": None,
        "normalized_symptoms": None,
        "predicted_conditions": None,
        "risk_assessment": None,
        "final_response": None
    }


# ── Emergency Fast-Path Node ─────────────────────────────────────────────────
# If risk level is EMERGENCY, we skip the normal advice node and go straight
# to an urgent emergency response. This is the conditional routing in action.
_emergency_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def emergency_response_node(state: HealthAgentState) -> dict:
    """
    Fast-path node for EMERGENCY risk level.
    Generates a direct, urgent response instructing the user to call emergency services.
    """
    print("  [Node] EMERGENCY fast-path triggered!")

    risk = state.get("risk_assessment", {})
    symptoms = state.get("normalized_symptoms", [])

    messages = [
        SystemMessage(content="""You are an emergency medical triage assistant.
The patient has EMERGENCY-level symptoms. Respond with URGENT, CLEAR instructions.
Start with a bold emergency alert. Tell them to call emergency services immediately.
List the specific dangerous symptoms detected. Keep it short and urgent.
End with the medical disclaimer."""),
        HumanMessage(content=f"""
Symptoms: {', '.join(symptoms)}
Risk reason: {risk.get('reason', '')}
Action needed: {risk.get('action', 'Call emergency services immediately')}
""")
    ]

    response = _emergency_llm.invoke(messages)
    return {"final_response": f"🚨 **EMERGENCY ALERT**\n\n{response.content}"}


# ── Conditional Routing Functions ────────────────────────────────────────────
# These functions are called by add_conditional_edges().
# They look at the current state and return the STRING NAME of the next node.

def route_after_extraction(state: HealthAgentState) -> str:
    """
    Called after symptom extraction.
    If extraction found an error (no symptoms), go straight to END.
    Otherwise, proceed to normalization.
    """
    if state.get("error"):
        print("[Router] No symptoms found — routing to END")
        return "end_with_error"
    print("[Router] Symptoms found — routing to normalize")
    return "normalize_symptoms"


def route_by_risk_level(state: HealthAgentState) -> str:
    """
    Called after risk assessment.
    Routes to emergency fast-path if EMERGENCY, otherwise normal advice.
    """
    risk = state.get("risk_assessment", {})
    level = risk.get("risk_level", "MEDIUM")

    if level == "EMERGENCY":
        print("[Router] EMERGENCY detected — routing to emergency node")
        return "emergency_response"
    
    print(f"[Router] Risk={level} — routing to normal advice")
    return "generate_advice"


# ── Error Termination Node ───────────────────────────────────────────────────
def end_with_error_node(state: HealthAgentState) -> dict:
    """Passes the error message as the final response so the UI can display it."""
    return {"final_response": state.get("error_message", "Something went wrong.")}


# ── Build the Graph ──────────────────────────────────────────────────────────
def build_health_graph():
    """
    Constructs, configures, and compiles the LangGraph StateGraph.
    Returns a compiled runnable that can be called with .invoke(state).
    """

    # 1. Create the graph with our state schema
    graph = StateGraph(HealthAgentState)

    # 2. Register all nodes (name → function)
    graph.add_node("supervisor",         supervisor_node)
    graph.add_node("extract_symptoms",   symptom_extraction_node)
    graph.add_node("normalize_symptoms", symptom_normalization_node)
    graph.add_node("predict_disease",    disease_prediction_node)
    graph.add_node("assess_risk",        risk_assessment_node)
    graph.add_node("generate_advice",    medical_advice_node)
    graph.add_node("emergency_response", emergency_response_node)
    graph.add_node("end_with_error",     end_with_error_node)

    # 3. Define edges — the flow between nodes

    # Entry point: START → supervisor
    graph.add_edge(START, "supervisor")

    # Supervisor always goes to extraction
    graph.add_edge("supervisor", "extract_symptoms")

    # After extraction: CONDITIONAL — branch on error vs success
    graph.add_conditional_edges(
        "extract_symptoms",           # from this node
        route_after_extraction,       # call this routing function
        {                             # map return values to node names
            "end_with_error":         "end_with_error",
            "normalize_symptoms":     "normalize_symptoms"
        }
    )

    # Error path terminates
    graph.add_edge("end_with_error", END)

    # Normal path: normalization → prediction → risk
    graph.add_edge("normalize_symptoms", "predict_disease")
    graph.add_edge("predict_disease",    "assess_risk")

    # After risk assessment: CONDITIONAL — emergency vs normal
    graph.add_conditional_edges(
        "assess_risk",
        route_by_risk_level,
        {
            "emergency_response": "emergency_response",
            "generate_advice":    "generate_advice"
        }
    )

    # Both advice paths terminate
    graph.add_edge("generate_advice",    END)
    graph.add_edge("emergency_response", END)

    # 4. Compile and return the executable graph
    return graph.compile()


# Singleton — build once, reuse across requests
health_graph = build_health_graph()
