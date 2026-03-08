"""
app.py — Streamlit Chat UI
===========================
The frontend for the AI Health Assistant.
Uses Streamlit's chat components to create a conversational interface.

HOW STREAMLIT WORKS WITH LANGGRAPH:
  - Streamlit re-runs the entire script top-to-bottom on every user interaction
  - st.session_state persists data (like chat history) across reruns
  - When user submits a message, we call health_graph.invoke(state) which
    runs the full LangGraph pipeline and returns the final state
"""

import streamlit as st
from agents.graph import health_graph
from langchain_groq import ChatGroq
import os
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


# LangSmith tracing — set env vars explicitly before any langchain imports
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGSMITH_PROJECT"] = "AI Health Assistant"

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Health Assistant",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=DM+Sans:wght@300;400;500&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* App background */
.stApp {
    background: linear-gradient(135deg, #0f1923 0%, #1a2a3a 50%, #0f1923 100%);
    min-height: 100vh;
}

/* Header */
.main-header {
    text-align: center;
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid rgba(100, 200, 180, 0.2);
    margin-bottom: 1.5rem;
}
.main-header h1 {
    font-family: 'Lora', serif;
    font-size: 2.2rem;
    font-weight: 600;
    color: #e8f4f0;
    margin: 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #7ab8a8;
    font-size: 0.95rem;
    margin: 0.4rem 0 0;
    font-weight: 300;
}

/* Disclaimer banner */
.disclaimer-banner {
    background: rgba(255, 180, 50, 0.08);
    border: 1px solid rgba(255, 180, 50, 0.3);
    border-radius: 10px;
    padding: 0.7rem 1.1rem;
    color: #f0c060;
    font-size: 0.82rem;
    text-align: center;
    margin-bottom: 1.5rem;
}

/* Risk level badges */
.risk-badge {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.risk-EMERGENCY { background: #ff3333; color: white; }
.risk-HIGH      { background: #ff7700; color: white; }
.risk-MEDIUM    { background: #ddaa00; color: #1a1a1a; }
.risk-LOW       { background: #33aa66; color: white; }

/* Pipeline details card */
.pipeline-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(100,200,180,0.15);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-top: 0.5rem;
}
.pipeline-label {
    color: #7ab8a8;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 500;
    margin-bottom: 0.3rem;
}
.symptom-chip {
    display: inline-block;
    background: rgba(100,200,180,0.12);
    border: 1px solid rgba(100,200,180,0.3);
    color: #a0ddd0;
    padding: 0.15rem 0.6rem;
    border-radius: 20px;
    font-size: 0.8rem;
    margin: 0.15rem;
}
.condition-item {
    background: rgba(255,255,255,0.04);
    border-left: 3px solid #4ab8a0;
    padding: 0.4rem 0.7rem;
    margin: 0.3rem 0;
    border-radius: 0 6px 6px 0;
    font-size: 0.85rem;
    color: #d0e8e0;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: transparent !important;
}

/* Input box */
[data-testid="stChatInput"] > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(100,200,180,0.3) !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    color: #e0f0ec !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(100,200,180,0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏥 AI Health Assistant</h1>
    <p>Powered by multi-agent AI · LangGraph + LangChain + Claude</p>
</div>
<div class="disclaimer-banner">
    ⚠️ For informational purposes only · Not a substitute for professional medical advice
</div>
""", unsafe_allow_html=True)


# ── Session State ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "👋 Hello! I'm your AI Health Assistant.\n\nDescribe your symptoms in plain language and my AI agents will analyze them for you.\n\n**Example:** *\"I've had a high fever and chills since yesterday, along with body aches.\"*\n\n> ⚠️ *I'm not a doctor. Always consult a medical professional for real advice.*",
        "meta": None
    })


# ── Render Chat History ────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🏥"):
        st.markdown(msg["content"])

        # If this assistant message has pipeline metadata, show the details
        meta = msg.get("meta")
        if meta and not meta.get("error"):
            with st.expander("🔬 View Agent Pipeline Analysis", expanded=False):
                _render_pipeline_details(meta) if False else None
                
                risk = meta.get("risk", {})
                level = risk.get("risk_level", "MEDIUM")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<div class="pipeline-label">🔎 Extracted Symptoms</div>', unsafe_allow_html=True)
                    chips = "".join([f'<span class="symptom-chip">{s}</span>' for s in meta.get("raw_symptoms", [])])
                    st.markdown(f'<div>{chips}</div>', unsafe_allow_html=True)

                    st.markdown('<br><div class="pipeline-label">🏥 Normalized (Medical Terms)</div>', unsafe_allow_html=True)
                    chips2 = "".join([f'<span class="symptom-chip">{s}</span>' for s in meta.get("normalized_symptoms", [])])
                    st.markdown(f'<div>{chips2}</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="pipeline-label">🦠 Predicted Conditions</div>', unsafe_allow_html=True)
                    for c in meta.get("conditions", []):
                        prob_color = {"High": "#ff7777", "Medium": "#ffcc44", "Low": "#77cc88"}.get(c.get("probability", ""), "#aaaaaa")
                        st.markdown(
                            f'<div class="condition-item">'
                            f'<strong>{c.get("name")}</strong> '
                            f'<span style="color:{prob_color};font-size:0.78rem;">({c.get("probability")})</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                st.markdown("---")
                st.markdown(f'<span class="risk-badge risk-{level}">⚡ {level} Risk</span>', unsafe_allow_html=True)
                st.markdown(f'<span style="color:#aaa;font-size:0.85rem;">{risk.get("reason","")}</span>', unsafe_allow_html=True)
                st.markdown(f'<br><strong style="color:#e0f0ec;">Action:</strong> <span style="color:#b0d8cc;">{risk.get("action","")}</span>', unsafe_allow_html=True)


# ── User Input Handler ─────────────────────────────────────────────────────────
if user_input := st.chat_input("Describe your symptoms here..."):

    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input, "meta": None})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    # Run the LangGraph pipeline
    with st.chat_message("assistant", avatar="🏥"):
        with st.spinner("🤖 AI agents are analyzing your symptoms..."):

            # ── LANGGRAPH INVOCATION ──────────────────────────────────────────
            # This single call runs the entire multi-agent graph:
            # supervisor → extract → normalize → predict → assess → advise
            # The graph manages all state transitions internally.
            final_state = health_graph.invoke({
                "user_input": user_input,
                "error": False
            })
            # ─────────────────────────────────────────────────────────────────

        response_text = final_state.get("final_response", "I encountered an issue processing your request.")
        st.markdown(response_text)

        # Build metadata for the expander
        meta = {
            "error": final_state.get("error", False),
            "raw_symptoms": final_state.get("raw_symptoms", []),
            "normalized_symptoms": final_state.get("normalized_symptoms", []),
            "conditions": final_state.get("predicted_conditions", []),
            "risk": final_state.get("risk_assessment", {})
        }

        # Show pipeline details inline for the latest message
        if not meta["error"] and meta["raw_symptoms"]:
            with st.expander("🔬 View Agent Pipeline Analysis", expanded=False):
                risk = meta.get("risk", {})
                level = risk.get("risk_level", "MEDIUM")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<div class="pipeline-label">🔎 Extracted Symptoms</div>', unsafe_allow_html=True)
                    chips = "".join([f'<span class="symptom-chip">{s}</span>' for s in meta.get("raw_symptoms", [])])
                    st.markdown(f'<div>{chips}</div>', unsafe_allow_html=True)

                    st.markdown('<br><div class="pipeline-label">🏥 Normalized (Medical Terms)</div>', unsafe_allow_html=True)
                    chips2 = "".join([f'<span class="symptom-chip">{s}</span>' for s in meta.get("normalized_symptoms", [])])
                    st.markdown(f'<div>{chips2}</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="pipeline-label">🦠 Predicted Conditions</div>', unsafe_allow_html=True)
                    for c in meta.get("conditions", []):
                        prob_color = {"High": "#ff7777", "Medium": "#ffcc44", "Low": "#77cc88"}.get(c.get("probability", ""), "#aaaaaa")
                        st.markdown(
                            f'<div class="condition-item">'
                            f'<strong>{c.get("name")}</strong> '
                            f'<span style="color:{prob_color};font-size:0.78rem;">({c.get("probability")})</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                st.markdown("---")
                st.markdown(f'<span class="risk-badge risk-{level}">⚡ {level} Risk</span>', unsafe_allow_html=True)
                st.markdown(f'<span style="color:#aaa;font-size:0.85rem;">{risk.get("reason","")}</span>', unsafe_allow_html=True)
                st.markdown(f'<br><strong style="color:#e0f0ec;">Action:</strong> <span style="color:#b0d8cc;">{risk.get("action","")}</span>', unsafe_allow_html=True)

        # Save to session history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "meta": meta
        })
