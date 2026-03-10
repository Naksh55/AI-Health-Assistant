"""
app.py — Streamlit Chat UI (Complete Corrected Version)
"""

# ── MUST BE FIRST — load .env before anything else ────────────────────────────
from dotenv import load_dotenv
load_dotenv()

import os

# ── LangSmith Tracing — set BEFORE importing langchain/langgraph ──────────────
os.environ["LANGSMITH_TRACING"]  = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"]  = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGSMITH_PROJECT"]  = "AI Health Assistant"

import base64
import io
import streamlit as st
import PyPDF2
from agents.graph import health_graph

st.set_page_config(
    page_title="AI Health Assistant",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp {
    background: linear-gradient(135deg, #0f1923 0%, #1a2a3a 50%, #0f1923 100%);
    min-height: 100vh;
}
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
.main-header p { color: #7ab8a8; font-size: 0.95rem; margin: 0.4rem 0 0; font-weight: 300; }
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
.report-badge {
    background: rgba(100, 180, 255, 0.12);
    border: 1px solid rgba(100, 180, 255, 0.3);
    border-radius: 8px;
    padding: 0.5rem 0.8rem;
    color: #90c8ff;
    font-size: 0.82rem;
    margin-bottom: 0.5rem;
}
[data-testid="stChatMessage"] { background: transparent !important; }
[data-testid="stChatInput"] > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(100,200,180,0.3) !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea { color: #e0f0ec !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(100,200,180,0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🏥 AI Health Assistant</h1>
    <p>Powered by multi-agent AI · LangGraph + LangChain + Groq</p>
</div>
<div class="disclaimer-banner">
    ⚠️ For informational purposes only · Not a substitute for professional medical advice
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Upload Medical Report")
    st.caption("Upload a blood test, scan report, prescription, or any medical document.")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "jpg", "jpeg", "png"],
        help="Supported: PDF, JPG, PNG"
    )

    if uploaded_file:
        st.success(f"✅ **{uploaded_file.name}** ready")
        st.caption("This report will be analyzed with your next message.")
        if uploaded_file.type.startswith("image"):
            st.image(uploaded_file, caption="Uploaded Report", use_column_width=True)

    st.markdown("---")
    st.markdown("## 💬 How to Use")
    st.markdown("""
1. **Option A:** Type your symptoms in the chat
2. **Option B:** Upload a report + ask about it
3. **Option C:** Do both together
""")
    st.markdown("---")
    st.caption("🔒 Your data is not stored permanently.")


# ── Helper: Process File ──────────────────────────────────────────────────────
def process_uploaded_file(file):
    if file is None:
        return None, None
    try:
        if "pdf" in file.type:
            file.seek(0)
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
            text = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            if text.strip():
                return text.strip(), "pdf"
            file.seek(0)
            return base64.b64encode(file.read()).decode(), "pdf"
        elif "image" in file.type:
            file.seek(0)
            return base64.b64encode(file.read()).decode(), "image"
    except Exception as e:
        st.error(f"Could not process file: {e}")
    return None, None


# ── Helper: Render Pipeline Expander ─────────────────────────────────────────
def render_pipeline_expander(meta: dict):
    if not meta or meta.get("error"):
        return
    with st.expander("🔬 View Agent Pipeline Analysis", expanded=False):

        # Report section
        report = meta.get("report_analysis")
        if report:
            st.markdown('<div class="pipeline-label">📋 Report Analysis</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="report-badge">'
                f'<strong>{report.get("report_type", "Medical Report")}</strong> · '
                f'Urgency: <strong>{report.get("urgency_level", "ROUTINE")}</strong>'
                f'</div>', unsafe_allow_html=True
            )
            abnormal = report.get("abnormal_findings", [])
            if abnormal:
                st.markdown(f'⚠️ **Abnormal:** {", ".join(abnormal)}')
            st.markdown("---")

        risk  = meta.get("risk", {})
        level = risk.get("risk_level", "MEDIUM")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="pipeline-label">🔎 Extracted Symptoms</div>', unsafe_allow_html=True)
            raw = meta.get("raw_symptoms", [])
            chips = "".join([f'<span class="symptom-chip">{s}</span>' for s in raw]) if raw else "<span style='color:#666'>None</span>"
            st.markdown(f'<div>{chips}</div>', unsafe_allow_html=True)

            st.markdown('<br><div class="pipeline-label">🏥 Normalized Terms</div>', unsafe_allow_html=True)
            norm = meta.get("normalized_symptoms", [])
            chips2 = "".join([f'<span class="symptom-chip">{s}</span>' for s in norm]) if norm else "<span style='color:#666'>None</span>"
            st.markdown(f'<div>{chips2}</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="pipeline-label">🦠 Predicted Conditions</div>', unsafe_allow_html=True)
            for c in meta.get("conditions", []):
                prob_color = {"High": "#ff7777", "Medium": "#ffcc44", "Low": "#77cc88"}.get(c.get("probability", ""), "#aaa")
                st.markdown(
                    f'<div class="condition-item"><strong>{c.get("name","")}</strong> '
                    f'<span style="color:{prob_color};font-size:0.78rem;">({c.get("probability","")})</span></div>',
                    unsafe_allow_html=True
                )

        st.markdown("---")
        st.markdown(f'<span class="risk-badge risk-{level}">⚡ {level} Risk</span>', unsafe_allow_html=True)
        st.markdown(f'<span style="color:#aaa;font-size:0.85rem;">{risk.get("reason","")}</span>', unsafe_allow_html=True)
        st.markdown(f'<br><strong style="color:#e0f0ec;">Action:</strong> <span style="color:#b0d8cc;">{risk.get("action","")}</span>', unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "👋 Hello! I'm your **AI Health Assistant**.\n\n"
            "You can:\n"
            "- 💬 Describe your symptoms in plain language\n"
            "- 📋 Upload a medical report from the sidebar and ask about it\n"
            "- 🔍 Do both together for a complete analysis\n\n"
            "**Example:** *\"I've had a high fever and chills since yesterday, along with body aches.\"*\n\n"
            "> ⚠️ *I'm not a doctor. Always consult a medical professional for real advice.*"
        ),
        "meta": None
    })


# ── Render Chat History ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🏥"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta"):
            render_pipeline_expander(msg["meta"])


# ── Input Handler ─────────────────────────────────────────────────────────────
if user_input := st.chat_input("Describe your symptoms or ask about your report..."):

    report_data, report_type = process_uploaded_file(uploaded_file)
    has_report = report_data is not None

    display_msg = user_input
    if has_report:
        display_msg += f"\n\n📎 *Report attached: `{uploaded_file.name}`*"

    st.session_state.messages.append({"role": "user", "content": display_msg, "meta": None})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(display_msg)

    with st.chat_message("assistant", avatar="🏥"):
        spinner_msg = "🤖 Analyzing your report and symptoms..." if has_report else "🤖 AI agents are analyzing your symptoms..."
        with st.spinner(spinner_msg):
            # Single call — runs entire LangGraph multi-agent pipeline
            final_state = health_graph.invoke({
                "user_input":  user_input,
                "has_report":  has_report,
                "report_data": report_data,
                "report_type": report_type,
                "error":       False
            })

        response_text = final_state.get("final_response", "I encountered an issue. Please try again.")
        st.markdown(response_text)

        meta = {
            "error":               final_state.get("error", False),
            "raw_symptoms":        final_state.get("raw_symptoms") or [],
            "normalized_symptoms": final_state.get("normalized_symptoms") or [],
            "conditions":          final_state.get("predicted_conditions") or [],
            "risk":                final_state.get("risk_assessment") or {},
            "report_analysis":     final_state.get("report_analysis"),
        }

        render_pipeline_expander(meta)

        st.session_state.messages.append({
            "role":    "assistant",
            "content": response_text,
            "meta":    meta
        })