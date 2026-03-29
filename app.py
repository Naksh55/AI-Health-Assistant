"""
app.py — AI Health Assistant v6
- No st.expander (causes _arrow CSS bleed)
- Sidebar controlled entirely via session_state + show/hide column layout
- Blue-green-white palette
"""

from dotenv import load_dotenv
load_dotenv()

import os
os.environ["LANGSMITH_TRACING"]  = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"]  = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGSMITH_PROJECT"]  = "AI Health Assistant"

import base64, io, re
import streamlit as st
import PyPDF2
from agents.graph import health_graph

st.set_page_config(
    page_title="DocMate · Clinical Assistant",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "messages"   not in st.session_state: st.session_state.messages   = []
if "open_pipes" not in st.session_state: st.session_state.open_pipes = {}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin:0; padding:0; }

html, body,
.stApp,
[data-testid="stAppViewContainer"],u
[data-testid="stMain"],
[data-testid="stVerticalBlock"],
.block-container {
    background: #f0f7f4 !important;
    font-family: 'Inter', sans-serif !important;
    color: #1a2e25 !important;
    padding: 0 !important;
    max-width: 100% !important;
    margin: 0 auto !important;
}

/* Kill dark bottom bar */
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] {
    background: #f0f7f4 !important;
    border-top: 1px solid #d0e8df !important;
    box-shadow: 0 -2px 12px rgba(0,120,80,0.05) !important;
}
[data-testid="stAppViewBlockContainer"] {
    padding-top: 0 !important;
}
/* Hide ALL chrome + every sidebar toggle variant */
#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton,
[data-testid="stDecoration"],
[data-testid="stToolbar"],
[data-testid="collapsedControl"],
[data-testid="baseButton-header"],
button[kind="header"],
[aria-label="Close sidebar"],
[aria-label="Open sidebar"],
button[aria-label="Close sidebar"],
button[aria-label="Open sidebar"] { display: none !important; }

/* Sidebar — permanently visible, no collapse */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #d0e8df !important;
    box-shadow: 3px 0 20px rgba(0,120,80,0.08) !important;
    min-width: 270px !important;
    max-width: 270px !important;
    transform: none !important;
    visibility: visible !important;
    pointer-events: all !important;
}
section[data-testid="stSidebar"][aria-expanded="false"] {
    transform: none !important;
    margin-left: 0 !important;
    display: block !important;
}
section[data-testid="stSidebar"] > div {
    background: #ffffff !important;
    padding: 0 0 24px 0 !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 10px 0 16px 0 !important;
    margin-left:10px
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
    background: #ffffff !important;
    border: 1px solid #d0e8df !important;
    border-left: 3px solid #10b981 !important;
    border-radius: 2px 13px 13px 13px !important;
    padding: 18px 20px !important;
    box-shadow: 0 2px 14px rgba(0,120,80,0.07) !important;
    max-width: 92% !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
    background: linear-gradient(135deg, #0d9488, #0891b2) !important;
    border: none !important;
    border-radius: 13px 2px 13px 13px !important;
    padding: 12px 16px !important;
    box-shadow: 0 3px 14px rgba(13,148,136,0.28) !important;
    max-width: 60% !important;
    margin-left: auto !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) span {
    color: #ffffff !important; font-size: 0.9rem !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) p {
    color: #2d4a3e !important; font-size: 0.87rem !important; line-height: 1.65 !important;
}

/* Chat input */
[data-testid="stChatInput"] > div {
    background: #ffffff !important;
    border: 1.5px solid #a7d7c5 !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 10px rgba(0,120,80,0.07) !important;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color: #10b981 !important;
    box-shadow: 0 0 0 3px rgba(16,185,129,0.1) !important;
}
[data-testid="stChatInput"] textarea {
    background: #ffffff !important;
    color: #1a2e25 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #7db59a !important; }

/* All buttons */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    border-radius: 8px !important;
    transition: all 0.15s !important;
    cursor: pointer !important;
    background: #ffffff !important;
    border: 1px solid #d0e8df !important;
    color: #047857 !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    padding: 8px 16px !important;
    width: 100% !important;
    margin-top: 10px !important;
}
.stButton > button:hover {
    background: #ecfdf5 !important;
    border-color: #10b981 !important;
    color: #065f46 !important;
}

/* File uploader zone */
[data-testid="stFileUploaderDropzone"] {
    background: #f2fbf7 !important;
    border: 1.5px dashed #a7d7c5 !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #10b981 !important;
    background: #e8f5ef !important;
}
/* Browse files button — target every possible selector */
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] .stButton button,
[data-testid="stFileUploaderDropzoneInput"] + div button {
    background: #ffffff !important;
    border: 1.5px solid #10b981 !important;
    color: #047857 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    border-radius: 7px !important;
    padding: 6px 18px !important;
    box-shadow: none !important;
    cursor: pointer !important;
}
[data-testid="stFileUploaderDropzone"] button:hover,
[data-testid="stFileUploader"] button:hover {
    background: #ecfdf5 !important;
    border-color: #059669 !important;
    color: #065f46 !important;
}
/* Uploader helper text */
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] p {
    color: #5a8a76 !important;
    font-size: 0.76rem !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="column"] {
    display: flex !important;
    flex-direction: column !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #f0f7f4; }
::-webkit-scrollbar-thumb { background: #a7d7c5; border-radius: 3px; }

@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ── Global override: Browse files button in sidebar uploader ── */
/* Streamlit renders this as a plain <button> with no testid, catch it broadly */
section[data-testid="stSidebar"] button:not([data-testid="baseButton-primary"]) {
    background: #ffffff !important;
    border: 1.5px solid #10b981 !important;
    color: #047857 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    border-radius: 7px !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] button:hover {
    background: #ecfdf5 !important;
    border-color: #059669 !important;stV
    color: #065f46 !important;
}
</style>
""", unsafe_allow_html=True)

# ── CSS for custom HTML components ───────────────────────────────────────────
COMPONENT_CSS = """
<style>
/* 🚨 REMOVE GAP BETWEEN STREAMLIT BLOCKS (REAL ISSUE) */
[data-testid="stVerticalBlock"] {
    gap: 0.25rem !important;  
     background: #ffffff /* 👈 reduce space here */
}

/* Remove margin from markdown wrapper */
[data-testid="stMarkdown"] {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* Remove extra spacing from container */
[data-testid="stMarkdownContainer"] {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

.topbar {
    background:#ffffff; 
    border-bottom:1px solid #d0e8df;
     height:80px;
    display:flex; align-items:center; justify-content:center;
    box-shadow:0 1px 8px rgba(0,120,80,0.06);
    margin-top: 0 !important;
    padding-top: 2px !important;   /* 👈 adjust 0–6px */
}

/* 🚨 REMOVE GAP BETWEEN STREAMLIT BLOCKS */
[data-testid="stVerticalBlock"] > div {
    margin-top: 0 !important;
    gap: 0 !important;
}
.brand-icon {
    width:50px; height:50px;
    border-radius:9px; display:flex; align-items:center;
    justify-content:center; font-size:25px;
    box-shadow:0 2px 8px rgba(16,185,129,0.3); flex-shrink:0;
}
.brand-name { font-size:1.5rem; font-weight:700; color:#0a2418; }
.brand-sub  { font-size:0.8rem; color:#5a8a76; font-family:'JetBrains Mono',monospace; text-align:center; }
.live-pill  {
    display:inline-flex; align-items:center; gap:5px;
    background:#ecfdf5; border:1px solid #6ee7b7; color:#047857;
    font-size:.63rem; font-weight:700; letter-spacing:.8px; text-transform:uppercase;
    padding:4px 11px; border-radius:20px; font-family:'JetBrains Mono',monospace;
}
.ldot { width:6px; height:6px; background:#10b981; border-radius:50%;
        animation:blink 2s infinite; display:inline-block; }
.model-tag {
    font-size:.64rem; color:#5a8a76; background:#f0f7f4;
    border:1px solid #d0e8df; padding:4px 9px; border-radius:5px;
    font-family:'JetBrains Mono',monospace;
}
.disc-bar {
    background:#fffbeb; border-bottom:1px solid #fde68a;
    padding:6px 24px; font-size:.80rem; color:#854d0e;
    text-align:center; font-weight:500; margin-bottom-10px;
}
.sb-header {
    background:linear-gradient(135deg,#059669 0%,#0891b2 100%);
    padding:22px 20px 18px;
}
.sb-name { font-size:1.05rem; font-weight:700; color:#fff; }
.sb-sub  { font-size:.67rem; color:rgba(255,255,255,.65); font-family:'JetBrains Mono',monospace; margin-top:3px; }
.sb-sec  { 
    padding:18px 18px;   /* 👈 more breathing room */
    border-bottom:1px solid #e8f5ef; 
    gap:12px;            /* 👈 space between elements */
    display:flex; 
    flex-direction:column; 
}
.sb-lbl  {
    font-size:.6rem; font-weight:700; letter-spacing:1.3px;
    text-transform:uppercase; color:#7db59a; margin-bottom:10px;
    font-family:'JetBrains Mono',monospace;
}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 1rem !important;   /* 👈 bigger spacing between blocks */
}
[data-testid="stFileUploader"] {
    margin-top: 8px !important;
    margin-bottom: 8px !important;
}
.stat-card { background:#f2fbf7; border:1px solid #d0e8df; border-radius:8px; padding:10px 12px; }
.stat-row  { display:flex; justify-content:space-between; align-items:center; padding:3px 0; }
.stat-k    { font-size:.7rem; color:#7db59a; font-family:'JetBrains Mono',monospace; }
.stat-v    { font-size:.75rem; color:#1a2e25; font-weight:500; }
.stat-on   { color:#059669; font-weight:700; }
.how-row   { display:flex; gap:10px; align-items:flex-start; padding:7px 0; border-bottom:1px solid #f0f7f4; }
.how-row:last-child { border-bottom:none; }
.how-t { font-size:.8rem; font-weight:600; color:#1a2e25; }
.how-s { font-size:.7rem; color:#7db59a; margin-top:1px; }

.rcard {
    background:#ffffff; 
    border:1px solid #d0e8df; 
    border-radius:11px;
    padding:15px 17px; 
    box-shadow:0 1px 5px rgba(0,120,80,0.05);
    transition:box-shadow .18s;
    position: relative !important;   /* 👈 prevents stacking issues */
    z-index: 1;
    flex: 1 !important;
}
[data-testid="stVerticalBlock"] > div {
    display: block !important;
}
.stMarkdown {
    overflow: visible !important;
}
.rcard:hover { box-shadow:0 4px 16px rgba(0,120,80,0.1); }
.rcard-hdr {
    display:flex; align-items:center; gap:8px;
    margin-bottom:11px; padding-bottom:9px; border-bottom:1px solid #e8f5ef;
}
.rcard-ttl {
    font-size:1rem; font-weight:700; letter-spacing:.9px;
    text-transform:uppercase; color:#5a8a76; font-family:'JetBrains Mono',monospace;
}
.rcard-cond { border-top:3px solid #8b5cf6; }
.rcard-self { border-top:3px solid #0891b2; }
.rcard-warn { border-top:3px solid #f59e0b; }
.rcard-next { border-top:3px solid #10b981; }
.cond-text  { font-size:1rem; color:#1a2e25; line-height:1.7; }
.blist { list-style:none; padding:0; margin:0; }
.blist li {
    display:flex; gap:8px; align-items:flex-start;
    padding:6px 0; border-bottom:1px solid #f0f7f4;
    font-size:1rem; color:#2d4a3e; line-height:1.55;
}
.blist li:last-child { border-bottom:none; }
.disc { font-size:.68rem; color:#7db59a; text-align:center; margin-top:10px; padding-top:8px; border-top:1px solid #e8f5ef; }

.pipe-box {
    background:#ffffff; border:1px solid #d0e8df; border-radius:10px;
    padding:16px; margin-top:10px;
}
.pipe-title {
    font-size:.72rem; font-weight:700; letter-spacing:.8px; text-transform:uppercase;
    color:#047857; font-family:'JetBrains Mono',monospace;
    margin-bottom:14px; padding-bottom:8px; border-bottom:1px solid #e8f5ef;
}
.pcrd { background:#ffffff; border:1px solid #d0e8df; border-radius:8px; padding:12px 14px; }
.plbl { font-size:.6rem; font-weight:700; letter-spacing:1.2px; text-transform:uppercase; color:#7db59a; margin-bottom:7px; font-family:'JetBrains Mono',monospace; }
.chip { display:inline-block; background:#e0f2fe; border:1px solid #7dd3fc; color:#0369a1; padding:3px 8px; border-radius:4px; font-size:.7rem; margin:2px 2px 2px 0; font-family:'JetBrains Mono',monospace; font-weight:500; }
.crow { display:flex; align-items:center; justify-content:space-between; padding:6px 0; border-bottom:1px solid #e8f5ef; font-size:.8rem; color:#1a2e25; }
.crow:last-child { border-bottom:none; }
.pH { background:#fee2e2;color:#dc2626;font-size:.62rem;font-weight:700;padding:2px 7px;border-radius:4px;font-family:'JetBrains Mono',monospace; }
.pM { background:#fffbeb;color:#b45309;font-size:.62rem;font-weight:700;padding:2px 7px;border-radius:4px;font-family:'JetBrains Mono',monospace; }
.pL { background:#ecfdf5;color:#047857;font-size:.62rem;font-weight:700;padding:2px 7px;border-radius:4px;font-family:'JetBrains Mono',monospace; }
.risk-row {
    display:flex; align-items:center; gap:14px;
    background:#ffffff; border:1px solid #d0e8df; border-radius:8px; padding:11px 14px; margin-top:12px;
}
.rpill { font-size:.62rem; font-weight:700; letter-spacing:1px; text-transform:uppercase; padding:5px 13px; border-radius:5px; font-family:'JetBrains Mono',monospace; flex-shrink:0; }
.rpill-EMERGENCY { background:#fee2e2;color:#dc2626;border:1px solid #fca5a5; }
.rpill-HIGH      { background:#fff7ed;color:#c2410c;border:1px solid #fdba74; }
.rpill-MEDIUM    { background:#fffbeb;color:#b45309;border:1px solid #fcd34d; }
.rpill-LOW       { background:#ecfdf5;color:#047857;border:1px solid #6ee7b7; }
.risk-act { font-size:.83rem; font-weight:600; color:#0a2418; }
.risk-rsn { font-size:.74rem; color:#5a8a76; margin-top:2px; }
.rep-hdr { display:flex;align-items:center;gap:9px; background:#e0f2fe;border:1px solid #7dd3fc; border-radius:8px;padding:8px 12px;margin-bottom:10px; }
.rh-t { font-size:.82rem;font-weight:600;color:#0369a1; }
.rh-u { font-size:.63rem;font-family:'JetBrains Mono',monospace;color:#5a8a76;margin-left:auto; }
/* Sidebar header spacing (clean + small) */
[data-testid="stSidebarHeader"] {
    height: 8px !important;   /* 👈 adjust this (6–12px ideal) */
    min-height: 8px !important;
    padding: 0 !important;
    margin: 0 !important;
}
[data-testid="stHorizontalBlock"] {
    display: flex !important;
    align-items: stretch !important;   /* 👈 equal height columns */
    gap: 16px !important;
    margin-bottom: 10px !important;    /* 👈 space after columns */
}
/* Ensure no extra padding above */
[data-testid="stSidebarContent"] {
    padding-top: 6px !important;
}

/* TARGET USER MESSAGE TEXT (INNER ELEMENT) */
div[data-testid="stChatMessageContent"][aria-label="Chat message from user"] p {
    background:#ffffff; border:1px solid #d0e8df; border-left:3px solid #10b981;
    border-radius:2px 14px 14px 14px; padding:20px 22px;
    color: black !important;
    padding: 10px 16px;
    # border-radius: 14px 4px 14px 14px;
    display: inline-block;        /* 👈 THIS IS THE KEY FIX */
    max-width: 60%;
    margin-left: auto;
    box-shadow: 0 3px 12px rgba(13,148,136,0.25);
}
/* FILE NAME TEXT */
[data-testid="stFileUploaderFileName"] {
    color: #000000 !important;   /* 👈 black */
    font-weight: 500;            /* optional: make it clearer */
}
/* FILE SIZE TEXT (6.6MB) */
[data-testid="stFileUploaderFile"] small {
    color: #000000 !important;   /* 👈 black */
    font-weight: 500;
}
.welcome {
    background:#ffffff; border:1px solid #d0e8df; border-left:3px solid #10b981;
    border-radius:2px 14px 14px 14px; padding:20px 22px;
    box-shadow:0 2px 14px rgba(0,120,80,0.07); max-width:500px;
}
.w-title { font-size:1rem; font-weight:700; color:#0a2418; margin-bottom:6px; }
.w-desc  { font-size:1rem; color:#5a8a76; line-height:1.65; margin-bottom:13px; }
.w-chips { display:flex; flex-wrap:wrap; gap:6px; }
.w-chip  { background:#ecfdf5; border:1px solid #6ee7b7; border-radius:6px; padding:5px 11px; font-size:.74rem; color:#047857; font-weight:500; }
</style>
"""
st.markdown(COMPONENT_CSS, unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def process_uploaded_file(file):
    if file is None:
        return None, None
    try:
        if "pdf" in file.type:
            file.seek(0)
            reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
            text = "".join(p.extract_text() + "\n" for p in reader.pages if p.extract_text())
            if text.strip():
                return text.strip(), "pdf"
            file.seek(0)
            return base64.b64encode(file.read()).decode(), "pdf"
        elif "image" in file.type:
            file.seek(0)
            return base64.b64encode(file.read()).decode(), "image"
    except Exception as e:
        st.error(f"File error: {e}")
    return None, None


def parse_sections(text: str) -> dict:
    text = re.sub(r"-{2,}.*?-{2,}", "", text, flags=re.DOTALL).strip()
    secs = {}
    patterns = {
        "condition": r"##\s*[^\n]*Going On[^\n]*\n(.*?)(?=##|\Z)",
        "selfcare":  r"##\s*[^\n]*Self[^\n]*\n(.*?)(?=##|\Z)",
        "warning":   r"##\s*[^\n]*(?:Warning|Seek)[^\n]*\n(.*?)(?=##|\Z)",
        "nextsteps": r"##\s*[^\n]*(?:Next|Steps|Recommended)[^\n]*\n(.*?)(?=##|\Z)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
            bullets = re.findall(r"[-•*]\s*(.+?)(?=\n\s*[-•*]|\Z)", raw, re.DOTALL)
            bullets = [
                b.strip().replace("\n", " ")
                for b in bullets
                if b.strip()
                and len(b.strip()) > 4
                and not b.strip().startswith("*AI")
                and not b.strip().startswith("AI-gen")
                and "not a medical" not in b.lower()
                and "consult a" not in b.lower()[:30]
            ]
            secs[key] = bullets if bullets else [raw[:280].strip()]
    return secs

def render_response(text: str):
    secs = parse_sections(text)
    if not secs:
        st.markdown(text)
        return

    cond = secs.get("condition", [])
    if cond:
        st.markdown(f"""
        <div class="rcard rcard-cond" style="margin-bottom:10px;">
            <div class="rcard-hdr"><span style="font-size:1rem">🩺</span>
            <span class="rcard-ttl">What Might Be Going On</span></div>
            <div class="cond-text">{" ".join(cond)}</div>
        </div>""", unsafe_allow_html=True)

    # ✅ ADD SPACE BEFORE COLUMNS (THIS WAS MISSING)
    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)

    # ✅ WRAP COLUMNS (better layout control)
    with st.container():
        c1, c2 = st.columns(2, gap="medium")

        with c1:
            sc = secs.get("selfcare", [])
            rows = "".join([
                f'<li><span style="color:#0891b2;font-weight:700;flex-shrink:0">→</span><span>{b}</span></li>'
                for b in sc[:3]
            ])
            st.markdown(f"""
            <div class="rcard rcard-self">
                <div class="rcard-hdr"><span style="font-size:1rem">💊</span>
                <span class="rcard-ttl">Self-Care Tips</span></div>
                <ul class="blist">{rows}</ul>
            </div>""", unsafe_allow_html=True)

        with c2:
            ws = secs.get("warning", [])
            rows = "".join([
                f'<li><span style="color:#f59e0b;font-weight:700;flex-shrink:0">!</span><span>{b}</span></li>'
                for b in ws[:3]
            ])
            st.markdown(f"""
            <div class="rcard rcard-warn">
                <div class="rcard-hdr"><span style="font-size:1rem">⚠️</span>
                <span class="rcard-ttl">Seek Help If...</span></div>
                <ul class="blist">{rows}</ul>
            </div>""", unsafe_allow_html=True)

    # ✅ SPACE AFTER COLUMNS
    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)

    ns = secs.get("nextsteps", [])
    rows = "".join([
        f'<li><span style="color:#10b981;font-weight:700;flex-shrink:0">✓</span><span>{b}</span></li>'
        for b in ns[:3]
    ])

    st.markdown(f"""
    <div class="rcard rcard-next">
        <div class="rcard-hdr"><span style="font-size:1rem">📋</span>
        <span class="rcard-ttl">Recommended Next Steps</span></div>
        <ul class="blist">{rows}</ul>
    </div>
    <div class="disc">AI-generated · Not a medical diagnosis · Always consult a qualified healthcare professional</div>
    """, unsafe_allow_html=True)

def render_pipeline(meta: dict, key: str):
    """Renders pipeline using a native st.button toggle — no st.expander."""
    if not meta or meta.get("error"):
        return

    toggle_key = f"pipe_{key}"
    if toggle_key not in st.session_state:
        st.session_state[toggle_key] = False

    # Small toggle button — styled via CSS
    btn_label = "Hide Pipeline Analysis" if st.session_state[toggle_key] else "View Pipeline Analysis"
    st.markdown('<div style="margin-top:10px;">', unsafe_allow_html=True)

    if st.button(btn_label, key=toggle_key + "_btn", use_container_width=True):
        st.session_state[toggle_key] = not st.session_state[toggle_key]
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    if not st.session_state[toggle_key]:
        return

    # ── Pipeline content ──
    report = meta.get("report_analysis")
    report_html = ""
    if report:
        abn = report.get("abnormal_findings", [])
        abn_html = "".join([f'<div style="font-size:.74rem;color:#b45309;font-family:JetBrains Mono,monospace;padding:2px 0;">! {f}</div>' for f in abn])
        report_html = f"""
        <div class="rep-hdr">
            <span>📋</span>
            <span class="rh-t">{report.get("report_type","Medical Report")}</span>
            <span class="rh-u">Urgency: {report.get("urgency_level","ROUTINE")}</span>
        </div>{abn_html}"""

    st.markdown(f'<div class="pipe-box"><div class="pipe-title">Agent Pipeline Analysis</div>{report_html}', unsafe_allow_html=True)

    pc1, pc2 = st.columns(2, gap="small")
    with pc1:
        raw  = meta.get("raw_symptoms", [])
        norm = meta.get("normalized_symptoms", [])
        chips1 = "".join([f'<span class="chip">{s}</span>' for s in raw]) or '<span style="font-size:.73rem;color:#7db59a;">None</span>'
        chips2 = "".join([f'<span class="chip">{s}</span>' for s in norm]) or '<span style="font-size:.73rem;color:#7db59a;">—</span>'
        st.markdown(f"""
        <div class="pcrd">
            <div class="plbl">Extracted Symptoms</div>{chips1}
            <div class="plbl" style="margin-top:10px;">Normalized Terms</div>{chips2}
        </div>""", unsafe_allow_html=True)

    with pc2:
        conds = meta.get("conditions", [])
        rows = ""
        for c in conds:
            p   = c.get("probability", "")
            cls = {"High": "pH", "Medium": "pM", "Low": "pL"}.get(p, "pL")
            rows += f'<div class="crow"><span>{c.get("name","")}</span><span class="{cls}">{p}</span></div>'
        st.markdown(f"""
        <div class="pcrd">
            <div class="plbl">Predicted Conditions</div>
            {rows or '<span style="font-size:.73rem;color:#7db59a;">None identified</span>'}
        </div>""", unsafe_allow_html=True)

    risk  = meta.get("risk", {})
    level = risk.get("risk_level", "MEDIUM")
    st.markdown(f"""
    <div class="risk-row">
        <span class="rpill rpill-{level}">{level} RISK</span>
        <div>
            <div class="risk-act">{risk.get("action","Consult a healthcare provider")}</div>
            <div class="risk-rsn">{risk.get("reason","")}</div>
        </div>
    </div></div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# SIDEBAR — full content, toggled via initial_sidebar_state
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-header">
        <div class="sb-name">⚕️ &nbsp;HealthAI</div>
        <div class="sb-sub">Clinical Assistant · v1.0</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sb-sec">', unsafe_allow_html=True)
    st.markdown('<div class="sb-lbl">Upload Medical Report</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "PDF, JPG or PNG", type=["pdf", "jpg", "jpeg", "png"],
        label_visibility="collapsed", key="file_uploader"
    )
    if uploaded_file:
        st.success(f"Ready: {uploaded_file.name}")
        if uploaded_file.type.startswith("image"):
            st.image(uploaded_file, use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-sec">', unsafe_allow_html=True)
    if st.button("Clear Conversation", use_container_width=True, key="clear_btn"):
        st.session_state.messages = []
        st.session_state.open_pipes = {}
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
  
# ══════════════════════════════════════════════════════════
# TOPBAR
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="topbar">
    <div style="display:flex;align-items:center;gap:12px;justify-content:center;">
        <div class="brand-icon">⚕️</div>
        <div>
            <div class="brand-name">DOCMATE-YOUR AI CLINICAL ASSISTANT</div>
            <div class="brand-sub">LangGraph · LangChain · Groq · Multi-Agent</div>
        </div>
    </div>
</div>
<div class="disc-bar">
    ⚠️ &nbsp;For informational purposes only · Not a substitute for professional medical advice · Always consult a qualified healthcare provider
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# CHAT
# ══════════════════════════════════════════════════════════
st.markdown('<div style="padding:20px 28px 8px;max-width:880px;margin:0 auto;">', unsafe_allow_html=True)

if not st.session_state.messages:
    with st.chat_message("assistant", avatar="⚕️"):
        st.markdown("""
        <div class="welcome">
            <div class="w-title">Hello — I'm your AI Clinical Assistant</div>
            <div class="w-desc">Describe your symptoms or upload a medical report for full multi-agent analysis: symptom extraction, condition prediction, and risk triage.</div>
            <div class="w-chips">
                <span class="w-chip">💬 Symptoms</span>
                <span class="w-chip">📋 Upload report</span>
            </div>
        </div>""", unsafe_allow_html=True)

for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "assistant":
        with st.chat_message("assistant", avatar="⚕️"):
            render_response(msg["content"])
            if msg.get("meta"):
                render_pipeline(msg["meta"], key=str(i))
    else:
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg["content"])

st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# INPUT
# ══════════════════════════════════════════════════════════
if user_input := st.chat_input("Describe your symptoms or ask about your report..."):
    report_data, report_type = process_uploaded_file(uploaded_file)
    has_report = report_data is not None
    display_msg = user_input + (f"\n\n`📎 {uploaded_file.name}`" if has_report else "")

    st.session_state.messages.append({"role": "user", "content": display_msg, "meta": None})
    with st.chat_message("user", avatar="👤"):
        st.markdown(display_msg)

    with st.chat_message("assistant", avatar="⚕️"):
        with st.spinner("Analyzing..." if has_report else "Running pipeline..."):
           from groq import RateLimitError

        try:
            final_state = health_graph.invoke({
                "user_input": user_input,
                "has_report": has_report,
                "report_data": report_data,
                "report_type": report_type,
                "error": False
            })

        except RateLimitError:
            final_state = {
                "final_response": "⚠️ API limit reached. Please try again after a few minutes."
            }

        response_text = final_state.get("final_response", "Something went wrong. Please try again.")
        render_response(response_text)

        meta = {
            "error":               final_state.get("error", False),
            "raw_symptoms":        final_state.get("raw_symptoms") or [],
            "normalized_symptoms": final_state.get("normalized_symptoms") or [],
            "conditions":          final_state.get("predicted_conditions") or [],
            "risk":                final_state.get("risk_assessment") or {},
            "report_analysis":     final_state.get("report_analysis"),
        }
        idx = len(st.session_state.messages)
        render_pipeline(meta, key=str(idx))
        st.session_state.messages.append({"role": "assistant", "content": response_text, "meta": meta})