# 🏥 AI Health Assistant — LangGraph + LangChain + Groq

A multi-agent AI health assistant that analyzes symptoms using a proper
directed agent graph built with LangGraph and LangChain.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 3. Run
streamlit run app.py
```

---

## 🏗️ Architecture

```
User Input (Streamlit)
        │
        ▼
[Supervisor Node]          ← Validates input, orchestrates
        │
        ▼
[Symptom Extraction]       ← LangChain: ChatGroq + SystemMessage/HumanMessage
        │
        ▼ (conditional: error → END)
[Symptom Normalization]    ← LangChain: LCEL chain (llm | StrOutputParser)
        │
        ▼
[Disease Prediction]       ← LangChain: ChatPromptTemplate + LCEL chain
        │
        ▼
[Risk Assessment]          ← LangChain: ChatPromptTemplate with multi-var prompt
        │
        ▼ (conditional: EMERGENCY → emergency node)
[Medical Advice]           ← LangChain: Rich multi-variable template synthesis
        │
        ▼
   Final State → Streamlit UI
```

---

## 📁 File Structure

```
health_assistant/
├── app.py                      # Streamlit chat UI
├── requirements.txt
├── .env.example
└── agents/
    ├── __init__.py
    ├── state.py                # HealthAgentState TypedDict (shared state)
    ├── graph.py                # THE CORE: LangGraph StateGraph definition
    ├── symptom_extractor.py    # Node 1: Extract symptoms from raw text
    ├── symptom_normalizer.py   # Node 2: Normalize to medical terms
    ├── disease_predictor.py    # Node 3: Predict possible conditions
    ├── risk_assessor.py        # Node 4: Assess severity/risk level
    └── medical_advisor.py      # Node 5: Generate patient-friendly response
```

---

## 🧠 Key LangChain Concepts Used

| Concept | File | Purpose |
|---|---|---|
| `ChatGroq` | All agents | LangChain's Groq wrapper |
| `SystemMessage / HumanMessage` | symptom_extractor.py | Structured message building |
| `StrOutputParser` | normalizer, predictor | Parse AIMessage → string |
| `LCEL chain (llm \| parser)` | normalizer, predictor | Composable pipeline |
| `ChatPromptTemplate` | predictor, risk, advisor | Reusable prompt templates with variables |

## 🔗 Key LangGraph Concepts Used

| Concept | File | Purpose |
|---|---|---|
| `StateGraph` | graph.py | The directed graph container |
| `HealthAgentState` | state.py | Shared typed state flowing between nodes |
| `add_node()` | graph.py | Register agent functions as graph nodes |
| `add_edge()` | graph.py | Unconditional A → B connections |
| `add_conditional_edges()` | graph.py | Branch based on state values |
| `START / END` | graph.py | Graph entry and exit points |
| `.compile()` | graph.py | Build executable runnable |
| `.invoke()` | app.py | Run the full graph with initial state |
