<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776ab?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-Agent_Graph-10b981?style=for-the-badge" />
  <img src="https://img.shields.io/badge/LangChain-LLM_Framework-1c3c3c?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Groq-Llama_3.3_70B-f55036?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Streamlit-Frontend-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Tests-85+-059669?style=for-the-badge&logo=pytest&logoColor=white" />
</p>

<h1 align="center">⚕️ DocMate — AI Clinical Assistant</h1>

<p align="center">
  <strong>A multi-agent AI health assistant powered by LangGraph, LangChain, and Groq.</strong><br/>
  Hybrid ML + LLM disease prediction · RAG-grounded medical advice · Medical report analysis · Emergency triage
</p>

<p align="center">
  <em>Built for educational and informational purposes — not a substitute for professional medical advice.</em>
</p>

---

## 🎯 What Is This?

DocMate is a **multi-agent clinical assistant** that analyzes symptoms, predicts diseases, and provides medically-grounded advice through an intelligent pipeline of 8 specialized AI agents coordinated via a LangGraph state graph.

Unlike simple chatbot wrappers, DocMate uses:
- **Hybrid Intelligence** — A RandomForest ML model trained on real Kaggle data provides statistical disease probabilities, while Llama-3.3-70B adds clinical reasoning
- **RAG-Grounded Advice** — All medical advice is grounded in 120+ verified medical articles via ChromaDB, reducing hallucinations
- **Multi-Turn Diagnosis** — A diagnostic interview flow asks targeted follow-up questions for vague symptoms, mimicking a real doctor
- **Smart Triage** — Automatic risk assessment (LOW → EMERGENCY) with emergency fast-path routing

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🩺 **Symptom Analysis** | Natural language symptom extraction → normalization → disease prediction |
| 🧬 **Hybrid ML + LLM** | RandomForest (Kaggle-trained) + Llama-3.3-70B clinical reasoning |
| 📚 **RAG Engine** | ChromaDB + Sentence Transformers for verified medical knowledge retrieval |
| 📋 **Report Analysis** | Upload PDF/image medical reports — extracts lab values, flags abnormals |
| 🔄 **Diagnostic Interview** | Multi-turn follow-up questions for vague symptoms (toggleable) |
| ⚠️ **Risk Triage** | 4-level severity grading: LOW / MEDIUM / HIGH / EMERGENCY |
| 🛡️ **Input Guardrails** | Blocks prompt injection, HTML injection, off-topic queries |
| 🎨 **Clinical UI** | Structured response cards, pipeline visualization, polished design |
| 🧪 **85+ Tests** | Comprehensive test suite covering agents, intent, guardrails, and ML |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [Groq API Key](https://console.groq.com/) (free tier available)
- [Google Gemini API Key](https://aistudio.google.com/) (for image-based report analysis)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Naksh55/AI-Health-Assistant.git
cd AI-Health-Assistant

# 2. Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
# Create a .env file with:
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_gemini_api_key_here
LANGSMITH_API_KEY=your_langsmith_key_here    # optional — for tracing

# 5. Run the application
streamlit run app.py
```

The app will open at **http://localhost:8501**.

---

## 🏗️ System Architecture

```
User Input (Streamlit UI)
        │
        ▼
┌─────────────────────┐
│   SUPERVISOR NODE    │ ← Input validation, guardrails, intent classification
│  (Orchestrator)      │
└────────┬────────────┘
         │ (conditional routing based on intent)
         ├──────────────────────┬────────────────────┐
         ▼                      ▼                    ▼
  ┌──────────────┐    ┌────────────────┐    ┌──────────────┐
  │ REPORT       │    │ SYMPTOM        │    │ ANSWER       │
  │ ANALYZER     │    │ EXTRACTOR      │    │ QUERY        │
  │ (PDF/Image)  │    │ (NLP)          │    │ (Direct)     │
  └──────┬───────┘    └───────┬────────┘    └──────┬───────┘
         │                    │                    │
         ▼                    ▼                    ▼ END
  ┌──────────────┐    ┌────────────────┐
  │ (routes to   │    │ DIAGNOSTIC     │ ← (if symptoms are vague)
  │  extractor   │    │ INTERVIEWER    │
  │  or answer)  │    └───────┬────────┘
  └──────────────┘            │
                              ▼
                    ┌────────────────┐
                    │ SYMPTOM        │
                    │ NORMALIZER     │
                    └───────┬────────┘
                            │
                            ▼
                    ┌────────────────┐
                    │ DISEASE        │ ← Hybrid: ML (RandomForest) + LLM
                    │ PREDICTOR      │
                    └───────┬────────┘
                            │
                            ▼
                    ┌────────────────┐
                    │ RISK           │
                    │ ASSESSOR       │
                    └───────┬────────┘
                            │ (conditional: EMERGENCY → fast-path)
                            ▼
                    ┌────────────────┐
                    │ MEDICAL        │ ← RAG-augmented advice generation
                    │ ADVISOR        │
                    └───────┬────────┘
                            │
                            ▼
                        END → Streamlit UI
```

### Intent Classification

The Supervisor Agent classifies each message using a **two-tier strategy** (keyword rules → LLM fallback):

| Intent | Example | Routing |
|--------|---------|---------|
| `SYMPTOM_ANALYSIS` | "I have fever and headache" | Full 6-agent pipeline |
| `REPORT_QUERY` | "What is my cholesterol?" | Report → Direct answer |
| `REPORT_OVERVIEW` | "Check my report" | Report → Full pipeline |
| `SIMPLE_QUESTION` | "What is hypothyroidism?" | Direct LLM answer |
| `ACTION_REQUEST` | "Schedule an appointment" | Action handler |

---

## 📁 Project Structure

```
AI Health Assistant/
├── app.py                          # Streamlit UI — chat interface, styling, pipeline visualization
├── Main.py                         # Standalone test runner
├── requirements.txt                # Python dependencies
├── diseases.json                   # 60+ diseases with symptoms and severity data
├── .env                            # API keys (GROQ, GOOGLE, LANGSMITH)
│
├── agents/                         # 🧠 Core multi-agent system
│   ├── state.py                    # HealthAgentState TypedDict — shared pipeline state
│   ├── graph.py                    # LangGraph StateGraph — 11 nodes, conditional routing
│   ├── symptom_extractor.py        # Agent 1: Extract symptoms from natural language
│   ├── symptom_normalizer.py       # Agent 2: Normalize to standard medical terminology
│   ├── disease_predictor.py        # Agent 3: Hybrid ML + LLM disease prediction
│   ├── risk_assessor.py            # Agent 4: Severity-based risk triage
│   ├── medical_advisor.py          # Agent 5: RAG-augmented medical advice generation
│   ├── report_analyzer.py          # Agent 6: PDF/image medical report parsing
│   ├── diagnostic_interviewer.py   # Agent 7: Multi-turn follow-up question generation
│   ├── guardrails.py               # Agent 0: Input validation and sanitization
│   └── rag_engine.py               # RAG engine — ChromaDB + Sentence Transformers
│
├── ml/                             # 🤖 Machine Learning subsystem
│   ├── train_model.py              # Training script (Kaggle CSV or synthetic fallback)
│   ├── predictor.py                # Inference engine with synonym mapping
│   ├── disease_model.pkl           # Trained RandomForest model (~6.4 MB)
│   ├── feature_names.pkl           # 132 symptom feature names
│   ├── label_encoder.pkl           # Disease label encoder
│   └── kaggle_data/                # Kaggle disease-symptom dataset
│
├── knowledge/                      # 📚 Medical knowledge base
│   ├── medical_knowledge.json      # 120+ curated medical articles
│   └── chroma_db/                  # Persistent ChromaDB vector index
│
├── tests/                          # 🧪 Test suite (85+ tests)
│   ├── conftest.py                 # Shared fixtures and mock configurations
│   ├── test_agents.py              # Agent behavior tests
│   ├── test_classify_intent.py     # Intent classification tests
│   ├── test_input_guardrails.py    # Security and guardrail tests
│   └── test_ml_predictor.py        # ML model accuracy tests
│
└── sample_reports/                 # 📄 Example medical reports for testing
    ├── blood_test_report.pdf
    └── urine_report.pdf
```

---

## 🧠 Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Agent Framework** | LangGraph | Multi-agent graph orchestration with conditional routing |
| **LLM Integration** | LangChain | Prompt templates, LCEL chains, output parsers |
| **Primary LLM** | Groq (Llama-3.3-70B) | All agent reasoning and clinical analysis |
| **Multimodal LLM** | Google Gemini | Image-based medical report analysis |
| **ML Model** | Scikit-learn (RandomForest) | Disease classification from symptom vectors |
| **Vector Database** | ChromaDB | RAG knowledge storage and semantic retrieval |
| **Embeddings** | Sentence Transformers | Document embedding (all-MiniLM-L6-v2) |
| **Web UI** | Streamlit | Frontend chat interface with custom CSS |
| **PDF Parsing** | PyPDF2 | Medical report text extraction |
| **Tracing** | LangSmith | Pipeline debugging and performance monitoring |

---

## 🧬 How the Hybrid Prediction Works

```
Patient Symptoms → Symptom Extraction → Normalization
                                            │
                     ┌──────────────────────┤
                     ▼                      ▼
              ┌─────────────┐      ┌─────────────────┐
              │ ML MODEL    │      │ LLM REASONING   │
              │ RandomForest│      │ Llama-3.3-70B   │
              │ 132 features│      │ + Report Data   │
              │ 41 diseases │      │ + RAG Context   │
              └──────┬──────┘      └────────┬────────┘
                     │                      │
                     ▼                      ▼
              Confidence Scores    Clinical Reasoning
                     │                      │
                     └──────────┬───────────┘
                                ▼
                    Combined Predictions
                 (Data-Driven + Contextual)
```

1. **ML Model**: Maps symptoms to binary feature vectors → RandomForest outputs top-3 diseases with confidence %
2. **LLM Analysis**: Validates ML predictions against report data + adds clinical reasoning
3. **RAG Grounding**: Medical advice is enriched with relevant verified literature from ChromaDB

---

## 🛡️ Safety & Guardrails

DocMate implements comprehensive input protection:

- **Prompt Injection Detection** — Blocks 17+ patterns ("ignore previous instructions", jailbreak, DAN mode, etc.)
- **HTML/Script Injection** — Detects `<script>`, `onerror=`, `javascript:`, and 10+ attack vectors
- **Off-Topic Filtering** — Rejects non-medical queries ("write a poem", "stock market", "solve math")
- **Input Validation** — Length checks (2–2000 chars), special character ratio limits
- **Input Sanitization** — Unicode normalization, HTML stripping, control character removal

---

## 🧪 Running Tests

```bash
# Run the full test suite (85+ tests)
pytest tests/ -v

# Run specific test categories
pytest tests/test_agents.py -v              # Agent behavior
pytest tests/test_classify_intent.py -v     # Intent classification
pytest tests/test_input_guardrails.py -v    # Security guardrails
pytest tests/test_ml_predictor.py -v        # ML model accuracy
```

---

## 📝 Usage Examples

### Symptom Analysis
> **User:** "I've been having severe headaches, fever, and stiff neck for 2 days"
>
> DocMate runs the full pipeline: extraction → normalization → ML + LLM prediction → risk assessment → RAG-grounded advice with structured cards

### Medical Report Analysis
> Upload a blood test PDF → DocMate extracts lab values, flags abnormals (e.g., low hemoglobin, high TSH), and correlates with symptoms

### Diagnostic Interview
> **User:** "I don't feel well"
>
> DocMate (with interview mode ON): "Can you describe what you're feeling? Do you have fever, pain, fatigue, or any specific symptoms?"

### General Medical Questions
> **User:** "What is hypothyroidism?"
>
> DocMate provides a concise, doctor-like explanation without running the full pipeline

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | ✅ Yes | API key for Groq Cloud (Llama-3.3-70B) |
| `GOOGLE_API_KEY` | ⚠️ For images | Google Gemini API key for image report analysis |
| `LANGSMITH_API_KEY` | ❌ Optional | LangSmith tracing for pipeline debugging |

---

## 📋 Requirements

```
langchain
langchain-groq
langgraph
streamlit
python-dotenv
PyPDF2
langchain_google_genai
chromadb
sentence-transformers
scikit-learn
pandas
numpy
pytest
```

---

## 🔮 Future Scope

- 🎙️ Voice-based symptom input (Whisper API)
- 🌍 Multi-language support (Hindi, Spanish, etc.)
- 📱 Mobile app (React Native / Flutter)
- ⌚ Wearable data integration (Apple Health, Google Fit)
- 👨‍⚕️ Doctor dashboard for review and second opinions
- 💊 Medication interaction checker
- 🧠 Mental health screening module (PHQ-9, GAD-7)

---

## ⚠️ Disclaimer

> **This project is developed for educational and informational purposes only.** It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical decisions.

---

<p align="center">
  <strong>DocMate — AI Clinical Assistant v1.0</strong><br/>
  Built with ❤️ using LangGraph · LangChain · Groq · Streamlit
</p>
