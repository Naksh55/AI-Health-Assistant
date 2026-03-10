"""
agents/report_analyzer.py — Medical Report Analysis Node
=========================================================
WHAT IT DOES:
  Accepts an uploaded medical report (PDF, image, or text),
  extracts key findings, flags abnormal values, and summarizes
  everything in patient-friendly language.

SUPPORTS:
  - Blood test reports
  - X-ray / scan reports (text description)
  - Prescription documents
  - Discharge summaries
  - Any general medical document

LANGCHAIN CONCEPT USED:
  Gemini's multimodal capability via LangChain —
  we can pass both text AND images/PDFs in the same message.
"""

from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()

import os
import base64
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from agents.state import HealthAgentState


llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


REPORT_SYSTEM_PROMPT = """You are an expert medical report analyzer.
Analyze the provided medical report and extract structured information.

Return ONLY a valid JSON object in this exact format:
{
  "report_type": "Blood Test / X-Ray / MRI / Prescription / Discharge Summary / Other",
  "summary": "2-3 sentence plain English summary of the report",
  "key_findings": [
    {
      "parameter": "Hemoglobin",
      "value": "8.5 g/dL",
      "normal_range": "13.5-17.5 g/dL",
      "status": "LOW",
      "significance": "Indicates anemia - reduced oxygen carrying capacity"
    }
  ],
  "abnormal_findings": ["List of parameters that are outside normal range"],
  "normal_findings": ["List of parameters that are within normal range"],
  "doctor_notes": "Any doctor observations or impressions found in report",
  "medications_mentioned": ["Any medications listed"],
  "recommended_tests": ["Any follow-up tests suggested"],
  "urgency_level": "ROUTINE / SOON / URGENT",
  "patient_friendly_summary": "Explain the report as if talking to a non-medical person"
}

Be thorough but accurate. If a value is not present, use null.
Return ONLY the JSON. No markdown. No explanation."""


def report_analysis_node(state: HealthAgentState) -> dict:
    """
    LangGraph Node: Medical Report Analysis

    Input  (from state): state["report_data"], state["report_type"]
    Output (to state)  : {"report_analysis": {...}}
    """
    print("  [Node] ReportAnalyzer running...")

    report_data = state.get("report_data")
    report_type = state.get("report_type", "text")

    if not report_data:
        return {
            "report_analysis": None,
            "error": True,
            "error_message": "No report data found to analyze."
        }

    try:
        # Build message based on report type
        if report_type == "image":
            # For image reports (JPG, PNG) — multimodal message
            message = HumanMessage(content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{report_data}"
                    }
                },
                {
                    "type": "text",
                    "text": REPORT_SYSTEM_PROMPT
                }
            ])

        elif report_type == "pdf":
            # For PDF reports — send as document
            message = HumanMessage(content=[
                {
                    "type": "text",
                    "text": f"{REPORT_SYSTEM_PROMPT}\n\nMedical Report Content:\n{report_data}"
                }
            ])

        else:
            # Plain text report
            message = HumanMessage(content=[
                {
                    "type": "text",
                    "text": f"{REPORT_SYSTEM_PROMPT}\n\nMedical Report:\n{report_data}"
                }
            ])

        response = llm.invoke([message])
        raw_text = response.content.strip()

        # Parse JSON response
        clean = raw_text.replace("```json", "").replace("```", "").strip()
        analysis = json.loads(clean)

        print(f"  [Node] Report analyzed: {analysis.get('report_type')}")
        print(f"  [Node] Abnormal findings: {analysis.get('abnormal_findings')}")

        return {"report_analysis": analysis}

    except json.JSONDecodeError as e:
        print(f"  [Node] JSON parse error: {e}")
        return {
            "report_analysis": {
                "summary": "Report was analyzed but structured extraction failed.",
                "patient_friendly_summary": raw_text,
                "abnormal_findings": [],
                "urgency_level": "ROUTINE"
            }
        }
    except Exception as e:
        print(f"  [Node] Report analysis error: {e}")
        return {
            "report_analysis": None,
            "error": True,
            "error_message": f"Could not analyze report: {str(e)}"
        }