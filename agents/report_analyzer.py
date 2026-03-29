"""
agents/report_analyzer.py — Medical Report Analysis Node
=========================================================
FIXES:
  1. Now extracts patient_name AND patient_age from report text
  2. Cleaner JSON prompt with explicit patient info fields
  3. Better fallback regex for name/age extraction
  4. Image support kept intact
"""

from dotenv import load_dotenv
load_dotenv()

import json
import re
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from agents.state import HealthAgentState


llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


REPORT_SYSTEM_PROMPT = """You are an expert medical report analyzer.
Analyze the provided medical report text and extract ALL structured information.

Return ONLY a valid JSON object in this exact format (no markdown, no extra text):
{
  "report_type": "Blood Test / X-Ray / MRI / Prescription / Discharge Summary / Other",
  "patient_name": "Full name of the patient as written in the report, or null",
  "patient_age": "Age of the patient as written in the report, or null",
  "patient_gender": "Gender of the patient, or null",
  "summary": "2-3 sentence plain English summary of the overall report findings",
  "key_findings": [
    {
      "parameter": "Hemoglobin",
      "value": "8.4 g/dL",
      "normal_range": "12.0-16.0 g/dL",
      "status": "LOW",
      "significance": "Indicates iron deficiency anemia — reduced oxygen in blood"
    }
  ],
  "abnormal_findings": ["Hemoglobin", "TSH", "Vitamin D"],
  "normal_findings": ["Platelets", "WBC", "Sodium"],
  "doctor_notes": "Any doctor observations, impressions, or notes found in report, or null",
  "medications_mentioned": [],
  "recommended_tests": [],
  "urgency_level": "ROUTINE / SOON / URGENT",
  "patient_friendly_summary": "Explain the whole report in simple language as if talking to a non-medical person"
}

IMPORTANT:
- Extract patient_name exactly as it appears in the report
- Extract patient_age exactly as it appears (e.g. '29 Years')
- If a field has no data, use null (not empty string)
- medications_mentioned and recommended_tests must always be arrays (use [] if none)
- abnormal_findings and normal_findings must always be arrays"""


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

    raw_text = ""

    try:
        # ── Build message based on report type ────────────────────────────────
        if report_type == "image":
            message = HumanMessage(content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{report_data}"}
                },
                {
                    "type": "text",
                    "text": REPORT_SYSTEM_PROMPT
                }
            ])
        else:
            # PDF text or plain text
            message = HumanMessage(content=(
                f"{REPORT_SYSTEM_PROMPT}\n\nMedical Report Content:\n{report_data}"
            ))

        response = llm.invoke([message])
        raw_text = response.content.strip()

        # ── Parse JSON ────────────────────────────────────────────────────────
        clean    = raw_text.replace("```json", "").replace("```", "").strip()
        analysis = json.loads(clean)

        # ── Ensure list fields are always lists (never null) ──────────────────
        for list_field in ["medications_mentioned", "recommended_tests",
                           "abnormal_findings", "normal_findings", "key_findings"]:
            if not isinstance(analysis.get(list_field), list):
                analysis[list_field] = []

        # ── Fallback: extract patient_name from raw text if LLM missed it ─────
        if not analysis.get("patient_name") and isinstance(report_data, str):
            patterns = [
                r"Patient\s+Name\s*[:\-]?\s*([A-Za-z][A-Za-z\s\.\-]{1,50}?)(?:\n|$|,|\|)",
                r"Name\s*[:\-]\s*([A-Za-z][A-Za-z\s\.\-]{1,50}?)(?:\n|$|,|\|)",
                r"Patient\s*[:\-]\s*([A-Za-z][A-Za-z\s\.\-]{1,50}?)(?:\n|$|,|\|)",
            ]
            for pattern in patterns:
                match = re.search(pattern, report_data, re.IGNORECASE)
                if match:
                    name = match.group(1).strip()
                    # Sanity check — reject if it looks like a label not a name
                    if len(name) > 2 and name.lower() not in ["name", "patient", "report"]:
                        analysis["patient_name"] = name
                        break

        # ── Fallback: extract patient_age from raw text if LLM missed it ──────
        if not analysis.get("patient_age") and isinstance(report_data, str):
            age_patterns = [
                r"Age\s*[:/\-]?\s*(\d{1,3})\s*(?:Years?|Yrs?|Y\.?O\.?)",
                r"(\d{1,3})\s*(?:Years?|Yrs?)\s*/\s*(?:Male|Female|M|F)",
                r"Age\s*[:\-]\s*(\d{1,3})",
            ]
            for pattern in age_patterns:
                match = re.search(pattern, report_data, re.IGNORECASE)
                if match:
                    analysis["patient_age"] = match.group(1) + " Years"
                    break

        print(f"  [Node] Report analyzed: {analysis.get('report_type')}")
        print(f"  [Node] Patient: {analysis.get('patient_name')} | Age: {analysis.get('patient_age')}")
        print(f"  [Node] Abnormal findings: {analysis.get('abnormal_findings')}")

        return {"report_analysis": analysis}

    except json.JSONDecodeError as e:
        print(f"  [Node] JSON parse error: {e}")
        # Return minimal usable analysis even if JSON failed
        return {
            "report_analysis": {
                "report_type": "Medical Report",
                "summary": "Report was read but structured extraction failed.",
                "patient_friendly_summary": raw_text[:500] if raw_text else "",
                "abnormal_findings": [],
                "normal_findings": [],
                "key_findings": [],
                "medications_mentioned": [],
                "recommended_tests": [],
                "urgency_level": "ROUTINE",
                "patient_name": None,
                "patient_age": None,
            }
        }

    except Exception as e:
        print(f"  [Node] Report analysis error: {e}")
        return {
            "report_analysis": None,
            "error": True,
            "error_message": f"Could not analyze report: {str(e)}"
        }