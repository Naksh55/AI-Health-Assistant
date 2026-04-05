"""
tests/test_agents.py — Tests for Agent Nodes
==============================================
Tests each LangGraph node function with mocked LLM responses.
Verifies correct state updates and error handling.
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from unittest.mock import patch, MagicMock


class TestSymptomExtractionNode:
    """Tests for symptom_extraction_node."""

    @patch("agents.symptom_extractor.llm")
    def test_extracts_symptoms(self, mock_llm, base_state):
        mock_response = MagicMock()
        mock_response.content = '["fever", "headache", "chills"]'
        mock_llm.invoke.return_value = mock_response

        from agents.symptom_extractor import symptom_extraction_node
        result = symptom_extraction_node(base_state)

        assert result["raw_symptoms"] == ["fever", "headache", "chills"]
        assert result["error"] is False

    @patch("agents.symptom_extractor.llm")
    def test_no_symptoms_found(self, mock_llm, base_state):
        mock_response = MagicMock()
        mock_response.content = "[]"
        mock_llm.invoke.return_value = mock_response

        from agents.symptom_extractor import symptom_extraction_node
        result = symptom_extraction_node(base_state)

        assert result["raw_symptoms"] == []
        assert result["error"] is True
        assert "couldn't identify" in result["error_message"].lower()

    @patch("agents.symptom_extractor.llm")
    def test_invalid_json_response(self, mock_llm, base_state):
        mock_response = MagicMock()
        mock_response.content = "not valid json at all"
        mock_llm.invoke.return_value = mock_response

        from agents.symptom_extractor import symptom_extraction_node
        result = symptom_extraction_node(base_state)

        assert result["raw_symptoms"] == []
        assert result["error"] is True


class TestSymptomNormalizationNode:
    """Tests for symptom_normalization_node."""

    @patch("agents.symptom_normalizer.chain")
    def test_normalizes_symptoms(self, mock_chain, base_state):
        mock_chain.invoke.return_value = '["pyrexia", "cephalalgia", "rigors"]'

        state = {**base_state, "raw_symptoms": ["fever", "headache", "chills"]}

        from agents.symptom_normalizer import symptom_normalization_node
        result = symptom_normalization_node(state)

        assert result["normalized_symptoms"] == ["pyrexia", "cephalalgia", "rigors"]

    @patch("agents.symptom_normalizer.chain")
    def test_empty_symptoms(self, mock_chain, base_state):
        state = {**base_state, "raw_symptoms": []}

        from agents.symptom_normalizer import symptom_normalization_node
        result = symptom_normalization_node(state)

        assert result["normalized_symptoms"] == []

    @patch("agents.symptom_normalizer.chain")
    def test_invalid_json_fallback(self, mock_chain, base_state):
        mock_chain.invoke.return_value = "invalid json"

        state = {**base_state, "raw_symptoms": ["fever", "headache"]}

        from agents.symptom_normalizer import symptom_normalization_node
        result = symptom_normalization_node(state)

        # Should fallback to raw symptoms
        assert result["normalized_symptoms"] == ["fever", "headache"]


class TestRiskAssessorNode:
    """Tests for risk_assessment_node — especially emergency keyword detection."""

    def test_emergency_keyword_chest_pain(self, base_state):
        """Chest pain should trigger EMERGENCY without LLM."""
        from agents.risk_assessor import check_emergency_keywords

        result = check_emergency_keywords("chest pain, shortness of breath", "I have chest pain")
        assert result is not None
        assert result["risk_level"] == "EMERGENCY"

    def test_emergency_keyword_stroke(self, base_state):
        """Stroke symptoms should trigger EMERGENCY."""
        from agents.risk_assessor import check_emergency_keywords

        result = check_emergency_keywords("face droopy, arm weakness", "my face is drooping")
        assert result is not None
        assert result["risk_level"] == "EMERGENCY"

    def test_no_emergency_for_mild(self, base_state):
        """Mild symptoms should NOT trigger emergency keywords."""
        from agents.risk_assessor import check_emergency_keywords

        result = check_emergency_keywords("headache, fatigue", "I have a headache")
        assert result is None

    def test_emergency_unconscious(self):
        """Loss of consciousness should trigger EMERGENCY."""
        from agents.risk_assessor import check_emergency_keywords

        result = check_emergency_keywords("unconscious", "he is unconscious")
        assert result is not None
        assert result["risk_level"] == "EMERGENCY"


class TestReportAnalyzerNode:
    """Tests for report_analysis_node."""

    @patch("agents.report_analyzer.llm")
    def test_analyzes_pdf_report(self, mock_llm, state_with_report):
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "report_type": "Blood Test",
            "patient_name": "John Doe",
            "patient_age": "29 Years",
            "summary": "Low hemoglobin and elevated TSH",
            "abnormal_findings": ["Hemoglobin", "TSH"],
            "normal_findings": ["WBC"],
            "key_findings": [],
            "urgency_level": "SOON",
            "medications_mentioned": [],
            "recommended_tests": [],
            "patient_friendly_summary": "Some values are abnormal"
        })
        mock_llm.invoke.return_value = mock_response

        from agents.report_analyzer import report_analysis_node
        result = report_analysis_node(state_with_report)

        assert result["report_analysis"] is not None
        assert result["report_analysis"]["patient_name"] == "John Doe"

    def test_no_report_data(self, base_state):
        """No report data should return error."""
        from agents.report_analyzer import report_analysis_node
        result = report_analysis_node(base_state)

        assert result.get("error") is True


class TestEndWithErrorNode:
    """Tests for end_with_error_node."""

    def test_default_error_message(self, base_state):
        from agents.graph import end_with_error_node
        state = {k: v for k, v in base_state.items() if k != "error_message"}
        result = end_with_error_node(state)

        assert "couldn't identify" in result["final_response"].lower()

    def test_custom_error_message(self, base_state):
        state = {**base_state, "error_message": "Custom error message"}

        from agents.graph import end_with_error_node
        result = end_with_error_node(state)

        assert result["final_response"] == "Custom error message"


class TestDiagnosticInterviewerNode:
    """Tests for diagnostic_interview_node."""

    @patch("agents.diagnostic_interviewer.llm")
    def test_generates_followup_questions(self, mock_llm, base_state):
        mock_response = MagicMock()
        mock_response.content = "I'd like to understand more:\n1. Where is the pain?\n2. How long?"
        mock_llm.invoke.return_value = mock_response

        state = {
            **base_state,
            "raw_symptoms": ["pain"],
            "enable_diagnostic_interview": True,
            "question_count": 0,
        }

        from agents.diagnostic_interviewer import diagnostic_interview_node
        result = diagnostic_interview_node(state)

        assert result["diagnostic_phase"] == "COLLECTING"
        assert result["question_count"] == 1
        assert "?" in result["final_response"]

    def test_needs_followup_vague_symptoms(self, base_state):
        from agents.diagnostic_interviewer import needs_followup

        state = {
            **base_state,
            "raw_symptoms": ["pain"],
            "enable_diagnostic_interview": True,
            "question_count": 0,
        }
        assert needs_followup(state) is True

    def test_no_followup_when_disabled(self, base_state):
        from agents.diagnostic_interviewer import needs_followup

        state = {
            **base_state,
            "raw_symptoms": ["pain"],
            "enable_diagnostic_interview": False,
            "question_count": 0,
        }
        assert needs_followup(state) is False

    def test_no_followup_enough_symptoms(self, base_state):
        from agents.diagnostic_interviewer import needs_followup

        state = {
            **base_state,
            "raw_symptoms": ["fever", "headache", "chills", "body aches"],
            "enable_diagnostic_interview": True,
            "question_count": 0,
        }
        assert needs_followup(state) is False

    def test_max_followup_rounds(self, base_state):
        from agents.diagnostic_interviewer import needs_followup

        state = {
            **base_state,
            "raw_symptoms": ["pain"],
            "enable_diagnostic_interview": True,
            "question_count": 3,  # already asked 3 times
        }
        assert needs_followup(state) is False
