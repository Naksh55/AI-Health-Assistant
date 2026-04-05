"""
tests/conftest.py — Shared Test Fixtures
=========================================
Provides reusable fixtures for mocking LLM responses
and creating test states.
"""

import sys
import os
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def mock_llm_response():
    """Factory fixture to create mock LLM responses."""
    def _create(content: str):
        mock_response = MagicMock()
        mock_response.content = content
        return mock_response
    return _create


@pytest.fixture
def base_state():
    """Minimal valid HealthAgentState for testing."""
    return {
        "user_input": "I have fever and headache",
        "intent": None,
        "has_report": False,
        "report_data": None,
        "report_type": None,
        "report_analysis": None,
        "raw_symptoms": None,
        "normalized_symptoms": None,
        "predicted_conditions": None,
        "risk_assessment": None,
        "ml_predictions": None,
        "final_response": None,
        "chat_history": None,
        "enable_diagnostic_interview": False,
        "diagnostic_phase": None,
        "question_count": 0,
        "error": False,
        "error_message": None,
    }


@pytest.fixture
def state_with_symptoms(base_state):
    """State after symptom extraction."""
    return {
        **base_state,
        "raw_symptoms": ["fever", "headache", "chills"],
        "normalized_symptoms": ["pyrexia", "cephalalgia", "rigors"],
        "intent": "SYMPTOM_ANALYSIS",
    }


@pytest.fixture
def state_with_conditions(state_with_symptoms):
    """State after disease prediction."""
    return {
        **state_with_symptoms,
        "predicted_conditions": [
            {"name": "Malaria", "probability": "High", "reasoning": "Cyclic fever with chills"},
            {"name": "Influenza", "probability": "Medium", "reasoning": "Fever with body aches"},
            {"name": "Common Cold", "probability": "Low", "reasoning": "Mild symptoms"},
        ],
        "risk_assessment": {
            "risk_level": "MEDIUM",
            "reason": "Fever with headache, could be infectious",
            "action": "Consult a doctor within 24 hours",
            "emergency_signs": []
        },
    }


@pytest.fixture
def state_with_report(base_state):
    """State with a medical report uploaded."""
    return {
        **base_state,
        "has_report": True,
        "report_data": "Patient: John Doe\nAge: 29\nHemoglobin: 8.4 g/dL (Low)\nTSH: 8.92 mIU/L (High)",
        "report_type": "pdf",
        "report_analysis": {
            "report_type": "Blood Test",
            "patient_name": "John Doe",
            "patient_age": "29 Years",
            "summary": "Low hemoglobin and elevated TSH",
            "abnormal_findings": ["Hemoglobin", "TSH"],
            "normal_findings": ["WBC", "Platelets"],
            "key_findings": [
                {"parameter": "Hemoglobin", "value": "8.4 g/dL", "normal_range": "12-16 g/dL", "status": "LOW", "significance": "Iron deficiency anemia"},
                {"parameter": "TSH", "value": "8.92 mIU/L", "normal_range": "0.4-4.0 mIU/L", "status": "HIGH", "significance": "Hypothyroidism"},
            ],
            "urgency_level": "SOON",
            "medications_mentioned": [],
            "recommended_tests": ["Iron studies", "Free T4", "Anti-TPO"],
        },
    }
