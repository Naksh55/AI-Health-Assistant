"""
tests/test_classify_intent.py — Tests for Intent Classification
================================================================
Tests the classify_intent() function which routes user messages
to the correct pipeline branch.

Coverage:
  - Symptom descriptions → SYMPTOM_ANALYSIS
  - Educational questions → SIMPLE_QUESTION
  - Report queries → REPORT_QUERY
  - Report overview → REPORT_OVERVIEW
  - Action requests → ACTION_REQUEST
  - Short confirmations → ACTION_REQUEST
  - Mixed messages → SYMPTOM_ANALYSIS
  - Edge cases
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from unittest.mock import patch, MagicMock
from agents.graph import classify_intent


class TestSymptomAnalysis:
    """Messages describing symptoms should route to SYMPTOM_ANALYSIS."""

    def test_basic_symptoms(self):
        assert classify_intent("I have fever and headache", False) == "SYMPTOM_ANALYSIS"

    def test_symptoms_with_duration(self):
        assert classify_intent("I've had a cough for 3 days", False) == "SYMPTOM_ANALYSIS"

    def test_feeling_descriptions(self):
        assert classify_intent("I feel dizzy and nauseous", False) == "SYMPTOM_ANALYSIS"

    def test_pain_descriptions(self):
        assert classify_intent("I have severe chest pain", False) == "SYMPTOM_ANALYSIS"

    def test_multiple_symptoms(self):
        assert classify_intent("fever, chills, body aches, and vomiting", False) == "SYMPTOM_ANALYSIS"

    def test_informal_language(self):
        assert classify_intent("my head is killing me and I feel hot", False) == "SYMPTOM_ANALYSIS"


class TestSimpleQuestion:
    """Educational/knowledge questions should route to SIMPLE_QUESTION."""

    def test_what_is(self):
        assert classify_intent("what is diabetes", False) == "SIMPLE_QUESTION"

    def test_explain(self):
        assert classify_intent("explain hypothyroidism", False) == "SIMPLE_QUESTION"

    def test_how_to_treat(self):
        assert classify_intent("how to treat anemia", False) == "SIMPLE_QUESTION"

    def test_what_causes(self):
        assert classify_intent("what causes high blood pressure", False) == "SIMPLE_QUESTION"


class TestReportQuery:
    """Questions about specific report values should route to REPORT_QUERY."""

    def test_cholesterol_query(self):
        assert classify_intent("what is my cholesterol level", True) == "REPORT_QUERY"

    def test_patient_name(self):
        assert classify_intent("patient name", True) == "REPORT_QUERY"

    def test_my_values(self):
        assert classify_intent("what are my hemoglobin levels", True) == "REPORT_QUERY"


class TestReportOverview:
    """Full report review requests should route to REPORT_OVERVIEW."""

    def test_check_report(self):
        assert classify_intent("check my report", True) == "REPORT_OVERVIEW"

    def test_analyze_report(self):
        assert classify_intent("analyze my report", True) == "REPORT_OVERVIEW"

    def test_whats_wrong(self):
        assert classify_intent("what's wrong with me", True) == "REPORT_OVERVIEW"


class TestActionRequest:
    """Action requests should route to ACTION_REQUEST."""

    def test_schedule(self):
        assert classify_intent("schedule an appointment", False) == "ACTION_REQUEST"

    def test_book_doctor(self):
        assert classify_intent("book a doctor visit", False) == "ACTION_REQUEST"


class TestShortConfirmations:
    """Short confirmations should route to ACTION_REQUEST for follow-up handling."""

    def test_yes(self):
        assert classify_intent("yes", False) == "ACTION_REQUEST"

    def test_ok(self):
        assert classify_intent("ok", False) == "ACTION_REQUEST"

    def test_sure(self):
        assert classify_intent("sure", False) == "ACTION_REQUEST"

    def test_please(self):
        assert classify_intent("please", False) == "ACTION_REQUEST"


class TestMixedMessages:
    """Messages with both symptoms and report references → SYMPTOM_ANALYSIS."""

    def test_symptoms_plus_report(self):
        result = classify_intent("I feel tired and cold, check my report", True)
        assert result == "SYMPTOM_ANALYSIS"


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_string_with_report(self):
        """Empty string with report should still work (won't match symptom keywords)."""
        # Empty string handled by supervisor, not classify_intent
        # classify_intent should return something without crashing
        result = classify_intent("", False)
        assert result is not None

    def test_very_long_input(self):
        """Very long input should not crash."""
        long_text = "I have pain " * 200
        result = classify_intent(long_text, False)
        assert result in ["SYMPTOM_ANALYSIS", "SIMPLE_QUESTION", "REPORT_QUERY",
                          "ACTION_REQUEST", "REPORT_OVERVIEW"]

    def test_special_characters(self):
        """Special characters should not crash."""
        result = classify_intent("I have a headache!!! @#$%", False)
        assert result is not None
