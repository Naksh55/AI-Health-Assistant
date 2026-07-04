from unittest.mock import patch

from agents.chat_runner import run_chat_turn


def test_run_chat_turn_returns_fallback_when_graph_raises():
    with patch("agents.chat_runner.health_graph") as mock_graph:
        mock_graph.invoke.side_effect = RuntimeError("boom")

        result = run_chat_turn(
            user_input="I have fever",
            has_report=False,
            report_data=None,
            report_type=None,
            chat_history=[],
            enable_diagnostic_interview=False,
            question_count=0,
        )

        assert result["error"] is True
        assert "couldn't process your request" in result["final_response"].lower()
