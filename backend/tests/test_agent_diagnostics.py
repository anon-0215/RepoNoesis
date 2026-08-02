from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from app.database import Database
from app.services.agent_core import run_bounded_agent
from app.services.smoke_diagnostics import SmokeDiagnosticsRecorder
from tests.m1_helpers import disabled_embedding_service, make_project
from tests.test_m2_agent import NoLlm, ScriptedPlanner, decision


class _FinalLlm:
    available = True

    def chat(self, _messages, **_kwargs):
        return "Grounded answer [E1] src/auth.py:1-2."


class AgentDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = Database(Path(self.directory.name) / "agent-diagnostics.sqlite")
        _project_id, self.bundle = make_project(
            self.database,
            [
                (
                    "src/auth.py",
                    "authenticate_user",
                    "def authenticate_user(password):\n    return verify(password)\n",
                )
            ],
        )

    def _run(self, planner, *, llm=None, question="authenticate_user"):
        recorder = SmokeDiagnosticsRecorder()
        result = run_bounded_agent(
            question,
            self.bundle,
            llm or NoLlm(),
            self.database,
            disabled_embedding_service(),
            planner=planner,
            diagnostics_recorder=recorder,
        )
        return result, recorder.snapshot()

    def test_planner_invalid_json_then_repair_success_is_recorded(self):
        result, diagnostics = self._run(
            ScriptedPlanner(
                [
                    "not-json",
                    decision("continue", "search_code", {"query": "authenticate_user"}),
                    decision("answer"),
                ]
            )
        )
        self.assertEqual(result["agent_mode"], "bounded")
        self.assertEqual(diagnostics["planner_requests_attempted"], 3)
        self.assertTrue(diagnostics["planner_response_received"])
        self.assertTrue(diagnostics["planner_json_valid"])
        self.assertEqual(diagnostics["planner_repair_attempts"], 1)

    def test_planner_fallback_uses_fixed_reason_code(self):
        result, diagnostics = self._run(ScriptedPlanner(["bad", "still bad"]))
        self.assertEqual(result["agent_mode"], "deterministic_fallback")
        self.assertEqual(diagnostics["fallback_reason_code"], "planner_validation_failed")
        self.assertFalse(diagnostics["planner_json_valid"])

    def test_tool_success_failure_and_not_called_are_distinguishable(self):
        scenarios = (
            (
                "success",
                [
                    decision("continue", "search_code", {"query": "authenticate_user"}),
                    decision("answer"),
                ],
                (1, 1, 0),
            ),
            (
                "failure",
                [decision("continue", "unknown_tool", {}), decision("answer")],
                (1, 0, 1),
            ),
            ("not_called", [decision("answer")], (0, 0, 0)),
        )
        for label, decisions, expected in scenarios:
            with self.subTest(label=label):
                _result, diagnostics = self._run(ScriptedPlanner(decisions))
                actual = (
                    diagnostics["tool_calls_attempted"],
                    diagnostics["tool_calls_succeeded"],
                    diagnostics["tool_calls_failed"],
                )
                self.assertEqual(actual, expected)

    def test_final_answer_and_validators_are_recorded_without_response_schema_change(self):
        result, diagnostics = self._run(
            ScriptedPlanner(
                [
                    decision("continue", "search_code", {"query": "authenticate_user"}),
                    decision("answer"),
                ]
            ),
            llm=_FinalLlm(),
        )
        self.assertTrue(diagnostics["final_answer_attempted"])
        self.assertTrue(diagnostics["final_answer_response_received"])
        self.assertTrue(diagnostics["citation_validation_completed"])
        self.assertTrue(diagnostics["relation_validation_completed"])
        self.assertTrue(diagnostics["post_generation_validation_completed"])
        self.assertEqual(diagnostics["answer_mode"], "llm_grounded")
        self.assertEqual(diagnostics["agent_mode"], "bounded")
        self.assertEqual(diagnostics["agent_status"], "completed")
        self.assertNotIn("diagnostics", result)

    def test_agent_logs_and_diagnostics_exclude_request_and_source_bodies(self):
        with self.assertLogs("app.services.agent_core", level="INFO") as captured:
            _result, diagnostics = self._run(
                ScriptedPlanner(
                    [
                        decision(
                            "continue", "search_code", {"query": "authenticate_user"}
                        ),
                        decision("answer"),
                    ]
                ),
                question="sensitive-question-body",
            )
        serialized = "\n".join(captured.output) + str(diagnostics)
        self.assertNotIn("sensitive-question-body", serialized)
        self.assertNotIn("def authenticate_user", serialized)
        self.assertNotIn("return verify", serialized)


if __name__ == "__main__":
    unittest.main()
