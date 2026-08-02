from __future__ import annotations

from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest
import urllib.error

from app.config import LLMSettings
from app.database import Database
from app.services.agent_contracts import AgentLimits
from app.services.agent_core import run_bounded_agent
from app.services.llm_client import LLMClient, ProviderError
from app.services.smoke_diagnostics import SmokeDiagnosticsRecorder
from tests.m1_helpers import disabled_embedding_service, make_project
from tests.test_m2_agent import ScriptedPlanner, decision


class _FinalAnswerLlm:
    available = True

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def chat(self, _messages, **_kwargs):
        self.calls += 1
        return self.response


class AgentFinalizationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = Database(Path(self.directory.name) / "agent-finalization.sqlite")
        _project_id, self.bundle = make_project(
            self.database,
            [
                (
                    "src/auth.py",
                    "authenticate_user",
                    "def authenticate_user(password):\n    return verify(password)\n",
                ),
                (
                    "src/upload.py",
                    "upload_file",
                    "def upload_file(path):\n    return save(path)\n",
                ),
            ],
        )

    def _run(self, planner, llm, *, limits, recorder=None):
        return run_bounded_agent(
            "authenticate_user",
            self.bundle,
            llm,
            self.database,
            disabled_embedding_service(),
            planner=planner,
            limits=limits,
            diagnostics_recorder=recorder,
        )

    def test_last_allowed_tool_finds_evidence_then_uses_one_final_answer_call(self):
        planner = ScriptedPlanner(
            [
                decision("continue", "search_code", {"query": "authenticate_user"}),
                decision("answer"),
            ]
        )
        llm = _FinalAnswerLlm("Grounded answer [E1] src/auth.py:1-2.")
        recorder = SmokeDiagnosticsRecorder()
        limits = replace(AgentLimits(), max_tool_calls=1)

        result = self._run(planner, llm, limits=limits, recorder=recorder)

        diagnostics = recorder.snapshot()
        self.assertEqual(result["agent_status"], "completed")
        self.assertEqual(result["answer_mode"], "llm_grounded")
        self.assertEqual(result["budget_usage"]["tool_calls_used"], 1)
        self.assertEqual(result["budget_usage"]["limits"]["max_tool_calls"], 1)
        self.assertEqual(planner.calls, 1)
        self.assertEqual(llm.calls, 1)
        self.assertTrue(diagnostics["tool_budget_exhausted"])
        self.assertTrue(diagnostics["final_answer_attempted"])
        self.assertTrue(diagnostics["final_answer_response_received"])
        self.assertEqual(diagnostics["agent_status"], "completed")

    def test_tool_budget_exhausted_without_evidence_does_not_call_final_answer(self):
        planner = ScriptedPlanner([decision("continue", "unknown_tool", {})])
        llm = _FinalAnswerLlm("must not be used")
        recorder = SmokeDiagnosticsRecorder()

        result = self._run(
            planner,
            llm,
            limits=replace(AgentLimits(), max_tool_calls=1),
            recorder=recorder,
        )

        diagnostics = recorder.snapshot()
        self.assertEqual(result["agent_status"], "insufficient_evidence")
        self.assertEqual(result["answer_mode"], "deterministic")
        self.assertEqual(result["citations"], [])
        self.assertEqual(llm.calls, 0)
        self.assertTrue(diagnostics["tool_budget_exhausted"])
        self.assertFalse(diagnostics["final_answer_attempted"])

    def test_fifth_tool_failure_after_evidence_still_finalizes_without_extra_planning(self):
        planner = ScriptedPlanner(
            [
                decision("continue", "search_code", {"query": "authenticate_user"}),
                decision(
                    "continue",
                    "read_source",
                    {"path": "src/auth.py", "start_line": 1, "end_line": 2},
                ),
                decision("continue", "lookup_symbol", {"symbol": "upload_file"}),
                decision(
                    "continue",
                    "read_source",
                    {"path": "src/upload.py", "start_line": 1, "end_line": 2},
                ),
                decision(
                    "continue",
                    "read_source",
                    {"path": "src/missing.py", "start_line": 1, "end_line": 1},
                ),
                decision("answer"),
            ]
        )
        llm = _FinalAnswerLlm("Grounded answer [E1] src/auth.py:1-2.")

        result = self._run(planner, llm, limits=AgentLimits())

        self.assertEqual(result["agent_status"], "completed")
        self.assertEqual(result["answer_mode"], "llm_grounded")
        self.assertEqual(result["budget_usage"]["steps_used"], 5)
        self.assertEqual(result["budget_usage"]["tool_calls_used"], 5)
        self.assertEqual(planner.calls, 5)
        self.assertEqual(llm.calls, 1)
        self.assertEqual(result["agent_trace"][-1]["tool_calls"][0]["status"], "failed")

    def test_all_tools_fail_without_evidence_never_fabricates_citations(self):
        planner = ScriptedPlanner(
            [
                decision("continue", "lookup_symbol", {"symbol": "missing_one"}),
                decision("continue", "lookup_symbol", {"symbol": "missing_two"}),
            ]
        )
        llm = _FinalAnswerLlm("must not be used")

        result = self._run(
            planner,
            llm,
            limits=replace(AgentLimits(), max_tool_calls=2),
        )

        self.assertEqual(result["agent_status"], "insufficient_evidence")
        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["citations"], [])
        self.assertEqual(llm.calls, 0)

    def test_invalid_final_references_are_rejected_and_mark_final_answer_failed(self):
        planner = ScriptedPlanner(
            [decision("continue", "search_code", {"query": "authenticate_user"})]
        )
        llm = _FinalAnswerLlm("Unsupported answer [E999] src/missing.py:1-1.")

        result = self._run(
            planner,
            llm,
            limits=replace(AgentLimits(), max_tool_calls=1),
        )

        self.assertEqual(result["agent_status"], "final_answer_failed")
        self.assertEqual(result["answer_mode"], "deterministic")
        self.assertEqual(len(result["evidence"]), 1)
        self.assertEqual(len(result["citations"]), 1)
        self.assertNotIn("E999", result["answer"])
        self.assertEqual(llm.calls, 1)

    def test_provider_failure_records_attempt_and_failure_status_then_propagates(self):
        recorder = SmokeDiagnosticsRecorder()

        def fail_before_response(_request, *, timeout):
            raise urllib.error.URLError("offline-test")

        llm = LLMClient(
            LLMSettings(
                provider="openai_compatible",
                base_url="https://provider.invalid/v1",
                api_key="test-placeholder",
                model="configured-model",
                max_retries=0,
            ),
            opener=fail_before_response,
            sleep=lambda _value: None,
            diagnostics_recorder=recorder,
        )
        planner = ScriptedPlanner(
            [decision("continue", "search_code", {"query": "authenticate_user"})]
        )

        with self.assertRaises(ProviderError):
            self._run(
                planner,
                llm,
                limits=replace(AgentLimits(), max_tool_calls=1),
                recorder=recorder,
            )

        diagnostics = recorder.snapshot()
        self.assertTrue(diagnostics["final_answer_attempted"])
        self.assertFalse(diagnostics["final_answer_response_received"])
        self.assertEqual(diagnostics["agent_status"], "final_answer_failed")
        provider_call = diagnostics["provider_calls"][0]
        self.assertEqual(provider_call["purpose"], "final_answer")
        self.assertTrue(provider_call["request_started"])
        self.assertFalse(provider_call["response_received"])

    def test_provider_http_failure_records_that_a_response_was_received(self):
        recorder = SmokeDiagnosticsRecorder()

        def fail_with_response(_request, *, timeout):
            raise urllib.error.HTTPError(
                "https://provider.invalid/v1/chat/completions",
                503,
                "unavailable",
                {},
                io.BytesIO(b"ignored-test-body"),
            )

        llm = LLMClient(
            LLMSettings(
                provider="openai_compatible",
                base_url="https://provider.invalid/v1",
                api_key="test-placeholder",
                model="configured-model",
                max_retries=0,
            ),
            opener=fail_with_response,
            sleep=lambda _value: None,
            diagnostics_recorder=recorder,
        )
        planner = ScriptedPlanner(
            [decision("continue", "search_code", {"query": "authenticate_user"})]
        )

        with self.assertRaises(ProviderError):
            self._run(
                planner,
                llm,
                limits=replace(AgentLimits(), max_tool_calls=1),
                recorder=recorder,
            )

        diagnostics = recorder.snapshot()
        self.assertTrue(diagnostics["final_answer_attempted"])
        self.assertTrue(diagnostics["final_answer_response_received"])
        self.assertEqual(diagnostics["agent_status"], "final_answer_failed")
        provider_call = diagnostics["provider_calls"][0]
        self.assertTrue(provider_call["response_received"])
        self.assertEqual(provider_call["http_status"], 503)

    def test_explicit_answer_and_unexhausted_bounded_path_remain_unchanged(self):
        planner = ScriptedPlanner(
            [
                decision("continue", "search_code", {"query": "authenticate_user"}),
                decision("answer"),
            ]
        )
        llm = _FinalAnswerLlm("Grounded answer [E1] src/auth.py:1-2.")

        result = self._run(planner, llm, limits=AgentLimits())

        self.assertEqual(result["agent_status"], "completed")
        self.assertEqual(result["answer_mode"], "llm_grounded")
        self.assertEqual(planner.calls, 2)
        self.assertEqual(llm.calls, 1)
        self.assertNotIn("diagnostics", result)


if __name__ == "__main__":
    unittest.main()
