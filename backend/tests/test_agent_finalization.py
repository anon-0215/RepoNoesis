from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
import urllib.error
from unittest.mock import patch

from app.config import LLMSettings
from app.database import Database
from app.services.agent_contracts import AgentLimits
from app.services.agent_core import run_bounded_agent
from app.services.evidence import EvidenceBuilder
from app.services.llm_client import LLMClient, ProviderError
from app.services.qa_agent import answer_from_evidence
from app.services.relation_graph import RelationValidator
from app.services.retrieval_v2 import retrieve_code
from app.services.smoke_diagnostics import SmokeDiagnosticsRecorder
from tests.m1_helpers import disabled_embedding_service, make_project
from tests.test_m2_agent import ScriptedPlanner, decision


def _structured(text="Grounded answer", aliases=None):
    return json.dumps(
        {
            "parts": [
                {
                    "text": text,
                    "evidence_aliases": aliases or ["A1"],
                }
            ]
        }
    )


class _FinalAnswerLlm:
    available = True

    def __init__(
        self,
        response: str | None | list[str | None | Exception],
        callback=None,
    ) -> None:
        self.response = response
        self.calls = 0
        self.callback = callback
        self.messages: list[list[dict[str, str]]] = []
        self.kwargs: list[dict] = []

    def chat(self, _messages, **_kwargs):
        self.calls += 1
        self.messages.append(_messages)
        self.kwargs.append(_kwargs)
        if self.callback is not None:
            self.callback()
        if isinstance(self.response, list):
            value = self.response[self.calls - 1]
            if isinstance(value, Exception):
                raise value
            return value
        return self.response


class AgentFinalizationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = Database(Path(self.directory.name) / "agent-finalization.sqlite")
        self.project_id, self.bundle = make_project(
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
        llm = _FinalAnswerLlm(_structured())
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
        self.assertTrue(diagnostics["citation_validation_passed"])
        self.assertTrue(diagnostics["relation_validation_passed"])
        self.assertTrue(diagnostics["post_generation_validation_passed"])
        self.assertTrue(diagnostics["grounded_answer_candidate_received"])
        self.assertTrue(diagnostics["grounded_answer_accepted"])
        self.assertTrue(diagnostics["grounded_reference_validation_completed"])
        self.assertTrue(diagnostics["grounded_reference_validation_passed"])
        self.assertEqual(diagnostics["grounded_candidate_citation_count"], 1)
        self.assertNotIn("final_answer_failure_reason_code", diagnostics)
        self.assertEqual(diagnostics["agent_status"], "completed")

    def test_location_failure_repairs_once_without_original_candidate(self):
        planner = ScriptedPlanner(
            [decision("continue", "search_code", {"query": "authenticate_user"})]
        )
        original = _structured("PRIVATE_CANDIDATE [E1]", ["A1"])
        llm = _FinalAnswerLlm([original, _structured("Grounded answer", ["A1"])])
        recorder = SmokeDiagnosticsRecorder()

        result = self._run(
            planner,
            llm,
            limits=replace(AgentLimits(), max_tool_calls=1),
            recorder=recorder,
        )

        diagnostics = recorder.snapshot()
        self.assertEqual(result["agent_status"], "completed")
        self.assertEqual(result["answer_mode"], "llm_grounded")
        self.assertTrue(result["citations"])
        self.assertEqual(llm.calls, 2)
        self.assertTrue(diagnostics["final_answer_repair_attempted"])
        self.assertTrue(diagnostics["final_answer_repair_succeeded"])
        protocol = diagnostics["final_answer_protocol_failure"]
        self.assertEqual(
            protocol["stable_code"], "model_supplied_location_forbidden"
        )
        self.assertEqual(protocol["violation_kind"], "evidence_marker")
        repair_messages = llm.messages[1]
        self.assertEqual([item["role"] for item in repair_messages], ["system", "user"])
        repair_payload = json.loads(repair_messages[1]["content"])
        self.assertEqual(
            repair_payload["failure"]["stable_code"],
            "model_supplied_location_forbidden",
        )
        self.assertEqual(
            repair_payload["failure"]["violation_kind"], "evidence_marker"
        )
        repair_serialized = json.dumps(repair_messages, sort_keys=True)
        self.assertNotIn("PRIVATE_CANDIDATE", repair_serialized)
        self.assertNotIn(original, repair_serialized)
        self.assertNotIn("src/auth.py", repair_serialized)

    def test_failed_location_repair_remains_non_grounded(self):
        planner = ScriptedPlanner(
            [decision("continue", "search_code", {"query": "authenticate_user"})]
        )
        invalid = _structured("Fact [E1]", ["A1"])
        llm = _FinalAnswerLlm([invalid, invalid])
        recorder = SmokeDiagnosticsRecorder()

        result = self._run(
            planner,
            llm,
            limits=replace(AgentLimits(), max_tool_calls=1),
            recorder=recorder,
        )

        diagnostics = recorder.snapshot()
        self.assertEqual(result["agent_status"], "final_answer_failed")
        self.assertEqual(result["answer_mode"], "deterministic")
        self.assertNotEqual(result["answer_mode"], "llm_grounded")
        self.assertEqual(llm.calls, 2)
        self.assertTrue(diagnostics["final_answer_repair_attempted"])
        self.assertFalse(diagnostics["final_answer_repair_succeeded"])

    def test_provider_failure_during_repair_is_not_citation_failure(self):
        planner = ScriptedPlanner(
            [decision("continue", "search_code", {"query": "authenticate_user"})]
        )
        invalid = _structured("Fact [E1]", ["A1"])
        provider_error = ProviderError(
            "provider_unavailable",
            "The generation provider is unavailable.",
            retryable=True,
            status_code=503,
        )
        llm = _FinalAnswerLlm([invalid, provider_error])
        recorder = SmokeDiagnosticsRecorder()

        with self.assertRaises(ProviderError) as raised:
            self._run(
                planner,
                llm,
                limits=replace(AgentLimits(), max_tool_calls=1),
                recorder=recorder,
            )

        diagnostics = recorder.snapshot()
        self.assertEqual(raised.exception.code, "provider_unavailable")
        self.assertEqual(llm.calls, 2)
        self.assertTrue(diagnostics["final_answer_repair_attempted"])
        self.assertFalse(diagnostics["final_answer_repair_protocol_succeeded"])
        self.assertFalse(diagnostics["final_answer_repair_succeeded"])
        self.assertEqual(
            diagnostics["final_answer_repair_failure"]["stable_code"],
            "provider_failed",
        )
        self.assertEqual(
            diagnostics["final_answer_failure_reason_code"], "provider_failed"
        )
        self.assertNotIn("citation_failure_reason_code", diagnostics)

    def test_deadline_before_repair_call_does_not_record_attempt(self):
        outcome = retrieve_code(
            self.database,
            disabled_embedding_service(),
            self.project_id,
            "authenticate_user",
            evidence_count=1,
        )
        evidence = EvidenceBuilder().build(
            outcome.results,
            self.bundle["project"],
            retrieval_strategy_version=outcome.retrieval_strategy_version,
        )
        llm = _FinalAnswerLlm(_structured("Rejected [E1]"))
        recorder = SmokeDiagnosticsRecorder()

        # Six checks/build-time reads remain within budget.  The seventh is
        # the repair logical-call boundary after its messages are complete.
        with patch(
            "app.services.qa_agent.time.monotonic",
            side_effect=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
        ):
            with self.assertRaises(ProviderError) as raised:
                answer_from_evidence(
                    "authenticate_user",
                    evidence,
                    llm,
                    self.database,
                    retrieval_mode=outcome.retrieval_mode,
                    diagnostics_recorder=recorder,
                    request_deadline_at=1.0,
                )

        diagnostics = recorder.snapshot()
        self.assertEqual(raised.exception.code, "deadline_exceeded")
        self.assertEqual(llm.calls, 1)
        self.assertFalse(diagnostics.get("final_answer_repair_attempted", False))
        self.assertFalse(
            diagnostics.get("final_answer_repair_protocol_succeeded", False)
        )
        self.assertFalse(diagnostics.get("final_answer_repair_succeeded", False))
        self.assertNotIn("final_answer_repair_failure", diagnostics)

    def test_repair_reuses_one_absolute_request_deadline(self):
        outcome = retrieve_code(
            self.database,
            disabled_embedding_service(),
            self.project_id,
            "authenticate_user",
            evidence_count=1,
        )
        evidence = EvidenceBuilder().build(
            outcome.results,
            self.bundle["project"],
            retrieval_strategy_version=outcome.retrieval_strategy_version,
        )
        class _DeadlineCaptureLlm(LLMClient):
            def __init__(self):
                super().__init__(
                    LLMSettings(
                        provider="openai_compatible",
                        base_url="https://provider.invalid/v1",
                        api_key="test-placeholder",
                        model="configured-model",
                        max_retries=0,
                    )
                )
                self.responses = [
                    _structured("Rejected [E1]"),
                    _structured("Repaired answer"),
                ]
                self.calls = 0
                self.messages = []
                self.kwargs = []

            def chat(self, messages, **kwargs):
                self.messages.append(messages)
                self.kwargs.append(kwargs)
                value = self.responses[self.calls]
                self.calls += 1
                return value

        llm = _DeadlineCaptureLlm()
        deadline = time.monotonic() + 60

        result = answer_from_evidence(
            "authenticate_user",
            evidence,
            llm,
            self.database,
            retrieval_mode=outcome.retrieval_mode,
            request_deadline_at=deadline,
        )

        self.assertEqual(result["answer_mode"], "llm_grounded")
        self.assertEqual(llm.calls, 2)
        self.assertEqual(
            [item["deadline_monotonic"] for item in llm.kwargs],
            [deadline, deadline],
        )
        repair_payload = json.loads(llm.messages[1][1]["content"])
        self.assertGreater(repair_payload["remaining_deadline_ms"], 0)
        self.assertLessEqual(repair_payload["remaining_deadline_ms"], 60_000)

    def test_concurrent_repairs_keep_request_ids_counts_and_flags_isolated(self):
        def execute(responses):
            planner = ScriptedPlanner(
                [decision("continue", "search_code", {"query": "authenticate_user"})]
            )
            llm = _FinalAnswerLlm(responses)
            recorder = SmokeDiagnosticsRecorder()
            result = self._run(
                planner,
                llm,
                limits=replace(AgentLimits(), max_tool_calls=1),
                recorder=recorder,
            )
            return result, recorder.snapshot(), llm.calls

        with ThreadPoolExecutor(max_workers=2) as executor:
            successful = executor.submit(
                execute,
                [_structured("First [E1]"), _structured("First repaired")],
            )
            failed = executor.submit(
                execute,
                [_structured("Second [E1]"), _structured("Second [E2]")],
            )
            first_result, first_diagnostics, first_calls = successful.result()
            second_result, second_diagnostics, second_calls = failed.result()

        self.assertNotEqual(first_result["request_id"], second_result["request_id"])
        self.assertEqual((first_calls, second_calls), (2, 2))
        self.assertTrue(first_diagnostics["final_answer_repair_attempted"])
        self.assertTrue(first_diagnostics["final_answer_repair_protocol_succeeded"])
        self.assertTrue(first_diagnostics["final_answer_repair_succeeded"])
        self.assertTrue(second_diagnostics["final_answer_repair_attempted"])
        self.assertFalse(second_diagnostics["final_answer_repair_protocol_succeeded"])
        self.assertFalse(second_diagnostics["final_answer_repair_succeeded"])
        self.assertEqual(first_result["agent_status"], "completed")
        self.assertEqual(second_result["agent_status"], "final_answer_failed")

    def test_tool_budget_exhausted_without_evidence_does_not_call_final_answer(self):
        planner = ScriptedPlanner([
            decision("continue", "search_code", {"query": "definitely_absent"})
        ])
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
        llm = _FinalAnswerLlm(_structured())

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
        llm = _FinalAnswerLlm(_structured(aliases=["A999"]))

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
        self.assertEqual(llm.calls, 2)

    def test_missing_candidate_citation_has_fixed_reason_and_keeps_fallback_count_separate(self):
        planner = ScriptedPlanner(
            [decision("continue", "search_code", {"query": "authenticate_user"})]
        )
        recorder = SmokeDiagnosticsRecorder()
        result = self._run(
            planner,
            _FinalAnswerLlm(json.dumps({"parts": [{"text": "Answer", "evidence_aliases": []}]})),
            limits=replace(AgentLimits(), max_tool_calls=1),
            recorder=recorder,
        )

        diagnostics = recorder.snapshot()
        self.assertEqual(result["agent_status"], "final_answer_failed")
        self.assertEqual(result["answer_mode"], "deterministic")
        self.assertEqual(diagnostics["final_answer_failure_reason_code"], "citation_missing")
        self.assertTrue(diagnostics["grounded_answer_candidate_received"])
        self.assertFalse(diagnostics["grounded_answer_accepted"])
        self.assertEqual(diagnostics["grounded_candidate_citation_count"], 0)
        self.assertEqual(diagnostics["citation_count"], 1)

    def test_unknown_and_malformed_candidate_citations_are_distinct(self):
        cases = (
            (
                "unknown",
                _structured(aliases=["A999"]),
                "citation_unknown",
            ),
            (
                "malformed",
                "not-json",
                "citation_format_invalid",
            ),
            (
                "invalid_type",
                json.dumps({"parts": [{"text": "Fact", "evidence_aliases": "A1"}]}),
                "citation_format_invalid",
            ),
            (
                "extra_location_field",
                json.dumps({"parts": [{"text": "Fact", "evidence_aliases": ["A1"], "path": "src/missing.py"}]}),
                "citation_format_invalid",
            ),
        )
        for label, response, expected_reason in cases:
            with self.subTest(label=label):
                recorder = SmokeDiagnosticsRecorder()
                planner = ScriptedPlanner(
                    [
                        decision(
                            "continue", "search_code", {"query": "authenticate_user"}
                        )
                    ]
                )
                result = self._run(
                    planner,
                    _FinalAnswerLlm(response),
                    limits=replace(AgentLimits(), max_tool_calls=1),
                    recorder=recorder,
                )
                diagnostics = recorder.snapshot()
                self.assertEqual(result["agent_status"], "final_answer_failed")
                self.assertEqual(
                    diagnostics["final_answer_failure_reason_code"], expected_reason
                )
                self.assertFalse(diagnostics["grounded_answer_accepted"])
                self.assertTrue(diagnostics["citation_validation_completed"])
                self.assertTrue(diagnostics["citation_validation_passed"])
                self.assertTrue(
                    diagnostics["grounded_reference_validation_completed"]
                )
                self.assertFalse(
                    diagnostics["grounded_reference_validation_passed"]
                )

    def test_evidence_id_and_location_binding_failure_is_distinct(self):
        planner = ScriptedPlanner(
            [decision("continue", "search_code", {"query": "authenticate_user upload_file"})]
        )
        recorder = SmokeDiagnosticsRecorder()
        result = self._run(
            planner,
            _FinalAnswerLlm(
                json.dumps({"parts": [{"text": "Fact", "evidence_aliases": ["A1"], "revision": "fake"}]})
            ),
            limits=replace(AgentLimits(), max_tool_calls=1),
            recorder=recorder,
        )
        diagnostics = recorder.snapshot()
        self.assertEqual(result["agent_status"], "final_answer_failed")
        self.assertEqual(
            diagnostics["final_answer_failure_reason_code"],
            "citation_format_invalid",
        )

    def test_empty_provider_result_records_response_empty_without_validation_fabrication(self):
        recorder = SmokeDiagnosticsRecorder()
        planner = ScriptedPlanner(
            [decision("continue", "search_code", {"query": "authenticate_user"})]
        )
        result = self._run(
            planner,
            _FinalAnswerLlm(None),
            limits=replace(AgentLimits(), max_tool_calls=1),
            recorder=recorder,
        )

        diagnostics = recorder.snapshot()
        self.assertEqual(result["agent_status"], "final_answer_failed")
        self.assertEqual(diagnostics["final_answer_failure_reason_code"], "response_empty")
        self.assertFalse(diagnostics["grounded_answer_candidate_received"])
        self.assertFalse(diagnostics["grounded_answer_accepted"])
        self.assertNotIn("grounded_candidate_citation_count", diagnostics)
        self.assertNotIn("grounded_reference_validation_completed", diagnostics)
        self.assertNotIn("grounded_reference_validation_passed", diagnostics)

    def test_citation_validation_completion_and_failure_are_distinct(self):
        outcome = retrieve_code(
            self.database,
            disabled_embedding_service(),
            self.project_id,
            "authenticate_user",
            evidence_count=1,
        )
        evidence = EvidenceBuilder().build(
            outcome.results,
            self.bundle["project"],
            retrieval_strategy_version=outcome.retrieval_strategy_version,
        )
        evidence[0].content_hash = "invalid-test-hash"
        recorder = SmokeDiagnosticsRecorder()

        result = answer_from_evidence(
            "authenticate_user",
            evidence,
            _FinalAnswerLlm("must not be used"),
            self.database,
            retrieval_mode=outcome.retrieval_mode,
            diagnostics_recorder=recorder,
        )

        diagnostics = recorder.snapshot()
        self.assertEqual(result["evidence"], [])
        self.assertTrue(diagnostics["citation_validation_completed"])
        self.assertFalse(diagnostics["citation_validation_passed"])
        self.assertNotIn("grounded_answer_candidate_received", diagnostics)

    def test_relation_validation_failure_has_its_own_reason_code(self):
        recorder = SmokeDiagnosticsRecorder()
        planner = ScriptedPlanner(
            [
                decision("continue", "search_code", {"query": "authenticate_user"}),
                decision("answer"),
            ]
        )
        with patch.object(
            RelationValidator,
            "validate_chains",
            return_value=([], ["safe-test-rejection"]),
        ):
            result = self._run(
                planner,
                _FinalAnswerLlm(_structured()),
                limits=AgentLimits(),
                recorder=recorder,
            )

        diagnostics = recorder.snapshot()
        self.assertEqual(result["answer_mode"], "llm_grounded")
        self.assertTrue(diagnostics["relation_validation_completed"])
        self.assertFalse(diagnostics["relation_validation_passed"])
        self.assertTrue(diagnostics["grounded_answer_accepted"])
        self.assertEqual(
            diagnostics["final_answer_failure_reason_code"],
            "relation_validation_failed",
        )

    def test_post_generation_validation_failure_keeps_sticky_citation_code(self):
        def change_persisted_source():
            with self.database.connect() as connection:
                connection.execute(
                    "UPDATE repo_files SET content = ? WHERE project_id = ? AND path = ?",
                    (
                        "def changed(password):\n    return False\n",
                        self.project_id,
                        "src/auth.py",
                    ),
                )

        recorder = SmokeDiagnosticsRecorder()
        planner = ScriptedPlanner(
            [decision("continue", "search_code", {"query": "authenticate_user"})]
        )
        result = self._run(
            planner,
            _FinalAnswerLlm(
                _structured(),
                callback=change_persisted_source,
            ),
            limits=replace(AgentLimits(), max_tool_calls=1),
            recorder=recorder,
        )

        diagnostics = recorder.snapshot()
        self.assertTrue(diagnostics["post_generation_validation_completed"])
        self.assertFalse(diagnostics["post_generation_validation_passed"])
        self.assertFalse(diagnostics["grounded_answer_accepted"])
        self.assertEqual(
            diagnostics["final_answer_failure_reason_code"],
            "citation_evidence_binding_failed",
        )
        self.assertEqual(
            diagnostics["citation_failure_reason_code"],
            "citation_evidence_binding_failed",
        )

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
        llm = _FinalAnswerLlm(_structured())

        result = self._run(planner, llm, limits=AgentLimits())

        self.assertEqual(result["agent_status"], "completed")
        self.assertEqual(result["answer_mode"], "llm_grounded")
        self.assertEqual(planner.calls, 2)
        self.assertEqual(llm.calls, 1)
        self.assertNotIn("diagnostics", result)


if __name__ == "__main__":
    unittest.main()
