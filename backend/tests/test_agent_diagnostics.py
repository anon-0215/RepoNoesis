from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.database import Database
import app.services.agent_core as agent_core
import app.services.agent_tools as agent_tools
from app.services.agent_core import (
    LLMPlanner,
    build_planner_json_schema,
    run_bounded_agent,
)
from app.services.agent_contracts import AgentLimits
from app.services.agent_tools import (
    ToolRegistry,
    ToolSpec,
    build_m2_tool_registry,
)
from app.services.agent_contracts import SearchCodeInput
from app.services.ask_diagnostics import ask_failure_http_status, build_ask_failure_detail
from app.services.llm_client import ProviderError
from app.services.smoke_diagnostics import SmokeDiagnosticsRecorder
from tests.m1_helpers import disabled_embedding_service, make_project
from tests.test_m2_agent import NoLlm, ScriptedPlanner, decision


class _FinalLlm:
    available = True

    def __init__(self):
        self.calls = 0

    def chat(self, _messages, **_kwargs):
        self.calls += 1
        return json.dumps({"parts": [{"text": "Grounded answer", "evidence_aliases": ["A1"]}]})


class _CapturingPlannerLlm:
    def __init__(self):
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return '{"status":"answer","action":null,"arguments":{},"decision_summary":"done"}'


class _PromptCapturingLlm:
    available = True
    settings = SimpleNamespace(planner_thinking=None)

    def __init__(self, planner_responses):
        self.planner_responses = list(planner_responses)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if kwargs.get("purpose") == "planner":
            return self.planner_responses.pop(0)
        return json.dumps({"parts": [{"text": "Grounded answer", "evidence_aliases": ["A1"]}]})


class _PlannerFailsAfterEvidence:
    def __init__(self):
        self.calls = 0

    def decide(self, _state, *, repair_hint=None):
        self.calls += 1
        if self.calls == 1:
            return decision(
                "continue", "search_code", {"query": "authenticate_user"}
            ), 10
        raise ProviderError(
            "provider_unavailable",
            "Safe synthetic provider failure.",
            retryable=True,
            status_code=503,
        )


class _TruncatedPlanner:
    def decide(self, _state, *, repair_hint=None):
        body = "PRIVATE_TRUNCATED_PLANNER_BODY"
        raise ProviderError(
            "provider_output_truncated",
            "Safe synthetic truncation.",
            diagnostics={
                "finish_reason_present": True,
                "finish_reason": "length",
                "content_present": True,
                "reasoning_content_present": True,
                "output_chars": len(body),
                "output_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                "markdown_fence_detected": False,
            },
        )


class _DeadlineClock:
    expired = False

    def __call__(self):
        return 1.0 if self.expired else 0.0


class _MutableClock:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        return self.value


class _AdvancingPlanner:
    def __init__(self, clock, values, responses):
        self.clock = clock
        self.values = list(values)
        self.responses = list(responses)
        self.states = []
        self.calls = 0

    def decide(self, state, *, repair_hint=None):
        self.states.append((state, repair_hint))
        self.clock.value = self.values[self.calls]
        response = self.responses[self.calls]
        self.calls += 1
        return response, 1


class _ExpiringRegistry:
    def __init__(self, delegate, clock):
        self.delegate = delegate
        self.clock = clock

    def list_tools(self):
        return self.delegate.list_tools()

    def get(self, name):
        return self.delegate.get(name)

    def execute(self, context, call):
        result = self.delegate.execute(context, call)
        self.clock.expired = True
        return result

    def execute_resolved(self, context, call, spec):
        result = self.delegate.execute_resolved(context, call, spec)
        self.clock.expired = True
        return result


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

    def _run(self, planner, *, llm=None, question="authenticate_user", limits=None):
        recorder = SmokeDiagnosticsRecorder()
        result = run_bounded_agent(
            question,
            self.bundle,
            llm or NoLlm(),
            self.database,
            disabled_embedding_service(),
            planner=planner,
            limits=limits,
            diagnostics_recorder=recorder,
        )
        return result, recorder.snapshot()

    def _failure_code(self, result, diagnostics):
        return build_ask_failure_detail(
            result=result,
            recorder_snapshot=diagnostics,
            retrieval_version="v1",
            hierarchy_mode="off",
            relation_mode="off",
        )["code"]

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
        attempts = diagnostics["planner_attempts"]
        self.assertEqual(
            [(item["stage"], item["stable_code"]) for item in attempts],
            [
                ("parser", "invalid_json"),
                ("semantic", "valid"),
                ("semantic", "valid"),
            ],
        )
        self.assertFalse(attempts[0]["repair_attempt"])
        self.assertTrue(attempts[1]["repair_attempt"])

    def test_initial_and_repair_prompts_share_schema_and_exclude_rejected_content(self):
        rejected = "SENSITIVE_INVALID_PLANNER_OUTPUT"
        llm = _PromptCapturingLlm(
            [
                rejected,
                json.dumps(
                    decision(
                        "continue",
                        "search_code",
                        {"query": "authenticate_user"},
                    )
                ),
                json.dumps(decision("answer")),
            ]
        )
        recorder = SmokeDiagnosticsRecorder()
        registry = build_m2_tool_registry(AgentLimits())
        result = run_bounded_agent(
            "authenticate_user",
            self.bundle,
            llm,
            self.database,
            disabled_embedding_service(),
            planner=LLMPlanner(llm, registry, AgentLimits(), recorder),
            registry=registry,
            diagnostics_recorder=recorder,
        )

        self.assertEqual(result["agent_status"], "completed")
        planner_calls = [call for call in llm.calls if call.get("purpose") == "planner"]
        self.assertEqual(len(planner_calls), 3)
        initial_payload = json.loads(planner_calls[0]["messages"][1]["content"])
        repair_payload = json.loads(planner_calls[1]["messages"][1]["content"])
        expected_schema = build_planner_json_schema(registry)
        self.assertEqual(
            initial_payload["server_constraints"]["planner_json_schema"],
            expected_schema,
        )
        self.assertEqual(
            repair_payload["repair_request"]["planner_json_schema"],
            expected_schema,
        )
        failure = repair_payload["repair_request"]["failure"]
        self.assertEqual(
            (failure["stage"], failure["stable_code"], failure["field_path"]),
            ("parser", "invalid_json", []),
        )
        repair_serialized = json.dumps(repair_payload, sort_keys=True)
        self.assertNotIn(rejected, repair_serialized)
        self.assertNotIn("JSONDecodeError", repair_serialized)
        self.assertNotIn("reasoning_content", repair_serialized)
        attempt = recorder.snapshot()["planner_attempts"][0]
        self.assertEqual(attempt["output_chars"], len(rejected))
        self.assertEqual(
            attempt["output_sha256"],
            hashlib.sha256(rejected.encode("utf-8")).hexdigest(),
        )

    def test_formal_product_repair_failure_short_circuits_final_provider(self):
        final_llm = _FinalLlm()
        planner = ScriptedPlanner(["bad", "still bad"])
        recorder = SmokeDiagnosticsRecorder()
        result = run_bounded_agent(
            "authenticate_user",
            self.bundle,
            final_llm,
            self.database,
            disabled_embedding_service(),
            planner=planner,
            diagnostics_recorder=recorder,
            allow_planner_failure_fallback=False,
        )
        diagnostics = recorder.snapshot()
        detail = build_ask_failure_detail(
            result=result,
            recorder_snapshot=diagnostics,
            retrieval_version="v1",
            hierarchy_mode="off",
            relation_mode="off",
        )
        self.assertEqual(planner.calls, 2)
        self.assertEqual(final_llm.calls, 0)
        self.assertEqual(result["agent_mode"], "bounded")
        self.assertEqual(result["agent_status"], "failed")
        self.assertEqual(detail["code"], "planner_repair_failed")
        self.assertFalse(detail["diagnostics"]["final_answer_attempted"])
        self.assertNotIn("fallback_reason_code", diagnostics)

    def test_adapter_truncation_has_safe_planner_stage_without_repair(self):
        recorder = SmokeDiagnosticsRecorder()
        with self.assertRaises(ProviderError) as raised:
            run_bounded_agent(
                "authenticate_user",
                self.bundle,
                _FinalLlm(),
                self.database,
                disabled_embedding_service(),
                planner=_TruncatedPlanner(),
                diagnostics_recorder=recorder,
            )
        self.assertEqual(raised.exception.code, "provider_output_truncated")
        attempt = recorder.snapshot()["planner_attempts"][0]
        self.assertEqual(
            (attempt["stage"], attempt["stable_code"]),
            ("adapter", "provider_output_truncated"),
        )
        self.assertEqual(attempt["finish_reason_value"], "length")
        self.assertTrue(attempt["finish_reason_present"])
        self.assertTrue(attempt["content_present"])
        self.assertTrue(attempt["reasoning_content_present"])
        self.assertFalse(attempt["repair_attempt"])
        self.assertNotIn(
            "PRIVATE_TRUNCATED_PLANNER_BODY", json.dumps(recorder.snapshot())
        )

    def test_planner_fallback_uses_fixed_reason_code(self):
        result, diagnostics = self._run(ScriptedPlanner(["bad", "still bad"]))
        self.assertEqual(result["agent_mode"], "deterministic_fallback")
        self.assertEqual(diagnostics["fallback_reason_code"], "planner_validation_failed")
        self.assertFalse(diagnostics["planner_json_valid"])

    def test_planner_token_budget_exhaustion_survives_status_projection(self):
        result, diagnostics = self._run(
            ScriptedPlanner([decision("answer")], token_usage=10),
            limits=replace(AgentLimits(), max_total_planner_output_tokens=1),
        )
        self.assertEqual(
            diagnostics["agent_failure_reason_code"], "planner_budget_exhausted"
        )
        self.assertEqual(
            self._failure_code(result, diagnostics), "planner_budget_exhausted"
        )

    def test_deadline_exhaustion_survives_status_projection(self):
        result, diagnostics = self._run(
            ScriptedPlanner([decision("answer")]),
            limits=replace(AgentLimits(), total_deadline_ms=0),
        )
        self.assertEqual(diagnostics["agent_failure_reason_code"], "deadline_exceeded")
        self.assertEqual(self._failure_code(result, diagnostics), "deadline_exceeded")

    def test_deadline_after_evidence_before_final_answer_is_public_504(self):
        recorder = SmokeDiagnosticsRecorder()
        limits = replace(
            AgentLimits(), total_deadline_ms=100, min_final_answer_budget_ms=0
        )
        clock = _DeadlineClock()
        registry = _ExpiringRegistry(build_m2_tool_registry(limits), clock)

        with (
            patch.object(agent_core.time, "monotonic", clock),
            patch.object(agent_tools.time, "monotonic", clock),
        ):
            result = run_bounded_agent(
                "authenticate_user",
                self.bundle,
                _FinalLlm(),
                self.database,
                disabled_embedding_service(),
                planner=ScriptedPlanner(
                    [
                        decision(
                            "continue",
                            "search_code",
                            {"query": "authenticate_user"},
                        )
                    ]
                ),
                limits=limits,
                registry=registry,
                diagnostics_recorder=recorder,
                request_id="deadline-request",
            )

        detail = build_ask_failure_detail(
            result=result,
            recorder_snapshot=recorder.snapshot(),
            retrieval_version="v1",
            hierarchy_mode="off",
            relation_mode="off",
        )
        self.assertEqual(result["answer"], "")
        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["citations"], [])
        self.assertFalse(detail["diagnostics"]["final_answer_attempted"])
        self.assertGreater(detail["diagnostics"]["evidence_count"], 0)
        self.assertEqual(detail["diagnostics"]["citation_count"], 0)
        self.assertEqual(detail["code"], "deadline_exceeded")
        self.assertEqual(detail["diagnostics"]["failure_stage"], "deadline")
        self.assertEqual(ask_failure_http_status(detail), 504)

    def test_deadline_after_tool_skips_nonessential_validators(self):
        recorder = SmokeDiagnosticsRecorder()
        limits = replace(
            AgentLimits(), total_deadline_ms=100, min_final_answer_budget_ms=0
        )
        clock = _DeadlineClock()
        registry = _ExpiringRegistry(build_m2_tool_registry(limits), clock)

        with (
            patch.object(agent_core.time, "monotonic", clock),
            patch.object(agent_tools.time, "monotonic", clock),
            patch.object(agent_core.CitationValidator, "validate_all") as citations,
            patch.object(agent_core.RelationValidator, "validate_chains") as relations,
        ):
            result = run_bounded_agent(
                "authenticate_user",
                self.bundle,
                _FinalLlm(),
                self.database,
                disabled_embedding_service(),
                planner=ScriptedPlanner(
                    [decision("continue", "search_code", {"query": "authenticate_user"})]
                ),
                limits=limits,
                registry=registry,
                diagnostics_recorder=recorder,
            )

        self.assertEqual(result["answer"], "")
        citations.assert_not_called()
        relations.assert_not_called()
        self.assertFalse(recorder.snapshot()["final_answer_attempted"])

    def test_slow_planner_stops_at_work_cutoff_without_claiming_request_deadline(self):
        clock = _MutableClock()
        limits = replace(
            AgentLimits(), total_deadline_ms=10_000, min_final_answer_budget_ms=4_000
        )
        planner = _AdvancingPlanner(clock, [6.1], [decision("answer")])
        recorder = SmokeDiagnosticsRecorder()
        with patch.object(agent_core.time, "monotonic", clock):
            result = run_bounded_agent(
                "authenticate_user",
                self.bundle,
                NoLlm(),
                self.database,
                disabled_embedding_service(),
                planner=planner,
                limits=limits,
                diagnostics_recorder=recorder,
            )
        detail = build_ask_failure_detail(
            result=result,
            recorder_snapshot=recorder.snapshot(),
            retrieval_version="v1",
            hierarchy_mode="off",
            relation_mode="off",
        )
        self.assertEqual(planner.states[0][0]["deadline_monotonic"], 6.0)
        self.assertEqual(planner.states[0][0]["remaining_budget"]["time_ms"], 6000)
        self.assertEqual(detail["code"], "planner_budget_exhausted")
        self.assertEqual(ask_failure_http_status(detail), 503)
        self.assertFalse(detail["diagnostics"]["request_deadline_reached"])
        self.assertEqual(result["answer"], "")

    def test_production_llm_planner_receives_work_cutoff_not_request_deadline(self):
        clock = _MutableClock()
        limits = replace(
            AgentLimits(), total_deadline_ms=10_000, min_final_answer_budget_ms=4_000
        )
        planner_llm = _CapturingPlannerLlm()
        registry = build_m2_tool_registry(limits)
        planner = LLMPlanner(planner_llm, registry, limits)
        with patch.object(agent_core.time, "monotonic", clock):
            run_bounded_agent(
                "authenticate_user",
                self.bundle,
                NoLlm(),
                self.database,
                disabled_embedding_service(),
                planner=planner,
                limits=limits,
                registry=registry,
            )
        self.assertEqual(len(planner_llm.calls), 1)
        self.assertEqual(planner_llm.calls[0]["deadline_monotonic"], 6.0)
        self.assertEqual(planner_llm.calls[0]["timeout_seconds"], 6.0)

    def test_slow_repair_uses_the_same_work_cutoff(self):
        clock = _MutableClock()
        limits = replace(
            AgentLimits(), total_deadline_ms=10_000, min_final_answer_budget_ms=4_000
        )
        planner = _AdvancingPlanner(
            clock,
            [1.0, 6.1],
            ["not-json", decision("answer")],
        )
        recorder = SmokeDiagnosticsRecorder()
        with patch.object(agent_core.time, "monotonic", clock):
            result = run_bounded_agent(
                "authenticate_user",
                self.bundle,
                NoLlm(),
                self.database,
                disabled_embedding_service(),
                planner=planner,
                limits=limits,
                diagnostics_recorder=recorder,
            )
        detail = build_ask_failure_detail(
            result=result,
            recorder_snapshot=recorder.snapshot(),
            retrieval_version="v1",
            hierarchy_mode="off",
            relation_mode="off",
        )
        self.assertEqual(planner.calls, 2)
        self.assertEqual(planner.states[0][0]["deadline_monotonic"], 6.0)
        self.assertEqual(planner.states[1][0]["deadline_monotonic"], 6.0)
        self.assertEqual(detail["code"], "planner_budget_exhausted")
        self.assertEqual(detail["diagnostics"]["planner_repair_calls"], 1)

    def test_multiple_planner_steps_do_not_receive_new_reserves(self):
        clock = _MutableClock()
        limits = replace(
            AgentLimits(), total_deadline_ms=10_000, min_final_answer_budget_ms=4_000
        )
        planner = _AdvancingPlanner(
            clock,
            [1.0, 2.0],
            [
                decision("continue", "search_code", {"query": "authenticate_user"}),
                decision("answer"),
            ],
        )
        with (
            patch.object(agent_core.time, "monotonic", clock),
            patch.object(agent_tools.time, "monotonic", clock),
        ):
            result = run_bounded_agent(
                "authenticate_user",
                self.bundle,
                NoLlm(),
                self.database,
                disabled_embedding_service(),
                planner=planner,
                limits=limits,
            )
        self.assertEqual([item[0]["deadline_monotonic"] for item in planner.states], [6.0, 6.0])
        self.assertTrue(result["evidence"])

    def test_total_budget_smaller_than_reserve_fails_before_planner(self):
        planner = ScriptedPlanner([decision("answer")])
        recorder = SmokeDiagnosticsRecorder()
        result = run_bounded_agent(
            "authenticate_user",
            self.bundle,
            NoLlm(),
            self.database,
            disabled_embedding_service(),
            planner=planner,
            limits=replace(
                AgentLimits(),
                total_deadline_ms=100,
                min_final_answer_budget_ms=200,
            ),
            diagnostics_recorder=recorder,
        )
        detail = build_ask_failure_detail(
            result=result,
            recorder_snapshot=recorder.snapshot(),
            retrieval_version="v1",
            hierarchy_mode="off",
            relation_mode="off",
        )
        self.assertEqual(planner.calls, 0)
        self.assertEqual(detail["code"], "final_answer_not_attempted")
        self.assertEqual(result["answer"], "")

    def test_tool_timeout_and_request_deadline_are_classified_separately(self):
        scenarios = (
            (41.0, 60_000, 5_000, 40_000, "tool_timeout", 503, False),
            (61.0, 60_000, 5_000, 40_000, "deadline_exceeded", 504, True),
            (20.0, 20_000, 0, 20_000, "deadline_exceeded", 504, True),
        )
        for end_time, total_ms, reserve_ms, tool_ms, code, http, deadline_reached in scenarios:
            with self.subTest(code=code, end_time=end_time):
                clock = _MutableClock()
                limits = replace(
                    AgentLimits(),
                    total_deadline_ms=total_ms,
                    min_final_answer_budget_ms=reserve_ms,
                    default_tool_timeout_ms=tool_ms,
                )
                registry = ToolRegistry()

                def handler(_context, _parameters, *, target=end_time):
                    clock.value = target
                    return {"retrieval_mode": "lexical", "evidence": []}, [], False

                registry.register(
                    ToolSpec(
                        name="search_code",
                        version="1",
                        description="test-only bounded handler",
                        input_model=SearchCodeInput,
                        handler=handler,
                        timeout_ms=tool_ms,
                        max_results=1,
                        max_bytes=1024,
                    )
                )
                recorder = SmokeDiagnosticsRecorder()
                with (
                    patch.object(agent_core.time, "monotonic", clock),
                    patch.object(agent_tools.time, "monotonic", clock),
                ):
                    result = run_bounded_agent(
                        "authenticate_user",
                        self.bundle,
                        NoLlm(),
                        self.database,
                        disabled_embedding_service(),
                        planner=ScriptedPlanner(
                            [decision("continue", "search_code", {"query": "x"})]
                        ),
                        limits=limits,
                        registry=registry,
                        diagnostics_recorder=recorder,
                    )
                detail = build_ask_failure_detail(
                    result=result,
                    recorder_snapshot=recorder.snapshot(),
                    retrieval_version="v1",
                    hierarchy_mode="off",
                    relation_mode="off",
                )
                self.assertEqual(detail["code"], code)
                self.assertEqual(ask_failure_http_status(detail), http)
                self.assertEqual(
                    detail["diagnostics"]["request_deadline_reached"],
                    deadline_reached,
                )
                if code == "tool_timeout":
                    self.assertTrue(detail["diagnostics"]["tool_deadline_overrun"])
                    self.assertEqual(detail["diagnostics"]["deadline_overrun_ms"], 0)

    def test_tool_failure_followed_by_request_deadline_reports_deadline(self):
        clock = _MutableClock()
        limits = replace(AgentLimits(), total_deadline_ms=60_000)
        registry = ToolRegistry()

        def handler(_context, _parameters):
            clock.value = 61.0
            raise RuntimeError("synthetic tool failure")

        registry.register(
            ToolSpec(
                name="search_code",
                version="1",
                description="test-only failing handler",
                input_model=SearchCodeInput,
                handler=handler,
                timeout_ms=15_000,
                max_results=1,
                max_bytes=1024,
            )
        )
        recorder = SmokeDiagnosticsRecorder()
        with (
            patch.object(agent_core.time, "monotonic", clock),
            patch.object(agent_tools.time, "monotonic", clock),
        ):
            result = run_bounded_agent(
                "authenticate_user",
                self.bundle,
                NoLlm(),
                self.database,
                disabled_embedding_service(),
                planner=ScriptedPlanner(
                    [decision("continue", "search_code", {"query": "x"})]
                ),
                limits=limits,
                registry=registry,
                diagnostics_recorder=recorder,
            )
        detail = build_ask_failure_detail(
            result=result,
            recorder_snapshot=recorder.snapshot(),
            retrieval_version="v1",
            hierarchy_mode="off",
            relation_mode="off",
        )
        self.assertEqual(detail["code"], "deadline_exceeded")
        self.assertTrue(detail["diagnostics"]["request_deadline_reached"])

    def test_planner_provider_error_after_evidence_keeps_request_local_count(self):
        recorder = SmokeDiagnosticsRecorder()
        planner = _PlannerFailsAfterEvidence()

        with self.assertRaises(ProviderError):
            run_bounded_agent(
                "authenticate_user",
                self.bundle,
                _FinalLlm(),
                self.database,
                disabled_embedding_service(),
                planner=planner,
                diagnostics_recorder=recorder,
                request_id="provider-error-request",
            )

        diagnostics = recorder.snapshot()
        detail = build_ask_failure_detail(
            result={"request_id": "provider-error-request"},
            recorder_snapshot=diagnostics,
            retrieval_version="v1",
            hierarchy_mode="off",
            relation_mode="off",
            provider_error=True,
            retryable=True,
        )
        self.assertEqual(planner.calls, 2)
        self.assertEqual(detail["code"], "provider_error")
        self.assertEqual(detail["diagnostics"]["evidence_count"], 1)
        self.assertGreater(detail["diagnostics"]["evidence_count"], 0)

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
                [
                    decision(
                        "continue",
                        "read_source",
                        {"path": "src/missing.py", "start_line": 1, "end_line": 1},
                    ),
                    decision("answer"),
                ],
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
