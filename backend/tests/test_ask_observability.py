from __future__ import annotations

import json
import logging
import asyncio
from collections import deque
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError

from app.database import Database
from app.services.ask_diagnostics import (
    ask_failure_http_status,
    build_ask_failure_detail,
    build_ask_success_diagnostics,
    format_ask_success_log,
)
from app.services.llm_client import ProviderError
from app.services.smoke_diagnostics import SmokeDiagnosticsRecorder
from tests.m1_helpers import make_project


class _AvailableLlm:
    available = True

    def require_available(self):
        return None


class _LearningService:
    def get_learning_context(self, _project_id):
        return None


class _PlannerRepairFailureLlm:
    available = True
    settings = SimpleNamespace(planner_thinking=None)

    def __init__(self):
        self.planner_calls = 0
        self.final_calls = 0

    def require_available(self):
        return None

    def chat(self, _messages, **kwargs):
        if kwargs.get("purpose") == "planner":
            self.planner_calls += 1
            return "SENSITIVE_INVALID_PLANNER_OUTPUT"
        self.final_calls += 1
        return "MUST_NOT_BE_CALLED"


class _PlannerRepairSuccessLlm(_PlannerRepairFailureLlm):
    def __init__(self):
        super().__init__()
        self.responses = [
            "not-json",
            json.dumps(
                {
                    "status": "continue",
                    "action": "search_code",
                    "arguments": {"query": "authenticate_user"},
                    "decision_summary": "collect evidence",
                }
            ),
            json.dumps(
                {
                    "status": "answer",
                    "action": None,
                    "arguments": {},
                    "decision_summary": "answer from evidence",
                }
            ),
        ]

    def chat(self, _messages, **kwargs):
        if kwargs.get("purpose") == "planner":
            self.planner_calls += 1
            return self.responses.pop(0)
        self.final_calls += 1
        return json.dumps({"parts": [{"text": "Grounded answer", "evidence_aliases": ["A1"]}]})


_METADATA_FIELDS = (
    "agent_trace",
    "budget_usage",
    "relation_summary",
    "learning_context_summary",
    "learning_plan_summary",
    "recommended_next_action",
)

_NON_FINITE_CASES = (
    ("nan", float("nan")),
    ("positive_infinity", float("inf")),
    ("negative_infinity", float("-inf")),
)


def _metadata_attack(field, carrier, value):
    if carrier == "set":
        nested = {"container": {value}}
    elif carrier == "frozenset":
        nested = {"container": frozenset({value})}
    elif carrier == "mapping_key":
        nested = {"container": {value: "fixed"}}
    else:
        raise AssertionError(f"unsupported carrier: {carrier}")
    return [nested] if field == "agent_trace" else nested


def _result(**overrides):
    value = {
        "request_id": "00000000-0000-0000-0000-000000000001",
        "answer": "safe answer",
        "citations": [
            {
                "path": "src/auth.py",
                "summary": "Authentication implementation",
                "snippet": "def authenticate_user(password):",
                "qualified_name": "authenticate_user",
                "start_line": 1,
                "end_line": 2,
            }
        ],
        "evidence_schema_version": 1,
        "evidence": [
            {
                "evidence_id": "E1",
                "project_id": "project-fixture",
                "repository_id": "repository-fixture",
                "repository_url": "local://fixture",
                "repository_revision": "fixture-revision",
                "path": "src/auth.py",
                "language": "python",
                "code_chunk_id": 1,
                "chunk_identity": "fixture:src/auth.py:authenticate_user",
                "chunk_type": "function",
                "symbol_name": "authenticate_user",
                "qualified_name": "authenticate_user",
                "start_line": 1,
                "end_line": 2,
                "content_hash": "fixture-hash",
                "excerpt": "def authenticate_user(password):",
                "retrieval_sources": ["lexical"],
                "lexical_score": 1.0,
                "lexical_rank": 1,
                "semantic_score": None,
                "semantic_rank": None,
                "fusion_score": 1.0,
                "fusion_rank": 1,
                "selection_reason": "fixture",
                "validation_status": "valid",
                "invalid_reason": None,
                "retrieval_strategy_version": "fixture-v1",
            }
        ],
        "grounding_status": "grounded",
        "retrieval_mode": "hybrid",
        "warnings": [],
        "answer_mode": "deterministic",
        "agent_schema_version": 1,
        "agent_mode": "bounded",
        "agent_status": "final_answer_failed",
        "agent_trace": [],
        "budget_usage": {
            "steps_used": 2,
            "tool_calls_used": 1,
            "elapsed_ms": 12,
            "limits": {"total_deadline_ms": 60_000},
        },
        "relation_schema_version": 1,
        "analysis_mode": "retrieval_only",
        "evidence_chains": [],
        "relation_summary": {
            "seed_count": 1,
            "resolved_edge_count": 0,
            "ambiguous_edge_count": 0,
            "unresolved_edge_count": 0,
            "external_edge_count": 0,
            "validated_chain_count": 0,
            "truncated": False,
            "warnings": [],
        },
        "learning_schema_version": 1,
        "learning_mode": "disabled",
        "learning_context_summary": {},
        "learning_plan_summary": {},
        "recommended_next_action": None,
        "learning_warnings": [],
    }
    value.update(overrides)
    return value


def _snapshot(*, reason=None, fallback=None, repairs=0, final_attempted=True):
    value = {
        "request_id": "00000000-0000-0000-0000-000000000001",
        "planner_requests_attempted": 2,
        "planner_repair_attempts": repairs,
        "final_answer_attempted": final_attempted,
        "evidence_count": 1,
        "citation_count": 1,
    }
    if reason:
        value["final_answer_failure_reason_code"] = reason
        if reason.startswith("citation_"):
            value["citation_failure_reason_code"] = reason
        if reason == "relation_validation_failed":
            value["relation_failure_reason_code"] = reason
    if fallback:
        value["fallback_reason_code"] = fallback
    return value


async def _asgi_post(app, path: str, payload: dict):
    request_messages = [
        {
            "type": "http.request",
            "body": json.dumps(payload).encode("utf-8"),
            "more_body": False,
        }
    ]
    response_messages = []

    async def receive():
        if request_messages:
            return request_messages.pop(0)
        return {"type": "http.disconnect"}

    async def send(message):
        response_messages.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": b"",
            "root_path": "",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 8000),
        },
        receive,
        send,
    )
    start = next(item for item in response_messages if item["type"] == "http.response.start")
    body = b"".join(
        item.get("body", b"")
        for item in response_messages
        if item["type"] == "http.response.body"
    )
    return start["status"], json.loads(body)


class AskFailureClassificationTests(unittest.TestCase):
    def _detail(self, result=None, snapshot=None, **kwargs):
        return build_ask_failure_detail(
            result=result or _result(),
            recorder_snapshot=snapshot or _snapshot(reason="citation_missing"),
            retrieval_version="v1",
            hierarchy_mode="off",
            relation_mode="off",
            **kwargs,
        )

    def test_planner_invalid_is_not_collapsed(self):
        detail = self._detail(
            result=_result(agent_mode="deterministic_fallback"),
            snapshot=_snapshot(
                fallback="planner_validation_failed", repairs=0, final_attempted=False
            ),
        )
        self.assertEqual(detail["code"], "planner_invalid")

    def test_planner_repair_failed_is_not_collapsed(self):
        detail = self._detail(
            result=_result(agent_mode="deterministic_fallback"),
            snapshot=_snapshot(
                fallback="planner_validation_failed", repairs=1, final_attempted=False
            ),
        )
        self.assertEqual(detail["code"], "planner_repair_failed")
        self.assertEqual(detail["diagnostics"]["planner_repair_calls"], 1)

    def test_step_budget_and_deadline_are_distinct(self):
        step_budget = self._detail(
            result=_result(
                agent_status="budget_exhausted",
                budget_usage={
                    "steps_used": 5,
                    "tool_calls_used": 1,
                    "elapsed_ms": 25,
                    "limits": {"total_deadline_ms": 60_000},
                },
            ),
            snapshot=_snapshot(reason=None),
        )
        deadline = self._detail(
            result=_result(
                agent_status="budget_exhausted",
                budget_usage={
                    "steps_used": 1,
                    "tool_calls_used": 0,
                    "elapsed_ms": 60_000,
                    "limits": {"total_deadline_ms": 60_000},
                },
            ),
            snapshot={**_snapshot(reason=None), "request_deadline_reached": True},
        )
        self.assertEqual(step_budget["code"], "planner_budget_exhausted")
        self.assertEqual(deadline["code"], "deadline_exceeded")

    def test_evidence_insufficient_is_not_collapsed(self):
        detail = self._detail(
            result=_result(
                evidence=[], citations=[], agent_status="insufficient_evidence"
            ),
            snapshot=_snapshot(reason=None, final_attempted=False),
        )
        self.assertEqual(detail["code"], "evidence_insufficient")

    def test_final_answer_not_attempted_is_not_collapsed(self):
        detail = self._detail(
            snapshot=_snapshot(reason=None, final_attempted=False)
        )
        self.assertEqual(detail["code"], "final_answer_not_attempted")

    def test_all_citation_failures_remain_distinct(self):
        reasons = (
            "citation_missing",
            "citation_format_invalid",
            "citation_unknown",
            "citation_location_missing",
            "citation_path_mismatch",
            "citation_line_range_mismatch",
            "citation_evidence_binding_failed",
        )
        for reason in reasons:
            with self.subTest(reason=reason):
                detail = self._detail(snapshot=_snapshot(reason=reason))
                self.assertEqual(detail["code"], reason)
                self.assertEqual(
                    detail["diagnostics"]["citation_failure_reason_code"], reason
                )

    def test_relation_validation_failed_is_not_collapsed(self):
        detail = self._detail(
            snapshot=_snapshot(reason="relation_validation_failed")
        )
        self.assertEqual(detail["code"], "relation_validation_failed")
        self.assertEqual(
            detail["diagnostics"]["relation_failure_reason_code"],
            "relation_validation_failed",
        )

    def test_provider_error_is_distinct_from_citation_failure(self):
        detail = self._detail(
            snapshot=_snapshot(reason=None), provider_error=True, retryable=True
        )
        self.assertEqual(detail["code"], "provider_error")
        self.assertTrue(detail["retryable"])

    def test_request_deadline_has_priority_over_earlier_validation_failures(self):
        for reason in ("citation_missing", "relation_validation_failed"):
            with self.subTest(reason=reason):
                snapshot = _snapshot(reason=reason)
                snapshot["request_deadline_reached"] = True
                detail = self._detail(snapshot=snapshot)
                self.assertEqual(detail["code"], "deadline_exceeded")
                self.assertEqual(detail["diagnostics"]["failure_stage"], "deadline")
                self.assertEqual(ask_failure_http_status(detail), 504)

    def test_tool_timeout_is_not_a_request_deadline(self):
        snapshot = _snapshot(reason=None, final_attempted=False)
        snapshot["agent_failure_reason_code"] = "tool_timeout"
        snapshot["tool_deadline_overrun"] = True
        snapshot["tool_deadline_overrun_ms"] = 7
        detail = self._detail(
            result=_result(agent_status="failed", answer="", citations=[], evidence=[]),
            snapshot=snapshot,
        )
        self.assertEqual(detail["code"], "tool_timeout")
        self.assertEqual(detail["diagnostics"]["failure_stage"], "tool")
        self.assertFalse(detail["diagnostics"]["request_deadline_reached"])
        self.assertEqual(detail["diagnostics"]["deadline_overrun_ms"], 0)
        self.assertEqual(detail["diagnostics"]["tool_deadline_overrun_ms"], 7)
        self.assertEqual(ask_failure_http_status(detail), 503)

    def test_diagnostics_have_fixed_types_and_bounded_size(self):
        detail = self._detail()
        diagnostics = detail["diagnostics"]
        self.assertLessEqual(len(json.dumps(diagnostics).encode("utf-8")), 4_096)
        self.assertTrue(all(key in diagnostics for key in (
            "request_id", "agent_mode", "agent_status", "answer_mode",
            "failure_stage", "failure_reason_code", "retrieval_version",
            "hierarchy_mode", "relation_mode", "steps_used", "tool_calls_used",
            "planner_logical_calls", "planner_repair_calls",
            "final_answer_attempted", "provider_logical_calls", "evidence_count",
            "citation_count", "citation_failure_reason_code",
            "relation_failure_reason_code", "elapsed_ms",
        )))


class AskResponseFiniteFloatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app import main

        cls.main = main

    def test_formal_response_model_rejects_every_non_finite_float_field(self):
        for field in ("lexical_score", "semantic_score", "fusion_score"):
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(field=field, value=value):
                    candidate = _result(
                        answer_mode="llm_grounded",
                        agent_mode="bounded",
                        agent_status="completed",
                    )
                    candidate["evidence"][0][field] = value
                    with self.assertRaises(ValidationError):
                        self.main.AskResponse.model_validate(candidate)

    def test_formal_response_model_preserves_finite_values_and_explicit_none(self):
        candidate = _result(
            answer_mode="llm_grounded",
            agent_mode="bounded",
            agent_status="completed",
        )
        candidate["evidence"][0].update(
            lexical_score=0.25,
            semantic_score=None,
            fusion_score=1.5,
        )

        validated = self.main.AskResponse.model_validate(candidate)

        self.assertEqual(validated.evidence[0].lexical_score, 0.25)
        self.assertIsNone(validated.evidence[0].semantic_score)
        self.assertEqual(validated.evidence[0].fusion_score, 1.5)

    def test_formal_response_model_rejects_nested_non_finite_metadata(self):
        for field in _METADATA_FIELDS:
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(field=field, value=value):
                    candidate = _result(
                        answer_mode="llm_grounded",
                        agent_mode="bounded",
                        agent_status="completed",
                    )
                    nested = {"level_1": [{"level_2": ("safe", value)}]}
                    candidate[field] = [nested] if field == "agent_trace" else nested
                    with self.assertRaisesRegex(
                        ValidationError,
                        "response metadata contains a non-finite float",
                    ) as raised:
                        self.main.AskResponse.model_validate(candidate)
                    message = str(raised.exception)
                    self.assertNotIn(field, message)
                    self.assertNotIn("safe answer", message)

    def test_formal_response_model_rejects_set_frozenset_and_mapping_key_matrix(self):
        for field in _METADATA_FIELDS:
            for carrier in ("set", "frozenset", "mapping_key"):
                for value_name, value in _NON_FINITE_CASES:
                    with self.subTest(
                        field=field, carrier=carrier, value=value_name
                    ):
                        candidate = _result(
                            answer_mode="llm_grounded",
                            agent_mode="bounded",
                            agent_status="completed",
                        )
                        candidate[field] = _metadata_attack(field, carrier, value)
                        with self.assertRaisesRegex(
                            ValidationError,
                            "response metadata contains a non-finite float",
                        ) as raised:
                            self.main.AskResponse.model_validate(candidate)
                        message = str(raised.exception)
                        self.assertNotIn(field, message)
                        self.assertNotIn("safe answer", message)

    def test_formal_response_model_rejects_non_finite_float_inside_composite_key(self):
        candidate = _result(
            answer_mode="llm_grounded",
            agent_mode="bounded",
            agent_status="completed",
            relation_summary={"container": {("fixed", float("nan")): "fixed"}},
        )

        with self.assertRaisesRegex(
            ValidationError,
            "response metadata contains a non-finite float",
        ):
            self.main.AskResponse.model_validate(candidate)

    def test_formal_response_model_preserves_metadata_strings_finite_values_and_none(self):
        candidate = _result(
            answer_mode="llm_grounded",
            agent_mode="bounded",
            agent_status="completed",
        )
        text_values = ["NaN", "Infinity", "-Infinity"]
        candidate["agent_trace"] = [{"values": text_values}]
        candidate["budget_usage"] = {"nested": (0.25, {"value": -4.5})}
        candidate["relation_summary"] = {"value": 0.0}
        candidate["learning_context_summary"] = {"value": None}
        candidate["learning_plan_summary"] = {"values": text_values}
        candidate["recommended_next_action"] = None

        validated = self.main.AskResponse.model_validate(candidate)

        self.assertEqual(validated.agent_trace[0]["values"], text_values)
        self.assertEqual(validated.budget_usage["nested"], (0.25, {"value": -4.5}))
        self.assertEqual(validated.learning_context_summary["value"], None)
        self.assertIsNone(validated.recommended_next_action)

    def test_formal_response_model_preserves_legal_sets_keys_and_named_strings(self):
        candidate = _result(
            answer_mode="llm_grounded",
            agent_mode="bounded",
            agent_status="completed",
            agent_trace=[{"finite": {1.25}, "text": {"NaN"}}],
            budget_usage={
                "finite": frozenset({-2.5}),
                "text": frozenset({"Infinity"}),
            },
            relation_summary={
                "finite": {3.5: "finite-key"},
                "text": {"-Infinity": "text-key"},
            },
            learning_context_summary={"value": None},
            learning_plan_summary={"values": {"NaN", "Infinity", "-Infinity"}},
            recommended_next_action={"container": {0.0: "value"}},
        )

        first = self.main.AskResponse.model_validate(candidate)
        second = self.main.AskResponse.model_validate_json(first.model_dump_json())

        self.assertEqual(first.agent_trace[0]["finite"], {1.25})
        self.assertEqual(first.budget_usage["finite"], frozenset({-2.5}))
        self.assertEqual(first.learning_context_summary["value"], None)
        self.assertIsNotNone(second)

    def test_explicit_finite_float_string_and_none_contracts_are_unchanged(self):
        for field in ("lexical_score", "semantic_score", "fusion_score"):
            for value in ("NaN", "Infinity", "-Infinity"):
                with self.subTest(field=field, value=value):
                    candidate = _result(
                        answer_mode="llm_grounded",
                        agent_mode="bounded",
                        agent_status="completed",
                    )
                    candidate["evidence"][0][field] = value
                    with self.assertRaises(ValidationError):
                        self.main.AskResponse.model_validate(candidate)

        candidate = _result(
            answer_mode="llm_grounded",
            agent_mode="bounded",
            agent_status="completed",
        )
        candidate["evidence"][0].update(
            lexical_score=None,
            semantic_score=None,
            fusion_score=None,
        )
        with self.assertRaises(ValidationError):
            self.main.AskResponse.model_validate(candidate)

    def test_metadata_walk_terminates_for_cycles_shared_values_and_deep_nesting(self):
        cycle = []
        cycle.append(cycle)
        finite_candidate = _result(
            answer_mode="llm_grounded",
            agent_mode="bounded",
            agent_status="completed",
            budget_usage={"cycle": cycle},
        )
        self.assertIsNotNone(self.main.AskResponse.model_validate(finite_candidate))

        illegal_cycle = []
        illegal_cycle.append(illegal_cycle)
        illegal_cycle.append(float("nan"))
        illegal_candidate = _result(
            answer_mode="llm_grounded",
            agent_mode="bounded",
            agent_status="completed",
            budget_usage={"cycle": illegal_cycle},
        )
        with self.assertRaisesRegex(
            ValidationError,
            "response metadata contains a non-finite float",
        ):
            self.main.AskResponse.model_validate(illegal_candidate)

        dict_cycle = {}
        dict_cycle["self"] = dict_cycle
        dict_candidate = _result(
            answer_mode="llm_grounded",
            agent_mode="bounded",
            agent_status="completed",
            relation_summary={"cycle": dict_cycle},
        )
        self.assertIsNotNone(self.main.AskResponse.model_validate(dict_candidate))

        illegal_dict_cycle = {}
        illegal_dict_cycle["self"] = illegal_dict_cycle
        illegal_dict_cycle["value"] = float("-inf")
        illegal_dict_candidate = _result(
            answer_mode="llm_grounded",
            agent_mode="bounded",
            agent_status="completed",
            relation_summary={"cycle": illegal_dict_cycle},
        )
        with self.assertRaisesRegex(
            ValidationError,
            "response metadata contains a non-finite float",
        ):
            self.main.AskResponse.model_validate(illegal_dict_candidate)

        shared = [0.25]
        shared_candidate = _result(
            answer_mode="llm_grounded",
            agent_mode="bounded",
            agent_status="completed",
            budget_usage={"first": shared, "second": shared},
        )
        self.assertIsNotNone(self.main.AskResponse.model_validate(shared_candidate))
        shared.append(float("inf"))
        with self.assertRaisesRegex(
            ValidationError,
            "response metadata contains a non-finite float",
        ):
            self.main.AskResponse.model_validate(shared_candidate)

        deep = 1.0
        for _index in range(2_000):
            deep = [deep]
        deep_candidate = _result(
            answer_mode="llm_grounded",
            agent_mode="bounded",
            agent_status="completed",
            budget_usage={"deep": deep},
        )
        self.assertIsNotNone(self.main.AskResponse.model_validate(deep_candidate))

    def test_deque_remains_a_safe_first_validation_failure(self):
        candidate = _result(
            answer_mode="llm_grounded",
            agent_mode="bounded",
            agent_status="completed",
            budget_usage={"container": deque([float("nan")])},
        )

        with self.assertRaisesRegex(
            ValidationError,
            "response metadata contains a non-finite float",
        ):
            self.main.AskResponse.model_validate(candidate)

    def test_formal_response_model_rejects_one_shot_iterators_without_consuming(self):
        class TrackingIterator:
            def __init__(self):
                self.next_calls = 0

            def __iter__(self):
                return self

            def __next__(self):
                self.next_calls += 1
                return "PRIVATE-ITERATOR-VALUE"

        for field in _METADATA_FIELDS:
            for iterator_kind in ("generator", "custom_iterator"):
                with self.subTest(field=field, iterator_kind=iterator_kind):
                    consumed = []

                    if iterator_kind == "generator":
                        def values():
                            consumed.append(True)
                            yield "PRIVATE-GENERATOR-VALUE"

                        iterator = values()
                        consumption_count = lambda: len(consumed)
                    else:
                        iterator = TrackingIterator()
                        consumption_count = lambda: iterator.next_calls

                    candidate = _result(
                        answer_mode="llm_grounded",
                        agent_mode="bounded",
                        agent_status="completed",
                    )
                    nested = {"iterator": iterator}
                    candidate[field] = [nested] if field == "agent_trace" else nested

                    with self.assertRaisesRegex(
                        ValidationError,
                        "response metadata contains a one-shot iterator",
                    ) as raised:
                        self.main.AskResponse.model_validate(candidate)

                    self.assertEqual(consumption_count(), 0)
                    self.assertNotIn(field, str(raised.exception))
                    self.assertNotIn("PRIVATE-", str(raised.exception))


class AskRouteSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app import main

        cls.main = main

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = Database(Path(self.directory.name) / "ask-route.sqlite")
        self.project_id, self.bundle = make_project(
            self.database,
            [
                (
                    "src/auth.py",
                    "authenticate_user",
                    "def authenticate_user(password):\n    return verify(password)\n",
                )
            ],
        )
        self.bundle["project"]["source_type"] = "local"

    def _call(self, result, *, source_type="local", record_failure=None):
        self.bundle["project"]["source_type"] = source_type

        def run(*_args, **kwargs):
            if record_failure is not None:
                record_failure(kwargs["diagnostics_recorder"])
            value = dict(result)
            value["request_id"] = kwargs["request_id"]
            return value

        with (
            patch.object(self.main, "db", self.database),
            patch.object(self.main, "llm", _AvailableLlm()),
            patch.object(self.main, "learning_service", _LearningService()),
            patch.object(self.main, "_bundle_or_404", return_value=self.bundle),
            patch.object(self.main, "run_bounded_agent", side_effect=run),
        ):
            return self.main.ask_project(
                self.project_id, self.main.AskRequest(question="safe question")
            )

    def _saved_answers(self):
        bundle = self.database.get_bundle(self.project_id)
        return bundle["chat_answers"] if bundle else []

    def _chat_answer_count(self):
        with self.database.connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM chat_answers").fetchone()
        return int(row[0])

    def _asgi_call(self, result, *, llm=None, run_error=None):
        self.bundle["project"]["source_type"] = "local"

        def run(*_args, **kwargs):
            if run_error is not None:
                raise run_error
            value = dict(result)
            value["request_id"] = kwargs["request_id"]
            return value

        with (
            patch.object(self.main, "db", self.database),
            patch.object(self.main, "llm", llm or _AvailableLlm()),
            patch.object(self.main, "learning_service", _LearningService()),
            patch.object(self.main, "_bundle_or_404", return_value=self.bundle),
            patch.object(self.main, "run_bounded_agent", side_effect=run),
        ):
            return asyncio.run(
                _asgi_post(
                    self.main.app,
                    f"/api/projects/{self.project_id}/ask",
                    {"question": "safe question"},
                )
            )

    def test_failed_answer_is_not_persisted(self):
        with self.assertRaises(HTTPException) as raised:
            self._call(_result())
        self.assertEqual(raised.exception.detail["code"], "final_answer_not_attempted")
        self.assertEqual(self._saved_answers(), [])

    def test_successful_grounded_answer_is_persisted(self):
        result = _result(
            answer_mode="llm_grounded", agent_mode="bounded", agent_status="completed"
        )
        returned = self._call(result)
        self.assertEqual(returned["answer"], result["answer"])
        self.assertEqual(len(self._saved_answers()), 1)

    def test_product_route_repair_failure_is_502_zero_write_and_no_final_call(self):
        llm = _PlannerRepairFailureLlm()
        before = self._chat_answer_count()
        with (
            patch.object(self.main, "db", self.database),
            patch.object(self.main, "llm", llm),
            patch.object(self.main, "learning_service", _LearningService()),
            patch.object(self.main, "_bundle_or_404", return_value=self.bundle),
        ):
            status, body = asyncio.run(
                _asgi_post(
                    self.main.app,
                    f"/api/projects/{self.project_id}/ask",
                    {"question": "safe question"},
                )
            )
        payload = body["detail"]
        self.assertEqual(status, 502)
        self.assertEqual(payload["code"], "planner_repair_failed")
        self.assertEqual(payload["diagnostics"]["planner_logical_calls"], 1)
        self.assertEqual(payload["diagnostics"]["planner_repair_calls"], 1)
        self.assertEqual(payload["diagnostics"]["provider_logical_calls"], 2)
        self.assertFalse(payload["diagnostics"]["final_answer_attempted"])
        self.assertEqual(llm.planner_calls, 2)
        self.assertEqual(llm.final_calls, 0)
        self.assertEqual(self._chat_answer_count(), before)
        serialized = json.dumps(payload, sort_keys=True)
        self.assertNotIn("SENSITIVE_INVALID_PLANNER_OUTPUT", serialized)
        self.assertNotIn("MUST_NOT_BE_CALLED", serialized)

    def test_product_route_repair_success_validates_and_persists_exactly_once(self):
        llm = _PlannerRepairSuccessLlm()
        before = self._chat_answer_count()
        captured_success: list[dict] = []
        with (
            patch.object(self.main, "db", self.database),
            patch.object(self.main, "llm", llm),
            patch.object(self.main, "learning_service", _LearningService()),
            patch.object(self.main, "_bundle_or_404", return_value=self.bundle),
            patch.object(
                self.database,
                "save_chat_answer",
                wraps=self.database.save_chat_answer,
            ) as saved,
            patch.object(
                self.main, "_log_ask_success", side_effect=captured_success.append
            ),
        ):
            status, payload = asyncio.run(
                _asgi_post(
                    self.main.app,
                    f"/api/projects/{self.project_id}/ask",
                    {"question": "authenticate_user"},
                )
            )
        self.assertEqual(status, 200)
        self.assertEqual(payload["grounding_status"], "degraded")
        self.assertEqual(payload["answer_mode"], "llm_grounded")
        self.assertEqual(payload["agent_mode"], "bounded")
        self.assertEqual(llm.planner_calls, 3)
        self.assertEqual(llm.final_calls, 1)
        saved.assert_called_once()
        self.assertEqual(self._chat_answer_count(), before + 1)
        self.assertEqual(len(captured_success), 1)
        diagnostics = captured_success[0]
        self.assertEqual(diagnostics["request_id"], payload["request_id"])
        self.assertEqual(diagnostics["planner_logical_calls"], 2)
        self.assertEqual(diagnostics["planner_repair_calls"], 1)
        self.assertEqual(diagnostics["provider_logical_calls"], 4)
        self.assertTrue(diagnostics["citation_validation_passed"])
        self.assertTrue(diagnostics["relation_validation_passed"])
        self.assertTrue(diagnostics["post_generation_validation_passed"])
        self.assertEqual(len(diagnostics["planner_attempts"]), 3)
        serialized = format_ask_success_log(diagnostics)
        self.assertLessEqual(
            len(json.dumps(diagnostics, sort_keys=True).encode("utf-8")), 4_096
        )
        self.assertNotIn("Grounded answer", serialized)
        self.assertNotIn("authenticate_user", serialized)

    def test_asgi_non_finite_fusion_scores_are_safe_500_and_strictly_zero_write(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                candidate = _result(
                    answer="NON-FINITE-CANDIDATE-MUST-NOT-LEAK",
                    answer_mode="llm_grounded",
                    agent_mode="bounded",
                    agent_status="completed",
                )
                candidate["evidence"][0]["fusion_score"] = value
                with (
                    patch.object(
                        self.database,
                        "save_chat_answer",
                        wraps=self.database.save_chat_answer,
                    ) as saved,
                    self.assertLogs("app.main", logging.WARNING) as captured,
                ):
                    status, body = self._asgi_call(candidate)

                detail = body["detail"]
                self.assertEqual(status, 500)
                self.assertEqual(detail["code"], "response_contract_invalid")
                self.assertEqual(
                    detail["diagnostics"]["failure_reason_code"],
                    "response_contract_invalid",
                )
                self.assertEqual(detail["diagnostics"]["failure_stage"], "response")
                self.assertRegex(detail["diagnostics"]["request_id"], r"^[0-9a-f-]{36}$")
                saved.assert_not_called()
                self.assertEqual(self._chat_answer_count(), 0)
                serialized = json.dumps(body) + "".join(
                    record.getMessage() for record in captured.records
                )
                for marker in (
                    "NaN",
                    "Infinity",
                    "ValidationError",
                    "NON-FINITE-CANDIDATE-MUST-NOT-LEAK",
                    "PRIVATE-NON-FINITE-MARKER",
                ):
                    self.assertNotIn(marker, serialized)

    def test_asgi_nested_non_finite_metadata_is_safe_500_and_strictly_zero_write(self):
        values = (float("nan"), float("inf"), float("-inf"))
        for field in _METADATA_FIELDS:
            for value in values:
                with self.subTest(field=field, value=value):
                    candidate = _result(
                        answer="PRIVATE-METADATA-CANDIDATE-MUST-NOT-LEAK",
                        answer_mode="llm_grounded",
                        agent_mode="bounded",
                        agent_status="completed",
                    )
                    nested = {"level_1": [{"level_2": ("PRIVATE-MARKER", value)}]}
                    candidate[field] = [nested] if field == "agent_trace" else nested
                    with (
                        patch.object(
                            self.database,
                            "save_chat_answer",
                            wraps=self.database.save_chat_answer,
                        ) as saved,
                        self.assertLogs("app.main", logging.WARNING) as captured,
                    ):
                        status, body = self._asgi_call(candidate)

                    detail = body["detail"]
                    event = json.loads(captured.records[0].getMessage())
                    self.assertEqual(status, 500)
                    self.assertEqual(detail["code"], "response_contract_invalid")
                    self.assertEqual(
                        detail["diagnostics"]["failure_reason_code"],
                        "response_contract_invalid",
                    )
                    self.assertEqual(detail["diagnostics"]["failure_stage"], "response")
                    self.assertEqual(
                        detail["diagnostics"]["request_id"], event["request_id"]
                    )
                    saved.assert_not_called()
                    self.assertEqual(self._chat_answer_count(), 0)
                    serialized = json.dumps(body) + "".join(
                        record.getMessage() for record in captured.records
                    )
                    for marker in (
                        "NaN",
                        "Infinity",
                        "ValidationError",
                        "PRIVATE-METADATA-CANDIDATE-MUST-NOT-LEAK",
                        "PRIVATE-MARKER",
                        field,
                    ):
                        self.assertNotIn(marker, serialized)

    def test_asgi_set_frozenset_and_mapping_key_matrix_is_safe_500_and_zero_write(self):
        for field in _METADATA_FIELDS:
            for carrier in ("set", "frozenset", "mapping_key"):
                for value_name, value in _NON_FINITE_CASES:
                    with self.subTest(
                        field=field, carrier=carrier, value=value_name
                    ):
                        candidate = _result(
                            answer="PRIVATE-CONTAINER-CANDIDATE-MUST-NOT-LEAK",
                            answer_mode="llm_grounded",
                            agent_mode="bounded",
                            agent_status="completed",
                        )
                        candidate[field] = _metadata_attack(field, carrier, value)
                        with (
                            patch.object(
                                self.database,
                                "save_chat_answer",
                                wraps=self.database.save_chat_answer,
                            ) as saved,
                            self.assertLogs("app.main", logging.WARNING) as captured,
                        ):
                            status, body = self._asgi_call(candidate)

                        detail = body["detail"]
                        event = json.loads(captured.records[0].getMessage())
                        self.assertEqual(status, 500)
                        self.assertEqual(detail["code"], "response_contract_invalid")
                        self.assertEqual(
                            detail["diagnostics"]["failure_reason_code"],
                            "response_contract_invalid",
                        )
                        self.assertEqual(
                            detail["diagnostics"]["failure_stage"], "response"
                        )
                        self.assertEqual(
                            detail["diagnostics"]["request_id"], event["request_id"]
                        )
                        saved.assert_not_called()
                        self.assertEqual(self._chat_answer_count(), 0)
                        serialized = json.dumps(body) + "".join(
                            record.getMessage() for record in captured.records
                        )
                        for marker in (
                            "NaN",
                            "Infinity",
                            "ValidationError",
                            "PRIVATE-CONTAINER-CANDIDATE-MUST-NOT-LEAK",
                            field,
                        ):
                            self.assertNotIn(marker, serialized)

    def test_asgi_one_shot_iterator_matrix_is_safe_500_and_strictly_zero_write(self):
        class TrackingIterator:
            def __init__(self):
                self.next_calls = 0

            def __iter__(self):
                return self

            def __next__(self):
                self.next_calls += 1
                return "PRIVATE-ITERATOR-VALUE"

        for field in _METADATA_FIELDS:
            for iterator_kind in ("generator", "custom_iterator"):
                with self.subTest(field=field, iterator_kind=iterator_kind):
                    consumed = []

                    if iterator_kind == "generator":
                        def values():
                            consumed.append(True)
                            yield "PRIVATE-GENERATOR-VALUE"

                        iterator = values()
                        consumption_count = lambda: len(consumed)
                    else:
                        iterator = TrackingIterator()
                        consumption_count = lambda: iterator.next_calls

                    candidate = _result(
                        answer="PRIVATE-ITERATOR-CANDIDATE-MUST-NOT-LEAK",
                        answer_mode="llm_grounded",
                        agent_mode="bounded",
                        agent_status="completed",
                    )
                    nested = {"iterator": iterator}
                    candidate[field] = [nested] if field == "agent_trace" else nested

                    with (
                        patch.object(
                            self.database,
                            "save_chat_answer",
                            wraps=self.database.save_chat_answer,
                        ) as saved,
                        self.assertLogs("app.main", logging.WARNING) as captured,
                    ):
                        status, body = self._asgi_call(candidate)

                    detail = body["detail"]
                    event = json.loads(captured.records[0].getMessage())
                    self.assertEqual(status, 500)
                    self.assertEqual(detail["code"], "response_contract_invalid")
                    self.assertEqual(
                        detail["diagnostics"]["failure_reason_code"],
                        "response_contract_invalid",
                    )
                    self.assertEqual(detail["diagnostics"]["failure_stage"], "response")
                    self.assertEqual(
                        detail["diagnostics"]["request_id"], event["request_id"]
                    )
                    self.assertEqual(consumption_count(), 0)
                    saved.assert_not_called()
                    self.assertEqual(self._chat_answer_count(), 0)
                    serialized = json.dumps(body) + "".join(
                        record.getMessage() for record in captured.records
                    )
                    for marker in (
                        "ValidationError",
                        "PRIVATE-ITERATOR-CANDIDATE-MUST-NOT-LEAK",
                        "PRIVATE-GENERATOR-VALUE",
                        "PRIVATE-ITERATOR-VALUE",
                        field,
                    ):
                        self.assertNotIn(marker, serialized)

    def test_json_round_trip_revalidates_the_actual_serialized_result_before_save(self):
        candidate = _result(
            answer="ROUND-TRIP-CANDIDATE-MUST-NOT-LEAK",
            answer_mode="llm_grounded",
            agent_mode="bounded",
            agent_status="completed",
        )
        validated_evidence = self.main.EvidenceResponse.model_validate(
            candidate["evidence"][0]
        )
        candidate["evidence"] = [
            validated_evidence.model_copy(update={"fusion_score": float("nan")})
        ]

        first_pass = self.main.AskResponse.model_validate(candidate)
        serialized = first_pass.model_dump_json()
        self.assertIn('"fusion_score":null', serialized)
        with self.assertRaises(ValidationError):
            self.main.AskResponse.model_validate_json(serialized)

        with patch.object(
            self.database,
            "save_chat_answer",
            wraps=self.database.save_chat_answer,
        ) as saved:
            status, body = self._asgi_call(candidate)

        self.assertEqual(status, 500)
        self.assertEqual(body["detail"]["code"], "response_contract_invalid")
        self.assertEqual(body["detail"]["diagnostics"]["failure_stage"], "response")
        saved.assert_not_called()
        self.assertEqual(self._chat_answer_count(), 0)
        self.assertNotIn("ROUND-TRIP-CANDIDATE-MUST-NOT-LEAK", json.dumps(body))

    def test_asgi_invalid_success_contracts_are_500_and_strictly_zero_write(self):
        valid = _result(
            answer="CANDIDATE-ANSWER-MUST-NOT-LEAK",
            answer_mode="llm_grounded",
            agent_mode="bounded",
            agent_status="completed",
        )
        missing_required = dict(valid)
        missing_required.pop("learning_schema_version")
        invalid_nested = dict(valid)
        invalid_nested["citations"] = [
            {
                "path": "src/auth.py",
                "summary": ["RAW-VALIDATION-DETAIL-MUST-NOT-LEAK"],
                "snippet": "def authenticate_user(password):",
            }
        ]

        for name, candidate in (
            ("missing_required", missing_required),
            ("invalid_nested", invalid_nested),
        ):
            with self.subTest(name=name):
                with (
                    patch.object(
                        self.database,
                        "save_chat_answer",
                        wraps=self.database.save_chat_answer,
                    ) as saved,
                    self.assertLogs("app.main", logging.WARNING) as captured,
                ):
                    status, body = self._asgi_call(candidate)

                detail = body["detail"]
                event = json.loads(captured.records[0].getMessage())
                self.assertEqual(status, 500)
                self.assertEqual(detail["code"], "response_contract_invalid")
                self.assertEqual(
                    detail["diagnostics"]["failure_reason_code"],
                    "response_contract_invalid",
                )
                self.assertEqual(detail["diagnostics"]["failure_stage"], "response")
                self.assertRegex(
                    detail["diagnostics"]["request_id"],
                    r"^[0-9a-f-]{36}$",
                )
                self.assertEqual(event["code"], "response_contract_invalid")
                self.assertEqual(event["failure_stage"], "response")
                saved.assert_not_called()
                self.assertEqual(self._chat_answer_count(), 0)
                serialized = json.dumps(body) + captured.records[0].getMessage()
                self.assertNotIn("CANDIDATE-ANSWER-MUST-NOT-LEAK", serialized)
                self.assertNotIn("RAW-VALIDATION-DETAIL-MUST-NOT-LEAK", serialized)

    def test_asgi_json_serialization_failure_is_safe_and_zero_write(self):
        candidate = _result(
            answer="SERIALIZATION-CANDIDATE-MUST-NOT-LEAK",
            answer_mode="llm_grounded",
            agent_mode="bounded",
            agent_status="completed",
            budget_usage={"unsafe": object()},
        )
        with patch.object(
            self.database,
            "save_chat_answer",
            wraps=self.database.save_chat_answer,
        ) as saved:
            status, body = self._asgi_call(candidate)

        self.assertEqual(status, 500)
        self.assertEqual(body["detail"]["code"], "response_contract_invalid")
        self.assertEqual(body["detail"]["diagnostics"]["failure_stage"], "response")
        saved.assert_not_called()
        self.assertEqual(self._chat_answer_count(), 0)
        self.assertNotIn("SERIALIZATION-CANDIDATE-MUST-NOT-LEAK", json.dumps(body))

    def test_asgi_valid_success_is_validated_and_persisted_exactly_once(self):
        candidate = _result(
            answer_mode="llm_grounded",
            agent_mode="bounded",
            agent_status="completed",
        )
        candidate["evidence"][0].update(
            lexical_score=0.25,
            semantic_score=0.5,
            fusion_score=1.25,
        )
        with patch.object(
            self.database,
            "save_chat_answer",
            wraps=self.database.save_chat_answer,
        ) as saved:
            status, body = self._asgi_call(candidate)

        self.assertEqual(status, 200)
        validated = self.main.AskResponse.model_validate(body)
        saved.assert_called_once()
        saved_args = saved.call_args.args
        self.assertEqual(saved_args[2], validated.answer)
        self.assertEqual(saved_args[3], body["citations"])
        self.assertEqual(self._chat_answer_count(), 1)
        stored = self._saved_answers()[0]
        self.assertEqual(stored["answer"], body["answer"])
        self.assertEqual(stored["citations"], body["citations"])
        self.assertEqual(body["request_id"], validated.request_id)
        self.assertEqual(body["evidence"][0]["lexical_score"], 0.25)
        self.assertEqual(body["evidence"][0]["semantic_score"], 0.5)
        self.assertEqual(body["evidence"][0]["fusion_score"], 1.25)

    def test_provider_failures_use_one_canonical_code_across_http_log_and_diagnostics(self):
        class NotConfiguredLlm:
            def require_available(self):
                raise ProviderError(
                    "provider_not_configured",
                    "UNSAFE-CONFIGURATION-DETAIL",
                    status_code=503,
                )

        cases = (
            (
                "require_available",
                NotConfiguredLlm(),
                None,
                "provider_not_configured",
                503,
            ),
            (
                "unknown_agent_provider_code",
                _AvailableLlm(),
                ProviderError(
                    "provider-secret-response-text",
                    "UNSAFE-PROVIDER-BODY",
                    status_code=502,
                ),
                "provider_error",
                502,
            ),
        )
        for name, llm, run_error, expected_code, expected_status in cases:
            with self.subTest(name=name):
                with self.assertLogs("app.main", logging.WARNING) as captured:
                    status, body = self._asgi_call(
                        _result(), llm=llm, run_error=run_error
                    )
                detail = body["detail"]
                event = json.loads(captured.records[0].getMessage())
                self.assertEqual(status, expected_status)
                self.assertEqual(detail["code"], expected_code)
                self.assertEqual(
                    detail["diagnostics"]["failure_reason_code"], expected_code
                )
                self.assertEqual(detail["diagnostics"]["failure_stage"], "provider")
                self.assertEqual(event["code"], expected_code)
                self.assertEqual(event["failure_reason_code"], expected_code)
                self.assertEqual(event["failure_stage"], "provider")
                self.assertEqual(self._chat_answer_count(), 0)
                serialized = json.dumps(body) + captured.records[0].getMessage()
                self.assertNotIn("UNSAFE-CONFIGURATION-DETAIL", serialized)
                self.assertNotIn("UNSAFE-PROVIDER-BODY", serialized)
                self.assertNotIn("provider-secret-response-text", serialized)

    def test_deadline_overrides_response_contract_failure_before_save(self):
        class Clock:
            value = 10.0

            def __call__(self):
                return self.value

        clock = Clock()
        candidate = _result(
            answer_mode="llm_grounded",
            agent_mode="bounded",
            agent_status="completed",
        )
        candidate["evidence"][0]["fusion_score"] = float("nan")

        def run(*_args, **kwargs):
            clock.value = kwargs["request_budget"].request_deadline_at + 0.001
            value = dict(candidate)
            value["request_id"] = kwargs["request_id"]
            return value

        with (
            patch.object(self.main, "db", self.database),
            patch.object(self.main, "llm", _AvailableLlm()),
            patch.object(self.main, "learning_service", _LearningService()),
            patch.object(self.main, "_bundle_or_404", return_value=self.bundle),
            patch.object(self.main, "run_bounded_agent", side_effect=run),
            patch.object(self.main.time, "monotonic", side_effect=clock),
            patch.object(self.database, "save_chat_answer") as saved,
        ):
            status, body = asyncio.run(
                _asgi_post(
                    self.main.app,
                    f"/api/projects/{self.project_id}/ask",
                    {"question": "safe question"},
                )
            )

        self.assertEqual(status, 504)
        self.assertEqual(body["detail"]["code"], "deadline_exceeded")
        self.assertEqual(
            body["detail"]["diagnostics"]["failure_reason_code"],
            "deadline_exceeded",
        )
        saved.assert_not_called()
        self.assertEqual(self._chat_answer_count(), 0)

    def test_deadline_overrides_illegal_set_metadata_before_save(self):
        class Clock:
            value = 10.0

            def __call__(self):
                return self.value

        clock = Clock()
        candidate = _result(
            answer_mode="llm_grounded",
            agent_mode="bounded",
            agent_status="completed",
            budget_usage={"container": {float("nan")}},
        )

        def run(*_args, **kwargs):
            clock.value = kwargs["request_budget"].request_deadline_at + 0.001
            value = dict(candidate)
            value["request_id"] = kwargs["request_id"]
            return value

        with (
            patch.object(self.main, "db", self.database),
            patch.object(self.main, "llm", _AvailableLlm()),
            patch.object(self.main, "learning_service", _LearningService()),
            patch.object(self.main, "_bundle_or_404", return_value=self.bundle),
            patch.object(self.main, "run_bounded_agent", side_effect=run),
            patch.object(self.main.time, "monotonic", side_effect=clock),
            patch.object(self.database, "save_chat_answer") as saved,
        ):
            status, body = asyncio.run(
                _asgi_post(
                    self.main.app,
                    f"/api/projects/{self.project_id}/ask",
                    {"question": "safe question"},
                )
            )

        self.assertEqual(status, 504)
        self.assertEqual(body["detail"]["code"], "deadline_exceeded")
        saved.assert_not_called()
        self.assertEqual(self._chat_answer_count(), 0)

    def test_legacy_deadline_failure_is_504_and_never_persisted(self):
        result = _result(
            answer="",
            citations=[],
            evidence=[],
            grounding_status="budget_exhausted",
            agent_status="budget_exhausted",
        )
        with self.assertRaises(HTTPException) as raised:
            self._call(
                result,
                source_type="legacy_github",
                record_failure=lambda recorder: recorder.record_agent_failure(
                    "deadline_exceeded"
                ),
            )
        self.assertEqual(raised.exception.status_code, 504)
        self.assertEqual(raised.exception.detail["code"], "deadline_exceeded")
        self.assertEqual(self._saved_answers(), [])

    def test_http_route_legacy_deadline_does_not_fall_into_response_validation_500(self):
        self.bundle["project"]["source_type"] = "legacy_github"

        def run(*_args, **kwargs):
            kwargs["diagnostics_recorder"].record_agent_failure(
                "deadline_exceeded"
            )
            return _result(
                request_id=kwargs["request_id"],
                answer="",
                citations=[],
                evidence=[],
                grounding_status="budget_exhausted",
                agent_status="budget_exhausted",
            )

        with (
            patch.object(self.main, "db", self.database),
            patch.object(self.main, "llm", _AvailableLlm()),
            patch.object(self.main, "learning_service", _LearningService()),
            patch.object(self.main, "_bundle_or_404", return_value=self.bundle),
            patch.object(self.main, "run_bounded_agent", side_effect=run),
        ):
            status, body = asyncio.run(_asgi_post(
                self.main.app,
                f"/api/projects/{self.project_id}/ask",
                {"question": "safe question"},
            ))

        self.assertEqual(status, 504)
        self.assertEqual(body["detail"]["code"], "deadline_exceeded")
        self.assertEqual(self._saved_answers(), [])

    def test_legacy_reserve_failure_is_safe_and_never_persisted(self):
        result = _result(
            answer="",
            citations=[],
            evidence=[],
            grounding_status="budget_exhausted",
            agent_status="budget_exhausted",
        )
        with self.assertRaises(HTTPException) as raised:
            self._call(
                result,
                source_type="legacy_github",
                record_failure=lambda recorder: recorder.record_agent_failure(
                    "final_answer_not_attempted"
                ),
            )
        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(raised.exception.detail["code"], "final_answer_not_attempted")
        self.assertEqual(self._saved_answers(), [])

    def test_legacy_success_is_persisted_exactly_once(self):
        result = _result(
            answer_mode="deterministic",
            agent_mode="deterministic_fallback",
            agent_status="degraded",
            grounding_status="degraded",
        )
        with patch.object(
            self.database,
            "save_chat_answer",
            wraps=self.database.save_chat_answer,
        ) as saved:
            returned = self._call(result, source_type="legacy_github")
        self.assertEqual(returned["answer"], result["answer"])
        saved.assert_called_once()
        self.assertEqual(len(self._saved_answers()), 1)

    def test_product_budget_failure_is_never_persisted(self):
        result = _result(
            answer="",
            citations=[],
            evidence=[],
            grounding_status="budget_exhausted",
            agent_status="budget_exhausted",
        )
        with self.assertRaises(HTTPException):
            self._call(
                result,
                source_type="local",
                record_failure=lambda recorder: recorder.record_agent_failure(
                    "planner_budget_exhausted"
                ),
            )
        self.assertEqual(self._saved_answers(), [])

    def test_route_creates_one_absolute_deadline_and_passes_it_to_agent(self):
        captured = {}

        def run(*_args, **kwargs):
            captured.update(kwargs)
            return _result(
                answer_mode="llm_grounded", agent_mode="bounded", agent_status="completed"
            )

        with (
            patch.object(self.main, "db", self.database),
            patch.object(self.main, "llm", _AvailableLlm()),
            patch.object(self.main, "learning_service", _LearningService()),
            patch.object(self.main, "_bundle_or_404", return_value=self.bundle),
            patch.object(self.main, "run_bounded_agent", side_effect=run),
            patch.object(self.main.time, "monotonic", side_effect=[10.0, 10.0, 10.01]),
        ):
            self.main.ask_project(
                self.project_id, self.main.AskRequest(question="safe question")
            )

        budget = captured["request_budget"]
        self.assertEqual(
            budget.request_deadline_at,
            10.0 + self.main.agent_limits.total_deadline_ms / 1000,
        )
        self.assertEqual(budget.request_started_at, 10.0)
        self.assertEqual(
            budget.work_cutoff_at,
            budget.request_deadline_at
            - self.main.agent_limits.min_final_answer_budget_ms / 1000,
        )

    def test_last_deadline_gate_runs_immediately_before_save(self):
        class Clock:
            value = 10.0

            def __call__(self):
                return self.value

        clock = Clock()
        result = _result(
            answer_mode="deterministic",
            agent_mode="deterministic_fallback",
            agent_status="degraded",
            grounding_status="degraded",
        )

        def run(*_args, **kwargs):
            clock.value = kwargs["request_budget"].request_deadline_at + 0.010
            value = dict(result)
            value["request_id"] = kwargs["request_id"]
            return value

        self.bundle["project"]["source_type"] = "legacy_github"
        with (
            patch.object(self.main, "db", self.database),
            patch.object(self.main, "llm", _AvailableLlm()),
            patch.object(self.main, "learning_service", _LearningService()),
            patch.object(self.main, "_bundle_or_404", return_value=self.bundle),
            patch.object(self.main, "run_bounded_agent", side_effect=run),
            patch.object(self.main.time, "monotonic", side_effect=clock),
            patch.object(self.database, "save_chat_answer") as saved,
        ):
            with self.assertRaises(HTTPException) as raised:
                self.main.ask_project(
                    self.project_id, self.main.AskRequest(question="safe question")
                )
        self.assertEqual(raised.exception.status_code, 504)
        self.assertEqual(raised.exception.detail["code"], "deadline_exceeded")
        saved.assert_not_called()

    def test_database_exception_never_returns_grounded_success(self):
        result = _result(
            answer_mode="deterministic",
            agent_mode="deterministic_fallback",
            agent_status="degraded",
            grounding_status="degraded",
        )
        with patch.object(
            self.database,
            "save_chat_answer",
            side_effect=RuntimeError("unsafe database detail"),
        ):
            with self.assertRaises(HTTPException) as raised:
                self._call(result, source_type="legacy_github")
        self.assertEqual(raised.exception.status_code, 500)
        self.assertEqual(raised.exception.detail["code"], "persistence_failed")
        self.assertNotIn("unsafe database detail", json.dumps(raised.exception.detail))

    def test_concurrent_route_failures_keep_request_local_diagnostics(self):
        barrier = Barrier(2)

        def run(question, *_args, **kwargs):
            recorder = kwargs["diagnostics_recorder"]
            recorder.record_planner_request(repair=False)
            if question == "first":
                recorder.record_tool_attempt("search_code")
                recorder.record_agent_failure("tool_timeout")
                expected = "tool_timeout"
            else:
                recorder.record_planner_request(repair=True)
                recorder.record_agent_failure("planner_budget_exhausted")
                expected = "planner_budget_exhausted"
            barrier.wait(timeout=5)
            value = _result(
                request_id=kwargs["request_id"],
                answer="",
                citations=[],
                evidence=[],
                grounding_status="budget_exhausted",
                agent_status="failed",
            )
            value["expected"] = expected
            return value

        self.bundle["project"]["source_type"] = "legacy_github"
        with (
            patch.object(self.main, "db", self.database),
            patch.object(self.main, "llm", _AvailableLlm()),
            patch.object(self.main, "learning_service", _LearningService()),
            patch.object(self.main, "_bundle_or_404", return_value=self.bundle),
            patch.object(self.main, "run_bounded_agent", side_effect=run),
        ):
            def call(question):
                try:
                    self.main.ask_project(
                        self.project_id, self.main.AskRequest(question=question)
                    )
                except HTTPException as exc:
                    return exc.detail
                self.fail("failure response was expected")

            with ThreadPoolExecutor(max_workers=2) as executor:
                first, second = list(executor.map(call, ("first", "second")))

        by_code = {item["code"]: item for item in (first, second)}
        self.assertEqual(set(by_code), {"tool_timeout", "planner_budget_exhausted"})
        self.assertNotEqual(
            first["diagnostics"]["request_id"], second["diagnostics"]["request_id"]
        )
        self.assertEqual(by_code["tool_timeout"]["diagnostics"]["tool_calls_used"], 1)
        self.assertEqual(
            by_code["planner_budget_exhausted"]["diagnostics"]["planner_repair_calls"],
            1,
        )
        self.assertEqual(self._saved_answers(), [])

    def test_http_and_log_share_safe_failure_code_without_bodies_or_secrets(self):
        unsafe = _result(
            answer="FULL-MODEL-OUTPUT",
            prompt="FULL-PROMPT",
            source="FULL-SOURCE",
            authorization="Bearer TOP-SECRET",
        )
        with self.assertLogs("app.main", logging.WARNING) as captured:
            with self.assertRaises(HTTPException) as raised:
                self._call(unsafe)
        detail = raised.exception.detail
        logged = captured.records[0].getMessage()
        log_event = json.loads(logged)
        self.assertEqual(detail["code"], detail["diagnostics"]["failure_reason_code"])
        self.assertEqual(captured.records[0].ask_failure["code"], detail["code"])
        self.assertEqual(log_event["event"], "ask_failed")
        self.assertEqual(log_event["code"], detail["code"])
        self.assertEqual(
            log_event["failure_reason_code"],
            detail["diagnostics"]["failure_reason_code"],
        )
        self.assertEqual(log_event["failure_stage"], detail["diagnostics"]["failure_stage"])
        self.assertEqual(log_event["request_id"], detail["diagnostics"]["request_id"])
        serialized = logged + json.dumps(detail)
        for secret in (
            "FULL-MODEL-OUTPUT",
            "FULL-PROMPT",
            "FULL-SOURCE",
            "TOP-SECRET",
            "Authorization",
        ):
            self.assertNotIn(secret, serialized)

    def test_later_deadline_wins_over_citation_and_relation_in_route_log_and_http(self):
        for earlier_reason in ("citation_missing", "relation_validation_failed"):
            with self.subTest(earlier_reason=earlier_reason):
                def record(recorder, reason=earlier_reason):
                    recorder.record_final_answer_failure(reason)
                    recorder.record_agent_failure("deadline_exceeded")
                    recorder.record_request_deadline_reached(True)

                with self.assertLogs("app.main", logging.WARNING) as captured:
                    with self.assertRaises(HTTPException) as raised:
                        self._call(
                            _result(
                                answer="",
                                citations=[],
                                evidence=[],
                                grounding_status="budget_exhausted",
                                agent_status="budget_exhausted",
                            ),
                            source_type="legacy_github",
                            record_failure=record,
                        )
                detail = raised.exception.detail
                event = json.loads(captured.records[0].getMessage())
                self.assertEqual(raised.exception.status_code, 504)
                self.assertEqual(detail["code"], "deadline_exceeded")
                self.assertEqual(
                    detail["diagnostics"]["failure_reason_code"], "deadline_exceeded"
                )
                self.assertEqual(event["code"], "deadline_exceeded")
                self.assertEqual(event["failure_stage"], "deadline")
                historical_key = (
                    "citation_failure_reason_code"
                    if earlier_reason.startswith("citation_")
                    else "relation_failure_reason_code"
                )
                self.assertEqual(detail["diagnostics"][historical_key], earlier_reason)


class RecorderSafetyTests(unittest.TestCase):
    def test_success_projection_rejects_free_text_and_stays_bounded(self):
        recorder = SmokeDiagnosticsRecorder()
        recorder.begin_request(deadline_budget_ms=60_000, remaining_ms=60_000)
        recorder.begin_agent(["search_code"], request_id="request-success")
        recorder.record_planner_request(repair=False)
        recorder.record_planner_attempt(
            {
                "stage": "semantic",
                "stable_code": "valid",
                "field_path": [],
                "output_chars": 42,
                "output_sha256": "b" * 64,
                "finish_reason_present": True,
                "finish_reason_value": "stop",
                "content_present": True,
                "reasoning_content_present": True,
                "markdown_fence_detected": False,
                "repair_attempt": False,
                "raw": "MUST_NOT_SURVIVE",
            },
            duration_ms=9,
        )
        recorder.record_final_answer_attempt()
        recorder.mark_citation_validation_completed(passed=True)
        recorder.mark_relation_validation_completed(passed=True)
        recorder.mark_post_generation_validation_completed(passed=True)
        diagnostics = build_ask_success_diagnostics(
            result=_result(
                request_id="request-success",
                answer="SENSITIVE_ANSWER",
                answer_mode="llm_grounded",
                agent_mode="bounded",
                agent_status="completed",
            ),
            recorder_snapshot=recorder.snapshot(),
            retrieval_version="v1",
            hierarchy_mode="off",
            relation_mode="off",
        )
        serialized = format_ask_success_log(diagnostics)
        self.assertLessEqual(len(json.dumps(diagnostics).encode("utf-8")), 4_096)
        self.assertNotIn("MUST_NOT_SURVIVE", serialized)
        self.assertNotIn("SENSITIVE_ANSWER", serialized)
        self.assertNotIn('"reasoning_content":', serialized)

    def test_request_id_and_counts_are_safe_without_recording_model_output(self):
        recorder = SmokeDiagnosticsRecorder()
        recorder.begin_agent(["search_code"], request_id="request-1")
        recorder.record_planner_request(repair=False)
        recorder.record_final_answer_attempt()
        recorder.record_grounded_answer_candidate(received=True, citation_count=1)
        snapshot = recorder.snapshot()
        self.assertEqual(snapshot["request_id"], "request-1")
        self.assertNotIn("answer", snapshot)
        self.assertNotIn("prompt", snapshot)

    def test_stage_timing_and_attempt_arrays_stay_under_four_kib(self):
        recorder = SmokeDiagnosticsRecorder()
        recorder.begin_request(deadline_budget_ms=60_000, remaining_ms=60_000)
        recorder.begin_agent(["search_code"], request_id="request-1")
        for index in range(20):
            call_id = recorder.start_provider_call("planner")
            recorder.record_provider_attempt(
                call_id,
                outcome="timeout",
                duration_ms=index + 1,
                timeout_ms=100,
            )
        recorder.record_stage_duration("planner", 12)
        recorder.record_stage_duration("tool", 34)
        recorder.record_stage_duration("finalization", 56)
        recorder.record_deadline_state(remaining_ms=0, overrun_ms=7)
        snapshot = recorder.snapshot()

        self.assertLessEqual(len(json.dumps(snapshot).encode("utf-8")), 4_096)
        self.assertEqual(snapshot["provider_http_attempt_count"], 20)
        self.assertLessEqual(len(snapshot["provider_attempt_outcomes"]), 8)
        self.assertEqual(snapshot["deadline_overrun_ms"], 7)

    def test_planner_attempt_overflow_is_bounded_without_secondary_failure(self):
        recorder = SmokeDiagnosticsRecorder()
        recorder.begin_agent(["search_code"], request_id="request-planner-bounds")
        for index in range(30):
            recorder.record_planner_attempt(
                {
                    "stage": "schema",
                    "stable_code": "schema_invalid_type",
                    "field_path": ["arguments", index],
                    "output_chars": 1_000_000,
                    "output_sha256": "a" * 64,
                    "finish_reason_present": True,
                    "finish_reason_value": "stop",
                    "content_present": True,
                    "reasoning_content_present": True,
                    "markdown_fence_detected": True,
                    "repair_attempt": index > 0,
                    "raw": "MUST_NOT_SURVIVE",
                },
                duration_ms=index,
            )
        snapshot = recorder.snapshot()
        detail = build_ask_failure_detail(
            result=_result(request_id="request-planner-bounds"),
            recorder_snapshot=snapshot,
            retrieval_version="v1",
            hierarchy_mode="off",
            relation_mode="off",
            terminal_reason="planner_repair_failed",
        )
        self.assertLessEqual(len(json.dumps(snapshot).encode("utf-8")), 4_096)
        self.assertLessEqual(
            len(json.dumps(detail["diagnostics"]).encode("utf-8")), 4_096
        )
        self.assertTrue(snapshot["diagnostics_truncated"])
        self.assertNotIn("MUST_NOT_SURVIVE", json.dumps(detail))

    def test_provider_logical_calls_survive_bounded_detail_truncation(self):
        recorder = SmokeDiagnosticsRecorder()
        recorder.begin_agent(["search_code"], request_id="request-17")
        for _step in range(8):
            recorder.record_planner_request(repair=False)
            recorder.start_provider_call("planner")
            recorder.record_planner_request(repair=True)
            recorder.start_provider_call("planner")
        recorder.record_final_answer_attempt()
        recorder.start_provider_call("final_answer")

        snapshot = recorder.snapshot()
        detail = build_ask_failure_detail(
            result=_result(request_id="request-17"),
            recorder_snapshot=snapshot,
            retrieval_version="v1",
            hierarchy_mode="off",
            relation_mode="off",
        )
        self.assertEqual(len(snapshot["provider_calls"]), 16)
        self.assertTrue(snapshot["diagnostics_truncated"])
        self.assertEqual(snapshot["provider_logical_calls"], 17)
        self.assertEqual(detail["diagnostics"]["provider_logical_calls"], 17)
        self.assertEqual(detail["diagnostics"]["planner_logical_calls"], 8)
        self.assertEqual(detail["diagnostics"]["planner_repair_calls"], 8)

    def test_independent_recorders_do_not_share_provider_or_evidence_counts(self):
        first = SmokeDiagnosticsRecorder()
        second = SmokeDiagnosticsRecorder()
        first.begin_agent([], request_id="request-a")
        second.begin_agent([], request_id="request-b")
        for _index in range(3):
            first.start_provider_call("planner")
        second.start_provider_call("final_answer")
        first.record_evidence_count(2)
        second.record_evidence_count(1)

        first_snapshot = first.snapshot()
        second_snapshot = second.snapshot()
        self.assertEqual(first_snapshot["provider_logical_calls"], 3)
        self.assertEqual(second_snapshot["provider_logical_calls"], 1)
        self.assertEqual(first_snapshot["evidence_count"], 2)
        self.assertEqual(second_snapshot["evidence_count"], 1)


if __name__ == "__main__":
    unittest.main()
