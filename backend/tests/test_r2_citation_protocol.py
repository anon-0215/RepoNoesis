from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from app import main
from app.config import LLMSettings
from app.database import Database
from app.services import ask_diagnostics, smoke_diagnostics
from app.services.agent_contracts import AgentLimits
from app.services.citation_protocol import (
    MAX_ALIASES_PER_PART,
    MAX_FINAL_ANSWER_OUTPUT_CHARS,
    MAX_FINAL_ANSWER_PARTS,
    build_canonical_citation_descriptors,
    build_final_answer_json_schema,
    render_structured_final_answer,
)
from app.services.evidence import CitationValidator, Evidence, EvidenceBuilder
from app.services.llm_client import LLMClient
from app.services.qa_agent import (
    _validate_grounded_answer_references,
    answer_from_evidence,
)
from app.services.relation_graph import RelationValidator
from app.services.smoke_diagnostics import SmokeDiagnosticsRecorder
from tests.m1_helpers import disabled_embedding_service, make_chunk, make_project
from tests.test_f12_agent_orchestration import _LearningService, _post


def _structured(text: str = "Validated fact", aliases: list[str] | None = None) -> str:
    return json.dumps(
        {
            "parts": [
                {
                    "text": text,
                    "evidence_aliases": aliases or ["A1"],
                }
            ]
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


class _Response:
    status = 200

    def __init__(self, payload: dict) -> None:
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


class _RouteProvider:
    available = True

    def __init__(
        self,
        final_output: str | list[str | Exception],
        callback=None,
    ) -> None:
        self.final_output = final_output
        self.planner_calls = 0
        self.final_calls = 0
        self.final_messages = None
        self.final_message_history: list[list[dict[str, str]]] = []
        self.callback = callback

    def require_available(self) -> None:
        return None

    def chat(self, _messages, **kwargs):
        if kwargs.get("purpose") == "planner":
            self.planner_calls += 1
            return json.dumps(
                {
                    "status": "continue",
                    "action": "search_code",
                    "arguments": {"query": "authenticate_user"},
                    "decision_summary": "collect one source Evidence",
                }
            )
        self.final_calls += 1
        self.final_messages = _messages
        self.final_message_history.append(_messages)
        if self.callback is not None:
            self.callback(self.final_calls)
        if isinstance(self.final_output, list):
            output = self.final_output[self.final_calls - 1]
            if isinstance(output, Exception):
                raise output
            return output
        return self.final_output


class _SafeLogCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


class _RaisingSuccessLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        if '"event":"ask_succeeded"' in record.getMessage():
            raise RuntimeError("PRIVATE-SUCCESS-LOGGER-FAILURE")


class CitationProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = Database(Path(self.directory.name) / "citation-r2.sqlite")
        self.project_id, self.bundle = make_project(
            self.database,
            [
                ("src/auth.py", "authenticate_user", "def authenticate_user():\n    return True\n"),
                ("src/space name.py", "space_name", "def space_name():\n    return True\n"),
                ("源码/模块.py", "unicode_name", "def unicode_name():\n    return True\n"),
                ("src/(group)[x]#tag.py", "punctuation", "def punctuation():\n    return True\n"),
                ("src/colon:12-34.py", "colon_name", "def colon_name():\n    return True\n"),
            ],
        )
        chunks = self.database.get_code_chunks(self.project_id)
        self.evidence = EvidenceBuilder().build_from_code_chunks(
            chunks,
            self.bundle["project"],
        )
        self.valid, warnings = CitationValidator(self.database).validate_all(
            self.evidence
        )
        self.assertEqual(warnings, [])
        self.descriptors = build_canonical_citation_descriptors(self.valid)

    def test_renderer_covers_paths_ranges_multiple_aliases_and_real_validators(self):
        for descriptor, evidence in zip(self.descriptors, self.valid):
            with self.subTest(path=evidence.path):
                result = render_structured_final_answer(
                    _structured(aliases=[descriptor.alias]),
                    self.descriptors,
                )
                self.assertTrue(result.valid)
                self.assertIn(descriptor.canonical_token(), result.answer)
                self.assertEqual(
                    _validate_grounded_answer_references(result.answer, self.valid),
                    (True, None, 1),
                )
                revalidated, warnings = CitationValidator(self.database).validate_all(
                    self.valid
                )
                self.assertEqual(len(revalidated), len(self.valid))
                self.assertEqual(warnings, [])

        multiple = render_structured_final_answer(
            _structured(aliases=["A1", "A2"]), self.descriptors
        )
        self.assertTrue(multiple.valid)
        self.assertEqual(
            _validate_grounded_answer_references(multiple.answer, self.valid),
            (True, None, 2),
        )

    def test_grounded_reference_validator_rejects_unknown_missing_and_malformed_markers(self):
        location = (
            f"{self.valid[0].path}:"
            f"{self.valid[0].start_line}-{self.valid[0].end_line}"
        )
        cases = (
            (f"Fact [E999] {location}", (False, "citation_unknown", 1)),
            (f"Fact without a marker {location}", (False, "citation_missing", 0)),
            (f"Fact E1 {location}", (False, "citation_format_invalid", 0)),
            (f"Fact [E1]{location}", (False, "citation_location_missing", 1)),
        )
        for answer, expected in cases:
            with self.subTest(answer=answer):
                self.assertEqual(
                    _validate_grounded_answer_references(answer, self.valid),
                    expected,
                )

    def test_windows_separator_is_deterministic_and_rejected_by_evidence_validator(self):
        windows = Evidence(**self.valid[0].to_dict())
        windows.path = "src\\auth.py"
        descriptor = build_canonical_citation_descriptors([windows])[0]
        rendered = render_structured_final_answer(_structured(), [descriptor])
        self.assertEqual(rendered.answer, "Validated fact [E1] src\\auth.py:1-2")
        valid, warnings = CitationValidator(self.database).validate_all([windows])
        self.assertEqual(valid, [])
        self.assertTrue(warnings)

    def test_aliases_are_request_local_unique_and_bound_to_complete_identity(self):
        for index, descriptor in enumerate(self.descriptors, start=1):
            evidence = self.valid[index - 1]
            self.assertEqual(descriptor.alias, f"A{index}")
            self.assertEqual(descriptor.evidence_id, evidence.evidence_id)
            self.assertEqual(descriptor.project_id, evidence.project_id)
            self.assertEqual(descriptor.repository_revision, evidence.repository_revision)
            self.assertEqual(descriptor.path, evidence.path)
            self.assertEqual(descriptor.content_hash, evidence.content_hash)
            self.assertEqual(descriptor.chunk_identity, evidence.chunk_identity)

    def test_prompt_schema_and_parser_share_the_exact_allowed_aliases(self):
        schema = build_final_answer_json_schema(self.descriptors)
        aliases = [item.alias for item in self.descriptors]
        part = schema["$defs"]["FinalAnswerPart"]
        self.assertEqual(
            part["properties"]["evidence_aliases"]["items"]["enum"], aliases
        )
        self.assertFalse(schema["additionalProperties"])
        self.assertFalse(part["additionalProperties"])

    def test_structured_protocol_accepts_single_multi_and_per_part_aliases(self):
        cases = (
            _structured(),
            _structured(aliases=["A1", "A2"]),
            json.dumps(
                {
                    "parts": [
                        {"text": "First fact", "evidence_aliases": ["A1"]},
                        {"text": "Second fact", "evidence_aliases": ["A2"]},
                        {"text": "Same source again", "evidence_aliases": ["A1"]},
                    ]
                }
            ),
        )
        for raw in cases:
            with self.subTest(raw=raw[:40]):
                result = render_structured_final_answer(raw, self.descriptors)
                self.assertTrue(result.valid)
                self.assertTrue(
                    _validate_grounded_answer_references(
                        result.answer, self.valid
                    )[0]
                )

    def test_structured_protocol_rejects_every_untrusted_shape(self):
        valid_part = {"text": "Fact", "evidence_aliases": ["A1"]}
        cases = (
            ("not json", "final_answer_invalid_json"),
            ("```json\n" + _structured() + "\n```", "final_answer_invalid_json"),
            ("prefix " + _structured(), "final_answer_invalid_json"),
            (json.dumps([]), "final_answer_schema_invalid"),
            (json.dumps({"wrong": []}), "final_answer_schema_invalid"),
            (json.dumps({"parts": [{"text": "Fact"}]}), "citation_alias_missing"),
            (json.dumps({"parts": [{"text": "Fact", "evidence_aliases": []}]}), "citation_alias_missing"),
            (json.dumps({"parts": [{"text": "Fact", "evidence_aliases": "A1"}]}), "citation_alias_invalid_type"),
            (json.dumps({"parts": [{"text": "Fact", "evidence_aliases": [1]}]}), "citation_alias_invalid_type"),
            (json.dumps({"parts": [{"text": "Fact", "evidence_aliases": ["A999"]}]}), "citation_alias_unknown"),
            (json.dumps({"parts": [{**valid_part, "path": "src/auth.py"}]}), "final_answer_schema_invalid"),
            (json.dumps({"parts": [{**valid_part, "line": 1}]}), "final_answer_schema_invalid"),
            (json.dumps({"parts": [{**valid_part, "revision": "fake"}]}), "final_answer_schema_invalid"),
            (json.dumps({"parts": [{"text": "Fact [E1] src/auth.py:1-2", "evidence_aliases": ["A1"]}]}), "model_supplied_location_forbidden"),
            (json.dumps({"parts": [{"text": "Fact", "evidence_aliases": ["A1", "A1"]}]}), "final_answer_schema_invalid"),
            (json.dumps({"parts": [{"text": "Fact", "evidence_aliases": ["A1"] * (MAX_ALIASES_PER_PART + 1)}]}), "citation_alias_limit_exceeded"),
            (json.dumps({"parts": [valid_part] * (MAX_FINAL_ANSWER_PARTS + 1)}), "final_answer_schema_invalid"),
            ("x" * (MAX_FINAL_ANSWER_OUTPUT_CHARS + 1), "final_answer_schema_invalid"),
        )
        for raw, expected in cases:
            with self.subTest(expected=expected, raw=raw[:30]):
                failure = render_structured_final_answer(
                    raw, self.descriptors
                ).failure
                self.assertIsNotNone(failure)
                self.assertEqual(failure.stable_code, expected)
                serialized = json.dumps(failure.to_safe_dict(), sort_keys=True)
                self.assertNotIn("src/auth.py:1-2", serialized)
                self.assertNotIn("Fact [E1]", serialized)

    def test_reserved_server_marker_has_one_safe_violation_kind(self):
        failure = render_structured_final_answer(
            _structured("Fact [E1]", ["A1"]), self.descriptors
        ).failure
        self.assertIsNotNone(failure)
        self.assertEqual(failure.stable_code, "model_supplied_location_forbidden")
        self.assertEqual(failure.field_path, ("parts", 0, "text"))
        self.assertEqual(failure.violation_kind, "evidence_marker")
        self.assertNotIn("Fact [E1]", json.dumps(failure.to_safe_dict()))

    def test_r4_ordinary_location_and_identity_text_is_allowed_without_repair(self):
        hash64 = "a" * 64
        allowed = (
            "main.py:10",
            "main.py : 10",
            "src/main.py#L10",
            "Jenkinsfile:10",
            "file:///repo/main.py:10",
            "vscode://file/C:/repo/main.py:10",
            "api.example.com:443",
            "revision: 0xdeadbeef",
            "content_hash: 0x" + hash64,
            "deadbeef",
            "0xdeadbeef",
            "#deadbeef",
            "src/auth.py:1-2",
            self.descriptors[0].repository_revision,
            self.descriptors[0].content_hash,
            self.descriptors[0].chunk_identity,
        )
        for text in allowed:
            with self.subTest(text=text):
                provider = _RouteProvider(_structured(text))
                result = answer_from_evidence(
                    "explain",
                    [self.valid[0]],
                    provider,
                    self.database,
                    retrieval_mode="hybrid",
                )
                self.assertEqual(provider.final_calls, 1)
                self.assertEqual(result["answer_mode"], "llm_grounded")
                self.assertIn(text, result["answer"])
                self.assertIn(self.descriptors[0].canonical_token(), result["answer"])
                self.assertEqual(
                    result["citations"][0]["path"], self.valid[0].path
                )

    def test_only_the_downstream_server_marker_namespace_is_reserved(self):
        markers = ("[E0]", "[E1]", "[E2]", "[E123]")
        for marker in markers:
            with self.subTest(marker=marker):
                failure = render_structured_final_answer(
                    _structured(f"Fact {marker}"), self.descriptors
                ).failure
                self.assertIsNotNone(failure)
                self.assertEqual(failure.violation_kind, "evidence_marker")
        for text in ("E1", "(E-1)", "【Evidence 1】", "{evidence_id:E1}", "Evidence ID E1"):
            with self.subTest(text=text):
                self.assertTrue(
                    render_structured_final_answer(
                        _structured(text), self.descriptors
                    ).valid
                )

    def test_cross_project_revision_hash_and_location_mismatches_still_fail(self):
        mutations = (
            ("project_id", "other-project"),
            ("repository_revision", "other-revision"),
            ("path", "src/missing.py"),
            ("start_line", 2),
            ("content_hash", "0" * 64),
            ("chunk_identity", "invalid-identity"),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                candidate = Evidence(**self.valid[0].to_dict())
                setattr(candidate, field, value)
                valid, warnings = CitationValidator(self.database).validate_all(
                    [candidate]
                )
                self.assertEqual(valid, [])
                self.assertTrue(warnings)

    def test_deepseek_shape_uses_content_not_reasoning_and_renders_canonical(self):
        raw = _structured("Provider-grounded fact", ["A1"])

        def opener(_request, *, timeout):
            self.assertGreater(timeout, 0)
            return _Response(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": raw,
                                "reasoning_content": "PRIVATE_REASONING_MUST_NOT_BE_USED",
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }
            )

        client = LLMClient(
            LLMSettings(
                provider="openai_compatible",
                base_url="https://provider.invalid/v1",
                api_key="test-placeholder",
                model="configured-model",
                max_retries=0,
            ),
            opener=opener,
        )
        result = answer_from_evidence(
            "explain",
            [self.valid[0]],
            client,
            self.database,
            retrieval_mode="hybrid",
        )
        self.assertEqual(result["answer_mode"], "llm_grounded")
        descriptor = build_canonical_citation_descriptors([self.valid[0]])[0]
        self.assertIn(descriptor.canonical_token(), result["answer"])
        self.assertNotIn("PRIVATE_REASONING", result["answer"])

    def test_repair_complete_messages_use_only_the_explicit_allowlist(self):
        rejected = _structured("PRIVATE_CANDIDATE [E1]")
        repaired = _structured("Repaired relation-aware fact")
        private_reasoning = "PRIVATE_REASONING_MUST_NEVER_REENTER_REPAIR"
        requests: list[dict] = []
        responses = [
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": rejected,
                            "reasoning_content": private_reasoning,
                        },
                    }
                ]
            },
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": repaired},
                    }
                ]
            },
        ]

        def opener(request, *, timeout):
            self.assertGreater(timeout, 0)
            requests.append(json.loads(request.data.decode("utf-8")))
            return _Response(responses[len(requests) - 1])

        recorder = SmokeDiagnosticsRecorder()
        client = LLMClient(
            LLMSettings(
                provider="openai_compatible",
                base_url="https://provider.invalid/v1",
                api_key="test-placeholder",
                model="configured-model",
                max_retries=0,
            ),
            opener=opener,
        )
        relation_context = [
            {
                "evidence_ids": [self.valid[0].evidence_id],
                "relation_type": "calls",
                "source_symbol": "authenticate_user",
                "target_symbol": "verify",
                "resolution_status": "resolved",
                "resolution_rule": "exact",
            }
        ]
        result = answer_from_evidence(
            "explain the relation",
            [self.valid[0]],
            client,
            self.database,
            retrieval_mode="hybrid",
            relation_context=relation_context,
            diagnostics_recorder=recorder,
        )

        self.assertEqual(result["answer_mode"], "llm_grounded")
        self.assertEqual(len(requests), 2)
        repair_messages = requests[1]["messages"]
        self.assertEqual([item["role"] for item in repair_messages], ["system", "user"])
        payload = json.loads(repair_messages[1]["content"])
        self.assertEqual(
            set(payload),
            {
                "question",
                "evidence_aliases",
                "evidence",
                "relation_summary",
                "learner_guidance",
                "final_answer_json_schema",
                "failure",
                "repair_rules",
                "remaining_deadline_ms",
            },
        )
        self.assertEqual(
            set(payload["failure"]),
            {
                "stable_code",
                "field_path",
                "violation_kind",
                "part_count",
                "alias_count",
                "markdown_fence_detected",
            },
        )
        self.assertTrue(payload["relation_summary"])
        serialized_messages = json.dumps(repair_messages, sort_keys=True)
        serialized_diagnostics = json.dumps(recorder.snapshot(), sort_keys=True)
        for forbidden in (
            "PRIVATE_CANDIDATE",
            rejected,
            private_reasoning,
            "test-placeholder",
            "Authorization",
        ):
            self.assertNotIn(forbidden, serialized_messages)
            self.assertNotIn(forbidden, serialized_diagnostics)
        self.assertNotIn("reasoning_content", serialized_messages)
        self.assertTrue(
            recorder.snapshot()["provider_calls"][0]["reasoning_content_present"]
        )

    def test_initial_and_different_repair_failure_are_both_preserved_safely(self):
        provider = _RouteProvider(
            [_structured("First", ["A999"]), _structured("Second [E1]", ["A1"])]
        )
        recorder = SmokeDiagnosticsRecorder()

        result = answer_from_evidence(
            "explain",
            [self.valid[0]],
            provider,
            self.database,
            retrieval_mode="hybrid",
            diagnostics_recorder=recorder,
        )

        diagnostics = recorder.snapshot()
        self.assertEqual(result["answer_mode"], "deterministic")
        self.assertEqual(provider.final_calls, 2)
        self.assertEqual(
            diagnostics["final_answer_initial_failure"]["stable_code"],
            "citation_alias_unknown",
        )
        self.assertEqual(
            diagnostics["final_answer_repair_failure"]["stable_code"],
            "model_supplied_location_forbidden",
        )
        self.assertNotIn("A999", json.dumps(diagnostics))
        self.assertNotIn("Second [E1]", json.dumps(diagnostics))


class DiagnosticsBoundaryTests(unittest.TestCase):
    _FORBIDDEN = (
        "TEST_API_KEY_SENTINEL",
        "Authorization_SENTINEL",
        "PROMPT_SENTINEL",
        "REASONING_SENTINEL",
        "CANDIDATE_ANSWER_SENTINEL",
        "SOURCE_BODY_SENTINEL",
        "中文敏感正文哨兵",
    )

    @staticmethod
    def _serialized(value: dict) -> bytes:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def _failure(cls, code: str) -> dict:
        return {
            "stage": "final_answer",
            "stable_code": code,
            "field_path": [f"segment_{index}_" + "x" * 48 for index in range(16)],
            "output_chars": 1_000_000,
            "part_count": 1_000_000,
            "alias_count": 1_000_000,
            "markdown_fence_detected": True,
            "output_sha256": "a" * 64,
            "violation_kind": "path",
            "raw": " ".join(cls._FORBIDDEN),
        }

    @classmethod
    def _planner_attempts(cls) -> list[dict]:
        return [
            {
                "stage": "schema",
                "stable_code": "schema_invalid_type",
                "field_path": [
                    f"planner_{attempt}_{index}_" + "y" * 42
                    for index in range(16)
                ],
                "output_chars": 1_000_000,
                "duration_ms": 86_400_000,
                "finish_reason_present": True,
                "finish_reason_value": "length",
                "content_present": True,
                "reasoning_content_present": True,
                "markdown_fence_detected": True,
                "repair_attempt": attempt > 0,
                "output_sha256": "b" * 64,
                "raw": " ".join(cls._FORBIDDEN),
            }
            for attempt in range(10)
        ]

    def _assert_safe_bounded(self, value: dict, maximum: int) -> int:
        serialized = self._serialized(value)
        self.assertLessEqual(len(serialized), maximum)
        self.assertLessEqual(
            len(
                json.dumps(value, ensure_ascii=True, sort_keys=True).encode(
                    "utf-8"
                )
            ),
            maximum,
        )
        self.assertEqual(json.loads(serialized.decode("utf-8")), value)
        for forbidden in self._FORBIDDEN:
            self.assertNotIn(forbidden, serialized.decode("utf-8"))
        return len(serialized)

    def test_ask_diagnostics_extreme_combination_has_strict_utf8_bound(self):
        snapshot = {
            "request_id": "request-ask-diagnostics-boundary",
            "agent_mode": "bounded",
            "agent_status": "final_answer_failed",
            "answer_mode": "deterministic",
            "planner_requests_attempted": 10,
            "planner_repair_attempts": 1,
            "steps_used": 1_000_000,
            "tool_calls_used": 1_000_000,
            "provider_logical_calls": 3,
            "evidence_count": 1_000_000,
            "grounded_candidate_citation_count": 1_000_000,
            "elapsed_ms": 86_400_000,
            "final_answer_attempted": True,
            "final_answer_repair_attempted": True,
            "final_answer_repair_protocol_succeeded": False,
            "final_answer_repair_succeeded": False,
            "planner_attempts": self._planner_attempts(),
            "final_answer_protocol_failure": self._failure(
                "final_answer_schema_invalid"
            ),
            "final_answer_initial_failure": self._failure(
                "citation_alias_unknown"
            ),
            "final_answer_repair_failure": self._failure(
                "model_supplied_location_forbidden"
            ),
            "unsafe_unicode": "中文敏感正文哨兵",
        }
        result = {
            "request_id": "request-ask-diagnostics-boundary",
            "agent_mode": "bounded",
            "agent_status": "final_answer_failed",
            "answer_mode": "deterministic",
            "budget_usage": {
                "steps_used": 1_000_000,
                "tool_calls_used": 1_000_000,
                "elapsed_ms": 86_400_000,
            },
        }

        first = ask_diagnostics.build_ask_failure_detail(
            result=result,
            recorder_snapshot=json.loads(json.dumps(snapshot)),
            retrieval_version="v2",
            hierarchy_mode="normalize_v1",
            relation_mode="expand_v1",
            terminal_reason="provider_output_truncated",
        )["diagnostics"]
        second = ask_diagnostics.build_ask_failure_detail(
            result=result,
            recorder_snapshot=json.loads(json.dumps(snapshot)),
            retrieval_version="v2",
            hierarchy_mode="normalize_v1",
            relation_mode="expand_v1",
            terminal_reason="provider_output_truncated",
        )["diagnostics"]

        self._assert_safe_bounded(first, ask_diagnostics.MAX_ASK_DIAGNOSTICS_BYTES)
        self.assertEqual(first, second)
        self.assertEqual(first["request_id"], result["request_id"])
        self.assertEqual(first["failure_reason_code"], "provider_output_truncated")
        self.assertEqual(first["agent_status"], "final_answer_failed")
        self.assertTrue(first["final_answer_repair_attempted"])
        self.assertFalse(first["final_answer_repair_protocol_succeeded"])
        self.assertFalse(first["final_answer_repair_succeeded"])
        self.assertTrue(first["diagnostics_truncated"])

    def test_smoke_diagnostics_extreme_combination_has_strict_utf8_bound(self):
        def build() -> dict:
            recorder = SmokeDiagnosticsRecorder()
            recorder.begin_request(
                deadline_budget_ms=86_400_000,
                remaining_ms=86_400_000,
            )
            recorder.begin_agent(
                [f"tool_{index}" for index in range(16)],
                request_id="request-smoke-diagnostics-boundary",
            )
            for attempt in self._planner_attempts():
                recorder.record_planner_request(repair=attempt["repair_attempt"])
                recorder.record_planner_attempt(attempt, duration_ms=86_400_000)
            for index in range(16):
                call_id = recorder.start_provider_call("final_answer")
                recorder.record_provider_response(
                    call_id,
                    {
                        "response_received": True,
                        "response_json_valid": True,
                        "choices_present": True,
                        "content_present": True,
                        "reasoning_content_present": True,
                        "finish_reason_present": True,
                        "finish_reason": "length",
                        "output_chars": 1_000_000,
                        "output_sha256": "c" * 64,
                        "usage": {
                            "prompt_tokens": 1_000_000_000,
                            "completion_tokens": 1_000_000_000,
                            "reasoning_tokens": 1_000_000_000,
                            "total_tokens": 1_000_000_000,
                        },
                        "raw": " ".join(self._FORBIDDEN),
                    },
                )
                if index < 8:
                    recorder.record_provider_attempt(
                        call_id,
                        outcome="timeout",
                        duration_ms=86_400_000,
                        timeout_ms=86_400_000,
                    )
            recorder.record_final_answer_attempt()
            recorder.record_final_answer_initial_failure(
                self._failure("citation_alias_unknown")
            )
            recorder.record_final_answer_repair_attempt()
            recorder.record_final_answer_repair_protocol_result(succeeded=False)
            recorder.record_final_answer_repair_failure(
                self._failure("model_supplied_location_forbidden")
            )
            recorder.record_final_answer_failure("citation_format_invalid")
            recorder.record_agent_result(
                {
                    "agent_mode": "bounded",
                    "agent_status": "final_answer_failed",
                    "answer_mode": "deterministic",
                    "evidence": [{}] * 1_000,
                    "citations": [{}] * 1_000,
                }
            )
            return recorder.snapshot()

        first = build()
        second = build()

        self._assert_safe_bounded(
            first, smoke_diagnostics.MAX_SMOKE_DIAGNOSTICS_BYTES
        )
        self.assertEqual(first, second)
        self.assertEqual(first["request_id"], "request-smoke-diagnostics-boundary")
        self.assertEqual(first["agent_status"], "final_answer_failed")
        self.assertTrue(first["final_answer_repair_attempted"])
        self.assertFalse(first["final_answer_repair_protocol_succeeded"])
        self.assertFalse(first["final_answer_repair_succeeded"])
        self.assertTrue(first["diagnostics_truncated"])

    def test_ordinary_diagnostics_keep_their_existing_shape(self):
        recorder = SmokeDiagnosticsRecorder()
        recorder.begin_agent(["search_code"], request_id="request-ordinary")
        recorder.record_planner_request(repair=False)
        recorder.record_planner_attempt(
            {
                "stage": "semantic",
                "stable_code": "valid",
                "field_path": [],
                "repair_attempt": False,
            },
            duration_ms=3,
        )

        smoke = recorder.snapshot()
        self.assertNotIn("diagnostics_truncated", smoke)
        self.assertEqual(smoke["request_id"], "request-ordinary")
        self.assertEqual(smoke["planner_attempts"][0]["stable_code"], "valid")
        detail = ask_diagnostics.build_ask_failure_detail(
            result={
                "request_id": "request-ordinary",
                "agent_mode": "bounded",
                "agent_status": "failed",
                "answer_mode": "not_available",
            },
            recorder_snapshot=smoke,
            retrieval_version="v1",
            hierarchy_mode="off",
            relation_mode="off",
            terminal_reason="provider_error",
        )
        diagnostics = detail["diagnostics"]
        self.assertNotIn("diagnostics_truncated", diagnostics)
        self.assertEqual(diagnostics["request_id"], "request-ordinary")
        self.assertEqual(diagnostics["failure_reason_code"], "provider_error")
        self.assertEqual(diagnostics["planner_attempts"], smoke["planner_attempts"])


class CitationProtocolRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = Database(Path(self.directory.name) / "citation-route.sqlite")
        self.project_id, self.bundle = make_project(
            self.database,
            [("src/auth.py", "authenticate_user", "def authenticate_user():\n    return True\n")],
        )
        self.bundle["project"]["source_type"] = "local"

    def _count(self) -> int:
        with self.database.connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM chat_answers").fetchone()[0])

    def _install_two_evidence_project(self) -> None:
        self.database = Database(Path(self.directory.name) / "citation-route-multi.sqlite")
        self.project_id = self.database.create_project(
            {
                "repo_url": "https://github.com/demo/reponoesis-fixture",
                "owner": "demo",
                "repo": "reponoesis-fixture",
                "default_branch": "main",
                "repository_revision": "revision-m1",
            }
        )
        a_chunk = "\n".join(
            ["def authenticate_user():"]
            + [f"    value_{index} = {index}" for index in range(1, 10)]
            + ["    return True"]
        )
        b_chunk = "\n".join(
            ["def authenticate_user():"]
            + [f"    alternate_{index} = {index}" for index in range(1, 10)]
            + ["    return False"]
        )
        file_specs = (
            ("backend/app/a.py", a_chunk, 10, 100),
            ("backend/app/b.py", b_chunk, 30, 99),
        )
        files = [
            {
                "path": path,
                "extension": ".py",
                "language": "Python",
                "size": len((("\n" * (start_line - 1)) + chunk).encode("utf-8")),
                "content": ("\n" * (start_line - 1)) + chunk,
                "summary": "authenticate_user",
                "importance": importance,
                "is_core": True,
                "imports": [],
                "exports": [],
                "symbols": ["authenticate_user"],
            }
            for path, chunk, start_line, importance in file_specs
        ]
        chunks = [
            make_chunk(path, "authenticate_user", chunk, start_line=start_line)
            for path, chunk, start_line, _importance in file_specs
        ]
        self.database.save_analysis(
            self.project_id,
            {
                "primary_language": "Python",
                "frameworks": [],
                "files": files,
                "modules": [],
                "overview": "Two Evidence production route fixture",
            },
            files,
            [],
            chunks,
        )
        bundle = self.database.get_bundle(self.project_id)
        assert bundle is not None
        bundle["project"]["source_type"] = "local"
        self.bundle = bundle

    def _route(
        self,
        final_output: str | list[str | Exception],
        *,
        callback=None,
        limits: AgentLimits | None = None,
    ):
        provider = _RouteProvider(final_output, callback=callback)
        collector = _SafeLogCollector()
        previous_failure_level = main.logger.level
        previous_success_level = main.success_logger.level
        main.logger.setLevel(logging.INFO)
        main.success_logger.setLevel(logging.INFO)
        self.addCleanup(main.logger.setLevel, previous_failure_level)
        self.addCleanup(main.success_logger.setLevel, previous_success_level)
        main.logger.addHandler(collector)
        self.addCleanup(main.logger.removeHandler, collector)
        main.success_logger.addHandler(collector)
        self.addCleanup(main.success_logger.removeHandler, collector)
        limits = limits or AgentLimits(max_tool_calls=1)
        with (
            patch.object(main, "db", self.database),
            patch.object(main, "llm", provider),
            patch.object(main, "embedding_service", disabled_embedding_service()),
            patch.object(main, "learning_service", _LearningService()),
            patch.object(main, "_bundle_or_404", return_value=self.bundle),
            patch.object(main, "agent_limits", limits),
        ):
            status, body = asyncio.run(
                _post(
                    main.app,
                    f"/api/projects/{self.project_id}/ask",
                    {"question": "explain authentication"},
                )
            )
        return status, body, provider, collector.messages

    def test_formal_uvicorn_logging_config_emits_success_at_info(self):
        request_id = "obs-r1-request"
        script = "\n".join(
            (
                "import logging.config",
                "from uvicorn.config import LOGGING_CONFIG",
                "logging.config.dictConfig(LOGGING_CONFIG)",
                "from app import main",
                "main._log_ask_success({",
                f"    'request_id': '{request_id}',",
                "    'provider_logical_calls': 2,",
                "    'final_answer_repair_attempted': False,",
                "    'citation_validation_passed': True,",
                "    'relation_validation_passed': True,",
                "})",
            )
        )

        completed = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        output = completed.stdout + completed.stderr
        self.assertEqual(output.count('"event":"ask_succeeded"'), 1)
        self.assertIn(request_id, output)
        self.assertIn('"provider_logical_calls":2', output)

    def test_production_entry_success_persists_once_after_both_validators(self):
        before = self._count()
        status, body, provider, logs = self._route(_structured())
        self.assertEqual(status, 200)
        self.assertEqual(self._count() - before, 1)
        self.assertTrue(body["request_id"])
        self.assertEqual(provider.planner_calls, 1)
        self.assertEqual(provider.final_calls, 1)
        self.assertEqual(body["answer_mode"], "llm_grounded")
        self.assertIn("[E1] src/auth.py:1-2", body["answer"])
        user_prompt = provider.final_messages[1]["content"]
        self.assertIn('"allowed_aliases":["A1"]', user_prompt)
        self.assertIn('"evidence_aliases"', user_prompt)
        self.assertNotIn("src/auth.py", user_prompt)
        self.assertNotIn(self.bundle["project"]["repository_revision"], user_prompt)
        success_logs = [item for item in logs if '"event":"ask_succeeded"' in item]
        self.assertEqual(len(success_logs), 1)
        success_log = success_logs[0]
        success_event = json.loads(success_log)
        self.assertEqual(success_event["request_id"], body["request_id"])
        self.assertEqual(
            success_event["diagnostics"]["request_id"], body["request_id"]
        )
        bounded_diagnostics = json.dumps(
            success_event["diagnostics"], separators=(",", ":")
        ).encode("utf-8")
        self.assertLessEqual(
            len(bounded_diagnostics), ask_diagnostics.MAX_ASK_DIAGNOSTICS_BYTES
        )
        self.assertNotIn("explain authentication", success_log)
        self.assertNotIn("def authenticate_user", success_log)
        self.assertNotIn("PRIVATE_PROMPT", success_log)
        self.assertNotIn("PRIVATE_REASONING", success_log)
        self.assertNotIn("PRIVATE_API_KEY", success_log)
        self.assertNotIn("Authorization", success_log)

    def test_production_entry_ordinary_other_evidence_location_is_not_a_citation(self):
        self._install_two_evidence_project()
        ordinary_location = "backend/app/b.py:30-40"
        final_output = _structured(
            f"As ordinary technical prose, see {ordinary_location}; "
            "this conclusion uses the selected Evidence.",
            ["A1"],
        )
        before = self._count()

        status, body, provider, logs = self._route(final_output)

        self.assertEqual(status, 200)
        self.assertEqual(self._count() - before, 1)
        self.assertEqual(provider.planner_calls, 1)
        self.assertEqual(provider.final_calls, 1)
        self.assertEqual(body["answer_mode"], "llm_grounded")
        self.assertEqual(body["agent_status"], "completed")
        self.assertIn(ordinary_location, body["answer"])
        self.assertIn("[E1] backend/app/a.py:10-20", body["answer"])
        self.assertNotIn("[E2]", body["answer"])
        self.assertEqual(body["answer"].count("[E"), 1)
        self.assertEqual(
            {item["path"] for item in body["citations"]},
            {"backend/app/a.py", "backend/app/b.py"},
        )
        self.assertEqual(
            {item["path"] for item in body["evidence"]},
            {"backend/app/a.py", "backend/app/b.py"},
        )

        success_event = json.loads(
            next(item for item in logs if '"event":"ask_succeeded"' in item)
        )
        diagnostics = success_event["diagnostics"]
        self.assertEqual(success_event["request_id"], body["request_id"])
        self.assertEqual(diagnostics["request_id"], body["request_id"])
        self.assertTrue(diagnostics["citation_validation_passed"])
        self.assertTrue(diagnostics["relation_validation_passed"])
        self.assertTrue(diagnostics["post_generation_validation_passed"])
        self.assertFalse(diagnostics["final_answer_repair_attempted"])

        with self.database.connect() as connection:
            saved = connection.execute(
                "SELECT answer, citations_json FROM chat_answers ORDER BY id DESC LIMIT 1"
            ).fetchone()
        self.assertIsNotNone(saved)
        self.assertEqual(saved["answer"], body["answer"])
        self.assertEqual(json.loads(saved["citations_json"]), body["citations"])
        self.assertIn(ordinary_location, saved["answer"])
        self.assertNotIn("[E2]", saved["answer"])

    def test_production_entry_unknown_alias_is_non_200_and_zero_write(self):
        before = self._count()
        status, body, provider, logs = self._route(_structured(aliases=["A999"]))
        self.assertNotEqual(status, 200)
        self.assertEqual(self._count() - before, 0)
        detail = body["detail"]
        self.assertEqual(detail["code"], "citation_unknown")
        self.assertEqual(
            detail["diagnostics"]["final_answer_protocol_failure"]["stable_code"],
            "citation_alias_unknown",
        )
        self.assertEqual(provider.final_calls, 2)
        failure_logs = [item for item in logs if '"event":"ask_failed"' in item]
        self.assertEqual(len(failure_logs), 1)
        self.assertFalse(any('"event":"ask_succeeded"' in item for item in logs))
        failure_log = failure_logs[0]
        self.assertEqual(json.loads(failure_log)["code"], detail["code"])
        self.assertIn(detail["diagnostics"]["request_id"], failure_log)
        self.assertNotIn("A999", failure_log)
        self.assertNotIn("def authenticate_user", failure_log)

    def test_production_entry_failed_repair_is_non_200_and_zero_write(self):
        leaking = _structured("PRIVATE_CANDIDATE [E1]", ["A1"])
        before = self._count()
        status, body, provider, logs = self._route([leaking, leaking])

        self.assertNotEqual(status, 200)
        self.assertEqual(self._count() - before, 0)
        self.assertEqual(provider.final_calls, 2)
        detail = body["detail"]
        self.assertEqual(detail["code"], "citation_format_invalid")
        diagnostics = detail["diagnostics"]
        self.assertTrue(diagnostics["final_answer_repair_attempted"])
        self.assertFalse(diagnostics["final_answer_repair_protocol_succeeded"])
        self.assertFalse(diagnostics["final_answer_repair_succeeded"])
        self.assertEqual(
            diagnostics["final_answer_initial_failure"]["stable_code"],
            "model_supplied_location_forbidden",
        )
        self.assertEqual(
            diagnostics["final_answer_repair_failure"]["stable_code"],
            "model_supplied_location_forbidden",
        )
        protocol = diagnostics["final_answer_protocol_failure"]
        self.assertEqual(
            protocol["stable_code"], "model_supplied_location_forbidden"
        )
        self.assertEqual(protocol["violation_kind"], "evidence_marker")
        failure_log = next(item for item in logs if '"event":"ask_failed"' in item)
        self.assertNotIn("PRIVATE_CANDIDATE", failure_log)

    def test_production_entry_successful_repair_persists_once(self):
        leaking = _structured("PRIVATE_CANDIDATE [E1]", ["A1"])
        repaired = _structured("Grounded repaired answer", ["A1"])
        before = self._count()
        status, body, provider, logs = self._route([leaking, repaired])

        self.assertEqual(status, 200)
        self.assertEqual(self._count() - before, 1)
        self.assertEqual(provider.final_calls, 2)
        self.assertEqual(body["answer_mode"], "llm_grounded")
        self.assertEqual(body["agent_status"], "completed")
        self.assertTrue(body["answer"])
        self.assertTrue(body["citations"])
        self.assertTrue(body["evidence"])
        success_log = next(item for item in logs if '"event":"ask_succeeded"' in item)
        self.assertIn('"final_answer_repair_attempted":true', success_log)
        self.assertIn('"final_answer_repair_protocol_succeeded":true', success_log)
        self.assertIn('"final_answer_repair_succeeded":true', success_log)
        self.assertNotIn("PRIVATE_CANDIDATE", success_log)

    def test_repair_protocol_success_then_citation_validator_failure_is_zero_write(self):
        def mutate_after_repair(call_number: int) -> None:
            if call_number != 2:
                return
            with self.database.connect() as connection:
                connection.execute(
                    "UPDATE repo_files SET content = ? WHERE project_id = ? AND path = ?",
                    ("def changed():\n    return False\n", self.project_id, "src/auth.py"),
                )

        before = self._count()
        status, body, provider, _logs = self._route(
            [_structured("Fact [E1]"), _structured("Repaired fact")],
            callback=mutate_after_repair,
        )

        self.assertNotEqual(status, 200)
        self.assertEqual(self._count() - before, 0)
        self.assertEqual(provider.final_calls, 2)
        diagnostics = body["detail"]["diagnostics"]
        self.assertTrue(diagnostics["final_answer_repair_protocol_succeeded"])
        self.assertFalse(diagnostics["final_answer_repair_succeeded"])
        self.assertEqual(
            diagnostics["final_answer_repair_failure"]["stable_code"],
            "citation_evidence_binding_failed",
        )

    def test_repair_protocol_success_then_relation_validator_failure_is_zero_write(self):
        before = self._count()
        with patch.object(
            RelationValidator,
            "validate_chains",
            return_value=([], ["safe-test-rejection"]),
        ):
            status, body, provider, _logs = self._route(
                [_structured("Fact [E1]"), _structured("Repaired fact")]
            )

        self.assertNotEqual(status, 200)
        self.assertEqual(self._count() - before, 0)
        self.assertEqual(provider.final_calls, 2)
        diagnostics = body["detail"]["diagnostics"]
        self.assertTrue(diagnostics["final_answer_repair_protocol_succeeded"])
        self.assertFalse(diagnostics["final_answer_repair_succeeded"])
        self.assertEqual(
            diagnostics["final_answer_repair_failure"]["stable_code"],
            "relation_validation_failed",
        )

    def test_repaired_answer_success_then_persistence_failure_stays_distinct(self):
        leaking = _structured("Fact [E1]")
        repaired = _structured("Repaired fact")
        before = self._count()
        with patch.object(
            self.database,
            "save_chat_answer",
            side_effect=RuntimeError("safe-test-persistence-failure"),
        ):
            status, body, provider, logs = self._route([leaking, repaired])

        self.assertEqual(status, 500)
        self.assertEqual(body["detail"]["code"], "persistence_failed")
        self.assertEqual(self._count() - before, 0)
        self.assertEqual(provider.final_calls, 2)
        diagnostics = body["detail"]["diagnostics"]
        self.assertTrue(diagnostics["final_answer_repair_protocol_succeeded"])
        self.assertTrue(diagnostics["final_answer_repair_succeeded"])
        self.assertIsNone(diagnostics["final_answer_repair_failure"])
        failure_log = next(item for item in logs if '"event":"ask_failed"' in item)
        self.assertEqual(json.loads(failure_log)["code"], body["detail"]["code"])
        self.assertFalse(any('"event":"ask_succeeded"' in item for item in logs))

    def test_repaired_answer_success_then_response_contract_failure_stays_distinct(self):
        with patch.object(
            main.AskResponse,
            "model_validate_json",
            side_effect=ValueError("safe-test-response-contract-failure"),
        ):
            status, body, provider, logs = self._route(
                [_structured("Fact [E1]"), _structured("Repaired fact")]
            )

        self.assertNotEqual(status, 200)
        self.assertEqual(body["detail"]["code"], "response_contract_invalid")
        self.assertEqual(provider.final_calls, 2)
        diagnostics = body["detail"]["diagnostics"]
        self.assertTrue(diagnostics["final_answer_repair_succeeded"])
        self.assertIsNone(diagnostics["final_answer_repair_failure"])
        failure_log = next(item for item in logs if '"event":"ask_failed"' in item)
        self.assertEqual(json.loads(failure_log)["code"], body["detail"]["code"])

    def test_repaired_answer_success_then_save_deadline_stays_distinct(self):
        original_validate_json = main.AskResponse.model_validate_json

        def delayed_validate_json(value):
            time.sleep(0.6)
            return original_validate_json(value)

        limits = replace(
            AgentLimits(max_tool_calls=1),
            total_deadline_ms=500,
            min_final_answer_budget_ms=50,
            default_tool_timeout_ms=200,
        )
        before = self._count()
        with patch.object(
            main.AskResponse,
            "model_validate_json",
            side_effect=delayed_validate_json,
        ):
            status, body, provider, logs = self._route(
                [_structured("Fact [E1]"), _structured("Repaired fact")],
                limits=limits,
            )

        self.assertEqual(status, 504)
        self.assertEqual(body["detail"]["code"], "deadline_exceeded")
        self.assertEqual(self._count() - before, 0)
        self.assertEqual(provider.final_calls, 2)
        diagnostics = body["detail"]["diagnostics"]
        self.assertTrue(diagnostics["final_answer_repair_succeeded"])
        self.assertIsNone(diagnostics["final_answer_repair_failure"])
        failure_log = next(item for item in logs if '"event":"ask_failed"' in item)
        self.assertEqual(json.loads(failure_log)["code"], body["detail"]["code"])

    def test_success_diagnostics_builder_failure_uses_safe_fallback_and_saves_once(self):
        before = self._count()
        with patch.object(
            main,
            "build_ask_success_diagnostics",
            side_effect=RuntimeError("PRIVATE-DIAGNOSTICS-BUILDER-FAILURE"),
        ):
            status, body, provider, logs = self._route(_structured())

        self.assertEqual(status, 200)
        self.assertEqual(self._count() - before, 1)
        self.assertEqual(provider.final_calls, 1)
        event = json.loads(next(item for item in logs if '"event":"ask_succeeded"' in item))
        self.assertEqual(event["request_id"], body["request_id"])
        self.assertEqual(event["diagnostics"]["success_stage"], "response_validated")
        self.assertTrue(event["diagnostics"]["core_validation_passed"])
        self.assertTrue(event["diagnostics"]["observability_degraded"])
        self.assertNotIn("PRIVATE-DIAGNOSTICS-BUILDER-FAILURE", json.dumps(event))

    def test_success_projection_and_serialization_failure_cannot_reverse_save(self):
        before = self._count()
        with patch.object(
            ask_diagnostics,
            "_bounded_payload",
            side_effect=RuntimeError("PRIVATE-DIAGNOSTICS-PROJECTION-FAILURE"),
        ):
            status, body, provider, logs = self._route(_structured())

        self.assertEqual(status, 200)
        self.assertEqual(self._count() - before, 1)
        self.assertEqual(provider.final_calls, 1)
        self.assertFalse(any('"event":"ask_failed"' in item for item in logs))
        self.assertNotIn("PRIVATE-DIAGNOSTICS-PROJECTION-FAILURE", json.dumps(body))

    def test_post_save_success_logger_failure_is_best_effort_and_never_retries_save(self):
        handler = _RaisingSuccessLogHandler()
        main.success_logger.addHandler(handler)
        self.addCleanup(main.success_logger.removeHandler, handler)
        before = self._count()

        status, body, provider, logs = self._route(_structured())

        self.assertEqual(status, 200)
        self.assertEqual(self._count() - before, 1)
        self.assertEqual(provider.final_calls, 1)
        self.assertEqual(body["answer_mode"], "llm_grounded")
        self.assertFalse(any('"event":"ask_failed"' in item for item in logs))
        self.assertNotIn("PRIVATE-SUCCESS-LOGGER-FAILURE", json.dumps(body))


if __name__ == "__main__":
    unittest.main()
