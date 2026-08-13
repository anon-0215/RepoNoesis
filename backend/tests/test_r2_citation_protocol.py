from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app import main
from app.config import LLMSettings
from app.database import Database
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
from app.services.smoke_diagnostics import SmokeDiagnosticsRecorder
from tests.m1_helpers import disabled_embedding_service, make_project
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

    def __init__(self, final_output: str) -> None:
        self.final_output = final_output
        self.planner_calls = 0
        self.final_calls = 0
        self.final_messages = None

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
        return self.final_output


class _SafeLogCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


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
            (json.dumps({"parts": [{"text": "Fact [E1] src/auth.py:1-2", "evidence_aliases": ["A1"]}]}), "final_answer_schema_invalid"),
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

    def _route(self, final_output: str):
        provider = _RouteProvider(final_output)
        collector = _SafeLogCollector()
        previous_level = main.logger.level
        main.logger.setLevel(logging.INFO)
        self.addCleanup(main.logger.setLevel, previous_level)
        main.logger.addHandler(collector)
        self.addCleanup(main.logger.removeHandler, collector)
        limits = AgentLimits(max_tool_calls=1)
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
        success_log = next(item for item in logs if '"event":"ask_succeeded"' in item)
        self.assertIn(body["request_id"], success_log)
        self.assertNotIn("explain authentication", success_log)
        self.assertNotIn("def authenticate_user", success_log)

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
        self.assertEqual(provider.final_calls, 1)
        failure_log = next(item for item in logs if '"event":"ask_failed"' in item)
        self.assertIn(detail["diagnostics"]["request_id"], failure_log)
        self.assertNotIn("A999", failure_log)
        self.assertNotIn("def authenticate_user", failure_log)


if __name__ == "__main__":
    unittest.main()
