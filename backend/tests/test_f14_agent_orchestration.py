from __future__ import annotations

import asyncio
import copy
from dataclasses import replace
import json
import logging
from pathlib import Path
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

from pydantic import ValidationError

from app import main
from app.config import LLMSettings
from app.database import Database
from app.services import agent_core
from app.services.agent_contracts import (
    AgentLimits,
    CancellationToken,
    ToolCall,
)
from app.services.agent_core import run_bounded_agent, validate_planner_decision
from app.services.agent_tools import (
    EvidenceStore,
    build_m2_tool_registry,
    build_tool_context,
)
from app.services.ask_diagnostics import format_ask_failure_log
from app.services.evidence import CitationValidator
from app.services.llm_client import LLMClient
from app.services.relation_graph import RelationValidator
from app.services.smoke_diagnostics import SmokeDiagnosticsRecorder
from tests.m1_helpers import disabled_embedding_service, make_chunk, make_project
from tests.m3_helpers import call_chain_sources, make_relation_project
from tests.test_f12_agent_orchestration import (
    _LearningService,
    _ScriptedProvider,
    _post,
)
from tests.test_m2_agent import NoLlm, ScriptedPlanner, decision


class _LogCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(dict(record.__dict__))


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class _Response:
    status = 200

    def __init__(self, content: str) -> None:
        self.body = json.dumps(
            {"choices": [{"message": {"content": content}}]}
        ).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self.body


class _WorkCutoffHarness:
    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.planner_calls = 0
        self.final_calls = 0

    def opener(self, request, *, timeout):
        del timeout
        payload = json.loads(request.data.decode("utf-8"))
        if "bounded_repository_planner" in payload["messages"][0]["content"]:
            self.planner_calls += 1
            if self.planner_calls == 1:
                return _Response(
                    json.dumps(
                        decision(
                            "continue",
                            "search_code",
                            {"query": "authenticate_user upload_file", "top_k": 2},
                        )
                    )
                )
            self.clock.value = 7.1
            raise TimeoutError("safe fake work cutoff")
        self.final_calls += 1
        return _Response('{"parts":[{"text":"Grounded","evidence_aliases":["A1"]}]}')


def _cutoff_client(harness: _WorkCutoffHarness) -> LLMClient:
    return LLMClient(
        LLMSettings(
            provider="openai_compatible",
            base_url="https://provider.invalid/v1",
            api_key="test-placeholder",
            model="configured-model",
            timeout_seconds=20.0,
            max_tokens=1600,
            max_retries=0,
        ),
        opener=harness.opener,
        sleep=lambda _seconds: None,
        monotonic=harness.clock,
    )


class F14AgentOrchestrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = Database(Path(self.directory.name) / "f14.sqlite")
        self.project_id, self.bundle = make_project(
            self.database,
            [
                (
                    "src/auth.py",
                    "authenticate_user",
                    "def authenticate_user(password):\n    return verify(password)\n",
                ),
                (
                    "src/admin.py",
                    "authenticate_user",
                    "def authenticate_user(token):\n    return bool(token)\n",
                ),
                (
                    "src/upload.py",
                    "upload_file",
                    "def upload_file(path):\n    return save(path)\n",
                ),
            ],
        )
        self.bundle["project"]["source_type"] = "local"
        self.limits = replace(
            AgentLimits(), max_agent_steps=5, total_deadline_ms=10_000,
            min_final_answer_budget_ms=3_000,
        )

    def _chat_count(self, database: Database | None = None) -> int:
        with (database or self.database).connect() as connection:
            return int(
                connection.execute("SELECT COUNT(*) FROM chat_answers").fetchone()[0]
            )

    def _route(self, provider, payload: dict, *, database=None, bundle=None):
        database = database or self.database
        bundle = bundle or self.bundle
        with (
            patch.object(main, "db", database),
            patch.object(main, "llm", provider),
            patch.object(main, "embedding_service", disabled_embedding_service()),
            patch.object(main, "learning_service", _LearningService()),
            patch.object(main, "_bundle_or_404", return_value=bundle),
            patch.object(main, "agent_limits", self.limits),
        ):
            return asyncio.run(
                _post(
                    main.app,
                    f"/api/projects/{self.project_id}/ask",
                    payload,
                )
            )

    def test_request_capacity_bounds_two_searches_seed_and_finalization(self):
        recorder = SmokeDiagnosticsRecorder()
        final_context_sizes: list[int] = []
        original = agent_core.answer_from_evidence

        def capture_final(question, evidence, *args, **kwargs):
            final_context_sizes.append(len(evidence))
            return original(question, evidence, *args, **kwargs)

        with patch.object(agent_core, "answer_from_evidence", side_effect=capture_final):
            result = run_bounded_agent(
                "authenticate_user and upload_file",
                self.bundle,
                NoLlm(),
                self.database,
                disabled_embedding_service(),
                planner=ScriptedPlanner(
                    [
                        decision("continue", "search_code", {"query": "authenticate_user"}),
                        decision("continue", "search_code", {"query": "upload_file"}),
                        decision("answer"),
                    ]
                ),
                evidence_count=1,
                diagnostics_recorder=recorder,
            )
        self.assertEqual(final_context_sizes, [1])
        self.assertEqual(len(result["evidence"]), 1)
        executions = recorder.snapshot()["tool_executions"]
        self.assertEqual([item["result_count"] for item in executions], [1, 1])
        self.assertEqual([item["evidence_added"] for item in executions], [1, 0])

        seed_db = Database(Path(self.directory.name) / "seed-capacity.sqlite")
        seed_project = seed_db.create_project(
            {
                "repo_url": "https://github.com/demo/seed-capacity",
                "owner": "demo",
                "repo": "seed-capacity",
                "default_branch": "main",
                "repository_revision": "revision-m1",
            }
        )
        source = (
            "def authenticate_user(password):\n"
            "    return verify(password)\n\n"
            "def audit_user(user):\n"
            "    return bool(user)\n"
        )
        files = [
            {
                "path": "src/auth.py", "extension": ".py", "language": "Python",
                "size": len(source.encode("utf-8")), "content": source,
                "summary": "auth", "importance": 100, "is_core": True,
                "imports": [], "exports": [],
                "symbols": ["authenticate_user", "audit_user"],
            }
        ]
        seed_db.save_analysis(
            seed_project,
            {
                "primary_language": "Python", "frameworks": [], "files": files,
                "modules": [], "overview": "seed capacity fixture",
            },
            files,
            [],
            [
                make_chunk(
                    "src/auth.py", "authenticate_user",
                    "def authenticate_user(password):\n    return verify(password)\n",
                    start_line=1,
                ),
                make_chunk(
                    "src/auth.py", "audit_user",
                    "def audit_user(user):\n    return bool(user)\n",
                    start_line=4,
                ),
            ],
        )
        seed_bundle = seed_db.get_bundle(seed_project)
        self.assertIsNotNone(seed_bundle)
        seed_recorder = SmokeDiagnosticsRecorder()
        seeded = run_bounded_agent(
            "authenticate_user",
            seed_bundle,
            NoLlm(),
            seed_db,
            disabled_embedding_service(),
            path="src/auth.py",
            planner=ScriptedPlanner(
                [
                    decision("continue", "search_code", {"query": "audit_user"}),
                    decision("answer"),
                ]
            ),
            evidence_count=1,
            diagnostics_recorder=seed_recorder,
        )
        self.assertEqual(len(seeded["evidence"]), 1)
        self.assertEqual(seeded["evidence"][0]["path"], "src/auth.py")
        seed_executions = seed_recorder.snapshot()["tool_executions"]
        self.assertEqual([item["evidence_added"] for item in seed_executions], [1, 0])

    def test_capacity_deduplicates_before_count_and_preserves_multi_item_behavior(self):
        captured: list = []
        original = agent_core.answer_from_evidence

        def capture(question, evidence, *args, **kwargs):
            captured.extend(copy.deepcopy(evidence))
            return original(question, evidence, *args, **kwargs)

        with patch.object(agent_core, "answer_from_evidence", side_effect=capture):
            result = run_bounded_agent(
                "authenticate_user",
                self.bundle,
                NoLlm(),
                self.database,
                disabled_embedding_service(),
                planner=ScriptedPlanner(
                    [
                        decision(
                            "continue", "search_code",
                            {"query": "authenticate_user", "top_k": 2},
                        ),
                        decision("answer"),
                    ]
                ),
                evidence_count=2,
            )
        self.assertEqual(len(result["evidence"]), 2)
        self.assertEqual(len(captured), 2)
        store = EvidenceStore(capacity=2)
        added = store.add("owner", [captured[0], copy.deepcopy(captured[0]), captured[1]])
        self.assertEqual(len(added), 2)
        self.assertEqual(len(store.all("owner")), 2)
        with self.assertRaises(AttributeError):
            store._capacity = 3

    def test_explicit_relation_expansion_shares_the_store_capacity(self):
        relation_db = Database(Path(self.directory.name) / "relation-capacity.sqlite")
        project_id, bundle = make_relation_project(relation_db, call_chain_sources())
        store = EvidenceStore(capacity=1)
        context = build_tool_context(
            request_id="relation-capacity",
            bundle=bundle,
            database=relation_db,
            embedding_service=disabled_embedding_service(),
            evidence_store=store,
            limits=AgentLimits(),
            cancellation=CancellationToken(),
            deadline_monotonic=agent_core.time.monotonic() + 60,
            retrieval_version="v2",
            relation_mode="off",
        )
        registry = build_m2_tool_registry(AgentLimits())
        search = registry.execute(
            context,
            ToolCall(
                "C1", "S1", "search_code", "1",
                {"query": "a", "top_k": 1}, 15_000, {},
            ),
        )
        self.assertEqual(search.status, "succeeded")
        self.assertEqual(len(store.all(context.request_id)), 1)
        expanded = registry.execute(
            context,
            ToolCall(
                "C2", "S2", "expand_relations", "1",
                {"seed_evidence_ids": ["E1"], "relation_types": ["calls"]},
                15_000, {},
            ),
        )
        self.assertEqual(expanded.status, "succeeded")
        self.assertGreater(expanded.metrics["node_count"], 1)
        self.assertEqual(len(store.all(context.request_id)), 1)

    def test_production_planner_formal_models_normalize_and_reject_paths(self):
        planner_arguments = [
            {"query": "authenticate_user", "path": "src\\auth.py", "top_k": 2},
            {"symbol": "authenticate_user", "path": "src\\auth.py"},
            {"path": "src\\auth.py", "start_line": 1, "end_line": 2},
        ]
        originals = copy.deepcopy(planner_arguments)
        provider = _ScriptedProvider(
            [
                decision("continue", "search_code", planner_arguments[0]),
                decision("continue", "lookup_symbol", planner_arguments[1]),
                decision("continue", "read_source", planner_arguments[2]),
                decision("answer"),
            ]
        )
        registry = build_m2_tool_registry(self.limits)
        captured: dict[str, list[dict]] = {
            "search_code": [], "lookup_symbol": [], "read_source": []
        }
        for action in captured:
            spec = registry.get(action)

            def capture(context, parameters, *, _action=action, _handler=spec.handler):
                captured[_action].append(parameters.model_dump())
                return _handler(context, parameters)

            registry._tools[action] = replace(spec, handler=capture)
        result = run_bounded_agent(
            "authenticate_user",
            self.bundle,
            provider,
            self.database,
            disabled_embedding_service(),
            evidence_count=2,
            registry=registry,
            limits=self.limits,
            diagnostics_recorder=SmokeDiagnosticsRecorder(),
        )
        self.assertEqual(result["agent_status"], "completed")
        self.assertTrue(all(items[0]["path"] == "src/auth.py" for items in captured.values()))
        self.assertEqual(planner_arguments, originals)

        unsafe_paths = (
            "../secret.py", "src/../secret.py", "/absolute/path.py",
            "C:\\repo\\file.py", "C:/repo/file.py",
            "\\\\server\\share\\file.py", "//server/share/file.py",
            "src//auth.py", "src/./auth.py", "", "   ", "src/\0auth.py",
        )
        for action, base in (
            ("search_code", {"query": "q"}),
            ("lookup_symbol", {"symbol": "authenticate_user"}),
            ("read_source", {"start_line": 1, "end_line": 1}),
        ):
            for unsafe in unsafe_paths:
                with self.subTest(action=action, path=repr(unsafe)):
                    arguments = {**base, "path": unsafe}
                    original_arguments = copy.deepcopy(arguments)
                    raw_decision = decision("continue", action, arguments)
                    validation = validate_planner_decision(
                        raw_decision, build_m2_tool_registry(self.limits)
                    )
                    self.assertFalse(validation.valid)
                    self.assertEqual(validation.failure.stage, "semantic")
                    self.assertEqual(
                        validation.failure.stable_code,
                        "semantic_invalid_tool_contract",
                    )
                    self.assertEqual(validation.failure.field_path, ("arguments", "path"))
                    self.assertEqual(arguments, original_arguments)

    def test_server_bound_normalized_path_overrides_all_conflicting_planner_paths(self):
        for action, arguments in (
            ("search_code", {"query": "authenticate_user", "path": "src/upload.py"}),
            ("lookup_symbol", {"symbol": "authenticate_user", "path": "src/upload.py"}),
            ("read_source", {"path": "src/upload.py", "start_line": 1, "end_line": 2}),
        ):
            with self.subTest(action=action):
                original_arguments = copy.deepcopy(arguments)
                registry = build_m2_tool_registry(self.limits)
                spec = registry.get(action)
                captured: list[dict] = []

                def capture(context, parameters):
                    captured.append(parameters.model_dump())
                    return spec.handler(context, parameters)

                registry._tools[action] = replace(spec, handler=capture)
                run_bounded_agent(
                    "authenticate_user", self.bundle,
                    _ScriptedProvider(
                        [decision("continue", action, arguments), decision("answer")]
                    ),
                    self.database, disabled_embedding_service(),
                    path="src\\auth.py", evidence_count=2,
                    registry=registry, diagnostics_recorder=SmokeDiagnosticsRecorder(),
                )
                self.assertTrue(captured)
                self.assertTrue(all(item["path"] == "src/auth.py" for item in captured))
                self.assertEqual(arguments, original_arguments)

    def test_unknown_tool_is_safe_in_logs_trace_success_and_failure(self):
        raw_unknown = "secret_token=abc\0" + "X" * 63
        provider = _ScriptedProvider(
            [
                decision("continue", raw_unknown, {}),
                decision(
                    "continue", "search_code",
                    {"query": "authenticate_user", "path": "src/auth.py"},
                ),
                decision("answer"),
            ]
        )
        collector = _LogCollector()
        log = logging.getLogger("app.services.agent_core")
        previous_level = log.level
        log.setLevel(logging.INFO)
        log.addHandler(collector)
        try:
            status, body = self._route(
                provider,
                {"question": "authenticate_user", "evidence_count": 1},
            )
        finally:
            log.removeHandler(collector)
            log.setLevel(previous_level)
        self.assertEqual(status, 200)
        self.assertEqual(self._chat_count(), 1)
        self.assertEqual(provider.final_calls, 1)
        self.assertEqual(body["budget_usage"]["tool_calls_used"], 1)
        self.assertEqual(body["agent_trace"][0]["action"], "search_code")
        serialized = json.dumps(
            {"body": body, "records": collector.records}, default=str
        )
        self.assertNotIn(raw_unknown, serialized)
        self.assertNotIn('"tool_version": "unknown"', serialized)

        failing_provider = _ScriptedProvider(
            [decision("continue", raw_unknown, {})]
        )
        captured_failures: list[dict] = []
        with patch.object(main, "_log_ask_failure", side_effect=captured_failures.append):
            status, body = self._route(
                failing_provider,
                {"question": "authenticate_user", "evidence_count": 1},
            )
        self.assertEqual(status, 502)
        self.assertEqual(self._chat_count(), 1)
        self.assertNotIn(raw_unknown, json.dumps(body))
        self.assertNotIn(raw_unknown, format_ask_failure_log(captured_failures[0]))
        detail = body["detail"]
        self.assertEqual(detail["code"], "planner_repair_failed")
        self.assertEqual(detail["diagnostics"]["tool_calls_used"], 0)
        self.assertEqual(detail["diagnostics"]["tool_executions"], [])
        attempts = detail["diagnostics"]["planner_attempts"]
        self.assertEqual(len(attempts), 2)
        self.assertTrue(attempts[1]["repair_attempt"])
        self.assertTrue(all(
            item["stable_code"] == "semantic_invalid_tool_contract"
            for item in attempts
        ))

    def test_partial_citation_rejection_and_count_mismatch_fail_before_provider(self):
        original_validate = CitationValidator.validate_all

        def mixed(validator, evidence):
            valid, warnings = original_validate(validator, evidence)
            if len(valid) >= 2:
                valid[1].validation_status = "invalid"
                valid[1].invalid_reason = "content hash mismatch"
                valid[1].excerpt = ""
                return valid[:1], ["one Evidence was rejected"]
            return valid, warnings

        provider = _ScriptedProvider(
            [
                decision(
                    "continue", "search_code",
                    {"query": "authenticate_user", "top_k": 2},
                ),
                decision("answer"),
            ]
        )
        failures: list[dict] = []
        with (
            patch.object(CitationValidator, "validate_all", autospec=True, side_effect=mixed),
            patch.object(main, "_log_ask_failure", side_effect=failures.append),
        ):
            status, body = self._route(
                provider,
                {"question": "authenticate_user", "evidence_count": 2},
            )
        self.assertEqual(status, 502)
        self.assertEqual(body["detail"]["code"], "citation_evidence_binding_failed")
        self.assertEqual(provider.final_calls, 0)
        self.assertEqual(self._chat_count(), 0)
        self.assertEqual(failures[0]["code"], body["detail"]["code"])
        self.assertEqual(
            json.loads(format_ask_failure_log(failures[0]))["failure_reason_code"],
            body["detail"]["code"],
        )

        def count_mismatch(validator, evidence):
            valid, _warnings = original_validate(validator, evidence)
            return (valid[:-1], []) if len(valid) >= 2 else (valid, [])

        recorder = SmokeDiagnosticsRecorder()
        with patch.object(
            CitationValidator, "validate_all", autospec=True, side_effect=count_mismatch
        ):
            result = run_bounded_agent(
                "authenticate_user", self.bundle, NoLlm(), self.database,
                disabled_embedding_service(),
                planner=ScriptedPlanner(
                    [
                        decision(
                            "continue", "search_code",
                            {"query": "authenticate_user", "top_k": 2},
                        ),
                        decision("answer"),
                    ]
                ),
                evidence_count=2, diagnostics_recorder=recorder,
            )
        self.assertEqual(result["agent_status"], "final_answer_failed")
        self.assertEqual(
            recorder.snapshot()["citation_failure_reason_code"],
            "citation_evidence_binding_failed",
        )

    def test_work_cutoff_mixed_citations_fails_asgi_without_final_call_or_save(self):
        clock = _Clock()
        harness = _WorkCutoffHarness(clock)
        client = _cutoff_client(harness)
        original_validate = CitationValidator.validate_all

        def mixed(validator, evidence):
            valid, warnings = original_validate(validator, evidence)
            if len(valid) >= 2:
                valid[1].validation_status = "invalid"
                valid[1].invalid_reason = "repository path mismatch"
                valid[1].excerpt = ""
                return valid[:1], ["one Evidence was rejected"]
            return valid, warnings

        with (
            patch("app.services.agent_core.time.monotonic", side_effect=clock),
            patch("app.services.agent_tools.time.monotonic", side_effect=clock),
            patch("app.services.qa_agent.time.monotonic", side_effect=clock),
            patch("app.main.time.monotonic", side_effect=clock),
            patch.object(CitationValidator, "validate_all", autospec=True, side_effect=mixed),
            patch.object(main, "db", self.database),
            patch.object(main, "llm", client),
            patch.object(main, "embedding_service", disabled_embedding_service()),
            patch.object(main, "learning_service", _LearningService()),
            patch.object(main, "_bundle_or_404", return_value=self.bundle),
            patch.object(main, "agent_limits", self.limits),
        ):
            status, body = asyncio.run(
                _post(
                    main.app,
                    f"/api/projects/{self.project_id}/ask",
                    {"question": "authenticate_user upload_file", "evidence_count": 2},
                )
            )
        self.assertEqual(status, 502)
        self.assertEqual(body["detail"]["code"], "citation_path_mismatch")
        self.assertEqual(harness.planner_calls, 2)
        self.assertEqual(harness.final_calls, 0)
        self.assertEqual(self._chat_count(), 0)

    def test_relation_chain_change_calls_finalization_once_and_never_persists(self):
        relation_db = Database(Path(self.directory.name) / "relation-final.sqlite")
        project_id, bundle = make_relation_project(relation_db, call_chain_sources())
        bundle["project"]["source_type"] = "local"
        provider = _ScriptedProvider(
            [
                decision("continue", "search_code", {"query": "a", "top_k": 3}),
                decision("answer"),
            ],
            final_answer='{"parts":[{"text":"Grounded","evidence_aliases":["A1"]}]}',
        )
        original_relations = RelationValidator.validate_chains
        relation_calls = 0

        def changing(validator, **kwargs):
            nonlocal relation_calls
            relation_calls += 1
            valid, warnings = original_relations(validator, **kwargs)
            if relation_calls == 2:
                return [], []
            return valid, warnings

        original_final = agent_core.answer_from_evidence
        with (
            patch.object(
                RelationValidator, "validate_chains", autospec=True, side_effect=changing
            ),
            patch.object(
                agent_core, "answer_from_evidence", wraps=original_final
            ) as finalization,
            patch.object(main, "db", relation_db),
            patch.object(main, "llm", provider),
            patch.object(main, "embedding_service", disabled_embedding_service()),
            patch.object(main, "learning_service", _LearningService()),
            patch.object(main, "_bundle_or_404", return_value=bundle),
            patch.object(main, "agent_limits", self.limits),
        ):
            status, body = asyncio.run(
                _post(
                    main.app,
                    f"/api/projects/{project_id}/ask",
                    {
                        "question": "a", "evidence_count": 3,
                        "retrieval_version": "v2", "relation_mode": "expand_v1",
                    },
                )
            )
        self.assertEqual(status, 502)
        self.assertEqual(body["detail"]["code"], "relation_validation_failed")
        self.assertEqual(finalization.call_count, 1)
        self.assertEqual(provider.final_calls, 1)
        self.assertEqual(self._chat_count(relation_db), 0)

        stable_provider = _ScriptedProvider(
            [
                decision("continue", "search_code", {"query": "a", "top_k": 3}),
                decision("answer"),
            ],
            final_answer='{"parts":[{"text":"Grounded","evidence_aliases":["A1"]}]}',
        )
        with (
            patch.object(main, "db", relation_db),
            patch.object(main, "llm", stable_provider),
            patch.object(main, "embedding_service", disabled_embedding_service()),
            patch.object(main, "learning_service", _LearningService()),
            patch.object(main, "_bundle_or_404", return_value=bundle),
            patch.object(main, "agent_limits", self.limits),
        ):
            stable_status, _stable_body = asyncio.run(
                _post(
                    main.app,
                    f"/api/projects/{project_id}/ask",
                    {
                        "question": "a", "evidence_count": 3,
                        "retrieval_version": "v2", "relation_mode": "expand_v1",
                    },
                )
            )
        self.assertEqual(stable_status, 200)
        self.assertEqual(stable_provider.final_calls, 1)
        self.assertEqual(self._chat_count(relation_db), 1)

    def test_formal_path_models_preserve_optional_and_posix_semantics(self):
        registry = build_m2_tool_registry(AgentLimits())
        self.assertIsNone(
            registry.get("search_code").input_model(query="q").path
        )
        self.assertEqual(
            registry.get("lookup_symbol").input_model(
                symbol="x", path="src/auth.py"
            ).path,
            "src/auth.py",
        )
        with self.assertRaises(ValidationError):
            registry.get("read_source").input_model(
                path=" ", start_line=1, end_line=1
            )


if __name__ == "__main__":
    unittest.main()
