from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from pydantic import BaseModel, ConfigDict

from app.database import Database
from app.services.agent_contracts import (
    AgentLimits,
    CancellationToken,
    ToolCall,
)
from app.services.agent_tools import (
    EvidenceStore,
    ToolContext,
    ToolRegistry,
    ToolSpec,
    build_m2_tool_registry,
    build_tool_context,
)
from tests.m1_helpers import REVISION, disabled_embedding_service, make_project


class EmptyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class M2ToolTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.directory.name) / "m2-tools.sqlite")
        self.project_id, self.bundle = make_project(
            self.db,
            [
                (
                    "src/auth.py",
                    "AuthService.authenticate_user",
                    "class AuthService:\n"
                    "    def authenticate_user(self, password):\n"
                    "        return verify(password)\n",
                ),
                (
                    "src/admin.py",
                    "authenticate_user",
                    "def authenticate_user(token):\n    return bool(token)\n",
                ),
            ],
        )
        self.limits = AgentLimits()
        self.cancellation = CancellationToken()
        self.store = EvidenceStore()
        self.context = build_tool_context(
            request_id="request-1",
            bundle=self.bundle,
            database=self.db,
            embedding_service=disabled_embedding_service(),
            evidence_store=self.store,
            limits=self.limits,
            cancellation=self.cancellation,
            deadline_monotonic=time.monotonic() + 60,
        )
        self.registry = build_m2_tool_registry(self.limits)

    def tearDown(self):
        self.directory.cleanup()

    def _call(self, tool, parameters, *, timeout_ms=15_000, version="1"):
        call = ToolCall(
            call_id="C1",
            step_id="S1",
            tool_name=tool,
            tool_version=version,
            parameters=parameters,
            timeout_ms=timeout_ms,
            budget={"max_results": 20, "max_bytes": 65_536},
        )
        return call, self.registry.execute(self.context, call)

    def test_registry_is_static_versioned_and_stably_sorted(self):
        tools = self.registry.list_tools()
        self.assertEqual(
            [item["name"] for item in tools],
            [
                "expand_relations",
                "lookup_symbol",
                "read_source",
                "search_code",
                "validate_evidence",
            ],
        )
        self.assertTrue(all(item["version"] == "1" for item in tools))
        with self.assertRaises(ValueError):
            self.registry.register(self.registry.get("search_code"))
        with self.assertRaises(KeyError):
            self.registry.get("shell")

    def test_unknown_tool_and_wrong_version_are_rejected(self):
        _call, observation = self._call("shell", {})
        self.assertEqual(observation.status, "rejected")
        self.assertEqual(observation.error["code"], "unknown_tool")
        _call, observation = self._call("search_code", {"query": "auth"}, version="2")
        self.assertEqual(observation.status, "failed")

    def test_schema_rejects_extra_identity_and_invalid_parameters_without_execution(self):
        _call, observation = self._call(
            "search_code",
            {"query": "auth", "project_id": "forged", "revision": "other"},
        )
        self.assertEqual(observation.status, "rejected")
        self.assertEqual(observation.metrics["result_count"], 0)
        _call, observation = self._call("read_source", {"path": "x", "start_line": 2, "end_line": 0})
        self.assertEqual(observation.status, "rejected")

    def test_registry_reports_serialization_timeout_cancellation_and_metrics(self):
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                "bad",
                "1",
                "bad",
                EmptyInput,
                lambda _context, _input: (object(), [], False),
                100,
                1,
                100,
            )
        )
        call = ToolCall("C", "S", "bad", "1", {}, 100, {})
        observation = registry.execute(self.context, call)
        self.assertEqual(observation.status, "failed")
        self.assertEqual(observation.error["code"], "tool_error")

        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                "slow",
                "1",
                "slow",
                EmptyInput,
                lambda _context, _input: ([], [], False),
                0,
                1,
                100,
            )
        )
        call = ToolCall("C", "S", "slow", "1", {}, 0, {})
        observation = registry.execute(self.context, call)
        self.assertEqual(observation.status, "timed_out")
        self.assertEqual(observation.metrics["timeout_enforcement"], "cooperative")

        self.cancellation.cancel()
        call = ToolCall("C", "S", "slow", "1", {}, 100, {})
        observation = registry.execute(self.context, call)
        self.assertEqual(observation.status, "cancelled")

    def test_registry_result_count_byte_limit_and_truncation(self):
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                "many",
                "1",
                "many",
                EmptyInput,
                lambda _context, _input: ([{"value": "x" * 20}] * 10, [], False),
                100,
                2,
                100,
            )
        )
        call = ToolCall("C", "S", "many", "1", {}, 100, {})
        observation = registry.execute(self.context, call)
        self.assertEqual(observation.status, "succeeded")
        self.assertTrue(observation.truncated)
        self.assertLessEqual(observation.metrics["result_count"], 2)
        self.assertLessEqual(observation.metrics["output_bytes"], 100)

    def test_search_code_reuses_m1_and_preserves_server_identity_and_filters(self):
        _call, observation = self._call(
            "search_code",
            {
                "query": "authenticate_user",
                "path": "src/admin.py",
                "language": "python",
                "symbol": "authenticate_user",
                "top_k": 20,
            },
        )
        self.assertEqual(observation.status, "succeeded")
        self.assertEqual(observation.structured_results["retrieval_mode"], "lexical")
        evidence = observation.structured_results["evidence"]
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["path"], "src/admin.py")
        self.assertEqual(evidence[0]["repository_revision"], REVISION)
        self.assertNotIn("content", evidence[0])
        self.assertTrue(any("disabled" in warning for warning in observation.warnings))

    def test_search_code_empty_result_and_semantic_failure_degrade_cleanly(self):
        _call, observation = self._call("search_code", {"query": "no-such-quantum-symbol"})
        self.assertEqual(observation.status, "succeeded")
        self.assertEqual(observation.structured_results["evidence"], [])

    def test_lookup_symbol_exact_qualified_filters_duplicates_and_stable_sort(self):
        _call, observation = self._call(
            "lookup_symbol",
            {"symbol": "AuthService.authenticate_user", "match_mode": "exact"},
        )
        self.assertEqual(observation.status, "succeeded")
        self.assertEqual(
            [item["qualified_name"] for item in observation.structured_results],
            ["AuthService.authenticate_user"],
        )
        _call, observation = self._call(
            "lookup_symbol",
            {"symbol": "auth", "match_mode": "prefix", "language": "python"},
        )
        self.assertEqual(
            [item["path"] for item in observation.structured_results],
            ["src/admin.py", "src/auth.py"],
        )
        self.assertTrue(all("references" not in item for item in observation.structured_results))

    def test_lookup_symbol_no_result_and_bound_revision(self):
        _call, observation = self._call(
            "lookup_symbol", {"symbol": "missing", "match_mode": "fuzzy"}
        )
        self.assertEqual(observation.structured_results, [])
        self.assertEqual(self.context.repository_revision, REVISION)

    def test_read_source_valid_hash_and_does_not_execute_source(self):
        marker = Path(self.directory.name) / "executed"
        source = f"open({str(marker)!r}, 'w').write('bad')\n"
        project_id, bundle = make_project(
            self.db,
            [("src/injection.py", "dangerous_string", source)],
        )
        context = build_tool_context(
            request_id="request-2",
            bundle=bundle,
            database=self.db,
            embedding_service=disabled_embedding_service(),
            evidence_store=EvidenceStore(),
            limits=self.limits,
            cancellation=CancellationToken(),
            deadline_monotonic=time.monotonic() + 60,
        )
        call = ToolCall(
            "C",
            "S",
            "read_source",
            "1",
            {"path": "src/injection.py", "start_line": 1, "end_line": 1},
            15_000,
            {},
        )
        observation = self.registry.execute(context, call)
        self.assertEqual(observation.status, "succeeded")
        self.assertEqual(
            observation.structured_results["content_hash"],
            hashlib.sha256(source.encode()).hexdigest(),
        )
        self.assertFalse(marker.exists())
        self.assertEqual(context.project_id, project_id)

    def test_read_source_rejects_paths_ranges_missing_files_and_revision_change(self):
        bad_paths = [
            "../secret.py",
            "src\\auth.py",
            "C:/secret.py",
            "/secret.py",
        ]
        for path in bad_paths:
            with self.subTest(path=path):
                _call, observation = self._call(
                    "read_source", {"path": path, "start_line": 1, "end_line": 1}
                )
                self.assertEqual(observation.status, "failed")
        for parameters in (
            {"path": "missing.py", "start_line": 1, "end_line": 1},
            {"path": "src/auth.py", "start_line": 9, "end_line": 9},
            {"path": "src/auth.py", "start_line": 2, "end_line": 1},
            {"path": "src/auth.py", "start_line": 1, "end_line": 201},
        ):
            _call, observation = self._call("read_source", parameters)
            self.assertIn(observation.status, {"failed", "rejected"})

        with self.db.connect() as conn:
            conn.execute(
                "UPDATE code_chunks SET repository_revision = 'changed' WHERE project_id = ?",
                (self.project_id,),
            )
        _call, observation = self._call(
            "read_source", {"path": "src/auth.py", "start_line": 1, "end_line": 1}
        )
        self.assertEqual(observation.status, "failed")

    def test_read_source_detects_content_change_and_byte_truncation(self):
        small_limits = replace(self.limits, max_source_read_bytes=20)
        context = replace(self.context, limits=small_limits)
        registry = build_m2_tool_registry(small_limits)
        call = ToolCall(
            "C",
            "S",
            "read_source",
            "1",
            {"path": "src/auth.py", "start_line": 1, "end_line": 3},
            15_000,
            {},
        )
        observation = registry.execute(context, call)
        self.assertEqual(observation.status, "succeeded")
        self.assertTrue(observation.truncated)

        original = __import__(
            "app.services.agent_tools", fromlist=["_snapshot_file"]
        )._snapshot_file
        calls = {"count": 0}

        def changing_snapshot(tool_context, path):
            result = dict(original(tool_context, path))
            calls["count"] += 1
            if calls["count"] == 2:
                result["content"] += "# changed\n"
            return result

        with patch("app.services.agent_tools._snapshot_file", side_effect=changing_snapshot):
            _call, observation = self._call(
                "read_source",
                {"path": "src/auth.py", "start_line": 1, "end_line": 1},
            )
        self.assertEqual(observation.status, "failed")

    def test_validate_evidence_rejects_forgery_cross_request_and_stale_source(self):
        self._call("search_code", {"query": "authenticate_user"})
        evidence_id = self.store.all("request-1")[0].evidence_id
        _call, observation = self._call(
            "validate_evidence", {"evidence_ids": [evidence_id, "E999"]}
        )
        results = observation.structured_results
        self.assertTrue(results[0]["validated"])
        self.assertFalse(results[1]["validated"])
        self.assertIn("not owned", results[1]["invalid_reason"])

        with self.db.connect() as conn:
            conn.execute(
                "UPDATE repo_files SET content = 'changed' WHERE project_id = ?",
                (self.project_id,),
            )
        _call, observation = self._call(
            "validate_evidence", {"evidence_ids": [evidence_id]}
        )
        self.assertFalse(observation.structured_results[0]["validated"])


if __name__ == "__main__":
    unittest.main()
