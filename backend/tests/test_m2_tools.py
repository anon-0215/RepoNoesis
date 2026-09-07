from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from pydantic import BaseModel, ConfigDict

import app.services.agent_tools as agent_tools
import app.services.agent_core as agent_core
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
from tests.m1_helpers import (
    REVISION,
    disabled_embedding_service,
    make_chunk,
    make_project,
)


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
                "get_learning_context",
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
        exact = observation.structured_results[0]
        self.assertEqual(exact["candidate_source"], "symbol")
        self.assertEqual(exact["symbol_match_type"], "exact_qualified")
        self.assertEqual(exact["symbol_rank"], 1)
        self.assertIn("qualified_symbol_exact", exact["match_reasons"])
        _call, observation = self._call(
            "lookup_symbol",
            {"symbol": "auth", "match_mode": "prefix", "language": "python"},
        )
        self.assertEqual(
            [item["path"] for item in observation.structured_results],
            ["src/admin.py", "src/auth.py"],
        )
        self.assertTrue(all("references" not in item for item in observation.structured_results))

    def test_lookup_symbol_promotes_via_canonical_chunk_and_duplicate_is_zero_progress(self):
        _call, first = self._call(
            "lookup_symbol",
            {"symbol": "AuthService.authenticate_user", "match_mode": "exact"},
        )
        self.assertEqual(first.metrics["raw_result_count"], 1)
        self.assertEqual(first.metrics["normalized_candidate_count"], 1)
        self.assertEqual(first.metrics["valid_candidate_count"], 1)
        self.assertEqual(first.metrics["new_evidence_count"], 1)
        self.assertEqual(first.metrics["result_count"], 1)
        evidence = self.store.all(self.context.request_id)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(first.structured_results[0]["evidence_id"], evidence[0].evidence_id)
        self.assertIn("return verify", evidence[0].excerpt)

        _call, duplicate = self._call(
            "lookup_symbol",
            {"symbol": "AuthService.authenticate_user", "match_mode": "exact"},
        )
        self.assertEqual(duplicate.metrics["raw_result_count"], 1)
        self.assertEqual(duplicate.metrics["new_evidence_count"], 0)
        self.assertEqual(duplicate.metrics["result_count"], 0)
        self.assertEqual(len(self.store.all(self.context.request_id)), 1)

    def test_lookup_symbol_invalid_locator_fails_closed_without_blocking_valid_peer(self):
        valid = agent_tools.SymbolRetriever(self.db).search(
            self.project_id,
            "AuthService.authenticate_user",
            top_k=1,
            match_mode="exact",
            explicit_symbol=True,
            repository_revision=REVISION,
        )[0]
        forged = replace(valid, chunk_identity="forged-identity")
        with patch.object(
            agent_tools.SymbolRetriever,
            "search",
            return_value=[forged, valid],
        ):
            _call, observation = self._call(
                "lookup_symbol",
                {"symbol": "AuthService.authenticate_user", "match_mode": "exact"},
            )
        self.assertEqual(observation.metrics["raw_result_count"], 2)
        self.assertEqual(observation.metrics["valid_candidate_count"], 1)
        self.assertEqual(observation.metrics["new_evidence_count"], 1)
        self.assertEqual(observation.metrics["candidate_rejection_chunk_identity_mismatch_count"], 1)
        self.assertEqual(len(self.store.all(self.context.request_id)), 1)

    def test_lookup_symbol_validates_before_final_limit_and_never_claims_existing_evidence(self):
        valid = agent_tools.SymbolRetriever(self.db).search(
            self.project_id,
            "AuthService.authenticate_user",
            top_k=1,
            match_mode="exact",
            explicit_symbol=True,
            repository_revision=REVISION,
        )[0]
        invalid_first = replace(valid, chunk_identity="forged-identity")
        with patch.object(
            agent_tools.SymbolRetriever,
            "search",
            return_value=[invalid_first, valid],
        ):
            _call, observation = self._call(
                "lookup_symbol",
                {"symbol": "AuthService.authenticate_user", "match_mode": "exact", "top_k": 1},
            )
        self.assertEqual(observation.metrics["raw_result_count"], 2)
        self.assertEqual(observation.metrics["valid_candidate_count"], 1)
        self.assertEqual(observation.metrics["new_evidence_count"], 1)
        self.assertEqual(observation.metrics["candidate_rejection_chunk_identity_mismatch_count"], 1)
        self.assertEqual(
            [item["qualified_name"] for item in observation.structured_results],
            [valid.qualified_name],
        )
        self.assertEqual(observation.structured_results[0]["evidence_id"], "E1")

        forged_existing_identity = replace(valid, content_hash="0" * 64)
        with patch.object(
            agent_tools.SymbolRetriever,
            "search",
            return_value=[forged_existing_identity],
        ):
            _call, rejected = self._call(
                "lookup_symbol",
                {"symbol": "AuthService.authenticate_user", "match_mode": "exact", "top_k": 1},
            )
        self.assertEqual(rejected.metrics["new_evidence_count"], 0)
        self.assertEqual(rejected.metrics["result_count"], 0)
        self.assertEqual(rejected.structured_results, [])
        self.assertEqual(agent_core._progress_keys("lookup_symbol", rejected.structured_results), set())

        _call, duplicate = self._call(
            "lookup_symbol",
            {"symbol": "AuthService.authenticate_user", "match_mode": "exact", "top_k": 1},
        )
        self.assertEqual(duplicate.metrics["new_evidence_count"], 0)
        self.assertEqual(duplicate.metrics["result_count"], 0)
        self.assertIsNone(duplicate.structured_results[0]["evidence_id"])
        self.assertEqual(agent_core._progress_keys("lookup_symbol", duplicate.structured_results), set())

    def test_lookup_symbol_rejects_raw_score_and_rank_type_confusion_without_blocking_peer(self):
        valid = agent_tools.SymbolRetriever(self.db).search(
            self.project_id,
            "AuthService.authenticate_user",
            top_k=1,
            match_mode="exact",
            explicit_symbol=True,
            repository_revision=REVISION,
        )[0]
        invalid_values = (
            ("score_false", {"symbol_score": False}),
            ("score_true", {"symbol_score": True}),
            ("score_string", {"symbol_score": "1.0"}),
            ("score_nan", {"symbol_score": float("nan")}),
            ("score_infinite", {"symbol_score": float("inf")}),
            ("rank_true", {"symbol_rank": True}),
            ("rank_float", {"symbol_rank": 1.0}),
            ("rank_string", {"symbol_rank": "1"}),
        )
        for name, values in invalid_values:
            with self.subTest(name=name):
                invalid = replace(valid, **values)
                with patch.object(
                    agent_tools.SymbolRetriever,
                    "search",
                    return_value=[invalid, valid],
                ):
                    _call, observation = self._call(
                        "lookup_symbol",
                        {"symbol": "AuthService.authenticate_user", "match_mode": "exact", "top_k": 1},
                    )
                self.assertEqual(observation.status, "succeeded")
                self.assertEqual(observation.metrics["raw_result_count"], 2)
                self.assertEqual(observation.metrics["valid_candidate_count"], 1)
                self.assertEqual(observation.metrics["candidate_rejection_score_invalid_count"], 1)
                self.assertEqual(
                    [item["qualified_name"] for item in observation.structured_results],
                    [valid.qualified_name],
                )

    def test_lookup_symbol_rejected_locator_cannot_claim_existing_evidence_id_or_progress(self):
        _call, first = self._call(
            "lookup_symbol",
            {"symbol": "AuthService.authenticate_user", "match_mode": "exact", "top_k": 1},
        )
        self.assertEqual(first.metrics["new_evidence_count"], 1)
        valid = agent_tools.SymbolRetriever(self.db).search(
            self.project_id,
            "AuthService.authenticate_user",
            top_k=1,
            match_mode="exact",
            explicit_symbol=True,
            repository_revision=REVISION,
        )[0]
        forged_existing_identity = replace(valid, content_hash="0" * 64)
        with patch.object(
            agent_tools.SymbolRetriever,
            "search",
            return_value=[forged_existing_identity],
        ):
            _call, observation = self._call(
                "lookup_symbol",
                {"symbol": "AuthService.authenticate_user", "match_mode": "exact", "top_k": 1},
            )
        self.assertEqual(observation.metrics["new_evidence_count"], 0)
        self.assertEqual(observation.metrics["result_count"], 0)
        self.assertEqual(observation.structured_results, [])
        self.assertEqual(agent_core._progress_keys("lookup_symbol", observation.structured_results), set())

    def test_read_source_mapping_cap_reports_pre_cap_count_separately_from_body_truncation(self):
        source = "".join(f"line_{index}\n" for index in range(1, 51))
        self.project_id, self.bundle = make_project(
            self.db, [("src/mapping.py", "mapping_container", source)]
        )
        self.store = EvidenceStore(capacity=1)
        self.context = build_tool_context(
            request_id="mapping-cap-only",
            bundle=self.bundle,
            database=self.db,
            embedding_service=disabled_embedding_service(),
            evidence_store=self.store,
            limits=self.limits,
            cancellation=CancellationToken(),
            deadline_monotonic=time.monotonic() + 60,
        )
        self.db.save_code_chunks_for_project(
            self.project_id,
            [
                make_chunk(
                    "src/mapping.py",
                    f"candidate_{index}",
                    source.splitlines(keepends=True)[index - 1],
                    start_line=index,
                )
                for index in range(1, 51)
            ],
        )
        _call, observation = self._call(
            "read_source", {"path": "src/mapping.py", "start_line": 1, "end_line": 50}
        )
        self.assertEqual(observation.metrics["raw_result_count"], 50)
        self.assertEqual(observation.metrics["mapping_candidate_count"], 50)
        self.assertEqual(observation.metrics["mapping_considered_count"], 20)
        self.assertTrue(observation.metrics["mapping_truncated"])
        self.assertFalse(observation.truncated)

    def test_read_source_prefers_complementary_siblings_and_reports_mapping_truncation(self):
        source = (
            "class AuthService:\n"
            "    def authenticate_user(self, password):\n"
            "        return verify(password)\n"
        )
        parent = make_chunk("src/auth.py", "AuthService", source)
        child_one = make_chunk(
            "src/auth.py", "AuthService.authenticate_user.signature", source.splitlines(keepends=True)[1], start_line=2
        )
        child_two = make_chunk(
            "src/auth.py", "AuthService.authenticate_user.body", source.splitlines(keepends=True)[2], start_line=3
        )
        self.db.save_code_chunks_for_project(self.project_id, [parent, child_one, child_two])
        _call, sibling_observation = self._call(
            "read_source", {"path": "src/auth.py", "start_line": 2, "end_line": 3}
        )
        self.assertEqual(sibling_observation.status, "succeeded")
        self.assertEqual(
            [item.qualified_name for item in self.store.all(self.context.request_id)],
            [child_one["qualified_name"], child_two["qualified_name"]],
        )

        hierarchy = self.db.get_code_chunks_for_hierarchy(
            self.project_id, REVISION, "src/auth.py", limit=20
        )
        forward = agent_tools._map_read_range_to_chunks(
            hierarchy, start_line=2, end_line=3, limit=2
        )
        backward = agent_tools._map_read_range_to_chunks(
            list(reversed(hierarchy)), start_line=2, end_line=3, limit=2
        )
        self.assertEqual(
            [item["id"] for item in forward], [item["id"] for item in backward]
        )

        mapping_source = "".join(f"line_{index}\n" for index in range(1, 51))
        self.project_id, self.bundle = make_project(
            self.db, [("src/mapping.py", "mapping_container", mapping_source)]
        )
        self.store = EvidenceStore(capacity=1)
        self.context = build_tool_context(
            request_id="mapping-cap",
            bundle=self.bundle,
            database=self.db,
            embedding_service=disabled_embedding_service(),
            evidence_store=self.store,
            limits=self.limits,
            cancellation=CancellationToken(),
            deadline_monotonic=time.monotonic() + 60,
        )
        self.db.save_code_chunks_for_project(
            self.project_id,
            [
                make_chunk(
                    "src/mapping.py",
                    f"candidate_{index}",
                    mapping_source.splitlines(keepends=True)[index - 1],
                    start_line=index,
                )
                for index in range(1, 51)
            ],
        )
        _call, capped = self._call(
            "read_source", {"path": "src/mapping.py", "start_line": 1, "end_line": 50}
        )
        self.assertEqual(capped.metrics["raw_result_count"], 50)
        self.assertEqual(capped.metrics["mapping_candidate_count"], 50)
        self.assertEqual(capped.metrics["mapping_considered_count"], 20)
        self.assertTrue(capped.metrics["mapping_truncated"])
        self.assertFalse(capped.truncated)

    def test_read_source_prefers_child_for_narrow_span_and_uses_parent_only_as_fallback(self):
        source = (
            "class AuthService:\n"
            "    def authenticate_user(self, password):\n"
            "        return verify(password)\n"
        )
        parent = make_chunk("src/auth.py", "AuthService", source)
        child = make_chunk(
            "src/auth.py",
            "AuthService.authenticate_user",
            source.splitlines(keepends=True)[1],
            start_line=2,
        )
        self.db.save_code_chunks_for_project(self.project_id, [parent, child])
        _call, narrow = self._call(
            "read_source", {"path": "src/auth.py", "start_line": 2, "end_line": 2}
        )
        self.assertEqual(narrow.status, "succeeded")
        self.assertEqual(
            [item.qualified_name for item in self.store.all(self.context.request_id)],
            [child["qualified_name"]],
        )

        fallback_store = EvidenceStore()
        fallback_context = build_tool_context(
            request_id="parent-fallback",
            bundle=self.bundle,
            database=self.db,
            embedding_service=disabled_embedding_service(),
            evidence_store=fallback_store,
            limits=self.limits,
            cancellation=CancellationToken(),
            deadline_monotonic=time.monotonic() + 60,
        )
        self.db.save_code_chunks_for_project(self.project_id, [parent])
        fallback_call = ToolCall(
            "parent-fallback-call",
            "S",
            "read_source",
            "1",
            {"path": "src/auth.py", "start_line": 2, "end_line": 2},
            15_000,
            {},
        )
        fallback = self.registry.execute(fallback_context, fallback_call)
        self.assertEqual(fallback.status, "succeeded")
        self.assertEqual(
            [item.qualified_name for item in fallback_store.all("parent-fallback")],
            [parent["qualified_name"]],
        )

    def test_lookup_duplicate_with_base_retrieval_merges_candidate_provenance(self):
        self._call("search_code", {"query": "verify"})
        _call, observation = self._call(
            "lookup_symbol",
            {"symbol": "AuthService.authenticate_user", "match_mode": "exact"},
        )
        self.assertEqual(observation.metrics["new_evidence_count"], 0)
        self.assertEqual(len(self.store.all(self.context.request_id)), 1)
        provenance = self.context.candidate_pool.accepted_candidates()[0].provenance
        self.assertEqual(
            provenance,
            ("planner_search_code", "planner_lookup_symbol"),
        )

    def test_read_source_maps_complete_canonical_chunks_and_never_uses_read_excerpt(self):
        _call, observation = self._call(
            "read_source", {"path": "src/auth.py", "start_line": 2, "end_line": 2}
        )
        self.assertEqual(observation.status, "succeeded")
        self.assertGreaterEqual(observation.metrics["normalized_candidate_count"], 1)
        self.assertEqual(observation.metrics["new_evidence_count"], 1)
        self.assertEqual(observation.metrics["result_count"], 1)
        read_content = observation.structured_results["content"]
        evidence = self.store.all(self.context.request_id)
        self.assertEqual(len(evidence), 1)
        self.assertNotEqual(evidence[0].excerpt, read_content)
        self.assertIn("return verify", evidence[0].excerpt)

        _call, duplicate = self._call(
            "read_source", {"path": "src/auth.py", "start_line": 2, "end_line": 2}
        )
        self.assertEqual(duplicate.metrics["new_evidence_count"], 0)
        self.assertEqual(duplicate.metrics["result_count"], 0)
        self.assertEqual(duplicate.structured_results["evidence"], [])

    def test_read_source_mapping_is_stable_for_nested_chunks_and_respects_capacity(self):
        self.db.save_code_chunks_for_project(
            self.project_id,
            [
                make_chunk(
                    "src/auth.py",
                    "AuthService",
                    "class AuthService:\n    def authenticate_user(self, password):\n        return verify(password)\n",
                ),
                make_chunk(
                    "src/auth.py",
                    "AuthService.authenticate_user",
                    "    def authenticate_user(self, password):\n        return verify(password)\n",
                    start_line=2,
                ),
            ],
        )
        _call, observation = self._call(
            "read_source", {"path": "src/auth.py", "start_line": 1, "end_line": 3}
        )
        self.assertEqual(observation.status, "succeeded")
        self.assertGreaterEqual(observation.metrics["raw_result_count"], 2)
        self.assertEqual(observation.metrics["new_evidence_count"], 1)
        self.assertEqual(
            [item.start_line for item in self.store.all(self.context.request_id)],
            [1],
        )

        limited_store = EvidenceStore(capacity=1)
        limited_context = build_tool_context(
            request_id="limited-read",
            bundle=self.bundle,
            database=self.db,
            embedding_service=disabled_embedding_service(),
            evidence_store=limited_store,
            limits=self.limits,
            cancellation=CancellationToken(),
            deadline_monotonic=time.monotonic() + 60,
        )
        call = ToolCall(
            "C",
            "S",
            "read_source",
            "1",
            {"path": "src/auth.py", "start_line": 1, "end_line": 3},
            15_000,
            {},
        )
        limited = self.registry.execute(limited_context, call)
        self.assertEqual(limited.metrics["new_evidence_count"], 1)
        self.assertEqual(len(limited_store.all("limited-read")), 1)

    def test_read_source_boundary_without_chunk_is_not_mapped_and_metrics_are_content_free(self):
        chunks = [
            {
                "id": 2,
                "path": "src/auth.py",
                "qualified_name": "AuthService.authenticate_user",
                "chunk_type": "function",
                "start_line": 2,
                "end_line": 3,
            },
        ]
        self.assertEqual(
            agent_tools._map_read_range_to_chunks(
                chunks, start_line=4, end_line=4, limit=20
            ),
            [],
        )

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
            "C:/secret.py",
            "/secret.py",
        ]
        for path in bad_paths:
            with self.subTest(path=path):
                _call, observation = self._call(
                    "read_source", {"path": path, "start_line": 1, "end_line": 1}
                )
                self.assertEqual(observation.status, "rejected")
        _call, normalized = self._call(
            "read_source", {"path": "src\\auth.py", "start_line": 1, "end_line": 1}
        )
        self.assertEqual(normalized.status, "succeeded")
        self.assertEqual(normalized.structured_results["path"], "src/auth.py")
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
