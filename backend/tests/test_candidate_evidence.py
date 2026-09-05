from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch

from app.database import Database
from app.services.agent_contracts import AgentLimits, CancellationToken, ToolCall
from app.services.agent_tools import EvidenceStore, build_m2_tool_registry, build_tool_context
from app.services.hybrid_retriever import HybridSearchResult
from tests.m1_helpers import disabled_embedding_service, make_project


class CandidateEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = Database(Path(self.directory.name) / "candidates.sqlite")
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
        self.revision = self.bundle["project"]["repository_revision"]

    def _retrieval_result(self, symbol: str, *, rank: int = 1) -> HybridSearchResult:
        chunk = self.database.get_code_chunks(self.project_id, symbol=symbol)[0]
        return HybridSearchResult(
            project_id=self.project_id,
            repository_revision=self.revision,
            code_chunk_id=chunk["id"],
            language=chunk["language"],
            path=chunk["path"],
            chunk_type=chunk["chunk_type"],
            symbol_name=chunk["symbol_name"],
            qualified_name=chunk["qualified_name"],
            start_line=chunk["start_line"],
            end_line=chunk["end_line"],
            content=chunk["content"],
            content_hash=chunk["content_hash"],
            retrieval_sources=["lexical"],
            lexical_score=float(10 - rank),
            lexical_rank=rank,
            fusion_score=float(10 - rank),
            fusion_rank=rank,
        )

    def _context(self, *, capacity: int, request_id: str = "candidate-request"):
        limits = AgentLimits()
        store = EvidenceStore(capacity=capacity)
        context = build_tool_context(
            request_id=request_id,
            bundle=self.bundle,
            database=self.database,
            embedding_service=disabled_embedding_service(),
            evidence_store=store,
            limits=limits,
            cancellation=CancellationToken(),
            deadline_monotonic=time.monotonic() + 60,
        )
        return context, store, build_m2_tool_registry(limits)

    def _execute(
        self,
        context,
        registry,
        results,
        *,
        provenance="deterministic_base_retrieval",
        query="candidate query",
        call_id="C1",
    ):
        context.candidate_provenance = provenance
        outcome = SimpleNamespace(
            results=list(results),
            retrieval_mode="lexical",
            retrieval_version="v1",
            retrieval_strategy_version="weighted-rrf-v1",
            audit={},
            warnings=[],
        )
        call = ToolCall(
            call_id=call_id,
            step_id="S1",
            tool_name="search_code",
            tool_version="1",
            parameters={"query": query, "top_k": 8},
            timeout_ms=40_000,
            budget={"max_results": 20, "max_bytes": 65_536},
        )
        with patch("app.services.agent_tools.retrieve_code", return_value=outcome):
            return registry.execute(context, call)

    def test_canonical_chunks_promote_once_with_stable_order(self) -> None:
        auth = self._retrieval_result("authenticate_user", rank=1)
        upload = self._retrieval_result("upload_file", rank=2)
        context, store, registry = self._context(capacity=8)

        observation = self._execute(context, registry, [upload, auth, auth])

        self.assertEqual(
            [item.symbol_name for item in store.all(context.request_id)],
            ["authenticate_user", "upload_file"],
        )
        self.assertEqual(observation.metrics["retrieval_hit_count"], 3)
        self.assertEqual(observation.metrics["normalized_candidate_count"], 3)
        self.assertEqual(observation.metrics["valid_candidate_count"], 3)
        self.assertEqual(observation.metrics["new_evidence_count"], 2)
        self.assertEqual(observation.metrics["result_count"], 2)
        self.assertEqual(observation.metrics["rejected_candidate_count"], 0)

    def test_forged_valid_orders_and_cross_call_recovery_use_production_bridge(self) -> None:
        valid = self._retrieval_result("authenticate_user")
        for label, batch in (
            ("forged-first", [replace(valid, path="src/forged.py"), valid]),
            ("valid-first", [valid, replace(valid, path="src/forged.py")]),
        ):
            with self.subTest(label=label):
                context, store, registry = self._context(capacity=1, request_id=label)
                observation = self._execute(context, registry, batch)
                self.assertEqual(len(store.all(context.request_id)), 1)
                self.assertEqual(observation.metrics["rejected_candidate_count"], 1)
                self.assertEqual(observation.metrics["result_count"], 1)

        context, store, registry = self._context(capacity=1, request_id="cross-call")
        first = self._execute(
            context,
            registry,
            [replace(valid, content_hash="0" * 64)],
            call_id="C1",
        )
        second = self._execute(context, registry, [valid], call_id="C2")
        self.assertEqual(first.metrics["result_count"], 0)
        self.assertEqual(first.metrics["rejected_candidate_count"], 1)
        self.assertEqual(second.metrics["result_count"], 1)
        self.assertEqual(len(store.all(context.request_id)), 1)

    def test_request_cap_applies_across_calls_and_invalid_never_consumes_it(self) -> None:
        auth = self._retrieval_result("authenticate_user", rank=1)
        upload = self._retrieval_result("upload_file", rank=1)
        context, store, registry = self._context(capacity=1)

        rejected = self._execute(
            context,
            registry,
            [replace(auth, repository_revision="stale")],
            call_id="C1",
        )
        accepted = self._execute(context, registry, [auth], call_id="C2")
        over_cap = self._execute(context, registry, [upload], call_id="C3")

        self.assertEqual(rejected.metrics["result_count"], 0)
        self.assertEqual(accepted.metrics["result_count"], 1)
        self.assertEqual(over_cap.metrics["valid_candidate_count"], 1)
        self.assertEqual(over_cap.metrics["result_count"], 0)
        self.assertEqual(
            [item.symbol_name for item in store.all(context.request_id)],
            ["authenticate_user"],
        )

    def test_duplicate_valid_merges_provenance_and_best_metadata_without_new_evidence(self) -> None:
        initial = replace(
            self._retrieval_result("authenticate_user", rank=2),
            retrieval_sources=["lexical"],
        )
        better = replace(
            self._retrieval_result("authenticate_user", rank=1),
            retrieval_sources=["semantic", "lexical"],
            semantic_score=0.75,
            semantic_rank=1,
        )
        context, store, registry = self._context(capacity=1)

        first = self._execute(context, registry, [initial], call_id="C1")
        second = self._execute(
            context,
            registry,
            [better, better, better],
            provenance="planner_search_code",
            call_id="C2",
        )

        self.assertEqual(first.metrics["result_count"], 1)
        self.assertEqual(second.metrics["retrieval_hit_count"], 3)
        self.assertEqual(second.metrics["valid_candidate_count"], 3)
        self.assertEqual(second.metrics["result_count"], 0)
        self.assertEqual(len(store.all(context.request_id)), 1)
        accepted = context.candidate_pool.accepted_candidates()
        self.assertEqual(len(accepted), 1)
        self.assertEqual(
            accepted[0].provenance,
            ("deterministic_base_retrieval", "planner_search_code"),
        )
        self.assertEqual(accepted[0].retrieval_sources, ("lexical", "semantic"))
        self.assertEqual(accepted[0].fusion_rank, 1)

        all_sources = [
            "relation",
            "hierarchy",
            "symbol",
            "semantic",
            "lexical",
            "dense",
        ]
        context, _store, registry = self._context(
            capacity=1,
            request_id="duplicate-source-cap",
        )
        self._execute(
            context,
            registry,
            [replace(initial, retrieval_sources=all_sources)],
            call_id="C1",
        )
        duplicate = self._execute(
            context,
            registry,
            [replace(better, retrieval_sources=list(reversed(all_sources)))],
            provenance="planner_search_code",
            call_id="C2",
        )
        self.assertEqual(duplicate.metrics["new_evidence_count"], 0)
        self.assertEqual(
            context.candidate_pool.accepted_candidates()[0].retrieval_sources,
            ("dense", "lexical", "semantic", "symbol", "hierarchy", "relation"),
        )
        self.assertLessEqual(
            len(context.candidate_pool.accepted_candidates()[0].retrieval_sources),
            8,
        )

    def test_retrieval_sources_are_allowlisted_deduplicated_and_bounded_on_first_accept(self) -> None:
        legal = replace(
            self._retrieval_result("authenticate_user"),
            retrieval_sources=["semantic", "lexical", "semantic", "lexical"],
        )
        context, store, registry = self._context(capacity=2, request_id="legal-sources")
        observation = self._execute(context, registry, [legal])
        accepted = context.candidate_pool.accepted_candidates()

        self.assertEqual(observation.metrics["new_evidence_count"], 1)
        self.assertEqual(accepted[0].retrieval_sources, ("lexical", "semantic"))
        self.assertEqual(
            store.all(context.request_id)[0].retrieval_sources,
            ["lexical", "semantic"],
        )
        self.assertLessEqual(len(accepted[0].retrieval_sources), 8)

        v2 = replace(
            self._retrieval_result("upload_file"),
            retrieval_sources=["symbol", "dense", "lexical", "dense"],
        )
        context, store, registry = self._context(capacity=1, request_id="v2-sources")
        observation = self._execute(context, registry, [v2])
        self.assertEqual(observation.metrics["new_evidence_count"], 1)
        self.assertEqual(
            context.candidate_pool.accepted_candidates()[0].retrieval_sources,
            ("dense", "lexical", "symbol"),
        )

    def test_oversized_or_invalid_sources_reject_one_candidate_without_blocking_valid_peer(self) -> None:
        auth = self._retrieval_result("authenticate_user")
        upload = self._retrieval_result("upload_file", rank=2)
        invalid_sources = (
            ["lexical"] * 1_000,
            ["lexical", "unknown"],
            [""],
            [1],
            [],
            "lexical",
            None,
        )
        for index, sources in enumerate(invalid_sources):
            with self.subTest(index=index):
                context, store, registry = self._context(
                    capacity=2,
                    request_id=f"invalid-sources-{index}",
                )
                observation = self._execute(
                    context,
                    registry,
                    [replace(auth, retrieval_sources=sources), upload],
                )
                accepted = context.candidate_pool.accepted_candidates()
                serialized = repr((observation.to_dict(), accepted, store.all(context.request_id)))

                self.assertEqual(observation.metrics["retrieval_hit_count"], 2)
                self.assertEqual(observation.metrics["valid_candidate_count"], 1)
                self.assertEqual(observation.metrics["new_evidence_count"], 1)
                self.assertEqual(observation.metrics["rejected_candidate_count"], 1)
                self.assertEqual(
                    observation.metrics["candidate_rejection_retrieval_sources_invalid_count"],
                    1,
                )
                self.assertEqual([item.symbol_name for item in accepted], ["upload_file"])
                self.assertNotIn("unknown", serialized)
                self.assertLess(len(serialized), 8_000)

    def test_invalid_rank_and_score_shapes_fail_closed_without_source_leakage(self) -> None:
        valid = self._retrieval_result("authenticate_user")
        invalid_cases = (
            replace(valid, fusion_score=float("nan")),
            replace(valid, fusion_rank=0),
            replace(valid, fusion_rank=True),
            replace(valid, lexical_score=float("inf")),
            replace(valid, lexical_rank=True),
            replace(valid, semantic_score=float("-inf")),
            replace(valid, semantic_rank=-1),
        )
        for index, malformed in enumerate(invalid_cases):
            with self.subTest(index=index):
                context, store, registry = self._context(
                    capacity=1,
                    request_id=f"invalid-{index}",
                )
                observation = self._execute(context, registry, [malformed])
                self.assertEqual(observation.metrics["result_count"], 0)
                self.assertEqual(observation.metrics["rejected_candidate_count"], 1)
                self.assertEqual(store.all(context.request_id), [])
                self.assertNotIn("return verify(password)", repr(context.candidate_pool))

    def test_zero_rank_sentinel_strictly_rejects_invalid_ranking_types(self) -> None:
        direct = self._retrieval_result("authenticate_user")
        derived = replace(
            direct,
            retrieval_sources=["relation"],
            lexical_score=None,
            lexical_rank=None,
            semantic_score=None,
            semantic_rank=None,
            fusion_score=0.0,
            fusion_rank=0,
        )
        valid_peer = self._retrieval_result("upload_file", rank=1)
        invalid_cases = (
            ("rank-bool", replace(derived, fusion_rank=False)),
            ("rank-float", replace(derived, fusion_rank=0.0)),
            ("score-bool", replace(derived, fusion_score=False)),
        )
        for label, malformed in invalid_cases:
            with self.subTest(label=label):
                context, store, registry = self._context(
                    capacity=2,
                    request_id=f"zero-rank-{label}",
                )
                rejected = self._execute(context, registry, [malformed])

                self.assertEqual(rejected.metrics["valid_candidate_count"], 0)
                self.assertEqual(rejected.metrics["new_evidence_count"], 0)
                self.assertEqual(rejected.metrics["result_count"], 0)
                self.assertEqual(rejected.metrics["rejected_candidate_count"], 1)
                self.assertEqual(
                    rejected.metrics["candidate_rejection_score_invalid_count"], 1
                )
                self.assertEqual(store.all(context.request_id), [])
                self.assertEqual(context.candidate_pool.accepted_candidates(), [])

                peer_context, peer_store, peer_registry = self._context(
                    capacity=1,
                    request_id=f"zero-rank-peer-{label}",
                )
                isolated = self._execute(
                    peer_context,
                    peer_registry,
                    [malformed, valid_peer],
                )
                self.assertEqual(isolated.metrics["valid_candidate_count"], 1)
                self.assertEqual(isolated.metrics["new_evidence_count"], 1)
                self.assertEqual(isolated.metrics["result_count"], 1)
                self.assertEqual(isolated.metrics["rejected_candidate_count"], 1)
                self.assertEqual(
                    isolated.metrics["candidate_rejection_score_invalid_count"], 1
                )
                self.assertEqual(
                    [item.symbol_name for item in peer_store.all(peer_context.request_id)],
                    ["upload_file"],
                )
                self.assertEqual(
                    [item.symbol_name for item in peer_context.candidate_pool.accepted_candidates()],
                    ["upload_file"],
                )

    def test_server_derived_zero_rank_sentinel_requires_the_complete_shape(self) -> None:
        direct = self._retrieval_result("authenticate_user")
        derived = replace(
            direct,
            retrieval_sources=["relation"],
            lexical_score=None,
            lexical_rank=None,
            semantic_score=None,
            semantic_rank=None,
            fusion_score=0.0,
            fusion_rank=0,
        )
        context, store, registry = self._context(
            capacity=1,
            request_id="derived-sentinel",
        )
        accepted = self._execute(context, registry, [derived])
        self.assertEqual(accepted.metrics["valid_candidate_count"], 1)
        self.assertEqual(accepted.metrics["new_evidence_count"], 1)
        self.assertEqual(len(store.all(context.request_id)), 1)

        context, store, registry = self._context(
            capacity=1,
            request_id="derived-malformed",
        )
        rejected = self._execute(
            context,
            registry,
            [replace(derived, lexical_rank=1)],
        )
        self.assertEqual(rejected.metrics["valid_candidate_count"], 0)
        self.assertEqual(rejected.metrics["rejected_candidate_count"], 1)
        self.assertEqual(store.all(context.request_id), [])

    def test_server_derived_zero_rank_requires_exact_raw_single_source_shape(self) -> None:
        direct = self._retrieval_result("authenticate_user")
        derived = replace(
            direct,
            lexical_score=None,
            lexical_rank=None,
            semantic_score=None,
            semantic_rank=None,
            fusion_score=0.0,
            fusion_rank=0,
        )
        cases = (
            (["hierarchy"], True),
            (["relation"], True),
            (["hierarchy", "relation"], False),
            (["hierarchy", "hierarchy"], False),
            (["relation", "relation"], False),
            (["unknown"], False),
            ([], False),
        )
        for index, (sources, expected_valid) in enumerate(cases):
            with self.subTest(sources=sources):
                context, store, registry = self._context(
                    capacity=1,
                    request_id=f"derived-source-shape-{index}",
                )
                observation = self._execute(
                    context,
                    registry,
                    [replace(derived, retrieval_sources=sources)],
                )
                self.assertEqual(
                    observation.metrics["valid_candidate_count"],
                    int(expected_valid),
                )
                self.assertEqual(
                    observation.metrics["rejected_candidate_count"],
                    int(not expected_valid),
                )
                self.assertEqual(len(store.all(context.request_id)), int(expected_valid))

    def test_equal_rank_score_tie_break_and_requests_are_stable_and_isolated(self) -> None:
        auth = replace(
            self._retrieval_result("authenticate_user"),
            fusion_score=1.0,
            lexical_score=1.0,
        )
        upload = replace(
            self._retrieval_result("upload_file"),
            fusion_score=1.0,
            lexical_score=1.0,
        )
        selected = []
        for index, batch in enumerate(([auth, upload], [upload, auth])):
            context, store, registry = self._context(
                capacity=1,
                request_id=f"tie-{index}",
            )
            self._execute(context, registry, batch)
            selected.append(store.all(context.request_id)[0].chunk_identity)
        self.assertEqual(selected[0], selected[1])

        first_context, first_store, first_registry = self._context(
            capacity=1,
            request_id="request-one",
        )
        second_context, second_store, second_registry = self._context(
            capacity=1,
            request_id="request-two",
        )
        self._execute(first_context, first_registry, [auth])
        self._execute(second_context, second_registry, [upload])
        self.assertEqual(first_store.all("request-two"), [])
        self.assertEqual(second_store.all("request-one"), [])
        self.assertNotEqual(
            first_store.all("request-one")[0].chunk_identity,
            second_store.all("request-two")[0].chunk_identity,
        )


if __name__ == "__main__":
    unittest.main()
