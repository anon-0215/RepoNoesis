from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from app.database import Database
from app.services.lexical_retriever import LexicalSearchResult
from app.services.query_analyzer import QueryAnalyzer, RetrievalRoutingHint
from app.services.retrieval_v2 import (
    RETRIEVAL_VERSION_V2,
    V2_FUSION_VERSION,
    RetrievalContractError,
    RetrievalV2Config,
    RetrievalV2Orchestrator,
)
from app.services.semantic_retriever import SemanticSearchOutcome, SemanticSearchResult
from app.services.symbol_retriever import SymbolSearchResult
from tests.m1_helpers import make_project


class RecordingLexicalRetriever:
    def __init__(self, results=None, error=None):
        self.results = list(results or [])
        self.error = error
        self.calls = []

    def search(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.error is not None:
            raise self.error
        return list(self.results)


class RecordingSemanticRetriever:
    def __init__(self, results=None, error=None):
        self.results = list(results or [])
        self.error = error
        self.calls = []

    def search(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.error is not None:
            raise self.error
        return SemanticSearchOutcome(
            status="ok" if self.results else "no_embeddings",
            results=list(self.results),
            model_name="fake-model",
            total_candidates=len(self.results),
            warnings=[] if self.results else ["fake dense source returned no candidates"],
        )


class RecordingSymbolRetriever:
    def __init__(self, results_by_query=None, default=None, error=None):
        self.results_by_query = dict(results_by_query or {})
        self.default = list(default or [])
        self.error = error
        self.calls = []

    def search(self, project_id, query, **kwargs):
        self.calls.append((project_id, query, kwargs))
        if self.error is not None:
            raise self.error
        return list(self.results_by_query.get(query, self.default))


class FixedAnalyzer:
    def __init__(self, analysis):
        self.analysis = analysis
        self.calls = []

    def analyze(self, query):
        self.calls.append(query)
        return self.analysis


def lexical_result(chunk, project_id, score=3.0, rank=1):
    return LexicalSearchResult(
        project_id=project_id,
        repository_revision=chunk["repository_revision"],
        code_chunk_id=int(chunk["id"]),
        language=chunk["language"],
        path=chunk["path"],
        chunk_type=chunk["chunk_type"],
        symbol_name=chunk["symbol_name"],
        qualified_name=chunk["qualified_name"],
        start_line=int(chunk["start_line"]),
        end_line=int(chunk["end_line"]),
        content=chunk["content"],
        content_hash=chunk["content_hash"],
        lexical_score=score,
        lexical_rank=rank,
    )


def semantic_result(chunk, project_id, score=0.9):
    return SemanticSearchResult(
        project_id=project_id,
        repository_revision=chunk["repository_revision"],
        code_chunk_id=int(chunk["id"]),
        language=chunk["language"],
        path=chunk["path"],
        chunk_type=chunk["chunk_type"],
        symbol_name=chunk["symbol_name"],
        qualified_name=chunk["qualified_name"],
        start_line=int(chunk["start_line"]),
        end_line=int(chunk["end_line"]),
        content=chunk["content"],
        content_hash=chunk["content_hash"],
        semantic_score=score,
        model_name="fake-model",
    )


def symbol_result(chunk, project_id, score=1.0, rank=1, reasons=("qualified_symbol_exact",)):
    identity = "|".join(
        str(value)
        for value in (
            project_id,
            chunk["repository_revision"],
            chunk["path"],
            chunk["start_line"],
            chunk["end_line"],
            chunk["content_hash"],
            chunk["id"],
        )
    )
    return SymbolSearchResult(
        project_id=project_id,
        repository_revision=chunk["repository_revision"],
        code_chunk_id=int(chunk["id"]),
        chunk_identity=identity,
        language=chunk["language"],
        path=chunk["path"],
        chunk_type=chunk["chunk_type"],
        symbol_name=chunk["symbol_name"],
        qualified_name=chunk["qualified_name"],
        start_line=int(chunk["start_line"]),
        end_line=int(chunk["end_line"]),
        content=chunk["content"],
        content_hash=chunk["content_hash"],
        symbol_match_type="exact_qualified",
        symbol_score=score,
        symbol_rank=rank,
        match_reasons=tuple(reasons),
    )


class RetrievalV2Tests(unittest.TestCase):
    def test_deadline_check_prevents_starting_dense_after_lexical(self):
        checks = 0

        def check_active():
            nonlocal checks
            checks += 1
            if checks >= 3:
                raise RuntimeError("deadline marker")

        lexical = RecordingLexicalRetriever([])
        dense = RecordingSemanticRetriever([])
        orchestrator = RetrievalV2Orchestrator(
            self.database,
            self.enabled_embedding,
            lexical_retriever=lexical,
            semantic_retriever=dense,
            symbol_retriever=RecordingSymbolRetriever(),
        )

        with self.assertRaisesRegex(RuntimeError, "deadline marker"):
            orchestrator.search(
                self.project_id,
                "target",
                check_active=check_active,
            )

        self.assertEqual(len(lexical.calls), 1)
        self.assertEqual(dense.calls, [])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.directory.name) / "retrieval-v2.sqlite")
        self.project_id, _bundle = make_project(
            self.database,
            [
                ("src/target.py", "target", "def target():\n    return helper()\n"),
                ("src/helper.py", "helper", "def helper():\n    return 1\n"),
                ("src/nested.py", "Outer.target", "def target(self):\n    return 2\n"),
            ],
        )
        self.chunks = {
            chunk["qualified_name"]: chunk
            for chunk in self.database.get_code_chunks(self.project_id)
        }
        self.enabled_embedding = SimpleNamespace(settings=SimpleNamespace(enabled=True))

    def tearDown(self):
        self.directory.cleanup()

    def _orchestrator(self, *, lexical=None, semantic=None, symbol=None, analyzer=None):
        return RetrievalV2Orchestrator(
            self.database,
            self.enabled_embedding,
            lexical_retriever=lexical or RecordingLexicalRetriever(),
            semantic_retriever=semantic or RecordingSemanticRetriever(),
            symbol_retriever=symbol or RecordingSymbolRetriever(),
            query_analyzer=analyzer,
        )

    def test_three_sources_merge_by_exact_identity_and_record_rrf_contributions(self):
        chunk = self.chunks["target"]
        lexical = RecordingLexicalRetriever([lexical_result(chunk, self.project_id)])
        semantic = RecordingSemanticRetriever([semantic_result(chunk, self.project_id)])
        symbol = RecordingSymbolRetriever(
            results_by_query={"target": [symbol_result(chunk, self.project_id)]}
        )
        outcome = self._orchestrator(
            lexical=lexical, semantic=semantic, symbol=symbol
        ).search(self.project_id, "How does `target` work?", evidence_count=5)

        self.assertEqual(outcome.retrieval_version, RETRIEVAL_VERSION_V2)
        self.assertEqual(outcome.retrieval_strategy_version, V2_FUSION_VERSION)
        self.assertEqual(len(outcome.results), 1)
        self.assertEqual(outcome.results[0].retrieval_sources, ["dense", "lexical", "symbol"])
        self.assertEqual(len(lexical.calls), 1)
        self.assertEqual(len(semantic.calls), 1)
        self.assertEqual(len(symbol.calls), 1)
        candidate = outcome.audit["candidates"][0]
        self.assertEqual(candidate["merged_sources"], ["dense", "lexical", "symbol"])
        self.assertEqual(set(candidate["source_records"]), {"dense", "lexical", "symbol"})
        self.assertAlmostEqual(
            sum(candidate["fusion_contributions"].values()),
            candidate["fused_score"],
        )
        expected = sum(
            outcome.audit["effective_policy"]["source_weights"][source]
            / (outcome.audit["rrf_k"] + 1)
            for source in ("dense", "lexical", "symbol")
        )
        self.assertAlmostEqual(candidate["fused_score"], expected)

    def test_all_intents_keep_three_positive_weights_and_budgets(self):
        queries = (
            "Where is `target` defined?",
            "How does `target` work?",
            "What changes if `target` is modified?",
            "Who calls `target`?",
            "Where is `target` defined and who calls it?",
            "Tell me about this project.",
        )
        for query in queries:
            with self.subTest(query=query):
                lexical = RecordingLexicalRetriever()
                semantic = RecordingSemanticRetriever()
                symbol = RecordingSymbolRetriever()
                outcome = self._orchestrator(
                    lexical=lexical, semantic=semantic, symbol=symbol
                ).search(self.project_id, query)
                policy = outcome.audit["effective_policy"]
                self.assertTrue(all(value > 0 for value in policy["source_weights"].values()))
                self.assertTrue(all(value > 0 for value in policy["source_budgets"].values()))
                self.assertEqual(len(lexical.calls), 1)
                self.assertEqual(len(semantic.calls), 1)
                self.assertGreaterEqual(len(symbol.calls), 1)
                if outcome.audit["query_analysis"]["primary_intent"] in {"mixed", "unknown"}:
                    self.assertTrue(outcome.audit["query_analysis"]["neutral_fallback"])
                    self.assertEqual(
                        policy["source_weights"],
                        {"dense": 1.0, "lexical": 1.0, "symbol": 1.0},
                    )

    def test_multiple_and_duplicate_symbol_hints_are_bounded_and_auditable(self):
        target = self.chunks["target"]
        helper = self.chunks["helper"]
        symbol = RecordingSymbolRetriever(
            results_by_query={
                "target": [symbol_result(target, self.project_id)],
                "helper": [
                    symbol_result(helper, self.project_id),
                    symbol_result(target, self.project_id),
                ],
            }
        )
        outcome = self._orchestrator(symbol=symbol).search(
            self.project_id,
            "Explain `target`, `helper`, and target().",
        )

        self.assertEqual([call[1] for call in symbol.calls], ["target", "helper"])
        self.assertLessEqual(
            outcome.audit["sources"]["symbol"]["candidate_count"],
            outcome.audit["effective_policy"]["source_budgets"]["symbol"],
        )
        target_audit = next(
            item for item in outcome.audit["candidates"] if item["qualified_name"] == "target"
        )
        self.assertEqual(
            target_audit["source_records"]["symbol"]["metadata"]["matched_hints"],
            ["target", "helper"],
        )

    def test_symbol_hint_call_count_has_an_independent_hard_limit(self):
        hints = [f"symbol_{index}" for index in range(12)]
        query = "Explain " + ", ".join(f"`{hint}`" for hint in hints)
        symbol = RecordingSymbolRetriever()
        outcome = RetrievalV2Orchestrator(
            self.database,
            self.enabled_embedding,
            lexical_retriever=RecordingLexicalRetriever(),
            semantic_retriever=RecordingSemanticRetriever(),
            symbol_retriever=symbol,
            config=RetrievalV2Config(max_symbol_hints=4),
        ).search(self.project_id, query)

        self.assertEqual(len(symbol.calls), 4)
        self.assertEqual(outcome.audit["sources"]["symbol"]["hint_count"], 12)
        self.assertEqual(outcome.audit["sources"]["symbol"]["executed_hint_count"], 4)
        self.assertTrue(outcome.audit["sources"]["symbol"]["hints_truncated"])
        self.assertTrue(any("truncated" in warning.lower() for warning in outcome.warnings))

    def test_no_reliable_symbol_hint_is_distinct_from_an_empty_match(self):
        symbol = RecordingSymbolRetriever()
        query = "Tell me something useful about this project."
        outcome = self._orchestrator(symbol=symbol).search(self.project_id, query)

        self.assertEqual(len(symbol.calls), 1)
        self.assertEqual(symbol.calls[0][1], query)
        self.assertEqual(outcome.audit["sources"]["symbol"]["status"], "no_reliable_hint")
        self.assertEqual(outcome.audit["sources"]["symbol"]["candidate_count"], 0)

    def test_invalid_analyzer_values_use_neutral_safe_policy_without_dropping_sources(self):
        analysis = QueryAnalyzer().analyze("How does `target` work?")
        invalid = replace(
            analysis,
            routing_hint=RetrievalRoutingHint(
                dense_weight=math.nan,
                lexical_weight=-1.0,
                symbol_weight=0.0,
                relation_direction="both",
                relation_budget=0,
                candidate_pool=0,
            ),
        )
        lexical = RecordingLexicalRetriever()
        semantic = RecordingSemanticRetriever()
        symbol = RecordingSymbolRetriever()
        outcome = self._orchestrator(
            lexical=lexical,
            semantic=semantic,
            symbol=symbol,
            analyzer=FixedAnalyzer(invalid),
        ).search(self.project_id, "How does `target` work?")

        self.assertEqual(
            outcome.audit["effective_policy"]["source_weights"],
            {"dense": 1.0, "lexical": 1.0, "symbol": 1.0},
        )
        self.assertTrue(
            all(
                value > 0
                for value in outcome.audit["effective_policy"]["source_budgets"].values()
            )
        )
        self.assertTrue(any("query analysis" in warning.lower() for warning in outcome.warnings))

    def test_dense_controlled_failure_is_visible_but_lexical_failure_is_not_swallowed(self):
        dense_failure = self._orchestrator(
            semantic=RecordingSemanticRetriever(error=RuntimeError("dense failed"))
        ).search(self.project_id, "target")
        self.assertEqual(dense_failure.audit["sources"]["dense"]["status"], "unavailable")
        self.assertEqual(dense_failure.audit["sources"]["dense"]["error_type"], "RuntimeError")
        self.assertTrue(any("RuntimeError" in warning for warning in dense_failure.warnings))

        with self.assertRaises(RuntimeError):
            self._orchestrator(
                lexical=RecordingLexicalRetriever(error=RuntimeError("lexical failed"))
            ).search(self.project_id, "target")

    def test_parent_child_and_same_named_chunks_remain_distinct(self):
        parent = self.chunks["target"]
        child = self.chunks["Outer.target"]
        outcome = self._orchestrator(
            lexical=RecordingLexicalRetriever(
                [
                    lexical_result(parent, self.project_id, rank=1),
                    lexical_result(child, self.project_id, rank=2),
                ]
            ),
            semantic=RecordingSemanticRetriever(
                [semantic_result(child, self.project_id)]
            ),
        ).search(self.project_id, "target", evidence_count=8)

        identities = [item["chunk_identity"] for item in outcome.audit["candidates"]]
        self.assertEqual(len(identities), 2)
        self.assertEqual(len(set(identities)), 2)

    def test_same_source_duplicates_are_idempotent_and_raw_scores_are_not_summed(self):
        chunk = self.chunks["target"]
        duplicate = lexical_result(chunk, self.project_id, score=999.0, rank=1)
        outcome = self._orchestrator(
            lexical=RecordingLexicalRetriever([duplicate, duplicate])
        ).search(self.project_id, "target", evidence_count=8)

        self.assertEqual(len(outcome.results), 1)
        audit = outcome.audit["candidates"][0]
        self.assertEqual(audit["source_records"]["lexical"]["raw_score"], 999.0)
        self.assertAlmostEqual(
            audit["fused_score"],
            outcome.audit["effective_policy"]["source_weights"]["lexical"]
            / (outcome.audit["rrf_k"] + 1),
        )
        self.assertNotAlmostEqual(audit["fused_score"], 999.0)

    def test_final_top_k_is_bounded_after_fusion(self):
        outcome = self._orchestrator(
            lexical=RecordingLexicalRetriever(
                [
                    lexical_result(self.chunks["target"], self.project_id, rank=1),
                    lexical_result(self.chunks["helper"], self.project_id, rank=2),
                    lexical_result(self.chunks["Outer.target"], self.project_id, rank=3),
                ]
            )
        ).search(self.project_id, "target", evidence_count=1)

        self.assertEqual(len(outcome.results), 1)
        self.assertEqual(outcome.audit["limits"]["final_top_k"], 1)

    def test_disabled_dense_source_is_auditable_controlled_unavailability(self):
        outcome = RetrievalV2Orchestrator(
            self.database,
            SimpleNamespace(settings=SimpleNamespace(enabled=False)),
            lexical_retriever=RecordingLexicalRetriever(),
            semantic_retriever=RecordingSemanticRetriever(
                error=AssertionError("disabled dense source must not encode")
            ),
            symbol_retriever=RecordingSymbolRetriever(),
        ).search(self.project_id, "target")

        self.assertEqual(outcome.audit["sources"]["dense"]["status"], "disabled")
        self.assertGreater(outcome.audit["sources"]["dense"]["budget"], 0)
        self.assertTrue(any("controlled-unavailable" in warning for warning in outcome.warnings))

    def test_same_database_chunk_with_conflicting_metadata_fails(self):
        chunk = self.chunks["target"]
        conflicting = replace(
            semantic_result(chunk, self.project_id),
            path="src/conflict.py",
        )
        with self.assertRaises(RetrievalContractError):
            self._orchestrator(
                lexical=RecordingLexicalRetriever(
                    [lexical_result(chunk, self.project_id)]
                ),
                semantic=RecordingSemanticRetriever([conflicting]),
            ).search(self.project_id, "target")

    def test_input_order_does_not_change_ranked_output_when_source_ranks_are_stable(self):
        target = lexical_result(self.chunks["target"], self.project_id, rank=2)
        helper = lexical_result(self.chunks["helper"], self.project_id, rank=1)
        first = self._orchestrator(
            lexical=RecordingLexicalRetriever([target, helper])
        ).search(self.project_id, "missing", evidence_count=8)
        second = self._orchestrator(
            lexical=RecordingLexicalRetriever([helper, target])
        ).search(self.project_id, "missing", evidence_count=8)

        self.assertEqual(
            [(item.code_chunk_id, item.fusion_rank) for item in first.results],
            [(item.code_chunk_id, item.fusion_rank) for item in second.results],
        )

    def test_config_rejects_unknown_fusion_and_invalid_numeric_limits(self):
        with self.assertRaises(ValueError):
            RetrievalV2Config(fusion_version="unknown")
        with self.assertRaises(ValueError):
            RetrievalV2Config(rrf_k=0)
        with self.assertRaises(ValueError):
            RetrievalV2Config(min_source_weight=math.inf)
        with self.assertRaises(ValueError):
            RetrievalV2Config(max_final_top_k=0)
        with self.assertRaises(ValueError):
            RetrievalV2Config(max_symbol_hints=0)


if __name__ == "__main__":
    unittest.main()
