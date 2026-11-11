from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch

from app.database import Database
from app.services.candidate_evidence import CandidateEvidencePool
from app.services.hybrid_retriever import HybridSearchResult
from app.services.llm_client import ProviderError, provider_cancellation_scope
from app.services.retrieval_enhancement import retrieve_initial_code
from app.services.retrieval_v2 import RetrievalExecutionOutcome
from app.services.retrieval_v2 import retrieve_code
from app.services.smoke_diagnostics import SmokeDiagnosticsRecorder
from app.services.evidence import EvidenceBuilder
from app.services.agent_tools import build_tool_context, EvidenceStore, _search_code
from app.services.agent_contracts import AgentLimits, CancellationToken, SearchCodeInput
from app.config import LLMSettings, get_retrieval_enhancement_settings
from app.services.llm_client import LLMClient
import urllib.error
from tests.m1_helpers import make_project
from tests.m1_helpers import disabled_embedding_service


class RetrievalEnhancementTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.db = Database(Path(self.directory.name) / "test.sqlite")
        self.pid, self.bundle = make_project(self.db, [
            ("a.py", "alpha", "def alpha():\n    return 1\n"),
            ("b.py", "beta", "def beta():\n    return 2\n"),
            ("c.py", "gamma", "def gamma():\n    return 3\n")])
        self.items = [HybridSearchResult(
            **{key: c[key] for key in ("project_id", "repository_revision", "language", "path",
                                      "chunk_type", "symbol_name", "qualified_name", "start_line",
                                      "end_line", "content", "content_hash")},
            code_chunk_id=c["id"], retrieval_sources=["lexical"], lexical_rank=i,
            lexical_score=1.0 / i, fusion_score=1.0 / (60+i), fusion_rank=i)
            for i, c in enumerate(self.db.get_code_chunks(self.pid), 1)]
        self.base = RetrievalExecutionOutcome(self.items[:2], "lexical", [], "v1", "test")
        self.recorder = SmokeDiagnosticsRecorder()
        self.embedding = SimpleNamespace(settings=SimpleNamespace(enabled=True))
        self.calls = []
        def chat(messages, **kwargs):
            self.calls.append((messages, kwargs))
            if kwargs["purpose"] == "hyde":
                return '{"hypothesis":"HYPOTHETICAL_ONLY"}'
            ids = [x["id"] for x in json.loads(messages[1]["content"])["candidates"]]
            return json.dumps({"ids": list(reversed(ids))})
        self.llm = SimpleNamespace(available=True, chat=chat)

    def run_search(self, hyde=False, rerank=False, **kwargs):
        with patch.dict(os.environ, {"RETRIEVAL_HYDE_ENABLED": str(hyde),
                                     "RETRIEVAL_RERANK_ENABLED": str(rerank)}), \
             patch("app.services.retrieval_enhancement.retrieve_code", return_value=self.base) as search, \
             patch("app.services.retrieval_enhancement.run_bounded_semantic",
                   return_value=("completed", SimpleNamespace(results=[
                       replace(self.items[0], semantic_score=0.9),
                       replace(self.items[2], semantic_score=0.8)]), 1)):
            value = retrieve_initial_code(self.db, self.embedding, self.pid, "original question",
                llm=self.llm, evidence_count=2, diagnostics_recorder=self.recorder, **kwargs)
            return value, search.call_args

    def test_off_preserves_original_flow(self):
        value, args = self.run_search()
        self.assertIs(value, self.base)
        self.assertNotIn("candidate_limit", args.kwargs)
        self.assertEqual(self.calls, [])

    def test_independent_switches(self):
        for hyde, rerank, expected in [(True, False, ["hyde"]), (False, True, ["rerank"]),
                                      (True, True, ["hyde", "rerank"])]:
            self.calls.clear()
            value, args = self.run_search(hyde, rerank)
            self.assertEqual([x[1]["purpose"] for x in self.calls], expected)
            self.assertEqual(args.kwargs["candidate_limit"], 40)
            self.assertEqual(len(value.results), 2)

    def test_real_source_dedup_and_hypothesis_isolation(self):
        value, _ = self.run_search(True, True)
        audit = value.audit["enhancement"]
        self.assertEqual(len(audit["rerank_input_ids"]), 3)
        self.assertEqual(len(set(audit["rerank_input_ids"])), 3)
        self.assertNotIn("HYPOTHETICAL_ONLY", json.dumps(self.calls[1][0]))
        self.assertNotIn("HYPOTHETICAL_ONLY", json.dumps(self.recorder.snapshot()))
        self.assertTrue(all(x.content.startswith("def ") for x in value.results))

    def test_ranking_survives_canonical_promotion_without_rewriting_scores(self):
        value, _ = self.run_search(False, True)
        pool = CandidateEvidencePool(2)
        candidates = pool.normalize_retrieval_results(value.results, provenance="deterministic_base_retrieval")
        promotion = pool.promote(candidates, database=self.db, project=self.bundle["project"],
            project_id=self.pid, repository_revision=self.bundle["project"]["repository_revision"],
            retrieval_strategy_version="test")
        self.assertEqual([x.code_chunk_id for x in promotion.evidence],
                         [x.code_chunk_id for x in reversed(self.items[:2])])
        self.assertEqual(promotion.evidence[0].fusion_rank, 2)
        self.assertEqual(promotion.evidence[0].rerank_rank, 1)
        self.assertEqual(promotion.evidence[0].fusion_score, self.items[1].fusion_score)

    def test_hyde_contributes_once_for_existing_and_new_candidates(self):
        # alpha is hit by the original lexical+dense routes and by HyDE;
        # gamma is found only by HyDE. beta's original rank must beat gamma
        # on an equal one-route score instead of losing to a doubled score.
        original_alpha = replace(self.items[0], semantic_score=0.7, semantic_rank=2,
            retrieval_sources=["lexical", "semantic"], fusion_score=1/61 + 1/62)
        self.base = replace(self.base, results=[original_alpha, self.items[1]])
        value, _ = self.run_search(True, False)
        self.assertEqual(value.audit["enhancement"]["rerank_input_ids"],
                         [x.code_chunk_id for x in self.items])
        alpha = value.results[0]
        self.assertEqual(alpha.lexical_score, original_alpha.lexical_score)
        self.assertEqual(alpha.lexical_rank, original_alpha.lexical_rank)
        self.assertEqual(alpha.semantic_score, original_alpha.semantic_score)
        self.assertEqual(alpha.semantic_rank, original_alpha.semantic_rank)
        self.assertEqual(alpha.fusion_score, original_alpha.fusion_score)
        self.assertEqual(alpha.fusion_rank, original_alpha.fusion_rank)

    def test_hyde_only_candidate_keeps_one_score_and_passes_canonical_read(self):
        # A weak original candidate should still lose to a genuinely stronger
        # HyDE-only hit; the fix must not turn new hits into zero-rank sentinels.
        self.base = replace(self.base, results=[self.items[0],
            replace(self.items[1], fusion_score=1/80)])
        value, _ = self.run_search(True, False)
        gamma = value.results[1]
        self.assertEqual(gamma.code_chunk_id, self.items[2].code_chunk_id)
        self.assertAlmostEqual(gamma.fusion_score, 1/62)
        self.assertEqual(gamma.semantic_rank, 2)
        self.assertEqual(gamma.fusion_rank, 3)
        pool = CandidateEvidencePool(2)
        candidates = pool.normalize_retrieval_results(value.results, provenance="deterministic_base_retrieval")
        promotion = pool.promote(candidates, database=self.db, project=self.bundle["project"],
            project_id=self.pid, repository_revision=self.bundle["project"]["repository_revision"],
            retrieval_strategy_version="test")
        self.assertEqual(promotion.rejection_counts, {})
        self.assertEqual(promotion.evidence[1].excerpt, self.items[2].content)
        self.assertEqual(promotion.evidence[1].content_hash, self.items[2].content_hash)

    def test_invalid_ids_duplicates_types_and_empty_fall_back(self):
        for ids in [[999], [True, False], [], [1, 1]]:
            self.llm.chat = lambda *a, **k: json.dumps({"ids": ids})
            value, _ = self.run_search(False, True)
            self.assertEqual([x.code_chunk_id for x in value.results], [x.code_chunk_id for x in self.items[:2]])
            self.assertEqual(value.audit["enhancement"]["rerank"]["status"], "fallback")

    def test_timeout_fallback_and_short_parameters(self):
        def fail(*a, **kw):
            self.assertEqual(kw["thinking"], "disabled")
            self.assertEqual(kw["temperature"], 0)
            self.assertLessEqual(kw["timeout_seconds"], 7)
            raise ProviderError("deadline_exceeded", "timeout")
        self.llm.chat = fail
        value, _ = self.run_search(True, True)
        self.assertEqual(value.audit["enhancement"]["hyde"]["status"], "fallback")
        self.assertEqual(value.audit["enhancement"]["rerank"]["status"], "fallback")

    def test_exhausted_shared_budget_skips_provider(self):
        value, _ = self.run_search(True, True, enhancement_deadline_at=time.monotonic()-1)
        self.assertEqual(self.calls, [])
        self.assertEqual(value.audit["enhancement"]["hyde"]["reason"], "budget_exhausted")

    def test_cancel_propagates(self):
        with provider_cancellation_scope(lambda: True):
            with self.assertRaises(ProviderError) as error:
                self.run_search(True, True)
        self.assertEqual(error.exception.code, "cancelled")

    def test_outer_json_fence_and_valid_prefix_are_usable_once(self):
        expected = self.items[1].code_chunk_id
        self.llm.chat = lambda *a, **k: '```json\n' + json.dumps({"ids": [expected]}) + '\n```'
        value, _ = self.run_search(False, True)
        self.assertEqual(value.results[0].code_chunk_id, expected)
        self.assertEqual(value.audit["enhancement"]["rerank"]["status"], "succeeded")
        self.assertEqual(value.audit["enhancement"]["rerank"]["appended_candidates"], 1)

    def test_only_primary_function_uses_bounded_complete_source(self):
        content = 'def primary():\n' + '    # bounded source body\n' * 90 + '    return "TAIL"\n'
        primary = replace(self.items[0], content=content, rerank_rank=1, start_line=1,
                          end_line=len(content.splitlines()))
        second = replace(primary, code_chunk_id=999, rerank_rank=2)
        built = EvidenceBuilder(primary_excerpt_limit=8192).build([primary, second], self.bundle["project"])
        self.assertEqual(built[0].excerpt, content)
        self.assertEqual(built[0].planner_excerpt_coverage()["truncated"], False)
        self.assertLess(len(built[1].excerpt), len(content))
        old = EvidenceBuilder().build([primary], self.bundle["project"])
        self.assertLess(len(old[0].excerpt), len(content))

    def test_provider_enhancement_has_one_attempt_despite_global_retries(self):
        attempts = []
        def opener(*a, **k):
            attempts.append(1)
            raise urllib.error.URLError("offline test")
        llm = LLMClient(LLMSettings(provider="openai_compatible", base_url="https://provider.invalid",
            api_key="test-placeholder", model="test", max_retries=2), opener=opener)
        for purpose in ("hyde", "rerank"):
            attempts.clear()
            with self.assertRaises(ProviderError):
                llm.chat([{"role": "user", "content": "test"}], purpose=purpose)
            self.assertEqual(len(attempts), 1)

    def test_nonfinite_budget_cannot_remove_shared_cap(self):
        for raw in ("nan", "inf", "100"):
            with patch.dict(os.environ, {"RETRIEVAL_ENHANCEMENT_SECONDS": raw}):
                self.assertLessEqual(get_retrieval_enhancement_settings().budget_seconds, 15)

    def test_planner_supplement_does_not_repeat_initial_enhancement(self):
        context = build_tool_context(request_id="first-only", bundle=self.bundle,
            database=self.db, embedding_service=disabled_embedding_service(),
            evidence_store=EvidenceStore(8), limits=AgentLimits(), cancellation=CancellationToken(),
            deadline_monotonic=time.monotonic()+60)
        with patch.dict(os.environ, {"RETRIEVAL_HYDE_ENABLED": "true", "RETRIEVAL_RERANK_ENABLED": "true"}), \
             patch("app.services.agent_tools.retrieve_initial_code", return_value=self.base) as initial, \
             patch("app.services.agent_tools.retrieve_code", return_value=self.base) as plain:
            context.candidate_provenance = "deterministic_base_retrieval"
            _search_code(context, SearchCodeInput(query="alpha", top_k=2))
            context.candidate_provenance = "planner_search_code"
            _search_code(context, SearchCodeInput(query="beta", top_k=2))
            self.assertEqual(initial.call_count, 1)
            self.assertEqual(plain.call_count, 1)

    def test_expanded_pool_keeps_v2_hierarchy_and_relation_size_contracts(self):
        for hierarchy in ("off", "normalize_v1"):
            for relation in ("off", "expand_v1"):
                value = retrieve_code(self.db, disabled_embedding_service(), self.pid, "alpha beta gamma",
                    retrieval_version="v2", evidence_count=2, candidate_limit=40,
                    hierarchy_mode=hierarchy, relation_mode=relation)
                self.assertGreaterEqual(len(value.results), 2)
                self.assertEqual(len({x.code_chunk_id for x in value.results}), len(value.results))


if __name__ == "__main__":
    unittest.main()
