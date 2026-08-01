from __future__ import annotations

import math
import unittest

from app.m5.contracts import Scenario
from app.retrieval_phase5.metrics import (
    aggregate_path_records,
    classify_failures,
    compare_paths,
    containment_match,
    evaluate_query,
    strict_gold_match,
)


def _scenario(*, multi_gold: bool = False, unanswerable: bool = False) -> Scenario:
    spans = [] if unanswerable else [
        {"path": "src/mod.py", "qualified_symbol": "target", "start_line": 10, "end_line": 12, "content_hash": "a" * 64}
    ]
    if multi_gold:
        spans.append(
            {"path": "src/other.py", "qualified_symbol": "other", "start_line": 20, "end_line": 24, "content_hash": "b" * 64}
        )
    return Scenario.model_validate(
        {
            "scenario_id": "click-case-1",
            "dataset_version": "pilot-v1",
            "repo_id": "click",
            "repository_revision": "1" * 40,
            "language": "python",
            "question": "Where is the target?",
            "category": "unanswerable" if unanswerable else "locate",
            "difficulty": "hard",
            "expected_target_type": "none" if unanswerable else "symbol",
            "expected_files": [] if unanswerable else [item["path"] for item in spans],
            "expected_symbols": [] if unanswerable else [item["qualified_symbol"] for item in spans],
            "expected_source_spans": spans,
            "expected_content_hashes": [] if unanswerable else [item["content_hash"] for item in spans],
            "expected_relation_edges": [],
            "expected_key_points": ["insufficient"] if unanswerable else ["target"],
            "unanswerable": unanswerable,
            "allowed_evidence_scope": {"paths": [], "repository_only": True},
            "maximum_steps": 5,
            "maximum_tool_calls": 8,
            "annotation_provenance": "user_confirmed",
            "annotation_status": "human_reviewed",
            "annotation_reviewed_at": "2026-07-27",
            "annotation_review_method": "codex_conversation",
            "annotation_note": "fixed",
        }
    )


def _candidate(**changes):
    value = {
        "repository_revision": "1" * 40,
        "path": "src/mod.py",
        "qualified_name": "target",
        "start_line": 10,
        "end_line": 12,
        "content_hash": "a" * 64,
        "chunk_identity": "chunk-1",
        "validation_status": "valid",
        "retrieval_sources": ["lexical"],
    }
    value.update(changes)
    return value


class RetrievalPhase5MetricsTests(unittest.TestCase):
    def test_strict_match_requires_revision_path_symbol_span_hash_and_validity(self):
        scenario = _scenario()
        self.assertTrue(strict_gold_match(_candidate(), scenario)[0])
        for changes in (
            {"repository_revision": "2" * 40}, {"path": "src/other.py"},
            {"qualified_name": "other"}, {"start_line": 9}, {"end_line": 13},
            {"content_hash": "b" * 64}, {"validation_status": "invalid"},
        ):
            with self.subTest(changes=changes):
                self.assertFalse(strict_gold_match(_candidate(**changes), scenario)[0])

    def test_containment_is_diagnostic_only(self):
        scenario = _scenario()
        containing = _candidate(start_line=1, end_line=30, content_hash="c" * 64)
        self.assertFalse(strict_gold_match(containing, scenario)[0])
        self.assertTrue(containment_match(containing, scenario)[0])

    def test_metrics_cover_multi_gold_empty_skip_and_top_k(self):
        record = evaluate_query(_scenario(multi_gold=True), [_candidate()], top_k=8)
        self.assertEqual(record["metrics"]["hit_at_1"], 1.0)
        self.assertEqual(record["metrics"]["hit_at_10"], 1.0)
        self.assertEqual(record["metrics"]["mrr_at_10"], 1.0)
        self.assertEqual(record["metrics"]["recall_at_8"], 0.5)
        self.assertTrue(record["metrics"]["hit_at_10_disclosure"])
        empty = evaluate_query(_scenario(), [], top_k=8)
        self.assertEqual(empty["metrics"]["hit_at_8"], 0.0)
        self.assertEqual(empty["metrics"]["mrr_at_10"], 0.0)
        skipped = evaluate_query(_scenario(unanswerable=True), [], top_k=8)
        self.assertTrue(skipped["skipped"])
        self.assertIsNone(skipped["metrics"])

    def test_aggregate_is_query_order_independent_and_rejects_nonfinite(self):
        first = evaluate_query(_scenario(), [_candidate()], top_k=8)
        miss_scenario = _scenario().model_copy(update={"scenario_id": "click-case-2"})
        second = evaluate_query(miss_scenario, [], top_k=8)
        self.assertEqual(
            aggregate_path_records([first, second]),
            aggregate_path_records([second, first]),
        )
        broken = {**first, "metrics": {**first["metrics"], "mrr_at_10": math.nan}}
        with self.assertRaises(ValueError):
            aggregate_path_records([broken])

    def test_paired_delta_relation_gain_loss_and_support_not_double_counted(self):
        scenario = _scenario()
        miss = evaluate_query(scenario, [], top_k=8)
        hit = evaluate_query(scenario, [_candidate(retrieval_sources=["relation"])], top_k=8)
        miss["path_id"] = "B"
        hit["path_id"] = "D"
        comparison = compare_paths([miss], [hit], left_path="B", right_path="D", seed=7, samples=200)
        self.assertEqual(comparison["improved_queries"], 1)
        self.assertEqual(comparison["relation_new_gold_gain_at_8"], 1)
        reverse = compare_paths([hit], [miss], left_path="D", right_path="B", seed=7, samples=200)
        self.assertEqual(reverse["relation_gold_loss_at_8"], 1)
        supported = evaluate_query(
            scenario,
            [_candidate(retrieval_sources=["lexical", "relation_support"])],
            top_k=8,
        )
        self.assertEqual(supported["relation_origin_gold_hits"], 0)
        self.assertEqual(supported["relation_assisted_gold_hits"], 1)

    def test_failure_taxonomy_is_deterministic(self):
        scenario = _scenario()
        base = evaluate_query(scenario, [], top_k=8)
        hit = evaluate_query(scenario, [_candidate()], top_k=8)
        records = []
        for path_id, record in zip("ABCDE", [base, hit, hit, base, hit]):
            records.append({**record, "path_id": path_id, "warnings": []})
        first = classify_failures(records)
        second = classify_failures(list(reversed(records)))
        self.assertEqual(first, second)
        self.assertIn("v2 fixes v1", first["categories"])
        self.assertIn("relation noise", first["categories"])


if __name__ == "__main__":
    unittest.main()
