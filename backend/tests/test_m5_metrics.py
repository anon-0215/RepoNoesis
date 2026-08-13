from __future__ import annotations

import math
import unittest

from app.m5.contracts import Scenario
from app.m5.metrics import aggregate_results, paired_delta, scenario_metrics
from m5_helpers import SOURCE_HASH


def scenario(unanswerable: bool = False) -> Scenario:
    return Scenario.model_validate({
        "scenario_id": "metric-scenario", "dataset_version": "v1", "repo_id": "repo-a",
        "repository_revision": "a" * 40, "language": "python", "question": "target",
        "category": "unanswerable" if unanswerable else "locate", "difficulty": "easy",
        "expected_target_type": "none" if unanswerable else "symbol",
        "expected_files": [] if unanswerable else ["app.py"],
        "expected_symbols": [] if unanswerable else ["target"],
        "expected_source_spans": [] if unanswerable else [{"path": "app.py", "qualified_symbol": "target", "start_line": 1, "end_line": 2, "content_hash": SOURCE_HASH}],
        "expected_content_hashes": [] if unanswerable else [SOURCE_HASH],
        "expected_relation_edges": [], "expected_key_points": ["target"], "unanswerable": unanswerable,
        "allowed_evidence_scope": {"paths": [], "repository_only": True},
        "maximum_steps": 5, "maximum_tool_calls": 8,
        "annotation_provenance": "agent_assisted_developer_curation",
        "annotation_status": "agent_curated_pending_human_review", "annotation_note": "fixture",
    })


def result(rank: int = 1):
    wrong = [{"evidence_id": f"E{i}", "path": "wrong.py", "qualified_name": "wrong", "start_line": 1, "end_line": 1,
              "content_hash": "0" * 64, "repository_revision": "a" * 40, "validation_status": "valid"} for i in range(1, rank)]
    target = {"evidence_id": f"E{rank}", "path": "app.py", "qualified_name": "target", "start_line": 1, "end_line": 2,
              "content_hash": SOURCE_HASH, "repository_revision": "a" * 40, "validation_status": "valid"}
    return {"answer": f"target [E{rank}]", "evidence": [*wrong, target], "grounding_status": "grounded",
            "scenario_status": "succeeded", "latency_ms": 10, "agent_trace": []}


class M5MetricTests(unittest.TestCase):
    def test_hand_calculated_hit_mrr_ndcg_and_recall(self):
        metrics = scenario_metrics(scenario(), result(2))
        self.assertEqual(metrics["hit_at_1"], 0)
        self.assertEqual(metrics["hit_at_5"], 1)
        self.assertEqual(metrics["mrr_at_10"], 0.5)
        self.assertAlmostEqual(metrics["evidence_precision"], 0.5)
        self.assertEqual(metrics["evidence_recall"], 1)
        self.assertEqual(metrics["expected_span_recall"], 1)

    def test_ndcg_stays_bounded_when_multiple_chunks_match_one_gold_file(self):
        payload = result(1)
        payload["evidence"].append(
            {
                **payload["evidence"][0],
                "evidence_id": "E2",
                "start_line": 3,
                "end_line": 4,
                "content_hash": "b" * 64,
            }
        )
        metrics = scenario_metrics(scenario(), payload)
        self.assertGreaterEqual(metrics["ndcg_at_10"], 0.0)
        self.assertLessEqual(metrics["ndcg_at_10"], 1.0)

    def test_correct_abstention(self):
        metrics = scenario_metrics(scenario(True), {
            "answer": "当前源码证据不足，无法可靠回答。", "evidence": [],
            "grounding_status": "insufficient_evidence", "latency_ms": 1,
        })
        self.assertEqual(metrics["correct_abstention"], 1)

    def test_failed_results_do_not_enter_success_denominator(self):
        records = [
            {"scenario_status": "succeeded", "repo_id": "r", "category": "c", "metrics": {"hit_at_5": 1, "latency_ms": 10}},
            {"scenario_status": "failed", "repo_id": "r", "category": "c", "metrics": {"hit_at_5": 0, "latency_ms": 999}},
        ]
        summary = aggregate_results(records)
        self.assertEqual(summary["means"]["hit_at_5"], 1)
        self.assertEqual(summary["successful_count"], 1)
        self.assertEqual(summary["failed_count"], 1)

    def test_latency_percentiles(self):
        records = [
            {"scenario_status": "succeeded", "repo_id": "r", "category": "c", "metrics": {"hit_at_5": 1, "latency_ms": value}}
            for value in [10, 20, 30, 40, 50]
        ]
        summary = aggregate_results(records)
        self.assertEqual(summary["p50_latency_ms"], 30)
        self.assertEqual(summary["p95_latency_ms"], 48)

    def test_paired_delta_and_bootstrap_are_reproducible(self):
        left = [{"scenario_id": "a", "scenario_status": "succeeded", "metrics": {"hit_at_5": 0}}]
        right = [{"scenario_id": "a", "scenario_status": "succeeded", "metrics": {"hit_at_5": 1}}]
        first = paired_delta(left, right, "hit_at_5", seed=7, samples=100)
        second = paired_delta(left, right, "hit_at_5", seed=7, samples=100)
        self.assertEqual(first, second)
        self.assertEqual(first["mean_delta"], 1)

    def test_empty_paired_data_is_explicit(self):
        self.assertIsNone(paired_delta([], [], "hit_at_5")["mean_delta"])

    def test_invalid_numeric_latency_is_rejected(self):
        with self.assertRaises(ValueError):
            scenario_metrics(scenario(), {**result(), "latency_ms": math.nan})


if __name__ == "__main__":
    unittest.main()
