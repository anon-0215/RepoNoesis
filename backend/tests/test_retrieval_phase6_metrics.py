from __future__ import annotations

import unittest

from app.retrieval_phase6.metrics import (
    aggregate_cross_repository,
    repository_stratified_compare,
)


def _record(repo: str, query: str, path: str, stratum: str, mrr: float, hit: float) -> dict:
    return {
        "repository_id": repo,
        "query_id": query,
        "path_id": path,
        "primary_stratum": stratum,
        "valid": True,
        "skipped": False,
        "metrics": {
            "hit_at_1": hit,
            "hit_at_3": hit,
            "hit_at_5": hit,
            "hit_at_8": hit,
            "mrr_at_8": mrr,
            "recall_at_8": hit,
            "ndcg_at_8": mrr,
        },
    }


class RetrievalPhase6MetricsTests(unittest.TestCase):
    def test_aggregate_reports_per_repo_micro_macro_and_strata(self):
        matrix = {
            "click": {
                "A": [_record("click", "c1", "A", "symbol_focused", 1.0, 1.0)],
                "B": [_record("click", "c1", "B", "symbol_focused", 0.5, 1.0)],
            },
            "httpx": {
                "A": [
                    _record("httpx", "h1", "A", "relation_dependent", 0.0, 0.0),
                    _record("httpx", "h2", "A", "hierarchy_sensitive", 0.0, 0.0),
                ],
                "B": [
                    _record("httpx", "h1", "B", "relation_dependent", 1.0, 1.0),
                    _record("httpx", "h2", "B", "hierarchy_sensitive", 0.0, 0.0),
                ],
            },
        }
        result = aggregate_cross_repository(matrix)
        self.assertEqual(result["per_repository"]["click"]["A"]["valid_answerable_count"], 1)
        self.assertAlmostEqual(result["micro"]["A"]["metrics"]["mrr_at_8"], 1.0 / 3.0)
        self.assertAlmostEqual(result["macro"]["A"]["metrics"]["mrr_at_8"], 0.5)
        self.assertEqual(result["strata"]["relation_dependent"]["B"]["metrics"]["hit_at_8"], 1.0)

    def test_repository_stratified_bootstrap_preserves_repository_units(self):
        left = [
            _record("click", "c1", "B", "symbol_focused", 0.0, 0.0),
            _record("httpx", "h1", "B", "relation_dependent", 0.0, 0.0),
        ]
        right = [
            _record("click", "c1", "D", "symbol_focused", 0.0, 0.0),
            _record("httpx", "h1", "D", "relation_dependent", 1.0, 1.0),
        ]
        first = repository_stratified_compare(
            left, right, left_path="B", right_path="D", seed=17, samples=2_000
        )
        second = repository_stratified_compare(
            list(reversed(left)), list(reversed(right)),
            left_path="B", right_path="D", seed=17, samples=2_000,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["paired_count"], 2)
        self.assertEqual(first["repository_counts"], {"click": 1, "httpx": 1})
        self.assertEqual(first["new_gold_gain_at_8"], 1)
        self.assertEqual(first["samples"], 2_000)


if __name__ == "__main__":
    unittest.main()
