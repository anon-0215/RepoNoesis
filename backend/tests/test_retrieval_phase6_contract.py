from __future__ import annotations

import unittest
from pathlib import Path

from app.retrieval_phase5.contracts import MATCHER_VERSION
from app.retrieval_phase6.contracts import load_phase6_benchmark


class RetrievalPhase6ContractTests(unittest.TestCase):
    def test_frozen_cross_repo_benchmark_loads_without_rewriting_phase5_gold(self):
        root = Path(__file__).resolve().parents[2]
        snapshot = load_phase6_benchmark(
            root / "benchmarks" / "retrieval_v2_phase6",
            root / "benchmarks" / "m5" / "datasets" / "pilot-v1",
        )
        self.assertEqual(snapshot.repository_ids, ("click", "httpx"))
        self.assertEqual(snapshot.new_repository_ids, ("httpx",))
        self.assertEqual(
            {repo: (len(snapshot.scenarios_by_repo[repo]), len(snapshot.answerable_by_repo[repo])) for repo in snapshot.repository_ids},
            {"click": (12, 11), "httpx": (22, 20)},
        )
        self.assertEqual(snapshot.matcher_version, MATCHER_VERSION)
        self.assertEqual(
            snapshot.matcher_hash,
            "169b0515ffcdd889212f88cc51430ac8b706eb25817e7647e7b064b45a405cb6",
        )
        self.assertEqual(
            snapshot.stratum_counts("httpx"),
            {
                "direct_behavior_location": 6,
                "hierarchy_sensitive": 4,
                "relation_dependent": 6,
                "symbol_focused": 4,
                "unanswerable": 2,
            },
        )
        self.assertEqual(len(snapshot.query_ids), 34)
        self.assertEqual(len(set(snapshot.query_ids)), 34)

    def test_relation_and_hierarchy_gold_survive_scenario_adaptation(self):
        root = Path(__file__).resolve().parents[2]
        snapshot = load_phase6_benchmark(
            root / "benchmarks" / "retrieval_v2_phase6",
            root / "benchmarks" / "m5" / "datasets" / "pilot-v1",
        )
        relation = next(
            item for item in snapshot.scenarios_by_repo["httpx"]
            if item.scenario_id == "httpx-phase6-relation-01"
        )
        hierarchy = next(
            item for item in snapshot.scenarios_by_repo["httpx"]
            if item.scenario_id == "httpx-phase6-hierarchy-01"
        )
        self.assertEqual(relation.expected_source_spans[0].qualified_symbol, "request")
        self.assertEqual(relation.expected_relation_edges[0].source_symbol, "get")
        self.assertEqual(relation.expected_relation_edges[0].target_symbol, "request")
        self.assertIn(".", hierarchy.expected_source_spans[0].qualified_symbol)


if __name__ == "__main__":
    unittest.main()
