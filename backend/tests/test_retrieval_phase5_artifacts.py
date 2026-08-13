from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.retrieval_phase5.artifacts import (
    build_relation_diagnostics,
    validate_result_matrix,
    write_result_artifacts,
)


def _record(path_id: str, query_id: str, hit: float, *, relation: bool = False) -> dict:
    audit = {}
    candidates = []
    if relation:
        audit = {
            "relation": {
                "controlled_unavailable": False,
                "edges_accepted": 1,
                "candidate_priorities": {"chunk-r": 1.0},
                "relation_paths": [
                    {"target_chunk_identity": "chunk-r", "relation_type": "calls", "seed_chunk_identity": "chunk-s"}
                ],
                "node_resolutions": [{"node_id": "n", "status": "unique", "candidate_identities": ["chunk-r"]}],
                "selection": {
                    "selected_relation_candidates": ["chunk-r"],
                    "selected_relation_paths": [
                        {"target_chunk_identity": "chunk-r", "relation_type": "calls", "seed_chunk_identity": "chunk-s"}
                    ],
                    "suppressed_relation_candidates": [{"chunk_identity": "x", "reason": "slot_cap"}],
                    "direct_backfill": 1,
                },
                "warnings": [],
                "truncated": False,
            }
        }
        candidates = [{"chunk_identity": "chunk-r", "rank": 1, "gold_match": bool(hit), "candidate_origin": "relation"}]
    return {
        "query_id": query_id,
        "query_text": "query",
        "category": "relation",
        "path_id": path_id,
        "path_label": path_id,
        "request_parameters": {},
        "valid": True,
        "skipped": False,
        "skip_reason": None,
        "latency_ms": 1.0,
        "top_k": 8,
        "metrics": {
            "hit_at_1": hit,
            "hit_at_3": hit,
            "hit_at_5": hit,
            "hit_at_8": hit,
            "hit_at_10": hit,
            "mrr_at_8": hit,
            "mrr_at_10": hit,
            "recall_at_1": hit,
            "recall_at_3": hit,
            "recall_at_5": hit,
            "recall_at_8": hit,
            "ndcg_at_1": hit,
            "ndcg_at_3": hit,
            "ndcg_at_5": hit,
            "ndcg_at_8": hit,
            "hit_at_10_disclosure": "computed-from-at-most-top-8-not-ten-retrieved",
        },
        "candidates": candidates,
        "warnings": [],
        "retrieval_audit": audit,
        "query_encode_count": 1,
        "citation_validation": {"valid": len(candidates), "invalid": 0, "warnings": []},
        "relation_validation": {"valid": int(relation), "invalid": 0, "warnings": []},
        "relation_origin_gold_hits": int(relation and hit),
        "relation_assisted_gold_hits": 0,
    }


class RetrievalPhase5ArtifactTests(unittest.TestCase):
    def _matrix(self):
        return {
            "A": [_record("A", "q1", 0.0)],
            "B": [_record("B", "q1", 0.0)],
            "C": [_record("C", "q1", 0.0)],
            "D": [_record("D", "q1", 1.0, relation=True)],
            "E": [_record("E", "q1", 1.0, relation=True)],
        }

    def test_result_matrix_requires_all_paths_same_queries_and_top_k(self):
        summary = validate_result_matrix(self._matrix())
        self.assertEqual(summary["query_count_per_path"], 1)
        broken = self._matrix()
        broken["E"] = []
        with self.assertRaises(ValueError):
            validate_result_matrix(broken)
        broken = self._matrix()
        broken["A"][0]["top_k"] = 7
        with self.assertRaises(ValueError):
            validate_result_matrix(broken)

    def test_relation_diagnostics_count_gain_selection_caps_and_backfill(self):
        diagnostics = build_relation_diagnostics(self._matrix())
        self.assertEqual(diagnostics["relation_enabled_query_count"], 2)
        self.assertEqual(diagnostics["relation_selected_query_count"], 2)
        self.assertEqual(diagnostics["relation_new_gold_gain_at_8"], 2)
        self.assertEqual(diagnostics["relation_gold_loss_at_8"], 0)
        self.assertEqual(diagnostics["suppression_reasons"]["slot_cap"], 2)
        self.assertEqual(diagnostics["direct_backfill_count"], 2)

    def test_artifacts_are_deterministic_schema_checked_and_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"evaluation_version": "retrieval-v2-phase5@1", "timestamp": "fixed"}
            first = write_result_artifacts(
                root,
                manifest=manifest,
                records_by_path=self._matrix(),
                determinism={"passed": True},
            )
            self.assertTrue((root / "aggregate.json").is_file())
            self.assertTrue((root / "query_results.jsonl").is_file())
            self.assertTrue((root / "query_results.csv").is_file())
            self.assertTrue((root / "report.md").is_file())
            self.assertRegex(first["result_hash"], r"^[0-9a-f]{64}$")
            aggregate = json.loads((root / "aggregate.json").read_text(encoding="utf-8"))
            self.assertEqual(list(aggregate["paths"]), list("ABCDE"))
            with self.assertRaises(FileExistsError):
                write_result_artifacts(
                    root,
                    manifest=manifest,
                    records_by_path=self._matrix(),
                    determinism={"passed": True},
                )


if __name__ == "__main__":
    unittest.main()
