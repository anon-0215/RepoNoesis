from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.retrieval_phase6.artifacts import (
    build_applicability_diagnostics,
    validate_cross_repository_matrix,
    write_phase6_artifacts,
)


def _record(repo: str, path: str, query: str, stratum: str, hit: float, *, relation=False, hierarchy=False) -> dict:
    audit = {}
    candidates = [{"chunk_identity": f"{repo}:{query}:gold", "rank": 1, "gold_match": bool(hit), "candidate_origin": "direct"}]
    if relation:
        audit["relation"] = {
            "controlled_unavailable": False,
            "edges_accepted": 2,
            "candidate_priorities": {"candidate": 0.2},
            "selection": {
                "selected_relation_candidates": ["candidate"],
                "selected_relation_paths": [{"target_chunk_identity": "candidate", "relation_type": "calls"}],
                "suppressed_relation_candidates": [{"chunk_identity": "other", "reason": "slot_cap"}],
                "direct_backfill": 1,
            },
            "truncated": False,
        }
    if hierarchy:
        audit["hierarchy"] = {
            "candidates": [
                {"chunk_identity": "hierarchy-derived", "origin": "hierarchy", "decision": "suppressed"}
            ],
            "warnings": [],
            "truncated": False,
        }
    return {
        "repository_id": repo,
        "path_id": path,
        "query_id": query,
        "primary_stratum": stratum,
        "valid": True,
        "skipped": False,
        "top_k": 8,
        "metrics": {"hit_at_8": hit, "mrr_at_8": hit},
        "candidates": candidates,
        "warnings": [],
        "retrieval_audit": audit,
        "citation_validation": {"invalid": 0},
        "relation_validation": {"invalid": 0},
    }


class RetrievalPhase6ArtifactTests(unittest.TestCase):
    def _matrix(self):
        matrix = {}
        for repo in ("click", "httpx"):
            query = f"{repo}-q"
            matrix[repo] = {
                path: [_record(repo, path, query, "relation_dependent", float(path in "DE"), relation=path in "DE", hierarchy=path in "CE")]
                for path in "ABCDE"
            }
        return matrix

    def test_matrix_requires_same_five_paths_and_globally_unique_query_ids(self):
        summary = validate_cross_repository_matrix(self._matrix())
        self.assertEqual(summary["repository_count"], 2)
        self.assertEqual(summary["query_count"], 2)
        broken = self._matrix()
        broken["httpx"]["E"] = []
        with self.assertRaises(ValueError):
            validate_cross_repository_matrix(broken)

    def test_diagnostics_are_reported_by_repo_and_stratum(self):
        diagnostics = build_applicability_diagnostics(self._matrix())
        self.assertEqual(diagnostics["relation"]["overall"]["new_gold_gain_at_8"], 2)
        self.assertEqual(diagnostics["relation"]["overall"]["selected_query_count"], 4)
        self.assertEqual(diagnostics["hierarchy"]["overall"]["derived_candidate_query_count"], 4)
        self.assertEqual(diagnostics["hierarchy"]["overall"]["retained_hierarchy_query_count"], 0)
        self.assertEqual(diagnostics["relation"]["effects"]["combined"]["paired_cell_count"], 4)
        self.assertEqual(diagnostics["relation"]["effects"]["combined"]["strict_gain_cell_count"], 4)
        self.assertEqual(diagnostics["hierarchy"]["effects"]["combined"]["paired_cell_count"], 4)
        self.assertEqual(diagnostics["hierarchy"]["effects"]["combined"]["strict_gain_cell_count"], 0)
        self.assertIn("httpx", diagnostics["relation"]["by_repository"])
        self.assertIn("relation_dependent", diagnostics["relation"]["by_stratum"])

    def test_cross_repo_artifacts_are_immutable_and_hash_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hashes = write_phase6_artifacts(
                root,
                manifest={"evaluation_version": "retrieval-v2-phase6@1", "timestamp": "fixed"},
                records_by_repo_path=self._matrix(),
                determinism={"passed": True, "mismatches": []},
            )
            self.assertRegex(hashes["result_hash"], r"^[0-9a-f]{64}$")
            aggregate = json.loads((root / "aggregate.json").read_text(encoding="utf-8"))
            self.assertEqual(set(aggregate), {"per_repository", "micro", "macro", "strata"})
            self.assertTrue((root / "applicability_diagnostics.json").is_file())
            self.assertTrue((root / "query_results.jsonl").is_file())
            with self.assertRaises(FileExistsError):
                write_phase6_artifacts(
                    root,
                    manifest={"evaluation_version": "retrieval-v2-phase6@1", "timestamp": "fixed"},
                    records_by_repo_path=self._matrix(),
                    determinism={"passed": True, "mismatches": []},
                )


if __name__ == "__main__":
    unittest.main()
