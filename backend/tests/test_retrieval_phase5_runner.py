from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from app.database import Database
from app.m5.contracts import Scenario
from app.m5.embedding import fake_embedding_service
from app.retrieval_phase5.contracts import FROZEN_PATHS
from app.retrieval_phase5.runner import (
    Phase5Harness,
    Phase5RunError,
    load_click_benchmark,
    relation_graph_identity,
)
from app.services.embedding_indexer import EmbeddingIndexer
from tests.m3_helpers import call_chain_sources, make_relation_project


REVISION = "a" * 40


def _scenario(symbol: str = "a") -> Scenario:
    source = call_chain_sources()[f"pkg/{symbol}.py"]
    lines = source.splitlines(keepends=True)
    start = 3
    end = 4
    content = "".join(lines[start - 1 : end])
    return Scenario.model_validate(
        {
            "scenario_id": f"click-{symbol}-fixture",
            "dataset_version": "pilot-v1",
            "repo_id": "click",
            "repository_revision": REVISION,
            "language": "python",
            "question": f"Where is `{symbol}` defined?",
            "category": "locate",
            "difficulty": "easy",
            "expected_target_type": "symbol",
            "expected_files": [f"pkg/{symbol}.py"],
            "expected_symbols": [symbol],
            "expected_source_spans": [
                {
                    "path": f"pkg/{symbol}.py",
                    "qualified_symbol": symbol,
                    "start_line": start,
                    "end_line": end,
                    "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                }
            ],
            "expected_content_hashes": [hashlib.sha256(content.encode("utf-8")).hexdigest()],
            "expected_relation_edges": [],
            "expected_key_points": [symbol],
            "unanswerable": False,
            "allowed_evidence_scope": {"paths": [f"pkg/{symbol}.py"], "repository_only": True},
            "maximum_steps": 5,
            "maximum_tool_calls": 8,
            "annotation_provenance": "user_confirmed",
            "annotation_status": "human_reviewed",
            "annotation_reviewed_at": "2026-07-27",
            "annotation_review_method": "codex_conversation",
            "annotation_note": "fixed test gold",
        }
    )


class RetrievalPhase5RunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.database = Database(root / "phase5.sqlite")
        self.project_id, self.bundle = make_relation_project(
            self.database,
            call_chain_sources(),
            revision=REVISION,
        )
        self.embedding_service = fake_embedding_service(root / "cache")
        EmbeddingIndexer(self.database, self.embedding_service).index_project(self.project_id)

    def tearDown(self):
        self.directory.cleanup()

    def test_click_benchmark_selection_is_frozen_without_rewriting_gold(self):
        dataset = Path(__file__).resolve().parents[2] / "benchmarks" / "m5" / "datasets" / "pilot-v1"
        snapshot = load_click_benchmark(dataset)
        self.assertEqual((len(snapshot.scenarios), len(snapshot.answerable)), (12, 11))
        self.assertEqual(snapshot.repository_revision, "00e592cea702e0b2caa0dee42489fdb1c22cd845")
        self.assertRegex(snapshot.dataset_hash, r"^[0-9a-f]{64}$")
        self.assertNotEqual(snapshot.query_hash, snapshot.gold_hash)

    def test_matrix_has_equal_query_counts_and_is_path_order_independent(self):
        harness = Phase5Harness(
            database=self.database,
            embedding_service=self.embedding_service,
            project_id=self.project_id,
            scenarios=[_scenario()],
            formal=False,
        )
        forward = harness.run_matrix(path_order=[item.path_id for item in FROZEN_PATHS])
        reverse = harness.run_matrix(path_order=[item.path_id for item in reversed(FROZEN_PATHS)])
        self.assertEqual({path: len(values) for path, values in forward.items()}, {path: 1 for path in "ABCDE"})
        for path in "ABCDE":
            self.assertEqual(
                [item["chunk_identity"] for item in forward[path][0]["candidates"]],
                [item["chunk_identity"] for item in reverse[path][0]["candidates"]],
            )
            self.assertTrue(all(item["citation_validation"] == "valid" for item in forward[path][0]["candidates"]))

    def test_formal_mode_rejects_fake_provider_and_revision_mismatch(self):
        with self.assertRaises(Phase5RunError):
            Phase5Harness(
                database=self.database,
                embedding_service=self.embedding_service,
                project_id=self.project_id,
                scenarios=[_scenario()],
                formal=True,
            )
        with self.assertRaises(Phase5RunError):
            Phase5Harness(
                database=self.database,
                embedding_service=self.embedding_service,
                project_id=self.project_id,
                scenarios=[_scenario().model_copy(update={"repository_revision": "b" * 40})],
                formal=False,
            )

    def test_relation_graph_identity_is_stable_and_unavailable_state_is_visible(self):
        first = relation_graph_identity(self.database.path, self.project_id)
        second = relation_graph_identity(self.database.path, self.project_id)
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "complete")

        missing_db = Database(Path(self.directory.name) / "missing.sqlite")
        project_id, _ = make_relation_project(
            missing_db,
            call_chain_sources(),
            revision=REVISION,
            index_relations=False,
        )
        identity = relation_graph_identity(missing_db.path, project_id)
        self.assertEqual(identity["status"], "missing")
        self.assertEqual(identity["node_count"], 0)


if __name__ == "__main__":
    unittest.main()
