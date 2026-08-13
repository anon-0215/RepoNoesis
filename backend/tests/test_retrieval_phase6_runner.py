from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from app.database import Database
from app.m5.contracts import Scenario
from app.m5.embedding import fake_embedding_service
from app.retrieval_phase6.runner import Phase6Harness, phase6_determinism_summary
from app.services.embedding_indexer import EmbeddingIndexer
from tests.m3_helpers import call_chain_sources, make_relation_project


def _scenario(repo: str, revision: str, query_id: str) -> Scenario:
    source = call_chain_sources()["pkg/a.py"]
    content = "".join(source.splitlines(keepends=True)[2:4])
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return Scenario.model_validate(
        {
            "scenario_id": query_id,
            "dataset_version": "cross-repo-v1",
            "repo_id": repo,
            "repository_revision": revision,
            "language": "python",
            "question": "Where is a defined?",
            "category": "locate",
            "difficulty": "easy",
            "expected_target_type": "symbol",
            "expected_files": ["pkg/a.py"],
            "expected_symbols": ["a"],
            "expected_source_spans": [{"path": "pkg/a.py", "qualified_symbol": "a", "start_line": 3, "end_line": 4, "content_hash": digest}],
            "expected_content_hashes": [digest],
            "expected_relation_edges": [],
            "expected_key_points": ["a"],
            "unanswerable": False,
            "allowed_evidence_scope": {"paths": ["pkg/a.py"], "repository_only": True},
            "maximum_steps": 5,
            "maximum_tool_calls": 8,
            "annotation_provenance": "agent_assisted_developer_curation",
            "annotation_status": "agent_curated_pending_human_review",
            "annotation_note": "test",
        }
    )


class RetrievalPhase6RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.database = Database(root / "phase6.sqlite")
        self.service = fake_embedding_service(root / "cache")
        self.projects = {}
        self.scenarios = {}
        self.strata = {}
        for repo, revision in (("click", "a" * 40), ("httpx", "b" * 40)):
            project_id, _ = make_relation_project(self.database, call_chain_sources(), revision=revision)
            self.projects[repo] = project_id
            EmbeddingIndexer(self.database, self.service).index_project(project_id)
            query_id = f"{repo}-phase6-test"
            self.scenarios[repo] = (_scenario(repo, revision, query_id),)
            self.strata[query_id] = "symbol_focused"

    def tearDown(self):
        self.temp.cleanup()

    def test_two_repo_matrix_has_explicit_attribution_and_no_query_collision(self):
        harness = Phase6Harness(
            database=self.database,
            embedding_service=self.service,
            projects_by_repo=self.projects,
            scenarios_by_repo=self.scenarios,
            strata_by_query=self.strata,
            formal=False,
        )
        forward = harness.run_matrix(repo_order=["click", "httpx"], path_order=list("ABCDE"))
        reverse = harness.run_matrix(repo_order=["httpx", "click"], path_order=list("ECADB"), reverse_queries=True)
        for repo in ("click", "httpx"):
            for path in "ABCDE":
                self.assertEqual(forward[repo][path][0]["repository_id"], repo)
                self.assertEqual(forward[repo][path][0]["primary_stratum"], "symbol_focused")
        self.assertTrue(phase6_determinism_summary(forward, reverse)["passed"])

    def test_global_query_identity_collision_is_rejected(self):
        duplicate = {"click": self.scenarios["click"], "httpx": (self.scenarios["httpx"][0].model_copy(update={"scenario_id": "click-phase6-test"}),)}
        with self.assertRaises(ValueError):
            Phase6Harness(
                database=self.database,
                embedding_service=self.service,
                projects_by_repo=self.projects,
                scenarios_by_repo=duplicate,
                strata_by_query={"click-phase6-test": "symbol_focused"},
                formal=False,
            )


if __name__ == "__main__":
    unittest.main()
