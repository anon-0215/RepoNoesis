from __future__ import annotations

import importlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.database import Database, SCHEMA_VERSION
from app.models import RepoFile, RepositorySnapshot
from tests.m1_helpers import disabled_embedding_service
from tests.m3_helpers import call_chain_sources, make_relation_project
from tests.test_m2_agent import NoLlm


class M3ApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.directory.name) / "api.sqlite")
        with patch.dict(os.environ, {"GITLEARN_DB": self.db_path}):
            self.main = importlib.import_module("app.main")
        self.db = Database(self.db_path)

    def tearDown(self):
        self.directory.cleanup()

    def test_analyze_route_runs_formal_relation_index_after_snapshot_save(self):
        sources = call_chain_sources()
        snapshot = RepositorySnapshot(
            repo_url="https://github.com/demo/reponoesis-m3-fixture",
            owner="demo",
            repo="reponoesis-m3-fixture",
            default_branch="main",
            repository_revision="revision-m3",
            files=[
                RepoFile(path, len(content.encode("utf-8")), content)
                for path, content in sorted(sources.items())
            ],
        )
        with (
            patch.object(self.main, "db", self.db),
            patch.object(self.main, "fetch_repository", return_value=snapshot),
            patch.object(self.main, "build_learning_path", return_value=[]),
            patch.object(
                self.main, "embedding_service", disabled_embedding_service()
            ),
        ):
            result = self.main.analyze_project(
                self.main.AnalyzeRequest(repo_url=snapshot.repo_url)
            )
        self.assertEqual(result["status"], "done")
        self.assertEqual(result["relation_index"]["status"], "complete")
        self.assertGreater(result["relation_index"]["edge_count"], 0)
        status = self.db.get_relation_index_status(
            result["project_id"], "revision-m3"
        )
        self.assertIsNotNone(status)

    def test_old_ask_request_shape_keeps_m1_m2_fields_and_adds_m3_fields(self):
        project_id, _bundle = make_relation_project(self.db, call_chain_sources())
        with (
            patch.object(self.main, "db", self.db),
            patch.object(self.main, "llm", NoLlm()),
            patch.object(
                self.main, "embedding_service", disabled_embedding_service()
            ),
        ):
            result = self.main.ask_project(
                project_id, self.main.AskRequest(question="def a")
            )
            validated = self.main.AskResponse.model_validate(result)
        self.assertEqual(validated.evidence_schema_version, 1)
        self.assertEqual(validated.agent_schema_version, 1)
        self.assertEqual(validated.relation_schema_version, 1)
        self.assertEqual(validated.analysis_mode, "retrieval_only")
        self.assertIn("answer", result)
        self.assertIn("citations", result)
        self.assertIn("budget_usage", result)
        self.assertIn("relation_summary", result)

    def test_health_reports_database_schema_version_separately(self):
        with (
            patch.object(self.main, "db", self.db),
            patch.object(
                self.main, "embedding_service", disabled_embedding_service()
            ),
        ):
            result = self.main.health()
        self.assertEqual(SCHEMA_VERSION, 7)
        self.assertEqual(result["database_schema_version"], 7)


if __name__ == "__main__":
    unittest.main()
