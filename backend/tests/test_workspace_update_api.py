from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import BackgroundTasks, HTTPException

from app.config import EmbeddingSettings, RepositorySettings
from app.database import Database
from app.models import RepoFile, RepositorySnapshot
from app.services.embedding_service import EmbeddingService
from app.services.repository_import import ImportedRepository
from app.services.workspace_update import WorkspaceUpdateService
from tests.p22_helpers import build_fixture_snapshot


class _Backend:
    def load_model(self, *_args, **_kwargs): return None
    def encode(self, texts, batch_size, normalize):
        del batch_size, normalize
        return [[1.0, 0.0] for _ in texts]
    def get_embedding_dimension(self): return 2
    def get_model_revision(self): return None
    def unload_model(self): return None


class WorkspaceUpdateApiTests(unittest.TestCase):
    def setUp(self) -> None:
        import app.main as main

        self.main = main
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.root = root / "repo"
        self.root.mkdir()
        self.database = Database(root / "api.sqlite")
        settings = EmbeddingSettings(
            enabled=True, model_name_or_path="fake-bge-m3", device="cpu",
            batch_size=4, max_length=128, normalize=True, cache_dir=root / "cache",
            query_prefix="", document_prefix="", model_revision="fake-revision",
            provider="local_bge_m3", offline=True,
        )
        embedding = EmbeddingService(settings, backend_factory=_Backend, cuda_available=lambda: False)
        self.current = self._imported("a" * 40, "def value():\n    return 1\n")
        self.service = WorkspaceUpdateService(
            self.database,
            RepositorySettings(root / "runtime"),
            embedding,
            importer=lambda *_args: self.current,
        )
        project_id = self.database.create_project(self.current.snapshot.to_dict())
        build_fixture_snapshot(self.database, embedding, project_id, self.current.snapshot)
        self.workspace_id = self.database.get_workspace_for_project(project_id)["id"]
        self.original_db = main.db
        main.db = self.database
        self.addCleanup(setattr, main, "db", self.original_db)

    def _imported(self, revision: str, content: str) -> ImportedRepository:
        snapshot = RepositorySnapshot(
            repo_url=str(self.root), owner="local", repo="repo", default_branch="main",
            files=[RepoFile(path="app.py", size=len(content.encode()), content=content, extension=".py")],
            repository_revision=revision, source_type="local", source_location=str(self.root),
            source_identity=f"identity-{revision}",
        )
        return ImportedRepository(snapshot, snapshot.source_identity, self.root)

    def test_check_and_explicit_refresh_use_safe_responses_and_activate(self) -> None:
        self.current = self._imported("b" * 40, "def value():\n    return 2\n")
        with patch.object(self.main, "_update_service", return_value=self.service):
            checked = self.main.check_workspace_revision(self.workspace_id)
            self.assertEqual(checked["state"], "update_available")
            self.assertEqual(self.database.count_workspace_revisions(self.workspace_id), 1)

            background = BackgroundTasks()
            with patch.object(
                self.main,
                "get_product_config_status",
                return_value={"embedding": {"ready": True}},
            ):
                started = self.main.start_workspace_refresh(self.workspace_id, background)
            self.assertEqual(started["status"], "pending")
            self.assertNotIn("base_project_id", started)
            self.assertNotIn("project_id", started)
            asyncio.run(background())
            completed = self.main.get_workspace_update_run(
                self.workspace_id, started["run_id"]
            )
        self.assertEqual(completed["status"], "succeeded")
        self.assertEqual(completed["result"], "activated")
        self.assertTrue(completed["active_project_id"])

    def test_run_404_retry_409_and_legacy_source_422_are_stable(self) -> None:
        with patch.object(self.main, "_update_service", return_value=self.service):
            with self.assertRaises(HTTPException) as missing:
                self.main.get_workspace_update_run(self.workspace_id, "missing")
        self.assertEqual(missing.exception.status_code, 404)
        self.assertEqual(missing.exception.detail["code"], "update_run_not_found")

        unchanged = self.service.start_refresh(self.workspace_id)
        with patch.object(self.main, "_update_service", return_value=self.service):
            with self.assertRaises(HTTPException) as conflict:
                self.main.retry_workspace_update_run(
                    self.workspace_id, unchanged["run_id"], BackgroundTasks()
                )
        self.assertEqual(conflict.exception.status_code, 409)
        self.assertEqual(conflict.exception.detail["code"], "update_not_retryable")

        legacy_project = self.database.create_project(
            {
                "repo_url": "https://example.test/legacy",
                "owner": "legacy", "repo": "legacy", "default_branch": "main",
                "repository_revision": "c" * 40, "source_type": "legacy_github",
                "source_location": "", "source_identity": "legacy-refresh",
            }
        )
        self.database.save_analysis(
            legacy_project,
            {"primary_language": "Python", "frameworks": [], "files": [], "modules": []},
            [], [], [],
        )
        legacy_workspace = self.database.get_workspace_for_project(legacy_project)["id"]
        with patch.object(self.main, "_update_service", return_value=self.service):
            with self.assertRaises(HTTPException) as unsupported:
                self.main.check_workspace_revision(legacy_workspace)
        self.assertEqual(unsupported.exception.status_code, 422)
        self.assertEqual(unsupported.exception.detail["code"], "workspace_source_unsupported")

    def test_openapi_declares_only_server_resolved_refresh_routes(self) -> None:
        schema = self.main.app.openapi()
        self.assertIn(f"/api/workspaces/{{workspace_id}}/revision/check", schema["paths"])
        self.assertIn(f"/api/workspaces/{{workspace_id}}/refresh", schema["paths"])
        self.assertIn(f"/api/workspaces/{{workspace_id}}/runs/{{run_id}}", schema["paths"])
        refresh = schema["paths"]["/api/workspaces/{workspace_id}/refresh"]["post"]
        self.assertNotIn("requestBody", refresh)
        properties = schema["components"]["schemas"]["WorkspaceUpdateRunResponse"]["properties"]
        self.assertNotIn("base_project_id", properties)
        self.assertNotIn("source_location", properties)


if __name__ == "__main__":
    unittest.main()
