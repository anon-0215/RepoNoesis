from __future__ import annotations

import tempfile
import unittest
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from app.config import EmbeddingSettings, LLMSettings, RepositorySettings
from app.database import Database
from app.services.embedding_service import EmbeddingService
from app.services.llm_client import LLMClient
from tests.m1_helpers import make_chunk


class ProductApiTests(unittest.TestCase):
    def setUp(self):
        import app.main as main

        self.main = main
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.database = Database(Path(self.temporary.name) / "product.sqlite")
        self.project_id = self.database.create_project(
            {
                "repo_url": str(Path(self.temporary.name) / "repo"),
                "owner": "local",
                "repo": "repo",
                "default_branch": "main",
                "repository_revision": "a" * 40,
                "source_type": "local",
                "source_location": str(Path(self.temporary.name) / "repo"),
                "source_identity": "source-sha256:" + "b" * 64,
            }
        )
        content = "def answer():\n    return 42\n"
        file = {
            "path": "app.py",
            "extension": ".py",
            "language": "Python",
            "size": len(content),
            "content": content,
            "summary": "answer",
            "importance": 100,
            "is_core": True,
            "imports": [],
            "exports": [],
            "symbols": ["answer"],
        }
        chunk = make_chunk("app.py", "answer", content, revision="a" * 40)
        self.database.save_analysis(
            self.project_id,
            {"primary_language": "Python", "frameworks": [], "files": [file], "modules": []},
            [file],
            [],
            [chunk],
        )

    def test_product_ask_rejects_missing_provider_without_fallback(self):
        missing = LLMClient(LLMSettings("", "", "", ""))
        original_db, original_llm = self.main.db, self.main.llm
        self.main.db, self.main.llm = self.database, missing
        self.addCleanup(setattr, self.main, "db", original_db)
        self.addCleanup(setattr, self.main, "llm", original_llm)
        with self.assertRaises(HTTPException) as raised:
            self.main.ask_project(
                self.project_id, self.main.AskRequest(question="What does answer return?")
            )
        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(raised.exception.detail["code"], "provider_not_configured")

    def test_configuration_status_never_exposes_api_key(self):
        status = self.main.configuration_status()
        self.assertNotIn("api_key", status["llm"])
        self.assertIn("api_key_configured", status["llm"])

    def test_repository_import_error_preserves_only_safe_contract(self):
        from app.services.repository_import import RepositoryImportError

        failure = RepositoryImportError(
            "git_tls_failed",
            "The public Git TLS connection could not be verified.",
            502,
            False,
            "clone",
            128,
            321,
        )
        with (
            patch.object(
                self.main,
                "get_product_config_status",
                return_value={"embedding": {"ready": True}},
            ),
            patch.object(self.main, "import_repository", side_effect=failure),
            self.assertRaises(HTTPException) as raised,
        ):
            self.main.analyze_project(
                self.main.AnalyzeRequest(
                    source_type="git_url", source="https://example.invalid/repo.git"
                )
            )
        detail = raised.exception.detail
        self.assertEqual(raised.exception.status_code, 502)
        self.assertEqual(detail["code"], "git_tls_failed")
        self.assertEqual(detail["retryable"], False)
        self.assertEqual(detail["safe_stage"], "clone")
        self.assertEqual(detail["exit_code"], 128)
        self.assertEqual(detail["elapsed_ms"], 321)
        self.assertFalse(detail["cleanup_pending"])
        self.assertRegex(detail["request_id"], r"^[a-f0-9-]{36}$")
        self.assertNotIn("source", detail)

    def test_repository_cleanup_failure_cannot_escape_or_replace_primary_error(self):
        from app.services.repository_import import RepositoryImportError

        failure = RepositoryImportError(
            "git_connection_failed",
            "The public Git connection was interrupted.",
            502,
            True,
            "clone",
            128,
            500,
            True,
        )
        with (
            patch.object(
                self.main,
                "get_product_config_status",
                return_value={"embedding": {"ready": True}},
            ),
            patch.object(self.main, "import_repository", side_effect=failure),
            self.assertRaises(HTTPException) as raised,
        ):
            self.main.analyze_project(
                self.main.AnalyzeRequest(
                    source_type="git_url", source="https://example.invalid/repo.git"
                )
            )
        self.assertEqual(raised.exception.status_code, 502)
        self.assertEqual(raised.exception.detail["code"], "git_connection_failed")
        self.assertTrue(raised.exception.detail["retryable"])
        self.assertTrue(raised.exception.detail["cleanup_pending"])

    def test_product_analysis_failure_rolls_back_project_workspace_and_indexes(self):
        from app.models import RepoFile, RepositorySnapshot
        from app.services.repository_import import ImportedRepository

        isolated = Database(Path(self.temporary.name) / "rollback.sqlite")
        checkout = Path(self.temporary.name) / "stable-checkout"
        checkout.mkdir()
        content = "def answer():\n    return 42\n"
        snapshot = RepositorySnapshot(
            repo_url="https://public.example.invalid/repository.git",
            owner="public.example.invalid",
            repo="repository",
            default_branch="main",
            files=[RepoFile(path="app.py", size=len(content), content=content)],
            repository_revision="c" * 40,
            source_type="git_url",
            source_location="https://public.example.invalid/repository.git",
            source_identity="source-sha256:" + "d" * 64,
        )
        imported = ImportedRepository(snapshot, snapshot.source_identity, checkout)
        marker = "PRIVATE-ANALYSIS-FAILURE"
        with (
            patch.object(self.main, "db", isolated),
            patch.object(
                self.main,
                "get_product_config_status",
                return_value={"embedding": {"ready": True}},
            ),
            patch.object(self.main, "import_repository", return_value=imported),
            patch.object(self.main, "analyze_snapshot", side_effect=RuntimeError(marker)),
            self.assertRaises(HTTPException) as raised,
        ):
            self.main.analyze_project(
                self.main.AnalyzeRequest(source_type="git_url", source="ignored")
            )
        self.assertEqual(raised.exception.status_code, 500)
        self.assertEqual(raised.exception.detail["code"], "repository_analysis_failed")
        self.assertFalse(raised.exception.detail["rollback_pending"])
        self.assertNotIn(marker, repr(raised.exception.detail))
        with isolated.connect() as conn:
            for table in (
                "projects",
                "repository_workspaces",
                "workspace_revisions",
                "repo_files",
                "code_chunks",
                "code_chunk_embeddings",
                "relation_nodes",
                "code_relations",
                "relation_index_runs",
            ):
                with self.subTest(table=table):
                    self.assertEqual(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0], 0)
        self.assertTrue(checkout.is_dir())

    def test_local_product_import_persists_and_reuses_same_revision(self):
        repository = Path(self.temporary.name) / "import-source"
        repository.mkdir()
        subprocess.run(["git", "init", "-b", "main", str(repository)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(repository), "config", "user.name", "Test"], check=True)
        subprocess.run(["git", "-C", str(repository), "config", "user.email", "test@example.invalid"], check=True)
        (repository / "service.py").write_text("def load_item():\n    return 1\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repository), "add", "service.py"], check=True)
        subprocess.run(["git", "-C", str(repository), "commit", "-m", "fixture"], check=True, capture_output=True)

        settings = EmbeddingSettings(
            enabled=True,
            model_name_or_path="fake-bge-m3",
            device="cpu",
            batch_size=2,
            max_length=128,
            normalize=True,
            cache_dir=Path(self.temporary.name) / "cache",
            query_prefix="",
            document_prefix="",
            model_revision="fake-revision",
            provider="local_bge_m3",
            offline=True,
        )
        service = EmbeddingService(
            settings,
            backend_factory=_FakeEmbeddingBackend,
            cuda_available=lambda: False,
        )
        original = (self.main.db, self.main.embedding_service, self.main.repository_settings)
        self.main.db = self.database
        self.main.embedding_service = service
        self.main.repository_settings = RepositorySettings(Path(self.temporary.name) / "runtime")
        self.addCleanup(setattr, self.main, "db", original[0])
        self.addCleanup(setattr, self.main, "embedding_service", original[1])
        self.addCleanup(setattr, self.main, "repository_settings", original[2])
        env = {
            "EMBEDDING_ENABLED": "true",
            "EMBEDDING_PROVIDER": "local_bge_m3",
            "EMBEDDING_MODEL": "fake-bge-m3",
            "EMBEDDING_OFFLINE": "true",
        }
        request = self.main.AnalyzeRequest(source_type="local", source=str(repository))
        with patch.dict(os.environ, env, clear=False):
            first = self.main.analyze_project(request)
            second = self.main.analyze_project(request)
        self.assertEqual(first["status"], "done")
        self.assertEqual(first["import_action"], "analyzed")
        self.assertEqual(second["project_id"], first["project_id"])
        self.assertEqual(second["workspace_id"], first["workspace_id"])
        self.assertEqual(second["import_action"], "reused")
        reopened = Database(self.database.path)
        bundle = reopened.get_bundle(first["project_id"])
        self.assertIsNotNone(bundle)
        self.assertEqual(bundle["project"]["source_type"], "local")
        self.assertGreater(len(bundle["code_chunks"]), 0)
        workspace = reopened.get_workspace_for_project(first["project_id"])
        self.assertEqual(workspace["id"], first["workspace_id"])


class _FakeEmbeddingBackend:
    def load_model(self, *_args, **_kwargs):
        return None

    def encode(self, texts, batch_size, normalize):
        del batch_size, normalize
        return [[1.0, 0.0] for _ in texts]

    def get_embedding_dimension(self):
        return 2

    def get_model_revision(self):
        return None

    def unload_model(self):
        return None


if __name__ == "__main__":
    unittest.main()
