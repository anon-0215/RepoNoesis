from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest

from app.config import EmbeddingSettings, RepositorySettings
from app.database import Database
from app.models import RepoFile, RepositorySnapshot
from app.services.embedding_service import EmbeddingService
from app.services.repository_import import ImportedRepository
from app.services.workspace_update import WorkspaceUpdateService
from app.services.learning_contracts import SubmitAttemptRequest
from app.services.learning_service import LearningService
from tests.m4_helpers import FakeEvaluator, create_goal_plan_task
from tests.p22_helpers import build_fixture_snapshot


class _FakeEmbeddingBackend:
    encode_calls = 0

    def load_model(self, *_args, **_kwargs):
        return None

    def encode(self, texts, batch_size, normalize):
        del batch_size, normalize
        type(self).encode_calls += len(texts)
        return [[1.0, 0.0] for _ in texts]

    def get_embedding_dimension(self):
        return 2

    def get_model_revision(self):
        return None

    def unload_model(self):
        return None


class _ThreeDimensionalBackend(_FakeEmbeddingBackend):
    def encode(self, texts, batch_size, normalize):
        del batch_size, normalize
        return [[1.0, 0.0, 0.0] for _ in texts]

    def get_embedding_dimension(self):
        return 3


def _identity(source: str, revision: str) -> str:
    digest = hashlib.sha256(f"local\0{source.casefold()}\0{revision}".encode()).hexdigest()
    return f"source-sha256:{digest}"


def _imported(root: Path, revision: str, files: dict[str, str]) -> ImportedRepository:
    snapshot = RepositorySnapshot(
        repo_url=str(root),
        owner="local",
        repo="fixture",
        default_branch="main",
        files=[RepoFile(path=path, size=len(content.encode()), content=content, extension=Path(path).suffix) for path, content in files.items()],
        repository_revision=revision,
        source_type="local",
        source_location=str(root),
        source_identity=_identity(str(root), revision),
    )
    return ImportedRepository(snapshot, snapshot.source_identity, root)


class WorkspaceUpdateServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.root = root / "repo"
        self.root.mkdir()
        self.database = Database(root / "updates.sqlite")
        settings = EmbeddingSettings(
            enabled=True,
            model_name_or_path="fake-bge-m3",
            device="cpu",
            batch_size=8,
            max_length=128,
            normalize=True,
            cache_dir=root / "cache",
            query_prefix="",
            document_prefix="",
            model_revision="fake-revision",
            provider="local_bge_m3",
            offline=True,
        )
        self.embedding = EmbeddingService(settings, backend_factory=_FakeEmbeddingBackend, cuda_available=lambda: False)
        self.imports: dict[str, ImportedRepository] = {}
        self.service = WorkspaceUpdateService(
            self.database,
            RepositorySettings(root / "runtime"),
            self.embedding,
            importer=lambda _source_type, _source, _settings: self.imports["current"],
        )
        _FakeEmbeddingBackend.encode_calls = 0

    def _seed(self, revision: str = "a" * 40, files: dict[str, str] | None = None) -> tuple[str, str]:
        imported = _imported(self.root, revision, files or {"app.py": "def stable():\n    return 1\n"})
        self.imports["current"] = imported
        project_id = self.database.create_project(imported.snapshot.to_dict())
        build_fixture_snapshot(self.database, self.embedding, project_id, imported.snapshot)
        workspace_id = self.database.get_workspace_for_project(project_id)["id"]
        return project_id, workspace_id

    def test_unchanged_refresh_reuses_one_run_and_never_encodes(self) -> None:
        _project_id, workspace_id = self._seed()
        before = _FakeEmbeddingBackend.encode_calls
        first = self.service.start_refresh(workspace_id)
        second = self.service.start_refresh(workspace_id)
        self.assertEqual(first["run_id"], second["run_id"])
        self.assertEqual(first["result"], "unchanged")
        self.assertEqual(_FakeEmbeddingBackend.encode_calls, before)
        self.assertEqual(self.database.count_workspace_revisions(workspace_id), 1)

    def test_added_modified_deleted_and_renamed_files_are_classified_by_content(self) -> None:
        _project_id, workspace_id = self._seed(
            files={
                "stable.py": "def stable():\n    return 1\n",
                "modify.py": "def value():\n    return 1\n",
                "delete.py": "def gone():\n    return 1\n",
                "old_name.py": "def moved():\n    return 1\n",
            }
        )
        self.imports["current"] = _imported(
            self.root,
            "b" * 40,
            {
                "stable.py": "def stable():\n    return 1\n",
                "modify.py": "def value():\n    return 2\n",
                "added.py": "def added():\n    return 1\n",
                "new_name.py": "def moved():\n    return 1\n",
            },
        )
        run = self.service.start_refresh(workspace_id)
        completed = self.service.execute_run(workspace_id, run["run_id"])
        stats = completed["stats"]["files"]
        self.assertEqual(stats, {"added": 1, "modified": 1, "deleted": 1, "renamed": 1, "unchanged": 1})

    def test_incremental_activation_reuses_only_safe_embeddings_and_has_no_stale_rows(self) -> None:
        old_project, workspace_id = self._seed(
            files={
                "stable.py": "def stable():\n    return 1\n",
                "change.py": "def value():\n    return 1\n",
                "delete.py": "def gone():\n    return 1\n",
            }
        )
        old_event_counts = self.database.learning_record_counts(old_project)
        self.imports["current"] = _imported(
            self.root,
            "b" * 40,
            {
                "stable.py": "def stable():\n    return 1\n",
                "change.py": "def value():\n    return 2\n",
            },
        )
        run = self.service.start_refresh(workspace_id)
        completed = self.service.execute_run(workspace_id, run["run_id"])
        self.assertEqual(completed["status"], "succeeded")
        self.assertEqual(completed["result"], "activated")
        new_project = completed["project_id"]
        self.assertNotEqual(new_project, old_project)
        self.assertEqual(self.database.get_workspace_record(workspace_id)["active_project_id"], new_project)
        new_chunks = self.database.get_code_chunks(new_project)
        self.assertEqual({chunk["path"] for chunk in new_chunks}, {"stable.py", "change.py"})
        self.assertEqual(completed["stats"]["embeddings"]["reused"], 1)
        self.assertEqual(completed["stats"]["embeddings"]["generated"], 1)
        self.assertEqual(self.database.learning_record_counts(old_project), old_event_counts)

    def test_failure_and_restart_recovery_keep_old_snapshot_active(self) -> None:
        old_project, workspace_id = self._seed()
        self.imports["current"] = _imported(self.root, "b" * 40, {"app.py": "def stable():\n    return 2\n"})
        run = self.service.start_refresh(workspace_id)
        self.service.fail_at_phase = "relation_update"
        failed = self.service.execute_run(workspace_id, run["run_id"])
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(self.database.get_workspace_record(workspace_id)["active_project_id"], old_project)

        pending = self.database.create_or_get_update_run(workspace_id, "c" * 40, "config-test")
        restarted = Database(self.database.path)
        recovered = WorkspaceUpdateService(restarted, self.service.repository_settings, self.embedding, importer=self.service.importer)
        self.assertEqual(recovered.recover_interrupted_runs(), 1)
        self.assertEqual(restarted.get_update_run(workspace_id, pending["run_id"])["error_code"], "update_interrupted")
        self.assertEqual(restarted.get_workspace_record(workspace_id)["active_project_id"], old_project)

    def test_concurrent_refreshes_share_one_run_and_one_active_snapshot(self) -> None:
        old_project, workspace_id = self._seed()
        self.imports["current"] = _imported(
            self.root, "b" * 40, {"app.py": "def stable():\n    return 2\n"}
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            started = list(pool.map(lambda _value: self.service.start_refresh(workspace_id), range(2)))
        self.assertEqual(started[0]["run_id"], started[1]["run_id"])
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda _value: self.service.execute_run(workspace_id, started[0]["run_id"]), range(2)))
        completed = self.service.get_run(workspace_id, started[0]["run_id"])
        self.assertEqual(completed["status"], "succeeded")
        self.assertEqual(self.database.count_workspace_revisions(workspace_id), 2)
        active = self.database.get_workspace_record(workspace_id)["active_project_id"]
        self.assertNotEqual(active, old_project)

    def test_activation_transaction_interruption_never_half_activates(self) -> None:
        old_project, workspace_id = self._seed()
        self.imports["current"] = _imported(
            self.root, "b" * 40, {"app.py": "def stable():\n    return 2\n"}
        )
        run = self.service.start_refresh(workspace_id)
        self.database._activation_test_hook = lambda: (_ for _ in ()).throw(RuntimeError("interrupt"))
        failed = self.service.execute_run(workspace_id, run["run_id"])
        self.database._activation_test_hook = None
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(self.database.get_workspace_record(workspace_id)["active_project_id"], old_project)
        with self.database.connect() as conn:
            statuses = {
                row["project_id"]: row["activation_status"]
                for row in conn.execute(
                    "SELECT project_id, activation_status FROM workspace_revisions WHERE workspace_id=?",
                    (workspace_id,),
                )
            }
        self.assertEqual(statuses[old_project], "active")
        self.assertEqual(statuses[failed["project_id"]], "failed")

    def test_chunker_change_forces_recompute_and_rename_forces_reencode(self) -> None:
        _old, workspace_id = self._seed(files={"old.py": "def moved():\n    return 1\n"})
        with self.database.connect() as conn:
            conn.execute(
                "UPDATE workspace_revisions SET chunker_version='obsolete' WHERE workspace_id=?",
                (workspace_id,),
            )
        self.imports["current"] = _imported(
            self.root, "b" * 40, {"new.py": "def moved():\n    return 1\n"}
        )
        run = self.service.start_refresh(workspace_id)
        completed = self.service.execute_run(workspace_id, run["run_id"])
        self.assertEqual(completed["stats"]["chunks"]["reused"], 0)
        self.assertEqual(completed["stats"]["chunks"]["recomputed"], 1)
        self.assertEqual(completed["stats"]["embeddings"]["reused"], 0)
        self.assertEqual(completed["stats"]["embeddings"]["generated"], 1)

    def test_corrupt_embedding_cache_is_recomputed_instead_of_reused(self) -> None:
        _old, workspace_id = self._seed()
        with self.database.connect() as conn:
            conn.execute("UPDATE code_chunk_embeddings SET vector_blob=x'00'")
        self.imports["current"] = _imported(
            self.root, "b" * 40, {"app.py": "def stable():\n    return 1\n"}
        )
        run = self.service.start_refresh(workspace_id)
        completed = self.service.execute_run(workspace_id, run["run_id"])
        self.assertEqual(completed["status"], "succeeded")
        self.assertEqual(completed["stats"]["embeddings"]["reused"], 0)
        self.assertEqual(completed["stats"]["embeddings"]["generated"], 1)

    def test_embedding_identity_change_in_dimension_revision_and_config_forces_recompute(self) -> None:
        _old, workspace_id = self._seed()
        changed_settings = EmbeddingSettings(
            enabled=True,
            model_name_or_path="fake-bge-m3-v2",
            device="cpu",
            batch_size=8,
            max_length=64,
            normalize=False,
            cache_dir=Path(self.directory.name) / "cache-v2",
            query_prefix="query:",
            document_prefix="document:",
            model_revision="fake-revision-v2",
            provider="local_bge_m3",
            offline=True,
        )
        changed_embedding = EmbeddingService(
            changed_settings,
            backend_factory=_ThreeDimensionalBackend,
            cuda_available=lambda: False,
        )
        changed_service = WorkspaceUpdateService(
            self.database,
            self.service.repository_settings,
            changed_embedding,
            importer=self.service.importer,
        )
        self.imports["current"] = _imported(
            self.root, "b" * 40, {"app.py": "def stable():\n    return 1\n"}
        )
        run = changed_service.start_refresh(workspace_id)
        completed = changed_service.execute_run(workspace_id, run["run_id"])
        self.assertEqual(completed["stats"]["embeddings"]["reused"], 0)
        self.assertEqual(completed["stats"]["embeddings"]["generated"], 1)
        new_project = completed["project_id"]
        with self.database.connect() as conn:
            dimension = conn.execute(
                """
                SELECT e.embedding_dimension FROM code_chunk_embeddings AS e
                JOIN code_chunks AS c ON c.id=e.code_chunk_id
                WHERE c.project_id=?
                """,
                (new_project,),
            ).fetchone()[0]
        self.assertEqual(dimension, 3)

    def test_incremental_result_equals_clean_full_build_for_same_revision(self) -> None:
        _old, workspace_id = self._seed(
            files={
                "stable.py": "def stable():\n    return 1\n",
                "change.py": "def value():\n    return 1\n",
            }
        )
        target = _imported(
            self.root,
            "b" * 40,
            {
                "stable.py": "def stable():\n    return 1\n",
                "change.py": "def value():\n    return 2\n",
                "added.py": "def added():\n    return stable()\n",
            },
        )
        self.imports["current"] = target
        run = self.service.start_refresh(workspace_id)
        completed = self.service.execute_run(workspace_id, run["run_id"])
        incremental_project = completed["project_id"]

        clean_path = Path(self.directory.name) / "clean.sqlite"
        clean_db = Database(clean_path)
        full_project = clean_db.create_project(target.snapshot.to_dict())
        build_fixture_snapshot(clean_db, self.embedding, full_project, target.snapshot)

        def chunk_view(database, project_id):
            return [
                (c["path"], c["chunk_type"], c["qualified_name"], c["start_line"], c["end_line"], c["content_hash"])
                for c in database.get_code_chunks(project_id)
            ]

        def relation_view(database, project_id, revision):
            return [
                (r["relation_type"], r["source_path"], r["source_symbol"], r["target_path"], r["target_symbol"], r["raw_target_name"], r["resolution_status"], r["resolution_rule"])
                for r in database.get_relations(project_id, revision)
            ]

        self.assertEqual(chunk_view(self.database, incremental_project), chunk_view(clean_db, full_project))
        self.assertEqual(
            relation_view(self.database, incremental_project, "b" * 40),
            relation_view(clean_db, full_project, "b" * 40),
        )
        self.assertTrue(
            all(r["repository_revision"] == "b" * 40 for r in self.database.get_relations(incremental_project, "b" * 40))
        )

    def test_refresh_preserves_nonempty_old_learning_history(self) -> None:
        old_project, workspace_id = self._seed()
        learning = LearningService(self.database)
        _goal, _plan, task = create_goal_plan_task(
            learning, old_project, qualified_name="stable"
        )
        learning.submit_attempt(
            old_project,
            task["task_id"],
            SubmitAttemptRequest(answer_text="valid", idempotency_key="p22-history"),
            evaluator=FakeEvaluator("pass"),
        )
        before = self.database.learning_record_counts(old_project)
        self.assertGreater(before["learning_events"], 0)
        self.assertGreater(before["learning_evaluations"], 0)
        self.imports["current"] = _imported(
            self.root, "b" * 40, {"app.py": "def stable():\n    return 2\n"}
        )
        run = self.service.start_refresh(workspace_id)
        completed = self.service.execute_run(workspace_id, run["run_id"])
        self.assertEqual(completed["status"], "succeeded")
        self.assertEqual(self.database.learning_record_counts(old_project), before)


if __name__ == "__main__":
    unittest.main()
