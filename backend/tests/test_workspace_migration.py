from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from app.database import Database, SCHEMA_VERSION
from app.services.learning_contracts import SubmitAttemptRequest
from app.services.learning_service import LearningService
from tests.m3_helpers import make_relation_project
from tests.m4_helpers import FakeEvaluator, create_goal_plan_task


class WorkspaceMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "workspace-migration.sqlite"

    def _legacy_database(self) -> Database:
        return Database(self.path)

    def _downgrade_to_v8(self, database: Database) -> None:
        conn = sqlite3.connect(database.path)
        try:
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute("DROP TABLE IF EXISTS repository_update_runs")
            conn.execute("DROP TABLE IF EXISTS workspace_revisions")
            conn.execute("DROP TABLE IF EXISTS repository_workspaces")
            conn.execute(
                "UPDATE schema_versions SET version = 8 WHERE key = 'database'"
            )
            conn.commit()
        finally:
            conn.close()

    def test_empty_v8_database_upgrades_through_v9_to_v10(self) -> None:
        database = self._legacy_database()
        self._downgrade_to_v8(database)
        reopened = Database(self.path)
        with reopened.connect() as conn:
            self.assertEqual(
                conn.execute(
                    "SELECT version FROM schema_versions WHERE key = 'database'"
                ).fetchone()[0],
                10,
            )
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM repository_workspaces").fetchone()[0],
                0,
            )
        self.assertEqual(SCHEMA_VERSION, 10)

    def test_each_legacy_project_gets_an_independent_stable_workspace(self) -> None:
        database = self._legacy_database()
        project_ids = []
        for index in range(3):
            project_ids.append(
                database.create_project(
                    {
                        "repo_url": "https://example.test/same/repo.git",
                        "owner": "same",
                        "repo": "repo",
                        "default_branch": "main",
                        "repository_revision": "a" * 40,
                        "source_type": "git_url",
                        "source_location": "https://example.test/same/repo.git",
                        "source_identity": f"legacy-{index}",
                    }
                )
            )
        self._downgrade_to_v8(database)

        first_open = Database(self.path)
        with first_open.connect() as conn:
            rows = conn.execute(
                "SELECT id, active_project_id FROM repository_workspaces ORDER BY id"
            ).fetchall()
            revision_rows = conn.execute(
                "SELECT workspace_id, project_id FROM workspace_revisions"
            ).fetchall()
        self.assertEqual(len(rows), 3)
        self.assertEqual({row["active_project_id"] for row in rows}, set(project_ids))
        self.assertEqual(len(revision_rows), 3)
        workspace_ids = {row["id"] for row in rows}

        second_open = Database(self.path)
        with second_open.connect() as conn:
            repeated_ids = {
                row[0]
                for row in conn.execute("SELECT id FROM repository_workspaces")
            }
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM workspace_revisions").fetchone()[0],
                3,
            )
        self.assertEqual(repeated_ids, workspace_ids)

    def test_migration_preserves_project_relation_and_learning_records(self) -> None:
        database = self._legacy_database()
        project_id, _bundle = make_relation_project(
            database,
            {"app.py": "def target():\n    return 1\n"},
        )
        service = LearningService(database)
        _goal, _plan, task = create_goal_plan_task(service, project_id)
        service.submit_attempt(
            project_id,
            task["task_id"],
            SubmitAttemptRequest(
                answer_text="valid",
                idempotency_key="workspace-migration-attempt",
            ),
            evaluator=FakeEvaluator("pass"),
        )
        protected_tables = (
            "projects",
            "repo_files",
            "code_chunks",
            "relation_nodes",
            "code_relations",
            "learning_goals",
            "learning_plans",
            "learning_tasks",
            "learning_task_evidence",
            "learning_attempts",
            "learning_evaluations",
            "learning_events",
            "learner_target_states",
        )
        with database.connect() as conn:
            before = {
                table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in protected_tables
            }
        self._downgrade_to_v8(database)

        reopened = Database(self.path)
        with reopened.connect() as conn:
            after = {
                table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in protected_tables
            }
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "UPDATE learning_events SET event_type = 'forged' WHERE project_id = ?",
                    (project_id,),
                )
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "DELETE FROM learning_events WHERE project_id = ?", (project_id,)
                )
        self.assertEqual(after, before)
        self.assertIsNotNone(reopened.get_project(project_id))

    def test_workspace_schema_has_required_constraints_foreign_keys_and_indexes(self) -> None:
        database = Database(self.path)
        with database.connect() as conn:
            workspace_sql = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='repository_workspaces'"
            ).fetchone()[0]
            revision_sql = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='workspace_revisions'"
            ).fetchone()[0]
            indexes = {
                row[1]
                for table in ("repository_workspaces", "workspace_revisions")
                for row in conn.execute(f"PRAGMA index_list({table})")
            }
            workspace_fks = conn.execute(
                "PRAGMA foreign_key_list(repository_workspaces)"
            ).fetchall()
            revision_fks = conn.execute(
                "PRAGMA foreign_key_list(workspace_revisions)"
            ).fetchall()
        self.assertIn("active_project_id", workspace_sql)
        self.assertIn("workspace_id", revision_sql)
        self.assertIn("project_id", revision_sql)
        self.assertIn("idx_repository_workspaces_order", indexes)
        self.assertIn("idx_workspace_revisions_project", indexes)
        self.assertTrue(workspace_fks)
        self.assertGreaterEqual(len(revision_fks), 2)

    def test_workspace_migration_failure_rolls_back_partial_schema_and_rows(self) -> None:
        database = self._legacy_database()
        database.create_project(
            {
                "repo_url": "https://example.test/owner/repo.git",
                "owner": "owner",
                "repo": "repo",
                "default_branch": "main",
                "repository_revision": "b" * 40,
                "source_type": "git_url",
                "source_location": "https://example.test/owner/repo.git",
                "source_identity": "rollback-project",
            }
        )
        self._downgrade_to_v8(database)

        original = Database._backfill_legacy_workspaces

        def fail_after_backfill(instance, conn):
            original(instance, conn)
            raise RuntimeError("forced workspace migration failure")

        with patch.object(
            Database, "_backfill_legacy_workspaces", new=fail_after_backfill
        ):
            with self.assertRaises(RuntimeError):
                Database(self.path)

        raw = sqlite3.connect(self.path)
        try:
            self.assertEqual(
                raw.execute(
                    "SELECT version FROM schema_versions WHERE key='database'"
                ).fetchone()[0],
                8,
            )
            names = {
                row[0]
                for row in raw.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            self.assertNotIn("repository_workspaces", names)
            self.assertNotIn("workspace_revisions", names)
        finally:
            raw.close()


if __name__ == "__main__":
    unittest.main()
