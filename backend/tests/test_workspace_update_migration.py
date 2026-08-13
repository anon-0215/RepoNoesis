from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import unittest

from app.database import Database, SCHEMA_VERSION


class WorkspaceUpdateMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "workspace-update.sqlite"

    def _downgrade_to_v9(self, database: Database) -> None:
        conn = sqlite3.connect(database.path)
        conn.row_factory = sqlite3.Row
        try:
            workspaces = conn.execute(
                "SELECT id, display_name, source_type, source_location, active_project_id, created_at, updated_at FROM repository_workspaces"
            ).fetchall()
            revisions = conn.execute(
                "SELECT workspace_id, project_id, repository_revision, created_at FROM workspace_revisions"
            ).fetchall()
            conn.execute("PRAGMA foreign_keys=OFF")
            conn.execute("DROP TRIGGER IF EXISTS trg_projects_delete_workspace")
            conn.execute("DROP TABLE IF EXISTS repository_update_runs")
            conn.execute("DROP TABLE workspace_revisions")
            conn.execute("DROP TABLE repository_workspaces")
            conn.execute(
                """
                CREATE TABLE repository_workspaces (
                    id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL CHECK(length(display_name) BETWEEN 1 AND 300),
                    source_type TEXT NOT NULL,
                    source_location TEXT NOT NULL DEFAULT '',
                    active_project_id TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(active_project_id) REFERENCES projects(id) ON DELETE RESTRICT,
                    FOREIGN KEY(id, active_project_id)
                        REFERENCES workspace_revisions(workspace_id, project_id)
                        DEFERRABLE INITIALLY DEFERRED
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE workspace_revisions (
                    workspace_id TEXT NOT NULL,
                    project_id TEXT NOT NULL UNIQUE,
                    repository_revision TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(workspace_id, project_id),
                    FOREIGN KEY(workspace_id) REFERENCES repository_workspaces(id)
                        ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED,
                    FOREIGN KEY(project_id) REFERENCES projects(id)
                        ON DELETE RESTRICT DEFERRABLE INITIALLY DEFERRED
                )
                """
            )
            conn.executemany(
                "INSERT INTO repository_workspaces VALUES (?, ?, ?, ?, ?, ?, ?)",
                [tuple(row) for row in workspaces],
            )
            conn.executemany(
                "INSERT INTO workspace_revisions VALUES (?, ?, ?, ?)",
                [tuple(row) for row in revisions],
            )
            conn.execute("UPDATE schema_versions SET version=9 WHERE key='database'")
            conn.commit()
        finally:
            conn.close()

    def test_v9_migrates_transactionally_and_idempotently_to_current_schema(self) -> None:
        database = Database(self.path)
        project_id = database.create_project(
            {
                "repo_url": "https://example.test/repo.git",
                "owner": "owner",
                "repo": "repo",
                "default_branch": "main",
                "repository_revision": "a" * 40,
                "source_type": "git_url",
                "source_location": "https://example.test/repo.git",
                "source_identity": "identity-v9",
            }
        )
        self._downgrade_to_v9(database)

        first = Database(self.path)
        second = Database(self.path)
        with second.connect() as conn:
            self.assertEqual(SCHEMA_VERSION, 11)
            self.assertEqual(
                conn.execute("SELECT version FROM schema_versions WHERE key='database'").fetchone()[0],
                11,
            )
            revision = conn.execute(
                "SELECT activation_status, parent_project_id, manifest_hash, chunker_version "
                "FROM workspace_revisions WHERE project_id=?",
                (project_id,),
            ).fetchone()
            self.assertEqual(revision["activation_status"], "active")
            self.assertIsNone(revision["parent_project_id"])
            self.assertTrue(revision["manifest_hash"])
            self.assertTrue(revision["chunker_version"])
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM repository_update_runs").fetchone()[0], 0)
        self.assertIsNotNone(first.get_project(project_id))

    def test_migration_failure_rolls_back_all_v10_changes(self) -> None:
        database = Database(self.path)
        self._downgrade_to_v9(database)
        original = Database._backfill_workspace_update_metadata

        def fail(instance, conn):
            original(instance, conn)
            raise RuntimeError("forced v10 failure")

        from unittest.mock import patch

        with patch.object(Database, "_backfill_workspace_update_metadata", new=fail):
            with self.assertRaises(RuntimeError):
                Database(self.path)
        conn = sqlite3.connect(self.path)
        try:
            self.assertEqual(conn.execute("SELECT version FROM schema_versions WHERE key='database'").fetchone()[0], 9)
            names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertNotIn("repository_update_runs", names)
            workspace_columns = {row[1] for row in conn.execute("PRAGMA table_info(repository_workspaces)")}
            revision_columns = {row[1] for row in conn.execute("PRAGMA table_info(workspace_revisions)")}
            self.assertNotIn("activation_version", workspace_columns)
            self.assertNotIn("activation_status", revision_columns)
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
