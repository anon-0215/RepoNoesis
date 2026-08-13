from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from app.database import Database


class WorkspaceApiTests(unittest.TestCase):
    def setUp(self) -> None:
        import app.main as main

        self.main = main
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = Database(Path(self.directory.name) / "workspace-api.sqlite")
        self.original_db = main.db
        main.db = self.database
        self.addCleanup(setattr, main, "db", self.original_db)

    def _create_done_project(self, name: str, revision: str) -> tuple[str, str]:
        project_id = self.database.create_project(
            {
                "repo_url": f"https://example.test/owner/{name}.git",
                "owner": "owner",
                "repo": name,
                "default_branch": "main",
                "repository_revision": revision,
                "source_type": "git_url",
                "source_location": f"https://example.test/owner/{name}.git",
                "source_identity": f"identity-{name}-{revision}",
            }
        )
        self.database.save_analysis(
            project_id,
            {
                "primary_language": "Python",
                "frameworks": [],
                "files": [],
                "modules": [],
            },
            [],
            [],
            [],
        )
        self.database.set_project_status(project_id, "done")
        workspace = self.database.get_workspace_for_project(project_id)
        return project_id, workspace["id"]

    def test_workspace_list_is_deterministically_paginated_without_sensitive_fields(self) -> None:
        created = [self._create_done_project(f"repo-{index}", str(index))[1] for index in range(5)]
        payloads = [
            self.main.list_workspaces(limit=2, offset=0),
            self.main.list_workspaces(limit=2, offset=2),
            self.main.list_workspaces(limit=2, offset=4),
        ]
        ids = [item["workspace_id"] for payload in payloads for item in payload["items"]]
        self.assertEqual(len(ids), 5)
        self.assertEqual(len(set(ids)), 5)
        self.assertEqual(set(ids), set(created))
        self.assertEqual(payloads[0]["total"], 5)
        self.assertEqual(payloads[0]["limit"], 2)
        self.assertEqual(payloads[0]["offset"], 0)
        rendered = str(payloads)
        self.assertNotIn("source_location", rendered)
        self.assertNotIn("analysis", rendered)
        self.assertNotIn("content", rendered)

    def test_workspace_list_has_safe_default_maximum_and_rejects_invalid_pagination(self) -> None:
        default = self.main.list_workspaces(limit=20, offset=0)
        self.assertEqual(default["limit"], 20)
        for limit, offset in ((0, 0), (101, 0), (20, -1)):
            with self.assertRaises(HTTPException) as raised:
                self.main.list_workspaces(limit=limit, offset=offset)
            self.assertEqual(raised.exception.status_code, 422)
            self.assertEqual(raised.exception.detail["code"], "invalid_pagination")

    def test_workspace_detail_resolves_current_snapshot_and_survives_database_restart(self) -> None:
        project_id, workspace_id = self._create_done_project("restart", "c" * 40)
        restarted = Database(self.database.path)
        self.main.db = restarted
        payload = self.main.get_workspace(workspace_id)
        self.assertEqual(payload["workspace_id"], workspace_id)
        self.assertEqual(payload["active_snapshot"]["project_id"], project_id)
        self.assertEqual(payload["active_snapshot"]["repository_revision"], "c" * 40)
        self.assertTrue(payload["openable"])

    def test_unknown_workspace_has_stable_404(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            self.main.get_workspace("00000000-0000-0000-0000-000000000000")
        self.assertEqual(raised.exception.status_code, 404)
        self.assertEqual(raised.exception.detail["code"], "workspace_not_found")

    def test_incomplete_workspace_is_listed_but_not_openable(self) -> None:
        project_id = self.database.create_project(
            {
                "repo_url": "https://example.test/owner/incomplete.git",
                "owner": "owner",
                "repo": "incomplete",
                "default_branch": "main",
                "repository_revision": "f" * 40,
                "source_type": "git_url",
                "source_location": "https://example.test/owner/incomplete.git",
                "source_identity": "identity-incomplete",
            }
        )
        workspace_id = self.database.get_workspace_for_project(project_id)["id"]
        listed = self.main.list_workspaces(limit=20, offset=0)["items"][0]
        self.assertFalse(listed["openable"])
        with self.assertRaises(HTTPException) as raised:
            self.main.get_workspace(workspace_id)
        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(raised.exception.detail["code"], "workspace_not_openable")

    def test_corrupt_workspace_link_fails_without_selecting_another_project(self) -> None:
        _project_a, workspace_a = self._create_done_project("a", "a" * 40)
        _project_b, _workspace_b = self._create_done_project("b", "b" * 40)
        conn = sqlite3.connect(self.database.path)
        try:
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute(
                "UPDATE repository_workspaces SET active_project_id = ? WHERE id = ?",
                ("missing-project", workspace_a),
            )
            conn.commit()
        finally:
            conn.close()
        with self.assertRaises(HTTPException) as raised:
            self.main.get_workspace(workspace_a)
        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(raised.exception.detail["code"], "workspace_corrupt")

    def test_reopen_is_read_only_and_never_runs_analysis_indexing_network_or_provider(self) -> None:
        _project_id, workspace_id = self._create_done_project("side-effects", "d" * 40)
        with self.database.connect() as conn:
            before_events = conn.execute("SELECT COUNT(*) FROM learning_events").fetchone()[0]
            before_projects = conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
            before_chunks = conn.execute("SELECT COUNT(*) FROM code_chunks").fetchone()[0]

        with (
            patch.object(self.main, "import_repository") as import_repository,
            patch.object(self.main, "analyze_snapshot") as analyze_snapshot,
            patch.object(self.main, "index_project_relations") as relation_index,
            patch.object(self.main.EmbeddingIndexer, "index_project") as embedding_index,
            patch.object(self.main.llm, "chat") as provider_call,
        ):
            response = self.main.get_workspace(workspace_id)
        self.assertTrue(response["openable"])
        for spy in (
            import_repository,
            analyze_snapshot,
            relation_index,
            embedding_index,
            provider_call,
        ):
            spy.assert_not_called()
        with self.database.connect() as conn:
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM learning_events").fetchone()[0],
                before_events,
            )
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0],
                before_projects,
            )
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM code_chunks").fetchone()[0],
                before_chunks,
            )

    def test_existing_project_route_and_project_id_remain_valid(self) -> None:
        project_id, _workspace_id = self._create_done_project("compat", "e" * 40)
        response = self.main.get_project(project_id)
        self.assertEqual(response["project"]["id"], project_id)


if __name__ == "__main__":
    unittest.main()
