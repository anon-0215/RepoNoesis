from __future__ import annotations

import hashlib
import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.parse import urlencode

from app.database import Database


class SourceFileApiTests(unittest.TestCase):
    def setUp(self):
        import app.main as main
        self.main = main
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.database = Database(Path(self.temp.name) / "source.sqlite")
        self.project = self.database.create_project({
            "repo_url": "https://example.test/source", "owner": "test", "repo": "source",
            "default_branch": "main", "repository_revision": "revision-a",
        })
        self.contents = {
            "app.py": "def answer():\r\n    return 42\r\n",
            "notes.md": "中文\n<script>unsafe()</script>\nlast",
            "empty.txt": "", "exact.txt": "x" * 1_048_576,
            "over.txt": "é" * 524_289, "cr.txt": "a\rb\r",
            "separators.txt": "a\vb\fc\x1cd\x85e\u2028f\u2029",
            "space name.txt": "one", "back\\slash.txt": "exact key",
        }
        self.database.save_analysis(self.project, {"files": [], "modules": []},
            [{"path": path, "content": text} for path, text in self.contents.items()], [], [])
        self.workspace = self.database.get_workspace_for_project(self.project)["id"]
        self.patcher = patch.object(main, "db", self.database)
        self.patcher.start(); self.addCleanup(self.patcher.stop)

    def http_get(self, url, params=None):
        # Exercise FastAPI query validation/response schema without optional httpx.
        async def run():
            messages = []
            async def send(message): messages.append(message)
            async def receive(): return {"type": "http.request", "body": b"", "more_body": False}
            await self.main.app({"type": "http", "http_version": "1.1", "method": "GET", "scheme": "http",
                "path": url, "raw_path": url.encode(), "query_string": urlencode(params or {}).encode(),
                "headers": [], "server": ("test", 80), "client": ("test", 1), "root_path": ""}, receive, send)
            status = next(item["status"] for item in messages if item["type"] == "http.response.start")
            body = b"".join(item.get("body", b"") for item in messages if item["type"] == "http.response.body").decode()
            return SimpleNamespace(status_code=status, text=body, json=lambda: json.loads(body))
        return asyncio.run(run())

    def get(self, path, revision="revision-a", project=None):
        return self.http_get(f"/api/projects/{project or self.project}/source-file",
                               params={"path": path, "revision": revision})

    def assert_error(self, response, status, code):
        self.assertEqual(response.status_code, status, response.text)
        self.assertEqual(response.json()["detail"]["code"], code)
        self.assertNotIn("content", response.json())

    def test_saved_python_and_non_python_text_contract(self):
        for path in ("app.py", "notes.md", "empty.txt", "cr.txt", "separators.txt", "space name.txt", "back\\slash.txt"):
            with self.subTest(path=path):
                response = self.get(path)
                self.assertEqual(response.status_code, 200, response.text)
                data = response.json(); text = self.contents[path]
                self.assertEqual(data["content"], text)
                self.assertEqual(data["total_lines"], len(text.splitlines()))
                self.assertEqual(data["utf8_bytes"], len(text.encode("utf-8")))
                self.assertEqual(data["content_sha256"], hashlib.sha256(text.encode("utf-8")).hexdigest())
                self.assertEqual((data["project_id"], data["revision"], data["path"], data["workspace_id"]),
                                 (self.project, "revision-a", path, self.workspace))
                self.assertEqual(data["activation_version"], 1)
                self.assertIs(data["complete"], True)
        self.assertEqual(self.database.get_code_chunks(self.project), [])

    def test_utf8_limit_inclusive_and_no_truncation(self):
        self.assertEqual(len(self.get("exact.txt").json()["content"]), 1_048_576)
        self.assert_error(self.get("over.txt"), 413, "source_file_too_large")

    def test_missing_exact_path_and_revision(self):
        self.assert_error(self.get("missing.txt"), 404, "source_file_not_found")
        self.assert_error(self.get("back/slash.txt"), 404, "source_file_not_found")
        self.assert_error(self.get("app.py", project="missing"), 404, "source_project_not_found")
        self.assert_error(self.get("app.py", revision="revision-b"), 409, "source_revision_mismatch")
        self.assert_error(self.get("app.py", revision=""), 422, "source_revision_required")

    def test_rejects_unsafe_paths_without_normalizing(self):
        for path in ("", "/app.py", "../app.py", "a/../app.py", "./app.py", "a//b", "C:\\app.py", "C:app.py", "\\\\server\\app.py", "a/./b", "a\0b", "app.py/"):
            with self.subTest(path=path):
                self.assert_error(self.get(path), 422, "source_path_invalid")
        self.assert_error(self.get("app.py' OR 1=1 --"), 404, "source_file_not_found")

    def test_deleted_or_unavailable_project_is_rejected(self):
        for status in ("cleanup_pending", "incomplete", "analyzing"):
            self.database.set_project_status(self.project, status)
            self.assert_error(self.get("app.py"), 409, "source_project_unavailable")

    def test_inactive_or_corrupt_association_is_rejected(self):
        with self.database.connect() as conn:
            conn.execute("UPDATE workspace_revisions SET activation_status='historical' WHERE project_id=?", (self.project,))
        self.assert_error(self.get("app.py"), 409, "source_snapshot_inactive")
        with self.database.connect() as conn:
            conn.execute("UPDATE workspace_revisions SET activation_status='active', repository_revision='wrong' WHERE project_id=?", (self.project,))
        self.assert_error(self.get("app.py"), 409, "source_snapshot_inactive")

    def test_switch_between_identity_and_text_uses_one_read_view(self):
        original_connect = self.database.connect
        with original_connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
        target_revision = "b" * 40
        run = self.database.create_or_get_update_run(self.workspace, target_revision, "fixture")
        self.database.claim_update_run(self.workspace, run["run_id"])
        target = self.database.create_staging_project(self.workspace, {
            "repo_url": "https://example.test/source", "owner": "test", "repo": "source",
            "default_branch": "main", "repository_revision": target_revision,
        }, self.project, "fixture-manifest", "fixture-chunker", run_id=run["run_id"])
        self.database.save_analysis(target, {"files": [], "modules": []},
                                    [{"path": "app.py", "content": "new revision text"}], [], [])
        self.assert_error(self.get("app.py", target_revision, target), 409, "source_snapshot_inactive")
        changed = False
        class InterleavedConnection:
            def __init__(inner): inner.conn = original_connect()
            def __enter__(inner): return inner
            def __exit__(inner, *args): return inner.conn.__exit__(*args)
            def execute(inner, sql, parameters=()):
                nonlocal changed
                if sql.startswith("SELECT content") and not changed:
                    changed = True
                    self.database.activate_workspace_snapshot(self.workspace, run["run_id"],
                        project_id=target, expected_active_project_id=self.project,
                        source_identity="fixture-active-b", stats={})
                    with original_connect() as writer:
                        writer.execute("UPDATE repo_files SET content='new text' WHERE project_id=? AND path='app.py'", (self.project,))
                return inner.conn.execute(sql, parameters)
        with patch.object(self.database, "connect", InterleavedConnection):
            response = self.get("app.py")
        self.assertTrue(changed)
        self.assertEqual(response.json()["content"], self.contents["app.py"])
        self.assert_error(self.get("app.py"), 409, "source_snapshot_inactive")
        activated = self.get("app.py", target_revision, target).json()
        self.assertEqual(activated["content"], "new revision text")
        self.assertEqual(activated["activation_version"], 2)

    def test_read_does_not_write_or_enter_provider_or_ask(self):
        before = self.database.path.read_bytes()
        with patch.object(self.main, "llm") as llm, patch.object(self.main, "_execute_ask") as ask, patch.object(self.database, "save_chat_answer") as save:
            self.assertEqual(self.get("notes.md").status_code, 200)
            self.assertEqual(llm.mock_calls, [])
            ask.assert_not_called(); save.assert_not_called()
        self.assertEqual(before, self.database.path.read_bytes())

    def test_required_query_and_duplicate_record_fail_closed(self):
        self.assertEqual(self.http_get(f"/api/projects/{self.project}/source-file").status_code, 422)
        with self.database.connect() as conn:
            conn.execute("INSERT INTO repo_files(project_id,path,content) VALUES(?,?,?)", (self.project, "app.py", "duplicate"))
        self.assert_error(self.get("app.py"), 409, "source_file_invalid")
