from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlencode

from app.database import Database, SCHEMA_VERSION
from app.services.chat_history import ChatHistoryError, HistoryCursor, RecoveryMetadata, prepare_recovery_metadata
from tests.m1_helpers import make_project, disabled_embedding_service
from tests.test_ask_execution_modes import _RouteLlm


def add_snapshot(db, workspace, parent, revision, *, activate=True):
    run = db.create_or_get_update_run(workspace, revision, "fixture")
    db.claim_update_run(workspace, run["run_id"])
    project = db.create_staging_project(workspace, {
        "repo_url": "https://example.test/history", "owner": "test", "repo": "history",
        "default_branch": "main", "repository_revision": revision,
    }, parent, "fixture-manifest", "fixture-chunker", run_id=run["run_id"])
    db.save_analysis(project, {"files": [], "modules": []}, [{"path": "src/auth.py", "content": "new text\n"}], [], [])
    if activate:
        db.activate_workspace_snapshot(workspace, run["run_id"], project_id=project,
            expected_active_project_id=parent, source_identity=f"fixture-{revision}", stats={})
    return project


class ChatHistoryTests(unittest.TestCase):
    def setUp(self):
        from app import main
        self.main = main
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "history.sqlite"
        self.db = Database(self.path)
        self.project, self.bundle = make_project(self.db, [
            ("src/auth.py", "authenticate_user", "def authenticate_user(password):\n    return verify(password)\n")])
        self.revision = self.bundle["project"]["repository_revision"]
        self.workspace = self.db.get_workspace_for_project(self.project)["id"]
        patcher = patch.object(main, "db", self.db)
        patcher.start(); self.addCleanup(patcher.stop)

    def history(self, **kwargs):
        return self.db.read_chat_history(self.workspace, self.project, self.revision, **kwargs)

    def get(self, workspace=None, **params):
        async def run():
            messages = []
            async def receive(): return {"type": "http.request", "body": b"", "more_body": False}
            async def send(message): messages.append(message)
            path = f"/api/workspaces/{workspace or self.workspace}/chat-history"
            query = {"active_project_id": self.project, "revision": self.revision, **params}
            await self.main.app({"type": "http", "http_version": "1.1", "method": "GET", "scheme": "http",
                "path": path, "raw_path": path.encode(), "query_string": urlencode(query).encode(),
                "headers": [], "server": ("test", 80), "client": ("test", 1), "root_path": ""}, receive, send)
            body = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
            return next(m["status"] for m in messages if m["type"] == "http.response.start"), json.loads(body)
        return asyncio.run(run())

    def ask(self, mode="rag", stream=False):
        class Learning:
            def get_learning_context(self, _project): return None
        with (patch.object(self.main, "llm", _RouteLlm()),
              patch.object(self.main, "embedding_service", disabled_embedding_service()),
              patch.object(self.main, "learning_service", Learning())):
            request = self.main.AskRequest(question="authenticate_user", execution_mode=mode,
                                          repository_revision=self.revision)
            if not stream:
                from tests.test_ask_observability import _asgi_post
                status, result = asyncio.run(_asgi_post(self.main.app, f"/api/projects/{self.project}/ask", request.model_dump()))
                self.assertEqual(status, 200, result)
                return result
            async def collect():
                response = await self.main.ask_project_stream(self.project, request)
                return [json.loads(frame) async for frame in response.body_iterator]
            frames = asyncio.run(collect())
            self.assertEqual(frames[-1]["type"], "completed", frames[-1])
            return frames[-1]["result"]

    def test_v11_migration_preserves_bytes_and_ids_without_backfill(self):
        self.db.save_chat_answer(self.project, "原问题", "原答案", [{"path": "src/auth.py"}])
        with self.db.connect() as conn:
            conn.execute("UPDATE chat_answers SET id=9007199254740993, created_at='2001-02-03 04:05:06'")
            old = tuple(conn.execute("SELECT id,project_id,question,answer,citations_json,created_at FROM chat_answers").fetchone())
            conn.execute("DROP TABLE chat_answers")
            conn.execute("""CREATE TABLE chat_answers(id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT NOT NULL,
                         question TEXT NOT NULL, answer TEXT NOT NULL, citations_json TEXT DEFAULT '[]',
                         created_at TEXT DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(project_id) REFERENCES projects(id))""")
            conn.execute("INSERT INTO chat_answers VALUES(?,?,?,?,?,?)", old)
            conn.execute("UPDATE schema_versions SET version=11")
        for _ in range(2):
            reopened = Database(self.path)
            with reopened.connect() as conn:
                row = conn.execute("SELECT * FROM chat_answers").fetchone()
                self.assertEqual(tuple(row)[:6], old)
                self.assertIsNone(row["repository_revision"])
                self.assertIsNone(row["recovery_metadata_json"])
                self.assertEqual(conn.execute("SELECT version FROM schema_versions").fetchone()[0], SCHEMA_VERSION)
                self.assertIn("idx_chat_answers_project_id", {r["name"] for r in conn.execute("PRAGMA index_list(chat_answers)")})
        record = self.history()["items"][0]
        self.assertEqual(record["record_id"], "9007199254740993")
        self.assertEqual(record["revision_source"], "snapshot_link")
        self.assertEqual(record["metadata_status"], "missing")
        self.assertTrue(record["full_file_available"])

    def test_real_route_and_stream_save_identity_metadata_once_and_failure_zero(self):
        for stream in (False, True):
            result = self.ask("agent" if stream else "rag", stream)
            self.assertRegex(result["record_id"], r"^[1-9]\d*$")
            with self.db.connect() as conn:
                row = conn.execute("SELECT * FROM chat_answers WHERE id=?", (result["record_id"],)).fetchone()
            self.assertEqual(result["created_at"], row["created_at"])
            self.assertEqual(row["repository_revision"], self.revision)
            metadata = json.loads(row["recovery_metadata_json"])
            self.assertEqual(metadata["execution_summary"], result["execution_summary"])
            self.assertEqual(metadata["execution_mode"], result["execution_mode"])
            self.assertEqual(metadata["citation_bindings"][0]["project_id"], self.project)
            self.assertNotIn("validation_status", metadata["citation_bindings"][0])
            self.assertNotIn("agent_trace", metadata)
        with patch.object(self.main, "run_bounded_agent", side_effect=ValueError("fixture failure")), patch.object(self.main, "learning_service"):
            with self.assertRaises(self.main.HTTPException):
                self.main.ask_project(self.project, self.main.AskRequest(question="failure"))
        self.assertEqual(len(self.history()["items"]), 2)

    def test_save_uses_original_context_even_when_activation_changes_before_insert(self):
        original_save = self.db.save_chat_answer
        def save(*args, **kwargs):
            self.assertEqual(kwargs["repository_revision"], self.revision)
            add_snapshot(self.db, self.workspace, self.project, "b" * 40)
            return original_save(*args, **kwargs)
        with patch.object(self.db, "save_chat_answer", side_effect=save):
            result = self.ask()
        with self.db.connect() as conn:
            row = conn.execute("SELECT project_id,repository_revision FROM chat_answers WHERE id=?", (result["record_id"],)).fetchone()
        self.assertEqual(tuple(row), (self.project, self.revision))

    def test_recovery_rejection_is_before_insert_not_after_save(self):
        with patch.object(self.main, "prepare_recovery_metadata", side_effect=ValueError("invalid")), patch.object(self.db, "save_chat_answer", wraps=self.db.save_chat_answer) as save:
            with self.assertRaises(AssertionError): self.ask()
            save.assert_not_called()
        self.assertEqual(self.history()["items"], [])

    def test_whitelist_trimming_keeps_false_distinct_from_missing(self):
        result = self.ask()
        result["prompt"] = "must not persist"
        result["warnings"] = ["x" * 600] * 40
        result["execution_summary"]["citation_validation_passed"] = False
        result["execution_summary"]["relation_validation_passed"] = None
        metadata = prepare_recovery_metadata(result, self.project, self.revision)
        self.assertNotIn("prompt", metadata)
        self.assertTrue(metadata["metadata_truncated"])
        self.assertEqual(len(metadata["warnings"]), 32)
        self.assertIs(metadata["execution_summary"]["citation_validation_passed"], False)
        self.assertIsNone(metadata["execution_summary"]["relation_validation_passed"])

    def test_snapshot_ownership_and_staging_failed_exclusion(self):
        old = self.db.save_chat_answer(self.project, "old", "old answer", [])
        new_project = add_snapshot(self.db, self.workspace, self.project, "b" * 40)
        self.db.save_chat_answer(new_project, "new", "new answer", [])
        staged = add_snapshot(self.db, self.workspace, new_project, "c" * 40, activate=False)
        self.db.save_chat_answer(staged, "staging", "excluded", [])
        other, _ = make_project(self.db, [])
        self.db.save_chat_answer(other, "another workspace", "excluded", [])
        page = self.db.read_chat_history(self.workspace, new_project, "b" * 40)
        self.assertEqual([r["question"] for r in page["items"]], ["new", "old"])
        record = page["items"][1]
        self.assertEqual(record["record_id"], old["record_id"])
        self.assertEqual(record["project_id"], self.project)
        self.assertFalse(record["full_file_available"])
        self.assertEqual(record["full_file_unavailable_reason"], "history_snapshot_old")
        with self.db.connect() as conn:
            conn.execute("UPDATE workspace_revisions SET activation_status='failed' WHERE project_id=?", (staged,))
            conn.execute("UPDATE projects SET status='failed' WHERE id=?", (staged,))
        self.assertEqual(len(self.db.read_chat_history(self.workspace, new_project, "b" * 40)["items"]), 2)

    def test_fixed_anchor_has_no_duplicates_or_omissions_after_insert(self):
        saved = [self.db.save_chat_answer(self.project, "same question", str(i), [])["record_id"] for i in range(55)]
        page = self.history(); collected = [item["record_id"] for item in page["items"]]
        self.assertEqual(len(collected), 20)
        new_id = self.db.save_chat_answer(self.project, "new", "after anchor", [])["record_id"]
        while page["next_cursor"]:
            page = self.history(cursor=page["next_cursor"])
            collected += [item["record_id"] for item in page["items"]]
        self.assertEqual(collected, list(reversed(saved)))
        self.assertNotIn(new_id, collected)
        self.assertEqual(len(set(collected)), 55)

    def test_cursor_and_context_errors_and_http_limits(self):
        for i in range(3): self.db.save_chat_answer(self.project, "q", str(i), [])
        cursor = self.history(limit=1)["next_cursor"]
        for value in ("bad", "", HistoryCursor.decode(cursor).model_copy(update={"anchor_id": "9999999999999999999"}).encode()):
            self.assertEqual(self.get(cursor=value)[0], 422)
        other, _ = make_project(self.db, [])
        other_workspace = self.db.get_workspace_for_project(other)["id"]
        self.assertEqual(self.get(other_workspace, active_project_id=other, cursor=cursor)[0], 422)
        self.assertEqual(self.get("missing")[0], 404)
        self.assertEqual(self.get(revision="wrong")[0], 409)
        self.assertEqual(self.get(activation_version=2)[0], 409)
        for limit in (0, 51): self.assertEqual(self.get(limit=limit)[0], 422)
        self.assertEqual(self.get(limit=50)[0], 200)

    def test_corrupt_records_degrade_individually_and_do_not_forge_bindings(self):
        self.ask()
        legacy = self.db.save_chat_answer(self.project, "legacy", "[E5] original", [{"path": "src/auth.py"}])
        broken = self.db.save_chat_answer(self.project, "broken", "retained", [])
        mismatch = self.db.save_chat_answer(self.project, "wrong revision", "retained", [], repository_revision="wrong")
        binding = self.ask()
        with self.db.connect() as conn:
            conn.execute("UPDATE chat_answers SET recovery_metadata_json='not-json' WHERE id=?", (broken["record_id"],))
            row = conn.execute("SELECT recovery_metadata_json FROM chat_answers WHERE id=?", (binding["record_id"],)).fetchone()
            metadata = json.loads(row[0]); metadata["citation_bindings"][0]["project_id"] = "other"
            conn.execute("UPDATE chat_answers SET recovery_metadata_json=? WHERE id=?", (json.dumps(metadata), binding["record_id"]))
        status, page = self.get()
        self.assertEqual(status, 200)
        by_id = {r["record_id"]: r for r in page["items"]}
        self.assertEqual(by_id[legacy["record_id"]]["metadata_status"], "missing")
        self.assertIsNone(by_id[legacy["record_id"]]["citations"][0]["start_line"])
        self.assertEqual(by_id[broken["record_id"]]["metadata_status"], "invalid")
        self.assertEqual(by_id[mismatch["record_id"]]["identity_status"], "conflict")
        self.assertEqual(by_id[binding["record_id"]]["metadata_status"], "binding_conflict")
        self.assertFalse(by_id[binding["record_id"]]["full_file_available"])
        self.assertEqual(sum(r["metadata_status"] == "available" for r in page["items"]), 1)

    def test_empty_or_conflicting_link_does_not_adopt_current_revision(self):
        self.db.save_chat_answer(self.project, "q", "a", [])
        with self.db.connect() as conn:
            conn.execute("UPDATE projects SET repository_revision='' WHERE id=?", (self.project,))
            conn.execute("UPDATE workspace_revisions SET repository_revision='' WHERE project_id=?", (self.project,))
        record = self.db.read_chat_history(self.workspace, self.project, "")["items"][0]
        self.assertIsNone(record["repository_revision"])
        self.assertEqual(record["identity_status"], "missing")
        self.assertFalse(record["full_file_available"])

    def test_read_is_one_read_view_and_never_writes_or_calls_provider(self):
        self.db.save_chat_answer(self.project, "q", "a", [])
        before = self.path.read_bytes()
        statements = []
        original = self.db.connect
        def connect():
            conn = original(); conn.set_trace_callback(statements.append); return conn
        with (patch.object(self.db, "connect", side_effect=connect), patch.object(self.main, "llm") as llm,
              patch.object(self.main, "_execute_ask") as ask, patch.object(self.db, "save_chat_answer") as save):
            self.assertEqual(self.get()[0], 200)
            self.assertEqual(llm.mock_calls, []); ask.assert_not_called(); save.assert_not_called()
        self.assertEqual(before, self.path.read_bytes())
        self.assertIn("PRAGMA query_only = ON", statements)
        self.assertIn("BEGIN", statements)
        self.assertFalse(any(sql.lstrip().upper().startswith(("UPDATE", "INSERT", "DELETE", "ALTER")) for sql in statements))

    def test_activation_between_context_and_rows_keeps_one_snapshot_read_view(self):
        self.db.save_chat_answer(self.project, "old question", "old answer", [])
        with self.db.connect() as conn: conn.execute("PRAGMA journal_mode=WAL")
        target = add_snapshot(self.db, self.workspace, self.project, "b" * 40, activate=False)
        run = self.db.get_latest_update_run(self.workspace)
        original = self.db.connect
        switched = False
        class Interleaved:
            def __init__(inner): inner.conn = original()
            def __enter__(inner): return inner
            def __exit__(inner, *args): return inner.conn.__exit__(*args)
            def execute(inner, sql, args=()):
                nonlocal switched
                if sql.startswith("SELECT ca.*") and not switched:
                    switched = True
                    with patch.object(self.db, "connect", original):
                        self.db.activate_workspace_snapshot(self.workspace, run["run_id"], project_id=target,
                            expected_active_project_id=self.project, source_identity="interleaved", stats={})
                return inner.conn.execute(sql, args)
        with patch.object(self.db, "connect", Interleaved):
            result = self.history()
        self.assertTrue(switched)
        self.assertEqual(result["active_project_id"], self.project)
        self.assertTrue(result["items"][0]["is_current_snapshot"])
        with self.assertRaises(ChatHistoryError): self.history()


if __name__ == "__main__":
    unittest.main()
