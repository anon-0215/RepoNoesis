import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.database import Database, SCHEMA_VERSION


def _project_id(db: Database) -> str:
    return db.create_project(
        {
            "repo_url": "https://github.com/demo/sample",
            "owner": "demo",
            "repo": "sample",
            "default_branch": "main",
        }
    )


def _chunk(
    path: str = "src/app.py",
    qualified_name: str = "target",
    content: str = "def target():\n    return 1\n",
    chunk_type: str = "function",
    symbol_name: str = "target",
    start_line: int = 1,
    end_line: int | None = None,
    repository_revision: str = "abc123",
) -> dict:
    return {
        "repository_revision": repository_revision,
        "language": "python",
        "path": path,
        "chunk_type": chunk_type,
        "symbol_name": symbol_name,
        "qualified_name": qualified_name,
        "parent_symbol": "",
        "start_line": start_line,
        "end_line": end_line if end_line is not None else start_line + len(content.splitlines()) - 1,
        "content": content,
        "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }


class DatabaseCodeChunkTests(unittest.TestCase):
    @staticmethod
    def _persistence_key(chunk: dict) -> tuple:
        return (
            chunk["repository_revision"],
            chunk["path"],
            chunk["chunk_type"],
            chunk["qualified_name"],
            chunk["start_line"],
            chunk["end_line"],
        )

    def _overloads(self) -> list[dict]:
        return [
            _chunk(
                qualified_name="Handler.process",
                symbol_name="process",
                chunk_type="method",
                start_line=start,
                end_line=end,
                content=f"def process(value_{index}):\n    return value_{index}\n",
            )
            for index, (start, end) in enumerate(((10, 11), (20, 21), (30, 31)), start=1)
        ]

    def test_legal_overloads_are_saved_as_distinct_persistent_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "overloads.sqlite")
            project_id = _project_id(db)
            target = self._overloads()

            db.save_code_chunks_for_project(project_id, target)

            stored = db.get_code_chunks(project_id)
            self.assertEqual(len(stored), len(target))
            self.assertEqual(
                {self._persistence_key(item) for item in stored},
                {self._persistence_key(item) for item in target},
            )
            self.assertEqual(
                {(item["content"], item["content_hash"]) for item in stored},
                {(item["content"], item["content_hash"]) for item in target},
            )

    def test_overload_subset_expands_repeats_and_reorders_with_stable_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "expand.sqlite")
            project_id = _project_id(db)
            target = self._overloads()
            subset = target[:2]
            db.save_code_chunks_for_project(project_id, subset)
            subset_ids = {
                self._persistence_key(item): item["id"]
                for item in db.get_code_chunks(project_id)
            }

            db.save_code_chunks_for_project(project_id, target)
            expanded = db.get_code_chunks(project_id)
            expanded_ids = {self._persistence_key(item): item["id"] for item in expanded}
            self.assertEqual(len(expanded), len(target))
            self.assertEqual(
                {key: expanded_ids[key] for key in subset_ids},
                subset_ids,
            )
            self.assertEqual(len(set(expanded_ids.values())), len(target))

            db.save_code_chunks_for_project(project_id, list(reversed(target)))
            reordered_ids = {
                self._persistence_key(item): item["id"]
                for item in db.get_code_chunks(project_id)
            }
            self.assertEqual(reordered_ids, expanded_ids)

            db.save_code_chunks_for_project(project_id, target)
            repeated_ids = {
                self._persistence_key(item): item["id"]
                for item in db.get_code_chunks(project_id)
            }
            self.assertEqual(repeated_ids, expanded_ids)

    def test_overload_reduction_only_deletes_removed_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "reduce.sqlite")
            project_id = _project_id(db)
            target = self._overloads()
            db.save_code_chunks_for_project(project_id, target)
            original_ids = {
                self._persistence_key(item): item["id"]
                for item in db.get_code_chunks(project_id)
            }
            reduced = [target[0], target[2]]

            db.save_code_chunks_for_project(project_id, reduced)

            stored = db.get_code_chunks(project_id)
            self.assertEqual(
                {self._persistence_key(item): item["id"] for item in stored},
                {
                    self._persistence_key(item): original_ids[self._persistence_key(item)]
                    for item in reduced
                },
            )

    def test_same_persistence_identity_updates_payload_in_place(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "update.sqlite")
            project_id = _project_id(db)
            original = self._overloads()[0]
            db.save_code_chunks_for_project(project_id, [original])
            original_id = db.get_code_chunks(project_id)[0]["id"]
            updated = dict(original)
            updated["language"] = "Python"
            updated["parent_symbol"] = "Handler"
            updated["content"] = "def process(value):\n    return value + 1\n"
            updated["content_hash"] = hashlib.sha256(updated["content"].encode()).hexdigest()

            db.save_code_chunks_for_project(project_id, [updated])

            stored = db.get_code_chunks(project_id)[0]
            self.assertEqual(stored["id"], original_id)
            for field in ("language", "parent_symbol", "content", "content_hash"):
                self.assertEqual(stored[field], updated[field])

    def test_span_change_replaces_persistent_chunk_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "span.sqlite")
            project_id = _project_id(db)
            original = self._overloads()[0]
            db.save_code_chunks_for_project(project_id, [original])
            original_id = db.get_code_chunks(project_id)[0]["id"]
            moved = dict(original, start_line=40, end_line=41)

            db.save_code_chunks_for_project(project_id, [moved])

            stored = db.get_code_chunks(project_id)
            self.assertEqual(len(stored), 1)
            self.assertNotEqual(stored[0]["id"], original_id)
            self.assertEqual(self._persistence_key(stored[0]), self._persistence_key(moved))

    def test_persistence_identity_isolated_by_project_revision_path_and_type(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "isolation.sqlite")
            first_project = _project_id(db)
            second_project = _project_id(db)
            base = self._overloads()[0]
            variants = [
                base,
                dict(base, path="src/other.py"),
                dict(base, chunk_type="function"),
                dict(base, repository_revision="different-revision"),
            ]
            db.save_code_chunks_for_project(first_project, variants)
            db.save_code_chunks_for_project(second_project, [base])
            first_ids = {item["id"] for item in db.get_code_chunks(first_project)}
            second_ids = {item["id"] for item in db.get_code_chunks(second_project)}

            self.assertEqual(len(first_ids), len(variants))
            self.assertEqual(len(second_ids), 1)
            self.assertTrue(first_ids.isdisjoint(second_ids))

    def test_replacement_write_failure_rolls_back_and_retry_succeeds(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "write-failure.sqlite")
            project_id = _project_id(db)
            original = self._overloads()[:2]
            target = self._overloads()
            db.save_code_chunks_for_project(project_id, original)
            before = db.get_code_chunks(project_id)
            real_insert = db._insert_prepared_code_chunk
            calls = 0

            def fail_after_first_insert(conn, chunk):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise sqlite3.OperationalError("forced replacement write failure")
                return real_insert(conn, chunk)

            extra = _chunk(
                qualified_name="Handler.extra",
                symbol_name="extra",
                start_line=50,
                end_line=51,
                content="def extra():\n    return None\n",
            )
            attempted = [*target, extra]
            with patch.object(db, "_insert_prepared_code_chunk", side_effect=fail_after_first_insert):
                with self.assertRaisesRegex(sqlite3.OperationalError, "forced replacement"):
                    db.save_code_chunks_for_project(project_id, attempted)

            self.assertEqual(db.get_code_chunks(project_id), before)
            db.save_code_chunks_for_project(project_id, attempted)
            self.assertEqual(
                {self._persistence_key(item) for item in db.get_code_chunks(project_id)},
                {self._persistence_key(item) for item in attempted},
            )

    def test_duplicate_target_persistence_identity_fails_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "duplicate-target.sqlite")
            project_id = _project_id(db)
            original = self._overloads()[0]
            db.save_code_chunks_for_project(project_id, [original])
            before = db.get_code_chunks(project_id)
            duplicate = dict(original)
            duplicate["content"] = "def process(other):\n    return other\n"
            duplicate["content_hash"] = hashlib.sha256(duplicate["content"].encode()).hexdigest()

            with patch.object(db, "_invalidate_relation_index") as invalidate, patch.object(
                db, "_update_code_chunk"
            ) as update, patch.object(db, "_insert_prepared_code_chunk") as insert:
                with self.assertRaisesRegex(ValueError, "duplicate code chunk persistence identity"):
                    db.save_code_chunks_for_project(project_id, [original, duplicate])
            invalidate.assert_not_called()
            update.assert_not_called()
            insert.assert_not_called()
            self.assertEqual(db.get_code_chunks(project_id), before)

    def test_legacy_database_initializes_new_code_chunk_structure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.sqlite"
            conn = sqlite3.connect(path)
            try:
                conn.executescript(
                    """
                    CREATE TABLE projects (
                        id TEXT PRIMARY KEY,
                        repo_url TEXT NOT NULL,
                        owner TEXT NOT NULL,
                        repo TEXT NOT NULL,
                        default_branch TEXT NOT NULL,
                        status TEXT NOT NULL,
                        primary_language TEXT DEFAULT '',
                        frameworks_json TEXT DEFAULT '[]',
                        analysis_json TEXT DEFAULT '{}',
                        error_message TEXT DEFAULT '',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    );
                    INSERT INTO projects (
                        id, repo_url, owner, repo, default_branch, status
                    )
                    VALUES (
                        'legacy-project', 'https://github.com/demo/legacy',
                        'demo', 'legacy', 'main', 'done'
                    );
                    """
                )
                conn.commit()
            finally:
                conn.close()

            db = Database(path)
            with db.connect() as conn:
                tables = {
                    row["name"]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    ).fetchall()
                }
                version = conn.execute(
                    "SELECT version FROM schema_versions WHERE key = 'database'"
                ).fetchone()["version"]

            self.assertIn("code_chunks", tables)
            self.assertIn("schema_versions", tables)
            self.assertGreaterEqual(version, 2)
            self.assertEqual(db.get_project("legacy-project")["repo"], "legacy")

    def test_schema_version_initialization_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schema.sqlite"
            db = Database(path)
            with db.connect() as conn:
                conn.execute(
                    """
                    UPDATE schema_versions
                    SET version = ?, updated_at = ?
                    WHERE key = ?
                    """,
                    (SCHEMA_VERSION, "fixed-timestamp", "database"),
                )

            reopened = Database(path)
            with reopened.connect() as conn:
                row = conn.execute(
                    "SELECT version, updated_at FROM schema_versions WHERE key = ?",
                    ("database",),
                ).fetchone()

            self.assertEqual(row["version"], SCHEMA_VERSION)
            self.assertEqual(row["updated_at"], "fixed-timestamp")

    def test_code_chunk_foreign_key_is_enabled_and_cascades(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "foreign-key.sqlite")
            project_id = _project_id(db)
            db.save_code_chunks_for_project(project_id, [_chunk()])

            with db.connect() as conn:
                foreign_keys_enabled = conn.execute("PRAGMA foreign_keys").fetchone()[0]
                conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))

            self.assertEqual(foreign_keys_enabled, 1)
            self.assertEqual(db.get_code_chunks(project_id), [])

    def test_saves_reads_and_filters_code_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "chunks.sqlite")
            project_id = _project_id(db)
            chunks = [
                _chunk("src/app.py", "target"),
                _chunk(
                    "src/service.py",
                    "Service.run",
                    "class Service:\n    def run(self):\n        return True\n",
                    "method",
                    "run",
                ),
            ]

            db.save_code_chunks_for_project(project_id, chunks)
            all_chunks = db.get_code_chunks(project_id)
            by_path = db.get_code_chunks(project_id, path="src\\app.py")
            by_symbol = db.get_code_chunks(project_id, symbol="Service.run")
            by_type = db.get_code_chunks(project_id, chunk_type="method")

            self.assertEqual(len(all_chunks), 2)
            self.assertEqual(by_path[0]["qualified_name"], "target")
            self.assertEqual(by_symbol[0]["path"], "src/service.py")
            self.assertEqual(by_type[0]["symbol_name"], "run")

    def test_repeated_project_save_does_not_duplicate_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "repeat.sqlite")
            project_id = _project_id(db)
            chunks = [_chunk()]

            db.save_code_chunks_for_project(project_id, chunks)
            db.save_code_chunks_for_project(project_id, chunks)

            self.assertEqual(len(db.get_code_chunks(project_id)), 1)

    def test_replaces_code_chunks_for_one_file(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "replace.sqlite")
            project_id = _project_id(db)
            db.save_code_chunks_for_project(
                project_id,
                [
                    _chunk("src/app.py", "old_name", "def old_name():\n    return 1\n", symbol_name="old_name"),
                    _chunk("src/other.py", "other", "def other():\n    return 2\n", symbol_name="other"),
                ],
            )

            db.replace_code_chunks_for_file(
                project_id,
                "src\\app.py",
                [_chunk("src/app.py", "new_name", "def new_name():\n    return 3\n", symbol_name="new_name")],
            )

            app_chunks = db.get_code_chunks(project_id, path="src/app.py")
            other_chunks = db.get_code_chunks(project_id, path="src/other.py")
            self.assertEqual([chunk["qualified_name"] for chunk in app_chunks], ["new_name"])
            self.assertEqual([chunk["qualified_name"] for chunk in other_chunks], ["other"])

    def test_empty_file_replacement_clears_previous_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "empty-replace.sqlite")
            project_id = _project_id(db)
            db.save_code_chunks_for_project(
                project_id,
                [
                    _chunk("src/app.py", "old_name", "def old_name():\n    return 1\n", symbol_name="old_name"),
                    _chunk("src/other.py", "other", "def other():\n    return 2\n", symbol_name="other"),
                ],
            )

            db.replace_code_chunks_for_file(project_id, "src/app.py", [])

            self.assertEqual(db.get_code_chunks(project_id, path="src/app.py"), [])
            self.assertEqual(
                [chunk["qualified_name"] for chunk in db.get_code_chunks(project_id, path="src/other.py")],
                ["other"],
            )

    def test_project_save_clears_stale_chunks_for_removed_python_files(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "stale.sqlite")
            project_id = _project_id(db)
            db.save_code_chunks_for_project(project_id, [_chunk("src/removed.py", "removed")])

            db.save_analysis(
                project_id,
                {
                    "primary_language": "Python",
                    "frameworks": [],
                    "files": [],
                    "modules": [],
                    "overview": "updated",
                },
                [
                    {
                        "path": "README.md",
                        "extension": ".md",
                        "language": "Markdown",
                        "size": 8,
                        "content": "# Demo\n",
                        "summary": "readme",
                        "importance": 1,
                        "is_core": True,
                        "imports": [],
                        "exports": [],
                        "symbols": [],
                    }
                ],
                [],
                [],
            )

            bundle = db.get_bundle(project_id)
            self.assertEqual(db.get_code_chunks(project_id), [])
            self.assertEqual([file["path"] for file in bundle["files"]], ["README.md"])

    def test_delete_project_removes_related_code_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "delete.sqlite")
            project_id = _project_id(db)
            db.save_code_chunks_for_project(project_id, [_chunk()])
            db.save_chat_answer(project_id, "question", "answer", [])

            db.delete_project(project_id)

            self.assertIsNone(db.get_project(project_id))
            self.assertEqual(db.get_code_chunks(project_id), [])
            self.assertIsNone(db.get_bundle(project_id))

    def test_code_chunk_save_rolls_back_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "rollback.sqlite")
            project_id = _project_id(db)
            original = _chunk("src/app.py", "original", "def original():\n    return 1\n", symbol_name="original")
            replacement = _chunk("src/app.py", "replacement", "def replacement():\n    return 2\n", symbol_name="replacement")
            invalid = _chunk("src/bad.py", "bad", "def bad():\n    return 3\n", symbol_name="bad")
            invalid["start_line"] = 0
            db.save_code_chunks_for_project(project_id, [original])

            with self.assertRaises(ValueError):
                db.save_code_chunks_for_project(project_id, [replacement, invalid])

            stored = db.get_code_chunks(project_id)
            self.assertEqual(len(stored), 1)
            self.assertEqual(stored[0]["qualified_name"], "original")

    def test_existing_project_qa_and_report_data_are_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "compat.sqlite")
            project_id = _project_id(db)
            analysis = {
                "primary_language": "Python",
                "frameworks": ["FastAPI"],
                "files": [],
                "modules": [{"name": "app", "responsibility": "backend"}],
                "overview": "Demo overview",
            }
            files = [
                {
                    "path": "app/main.py",
                    "extension": ".py",
                    "language": "Python",
                    "size": 12,
                    "content": "print('ok')\n",
                    "summary": "entry",
                    "importance": 1,
                    "is_core": True,
                    "imports": [],
                    "exports": [],
                    "symbols": [],
                }
            ]
            steps = [{"title": "Step", "goal": "Goal", "files": [], "tasks": [], "quiz": []}]

            db.save_analysis(project_id, analysis, files, steps)
            db.save_chat_answer(project_id, "入口在哪", "看 app/main.py", [{"path": "app/main.py"}])
            bundle = db.get_bundle(project_id)

            self.assertEqual(bundle["files"][0]["path"], "app/main.py")
            self.assertEqual(bundle["modules"][0]["name"], "app")
            self.assertEqual(bundle["learning_steps"][0]["title"], "Step")
            self.assertEqual(bundle["chat_answers"][0]["question"], "入口在哪")
            self.assertEqual(bundle["code_chunks"], [])


if __name__ == "__main__":
    unittest.main()
