import hashlib
import pickle
import sqlite3
import tempfile
import unittest
from pathlib import Path

from app.config import EmbeddingSettings
from app.database import Database
from app.services.embedding_service import (
    CODE_CHUNK_TEXT_FORMAT_VERSION,
    build_code_chunk_embedding_input_hash,
    build_embedding_config_hash,
)


MODEL_NAME = "fake-model"
MODEL_REVISION = ""


def _settings(**overrides) -> EmbeddingSettings:
    values = {
        "enabled": True,
        "model_name_or_path": MODEL_NAME,
        "device": "cpu",
        "batch_size": 2,
        "max_length": 128,
        "normalize": True,
        "cache_dir": Path("embedding-cache"),
        "query_prefix": "",
        "document_prefix": "",
        "model_revision": MODEL_REVISION,
    }
    values.update(overrides)
    return EmbeddingSettings(**values)


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
) -> dict:
    return {
        "repository_revision": "abc123",
        "language": "python",
        "path": path,
        "chunk_type": chunk_type,
        "symbol_name": symbol_name,
        "qualified_name": qualified_name,
        "parent_symbol": "",
        "start_line": start_line,
        "end_line": start_line + len(content.splitlines()) - 1,
        "content": content,
        "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }


def _save_chunk(db: Database, project_id: str, chunk: dict | None = None) -> dict:
    db.save_code_chunks_for_project(project_id, [chunk or _chunk()])
    return db.get_code_chunks(project_id)[0]


def _embedding_record(
    chunk: dict,
    vector=None,
    model_name: str = MODEL_NAME,
    text_format=None,
    settings: EmbeddingSettings | None = None,
):
    values = vector if vector is not None else [1.0, 0.0, 0.0]
    embedding_settings = settings or _settings()
    return {
        "code_chunk_id": chunk["id"],
        "content_hash": chunk["content_hash"],
        "embedding_input_hash": build_code_chunk_embedding_input_hash(
            chunk,
            embedding_settings,
        ),
        "model_name": model_name,
        "model_revision": embedding_settings.model_revision,
        "text_format_version": text_format or CODE_CHUNK_TEXT_FORMAT_VERSION,
        "embedding_config_hash": build_embedding_config_hash(embedding_settings),
        "embedding_dimension": len(values),
        "embedding_dtype": "float32",
        "normalized": embedding_settings.normalize,
        "vector": values,
    }


class DatabaseEmbeddingTests(unittest.TestCase):
    def test_legacy_database_initializes_embedding_table_without_data_loss(self):
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
            with db.connect() as reopened:
                tables = {
                    row["name"]
                    for row in reopened.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    ).fetchall()
                }
                version = reopened.execute(
                    "SELECT version FROM schema_versions WHERE key = 'database'"
                ).fetchone()["version"]

            self.assertIn("code_chunk_embeddings", tables)
            self.assertGreaterEqual(version, 3)
            self.assertEqual(db.get_project("legacy-project")["repo"], "legacy")

    def test_feature_schema_three_database_adds_embedding_hash_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feature-v3.sqlite"
            conn = sqlite3.connect(path)
            try:
                conn.executescript(
                    """
                    CREATE TABLE schema_versions (
                        key TEXT PRIMARY KEY,
                        version INTEGER NOT NULL,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    );
                    INSERT INTO schema_versions (key, version) VALUES ('database', 3);
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
                    CREATE TABLE code_chunks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        project_id TEXT NOT NULL,
                        repository_revision TEXT NOT NULL DEFAULT '',
                        language TEXT NOT NULL DEFAULT 'python',
                        path TEXT NOT NULL,
                        chunk_type TEXT NOT NULL,
                        symbol_name TEXT NOT NULL,
                        qualified_name TEXT NOT NULL,
                        parent_symbol TEXT DEFAULT '',
                        start_line INTEGER NOT NULL,
                        end_line INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
                    );
                    CREATE TABLE code_chunk_embeddings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        code_chunk_id INTEGER NOT NULL,
                        content_hash TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        model_revision TEXT NOT NULL DEFAULT '',
                        text_format_version TEXT NOT NULL,
                        embedding_dimension INTEGER NOT NULL,
                        embedding_dtype TEXT NOT NULL DEFAULT 'float32',
                        normalized INTEGER NOT NULL DEFAULT 1,
                        vector_blob BLOB NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(code_chunk_id) REFERENCES code_chunks(id) ON DELETE CASCADE
                    );
                    CREATE UNIQUE INDEX idx_code_chunk_embeddings_unique
                    ON code_chunk_embeddings (
                        code_chunk_id, content_hash, model_name,
                        model_revision, text_format_version
                    );
                    """
                )
                conn.commit()
            finally:
                conn.close()

            db = Database(path)
            with db.connect() as reopened:
                columns = {
                    row["name"]
                    for row in reopened.execute(
                        "PRAGMA table_info(code_chunk_embeddings)"
                    ).fetchall()
                }
                version = reopened.execute(
                    "SELECT version FROM schema_versions WHERE key = 'database'"
                ).fetchone()["version"]

            self.assertIn("embedding_input_hash", columns)
            self.assertIn("embedding_config_hash", columns)
            self.assertEqual(version, 5)

    def test_saves_and_reads_float32_vector_blob(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "vectors.sqlite")
            project_id = _project_id(db)
            chunk = _save_chunk(db, project_id)

            db.upsert_code_chunk_embeddings([_embedding_record(chunk, [0.0, 0.6, 0.8])])
            stored = db.get_code_chunk_embeddings_for_project(
                project_id,
                MODEL_NAME,
                MODEL_REVISION,
                CODE_CHUNK_TEXT_FORMAT_VERSION,
                build_embedding_config_hash(_settings()),
                True,
            )

            self.assertEqual(len(stored), 1)
            self.assertEqual(stored[0]["embedding_dtype"], "float32")
            self.assertEqual(stored[0]["embedding_dimension"], 3)
            self.assertAlmostEqual(stored[0]["vector"][2], 0.8)

    def test_vector_blob_is_not_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "not-pickle.sqlite")
            project_id = _project_id(db)
            chunk = _save_chunk(db, project_id)
            db.upsert_code_chunk_embeddings([_embedding_record(chunk)])

            with db.connect() as conn:
                blob = conn.execute(
                    "SELECT vector_blob FROM code_chunk_embeddings"
                ).fetchone()["vector_blob"]

            with self.assertRaises(Exception):
                pickle.loads(blob)

    def test_rejects_blob_with_wrong_byte_length(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "corrupt.sqlite")
            project_id = _project_id(db)
            chunk = _save_chunk(db, project_id)
            with db.connect() as conn:
                conn.execute(
                    """
                    INSERT INTO code_chunk_embeddings (
                        code_chunk_id, content_hash, embedding_input_hash,
                        model_name, model_revision, text_format_version,
                        embedding_config_hash, embedding_dimension,
                        embedding_dtype, normalized, vector_blob
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'float32', 1, ?)
                    """,
                    (
                        chunk["id"],
                        chunk["content_hash"],
                        build_code_chunk_embedding_input_hash(chunk, _settings()),
                        MODEL_NAME,
                        MODEL_REVISION,
                        CODE_CHUNK_TEXT_FORMAT_VERSION,
                        build_embedding_config_hash(_settings()),
                        3,
                        b"\x00\x00\x00\x00",
                    ),
                )

            with self.assertRaises(ValueError):
                db.get_code_chunk_embeddings_for_project(
                    project_id,
                    MODEL_NAME,
                    MODEL_REVISION,
                    CODE_CHUNK_TEXT_FORMAT_VERSION,
                    build_embedding_config_hash(_settings()),
                    True,
                )

    def test_rejects_blob_with_nan_value(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "nan.sqlite")
            project_id = _project_id(db)
            chunk = _save_chunk(db, project_id)
            nan_blob = b"\x00\x00\xc0\x7f" + b"\x00\x00\x00\x00" * 2
            with db.connect() as conn:
                conn.execute(
                    """
                    INSERT INTO code_chunk_embeddings (
                        code_chunk_id, content_hash, embedding_input_hash,
                        model_name, model_revision, text_format_version,
                        embedding_config_hash, embedding_dimension,
                        embedding_dtype, normalized, vector_blob
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, 3, 'float32', 1, ?)
                    """,
                    (
                        chunk["id"],
                        chunk["content_hash"],
                        build_code_chunk_embedding_input_hash(chunk, _settings()),
                        MODEL_NAME,
                        MODEL_REVISION,
                        CODE_CHUNK_TEXT_FORMAT_VERSION,
                        build_embedding_config_hash(_settings()),
                        nan_blob,
                    ),
                )

            with self.assertRaises(ValueError):
                db.get_code_chunk_embeddings_for_project(
                    project_id,
                    MODEL_NAME,
                    MODEL_REVISION,
                    CODE_CHUNK_TEXT_FORMAT_VERSION,
                    build_embedding_config_hash(_settings()),
                    True,
                )

    def test_rejects_non_unit_vector_marked_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "bad-normalized.sqlite")
            project_id = _project_id(db)
            chunk = _save_chunk(db, project_id)

            with self.assertRaises(ValueError):
                db.upsert_code_chunk_embeddings([_embedding_record(chunk, [2.0, 0.0, 0.0])])

    def test_repeated_cache_upsert_does_not_grow_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "dedupe.sqlite")
            project_id = _project_id(db)
            chunk = _save_chunk(db, project_id)
            record = _embedding_record(chunk)

            db.upsert_code_chunk_embeddings([record])
            db.upsert_code_chunk_embeddings([record])

            with db.connect() as conn:
                count = conn.execute("SELECT COUNT(*) FROM code_chunk_embeddings").fetchone()[0]
            self.assertEqual(count, 1)

    def test_stale_same_config_embeddings_are_pruned_on_new_upsert(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "prune.sqlite")
            project_id = _project_id(db)
            chunk = _save_chunk(db, project_id)
            old_record = _embedding_record(chunk)
            old_record["embedding_input_hash"] = "0" * 64
            db.upsert_code_chunk_embeddings([old_record])
            db.upsert_code_chunk_embeddings([_embedding_record(chunk)])

            with db.connect() as conn:
                rows = conn.execute(
                    """
                    SELECT embedding_input_hash
                    FROM code_chunk_embeddings
                    WHERE code_chunk_id = ?
                    """,
                    (chunk["id"],),
                ).fetchall()

            self.assertEqual(
                [row["embedding_input_hash"] for row in rows],
                [build_code_chunk_embedding_input_hash(chunk, _settings())],
            )

    def test_content_hash_change_makes_cache_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "hash-stale.sqlite")
            project_id = _project_id(db)
            chunk = _save_chunk(db, project_id)
            db.upsert_code_chunk_embeddings([_embedding_record(chunk)])
            with db.connect() as conn:
                conn.execute(
                    "UPDATE code_chunks SET content_hash = ? WHERE id = ?",
                    ("changed", chunk["id"]),
                )

            missing = db.get_code_chunks_missing_embeddings(
                project_id,
                MODEL_NAME,
                MODEL_REVISION,
                CODE_CHUNK_TEXT_FORMAT_VERSION,
                build_embedding_config_hash(_settings()),
                True,
            )

            self.assertEqual([item["id"] for item in missing], [chunk["id"]])

    def test_embedding_input_hash_change_makes_cache_stale_with_same_content_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "input-stale.sqlite")
            project_id = _project_id(db)
            chunk = _save_chunk(db, project_id)
            db.upsert_code_chunk_embeddings([_embedding_record(chunk)])
            with db.connect() as conn:
                conn.execute(
                    "UPDATE code_chunks SET qualified_name = ? WHERE id = ?",
                    ("renamed_target", chunk["id"]),
                )
            renamed = db.get_code_chunks(project_id)[0]
            expected_hashes = {
                renamed["id"]: build_code_chunk_embedding_input_hash(renamed, _settings())
            }

            missing = db.get_code_chunks_missing_embeddings(
                project_id,
                MODEL_NAME,
                MODEL_REVISION,
                CODE_CHUNK_TEXT_FORMAT_VERSION,
                build_embedding_config_hash(_settings()),
                True,
                expected_hashes,
            )

            self.assertEqual([item["id"] for item in missing], [renamed["id"]])

    def test_prefix_config_change_makes_cache_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "prefix-stale.sqlite")
            project_id = _project_id(db)
            chunk = _save_chunk(db, project_id)
            db.upsert_code_chunk_embeddings([_embedding_record(chunk, settings=_settings())])
            changed_settings = _settings(document_prefix="passage: ")

            missing = db.get_code_chunks_missing_embeddings(
                project_id,
                MODEL_NAME,
                MODEL_REVISION,
                CODE_CHUNK_TEXT_FORMAT_VERSION,
                build_embedding_config_hash(changed_settings),
                True,
                {
                    chunk["id"]: build_code_chunk_embedding_input_hash(
                        chunk,
                        changed_settings,
                    )
                },
            )

            self.assertEqual([item["id"] for item in missing], [chunk["id"]])

    def test_normalized_flag_mismatch_makes_cache_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "normalize-stale.sqlite")
            project_id = _project_id(db)
            chunk = _save_chunk(db, project_id)
            db.upsert_code_chunk_embeddings([_embedding_record(chunk, settings=_settings())])

            missing = db.get_code_chunks_missing_embeddings(
                project_id,
                MODEL_NAME,
                MODEL_REVISION,
                CODE_CHUNK_TEXT_FORMAT_VERSION,
                build_embedding_config_hash(_settings(normalize=False)),
                False,
                {
                    chunk["id"]: build_code_chunk_embedding_input_hash(
                        chunk,
                        _settings(normalize=False),
                    )
                },
            )

            self.assertEqual([item["id"] for item in missing], [chunk["id"]])

    def test_model_change_makes_cache_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "model-stale.sqlite")
            project_id = _project_id(db)
            chunk = _save_chunk(db, project_id)
            db.upsert_code_chunk_embeddings([_embedding_record(chunk, model_name="old-model")])

            missing = db.get_code_chunks_missing_embeddings(
                project_id,
                MODEL_NAME,
                MODEL_REVISION,
                CODE_CHUNK_TEXT_FORMAT_VERSION,
                build_embedding_config_hash(_settings()),
                True,
            )

            self.assertEqual(len(missing), 1)

    def test_model_revision_change_makes_cache_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "revision-stale.sqlite")
            project_id = _project_id(db)
            chunk = _save_chunk(db, project_id)
            db.upsert_code_chunk_embeddings(
                [_embedding_record(chunk, settings=_settings(model_revision="rev-a"))]
            )

            missing = db.get_code_chunks_missing_embeddings(
                project_id,
                MODEL_NAME,
                "rev-b",
                CODE_CHUNK_TEXT_FORMAT_VERSION,
                build_embedding_config_hash(_settings(model_revision="rev-b")),
                True,
                {
                    chunk["id"]: build_code_chunk_embedding_input_hash(
                        chunk,
                        _settings(model_revision="rev-b"),
                    )
                },
            )

            self.assertEqual(len(missing), 1)

    def test_text_format_change_makes_cache_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "format-stale.sqlite")
            project_id = _project_id(db)
            chunk = _save_chunk(db, project_id)
            db.upsert_code_chunk_embeddings([_embedding_record(chunk, text_format="old-format")])

            missing = db.get_code_chunks_missing_embeddings(
                project_id,
                MODEL_NAME,
                MODEL_REVISION,
                CODE_CHUNK_TEXT_FORMAT_VERSION,
                build_embedding_config_hash(_settings()),
                True,
            )

            self.assertEqual(len(missing), 1)

    def test_deleting_code_chunk_cascades_embedding(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "chunk-cascade.sqlite")
            project_id = _project_id(db)
            chunk = _save_chunk(db, project_id)
            db.upsert_code_chunk_embeddings([_embedding_record(chunk)])

            with db.connect() as conn:
                conn.execute("DELETE FROM code_chunks WHERE id = ?", (chunk["id"],))
                count = conn.execute("SELECT COUNT(*) FROM code_chunk_embeddings").fetchone()[0]

            self.assertEqual(count, 0)

    def test_batch_upsert_failure_rolls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "rollback.sqlite")
            project_id = _project_id(db)
            chunks = [
                _chunk("src/a.py", "a", "def a():\n    return 1\n", symbol_name="a"),
                _chunk("src/b.py", "b", "def b():\n    return 2\n", symbol_name="b"),
            ]
            db.save_code_chunks_for_project(project_id, chunks)
            stored = db.get_code_chunks(project_id)
            valid = _embedding_record(stored[0])
            invalid = _embedding_record(stored[1], [])

            with self.assertRaises(ValueError):
                db.upsert_code_chunk_embeddings([valid, invalid])

            with db.connect() as conn:
                count = conn.execute("SELECT COUNT(*) FROM code_chunk_embeddings").fetchone()[0]
            self.assertEqual(count, 0)

    def test_project_delete_removes_embeddings(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "project-delete.sqlite")
            project_id = _project_id(db)
            chunk = _save_chunk(db, project_id)
            db.upsert_code_chunk_embeddings([_embedding_record(chunk)])

            db.delete_project(project_id)

            with db.connect() as conn:
                count = conn.execute("SELECT COUNT(*) FROM code_chunk_embeddings").fetchone()[0]
            self.assertEqual(count, 0)

    def test_embedding_query_order_is_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "stable.sqlite")
            project_id = _project_id(db)
            chunks = [
                _chunk("src/b.py", "b", "def b():\n    return 2\n", symbol_name="b"),
                _chunk("src/a.py", "late", "def late():\n    return 3\n", symbol_name="late", start_line=10),
                _chunk("src/a.py", "early", "def early():\n    return 1\n", symbol_name="early", start_line=1),
            ]
            db.save_code_chunks_for_project(project_id, chunks)
            stored = db.get_code_chunks(project_id)
            db.upsert_code_chunk_embeddings([_embedding_record(chunk) for chunk in stored])

            rows = db.get_code_chunk_embeddings_for_project(
                project_id,
                MODEL_NAME,
                MODEL_REVISION,
                CODE_CHUNK_TEXT_FORMAT_VERSION,
                build_embedding_config_hash(_settings()),
                True,
            )

            self.assertEqual(
                [(row["path"], row["qualified_name"]) for row in rows],
                [("src/a.py", "early"), ("src/a.py", "late"), ("src/b.py", "b")],
            )


if __name__ == "__main__":
    unittest.main()
