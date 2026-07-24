from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import struct
import tempfile
import uuid
from pathlib import Path
from typing import Any, Sequence


DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "gitlearn.sqlite"
SCHEMA_VERSION = 4


SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_versions (
    key TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS projects (
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

CREATE TABLE IF NOT EXISTS repo_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    path TEXT NOT NULL,
    extension TEXT DEFAULT '',
    language TEXT DEFAULT '',
    size INTEGER DEFAULT 0,
    content TEXT DEFAULT '',
    summary TEXT DEFAULT '',
    importance REAL DEFAULT 0,
    is_core INTEGER DEFAULT 0,
    imports_json TEXT DEFAULT '[]',
    exports_json TEXT DEFAULT '[]',
    symbols_json TEXT DEFAULT '[]',
    FOREIGN KEY(project_id) REFERENCES projects(id)
);

CREATE TABLE IF NOT EXISTS modules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    name TEXT NOT NULL,
    responsibility TEXT DEFAULT '',
    files_json TEXT DEFAULT '[]',
    depends_on_json TEXT DEFAULT '[]',
    FOREIGN KEY(project_id) REFERENCES projects(id)
);

CREATE TABLE IF NOT EXISTS learning_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    step_order INTEGER NOT NULL,
    title TEXT NOT NULL,
    goal TEXT DEFAULT '',
    files_json TEXT DEFAULT '[]',
    tasks_json TEXT DEFAULT '[]',
    quiz_json TEXT DEFAULT '[]',
    FOREIGN KEY(project_id) REFERENCES projects(id)
);

CREATE TABLE IF NOT EXISTS chat_answers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    citations_json TEXT DEFAULT '[]',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(project_id) REFERENCES projects(id)
);

CREATE TABLE IF NOT EXISTS code_chunks (
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

CREATE TABLE IF NOT EXISTS code_chunk_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code_chunk_id INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    embedding_input_hash TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_revision TEXT NOT NULL DEFAULT '',
    text_format_version TEXT NOT NULL,
    embedding_config_hash TEXT NOT NULL DEFAULT '',
    embedding_dimension INTEGER NOT NULL,
    embedding_dtype TEXT NOT NULL DEFAULT 'float32',
    normalized INTEGER NOT NULL DEFAULT 1,
    vector_blob BLOB NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(code_chunk_id) REFERENCES code_chunks(id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_code_chunks_unique
ON code_chunks (
    project_id, repository_revision, path, chunk_type,
    qualified_name, start_line, end_line
);

CREATE INDEX IF NOT EXISTS idx_code_chunks_project_path
ON code_chunks (project_id, path);

CREATE INDEX IF NOT EXISTS idx_code_chunks_project_symbol
ON code_chunks (project_id, qualified_name);

"""


class _ManagedConnection(sqlite3.Connection):
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        try:
            return bool(super().__exit__(exc_type, exc_value, traceback))
        finally:
            self.close()


class Database:
    def __init__(self, path: Path | str | None = None) -> None:
        explicit_path = path or os.getenv("GITLEARN_DB")
        self.path = Path(explicit_path) if explicit_path else DEFAULT_DB_PATH
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._init_schema()
        except (PermissionError, sqlite3.OperationalError):
            if explicit_path:
                raise
            self.path = Path(tempfile.gettempdir()) / "gitlearnagent.sqlite"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._init_schema()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, factory=_ManagedConnection)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            self._migrate_schema(conn)
            conn.execute(
                "INSERT OR IGNORE INTO schema_versions (key, version) VALUES (?, ?)",
                ("database", SCHEMA_VERSION),
            )
            conn.execute(
                """
                UPDATE schema_versions
                SET version = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE key = ? AND version < ?
                """,
                (SCHEMA_VERSION, "database", SCHEMA_VERSION),
            )

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        embedding_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(code_chunk_embeddings)").fetchall()
        }
        if "embedding_input_hash" not in embedding_columns:
            conn.execute(
                """
                ALTER TABLE code_chunk_embeddings
                ADD COLUMN embedding_input_hash TEXT NOT NULL DEFAULT ''
                """
            )
        if "embedding_config_hash" not in embedding_columns:
            conn.execute(
                """
                ALTER TABLE code_chunk_embeddings
                ADD COLUMN embedding_config_hash TEXT NOT NULL DEFAULT ''
                """
            )
        conn.execute("DROP INDEX IF EXISTS idx_code_chunk_embeddings_unique")
        conn.execute("DROP INDEX IF EXISTS idx_code_chunk_embeddings_lookup")
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_code_chunk_embeddings_unique
            ON code_chunk_embeddings (
                code_chunk_id, embedding_input_hash, model_name, model_revision,
                text_format_version, embedding_config_hash, normalized
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_code_chunk_embeddings_lookup
            ON code_chunk_embeddings (
                model_name, model_revision, text_format_version,
                embedding_config_hash, normalized, content_hash
            )
            """
        )

    def create_project(self, snapshot: dict[str, Any]) -> str:
        project_id = str(uuid.uuid4())
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO projects (id, repo_url, owner, repo, default_branch, status)
                VALUES (?, ?, ?, ?, ?, 'analyzing')
                """,
                (
                    project_id,
                    snapshot["repo_url"],
                    snapshot["owner"],
                    snapshot["repo"],
                    snapshot["default_branch"],
                ),
            )
        return project_id

    def mark_failed(self, project_id: str, message: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE projects
                SET status = 'failed', error_message = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (message[:2000], project_id),
            )

    def save_analysis(
        self,
        project_id: str,
        analysis: dict[str, Any],
        files: list[dict[str, Any]],
        learning_steps: list[dict[str, Any]],
        code_chunks: list[dict[str, Any]] | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM repo_files WHERE project_id = ?", (project_id,))
            conn.execute("DELETE FROM modules WHERE project_id = ?", (project_id,))
            conn.execute("DELETE FROM learning_steps WHERE project_id = ?", (project_id,))

            for file in files:
                conn.execute(
                    """
                    INSERT INTO repo_files (
                        project_id, path, extension, language, size, content, summary,
                        importance, is_core, imports_json, exports_json, symbols_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        project_id,
                        file["path"],
                        file.get("extension", ""),
                        file.get("language", ""),
                        int(file.get("size", 0)),
                        file.get("content", ""),
                        file.get("summary", ""),
                        float(file.get("importance", 0)),
                        1 if file.get("is_core") else 0,
                        json.dumps(file.get("imports", []), ensure_ascii=False),
                        json.dumps(file.get("exports", []), ensure_ascii=False),
                        json.dumps(file.get("symbols", []), ensure_ascii=False),
                    ),
                )

            for module in analysis.get("modules", []):
                conn.execute(
                    """
                    INSERT INTO modules (project_id, name, responsibility, files_json, depends_on_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        project_id,
                        module["name"],
                        module.get("responsibility", ""),
                        json.dumps(module.get("files", []), ensure_ascii=False),
                        json.dumps(module.get("depends_on", []), ensure_ascii=False),
                    ),
                )

            for index, step in enumerate(learning_steps, start=1):
                conn.execute(
                    """
                    INSERT INTO learning_steps (
                        project_id, step_order, title, goal, files_json, tasks_json, quiz_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        project_id,
                        index,
                        step["title"],
                        step.get("goal", ""),
                        json.dumps(step.get("files", []), ensure_ascii=False),
                        json.dumps(step.get("tasks", []), ensure_ascii=False),
                        json.dumps(step.get("quiz", []), ensure_ascii=False),
                    ),
                )

            if code_chunks is not None:
                self._replace_code_chunks_in_scope(conn, project_id, code_chunks)

            conn.execute(
                """
                UPDATE projects
                SET status = 'done',
                    primary_language = ?,
                    frameworks_json = ?,
                    analysis_json = ?,
                    error_message = '',
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    analysis.get("primary_language", ""),
                    json.dumps(analysis.get("frameworks", []), ensure_ascii=False),
                    json.dumps(analysis, ensure_ascii=False),
                    project_id,
                ),
            )

    def save_code_chunks_for_project(
        self,
        project_id: str,
        code_chunks: list[dict[str, Any]],
    ) -> None:
        with self.connect() as conn:
            self._replace_code_chunks_in_scope(conn, project_id, code_chunks)

    def replace_code_chunks_for_file(
        self,
        project_id: str,
        path: str,
        code_chunks: list[dict[str, Any]],
        repository_revision: str | None = None,
    ) -> None:
        normalized_path = self._normalize_repo_path(path)
        with self.connect() as conn:
            self._replace_code_chunks_in_scope(
                conn,
                project_id,
                code_chunks,
                path=normalized_path,
                repository_revision=repository_revision,
            )

    def get_code_chunks(
        self,
        project_id: str,
        path: str | None = None,
        symbol: str | None = None,
        chunk_type: str | None = None,
        language: str | None = None,
    ) -> list[dict[str, Any]]:
        conditions = ["project_id = ?"]
        params: list[Any] = [project_id]
        if path:
            conditions.append("path = ?")
            params.append(self._normalize_repo_path(path))
        if symbol:
            conditions.append("(symbol_name = ? OR qualified_name = ?)")
            params.extend([symbol, symbol])
        if chunk_type:
            conditions.append("chunk_type = ?")
            params.append(chunk_type)
        if language:
            conditions.append("lower(language) = lower(?)")
            params.append(language)
        where_clause = " AND ".join(conditions)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM code_chunks
                WHERE {where_clause}
                ORDER BY path, start_line, qualified_name
                """,
                params,
            ).fetchall()
        return [self._code_chunk_from_row(row) for row in rows]

    def get_code_chunks_missing_embeddings(
        self,
        project_id: str,
        model_name: str,
        model_revision: str,
        text_format_version: str,
        embedding_config_hash: str = "",
        normalized: bool = True,
        embedding_input_hashes: dict[int, str] | None = None,
    ) -> list[dict[str, Any]]:
        with self.connect() as conn:
            if embedding_input_hashes is None:
                rows = conn.execute(
                    """
                    SELECT c.*
                    FROM code_chunks c
                    WHERE c.project_id = ?
                      AND NOT EXISTS (
                        SELECT 1
                        FROM code_chunk_embeddings e
                        WHERE e.code_chunk_id = c.id
                          AND e.content_hash = c.content_hash
                          AND e.model_name = ?
                          AND e.model_revision = ?
                          AND e.text_format_version = ?
                          AND e.embedding_config_hash = ?
                          AND e.normalized = ?
                      )
                    ORDER BY c.path, c.start_line, c.id
                    """,
                    (
                        project_id,
                        model_name,
                        model_revision,
                        text_format_version,
                        embedding_config_hash,
                        1 if normalized else 0,
                    ),
                ).fetchall()
                return [self._code_chunk_from_row(row) for row in rows]

            rows = conn.execute(
                """
                SELECT c.*
                FROM code_chunks c
                WHERE c.project_id = ?
                ORDER BY c.path, c.start_line, c.id
                """,
                (project_id,),
            ).fetchall()
            chunks = [self._code_chunk_from_row(row) for row in rows]
            stale: list[dict[str, Any]] = []
            for chunk in chunks:
                chunk_id = int(chunk["id"])
                expected_input_hash = embedding_input_hashes.get(chunk_id)
                if not expected_input_hash:
                    stale.append(chunk)
                    continue
                fresh = conn.execute(
                    """
                    SELECT 1
                    FROM code_chunk_embeddings e
                    WHERE e.code_chunk_id = ?
                      AND e.content_hash = ?
                      AND e.embedding_input_hash = ?
                      AND e.model_name = ?
                      AND e.model_revision = ?
                      AND e.text_format_version = ?
                      AND e.embedding_config_hash = ?
                      AND e.normalized = ?
                    LIMIT 1
                    """,
                    (
                        chunk_id,
                        chunk["content_hash"],
                        expected_input_hash,
                        model_name,
                        model_revision,
                        text_format_version,
                        embedding_config_hash,
                        1 if normalized else 0,
                    ),
                ).fetchone()
                if fresh is None:
                    stale.append(chunk)
            return stale

    def get_fresh_embedding_dimensions_for_project(
        self,
        project_id: str,
        model_name: str,
        model_revision: str,
        text_format_version: str,
        embedding_config_hash: str = "",
        normalized: bool = True,
    ) -> list[int]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT e.embedding_dimension
                FROM code_chunks c
                JOIN code_chunk_embeddings e ON e.code_chunk_id = c.id
                WHERE c.project_id = ?
                  AND e.content_hash = c.content_hash
                  AND e.model_name = ?
                  AND e.model_revision = ?
                  AND e.text_format_version = ?
                  AND e.embedding_config_hash = ?
                  AND e.normalized = ?
                ORDER BY e.embedding_dimension
                """,
                (
                    project_id,
                    model_name,
                    model_revision,
                    text_format_version,
                    embedding_config_hash,
                    1 if normalized else 0,
                ),
            ).fetchall()
        return [int(row["embedding_dimension"]) for row in rows]

    def upsert_code_chunk_embeddings(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        with self.connect() as conn:
            for record in records:
                values = self._embedding_values(record)
                conn.execute(
                    """
                    INSERT INTO code_chunk_embeddings (
                        code_chunk_id, content_hash, embedding_input_hash,
                        model_name, model_revision, text_format_version,
                        embedding_config_hash, embedding_dimension,
                        embedding_dtype, normalized, vector_blob
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (
                        code_chunk_id, embedding_input_hash, model_name,
                        model_revision, text_format_version,
                        embedding_config_hash, normalized
                    )
                    DO UPDATE SET
                        content_hash = excluded.content_hash,
                        embedding_dimension = excluded.embedding_dimension,
                        embedding_dtype = excluded.embedding_dtype,
                        vector_blob = excluded.vector_blob,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    values,
                )
                self._delete_stale_embeddings_for_record(conn, record)

    def get_code_chunk_embeddings_for_project(
        self,
        project_id: str,
        model_name: str,
        model_revision: str,
        text_format_version: str,
        embedding_config_hash: str = "",
        normalized: bool = True,
        path: str | None = None,
        chunk_type: str | None = None,
        language: str | None = None,
        symbol: str | None = None,
    ) -> list[dict[str, Any]]:
        conditions = [
            "c.project_id = ?",
            "e.content_hash = c.content_hash",
            "e.model_name = ?",
            "e.model_revision = ?",
            "e.text_format_version = ?",
            "e.embedding_config_hash = ?",
            "e.normalized = ?",
        ]
        params: list[Any] = [
            project_id,
            model_name,
            model_revision,
            text_format_version,
            embedding_config_hash,
            1 if normalized else 0,
        ]
        if path:
            conditions.append("c.path = ?")
            params.append(self._normalize_repo_path(path))
        if chunk_type:
            conditions.append("c.chunk_type = ?")
            params.append(chunk_type)
        if language:
            conditions.append("lower(c.language) = lower(?)")
            params.append(language)
        if symbol:
            conditions.append("(c.symbol_name = ? OR c.qualified_name = ?)")
            params.extend([symbol, symbol])
        where_clause = " AND ".join(conditions)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    c.*,
                    e.model_name AS embedding_model_name,
                    e.model_revision AS embedding_model_revision,
                    e.text_format_version AS embedding_text_format_version,
                    e.embedding_input_hash,
                    e.embedding_config_hash,
                    e.embedding_dimension,
                    e.embedding_dtype,
                    e.normalized,
                    e.vector_blob
                FROM code_chunks c
                JOIN code_chunk_embeddings e ON e.code_chunk_id = c.id
                WHERE {where_clause}
                ORDER BY c.path, c.start_line, c.id
                """,
                params,
            ).fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            item = self._code_chunk_from_row(row)
            dimension = int(row["embedding_dimension"])
            item.update(
                {
                    "model_name": row["embedding_model_name"],
                    "model_revision": row["embedding_model_revision"],
                    "text_format_version": row["embedding_text_format_version"],
                    "embedding_input_hash": row["embedding_input_hash"],
                    "embedding_config_hash": row["embedding_config_hash"],
                    "embedding_dimension": dimension,
                    "embedding_dtype": row["embedding_dtype"],
                    "normalized": bool(row["normalized"]),
                    "vector": unpack_float32_vector(row["vector_blob"], dimension),
                }
            )
            results.append(item)
        return results

    def get_evidence_source(
        self,
        project_id: str,
        code_chunk_id: int,
        path: str,
    ) -> dict[str, Any] | None:
        """Read project, chunk and stored source in one SQLite snapshot."""
        normalized_path = self._normalize_repo_path(path)
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    p.id AS project_id,
                    p.repo_url,
                    p.owner,
                    p.repo,
                    p.default_branch,
                    c.id AS code_chunk_id,
                    c.repository_revision,
                    c.language AS chunk_language,
                    c.path AS chunk_path,
                    c.chunk_type,
                    c.symbol_name,
                    c.qualified_name,
                    c.parent_symbol,
                    c.start_line,
                    c.end_line,
                    c.content AS chunk_content,
                    c.content_hash,
                    f.language AS file_language,
                    f.content AS file_content
                FROM projects p
                JOIN code_chunks c ON c.project_id = p.id
                JOIN repo_files f
                  ON f.project_id = p.id
                 AND f.path = c.path
                WHERE p.id = ?
                  AND c.id = ?
                  AND c.path = ?
                LIMIT 1
                """,
                (project_id, int(code_chunk_id), normalized_path),
            ).fetchone()
        return dict(row) if row is not None else None

    def delete_project(self, project_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM code_chunks WHERE project_id = ?", (project_id,))
            conn.execute("DELETE FROM chat_answers WHERE project_id = ?", (project_id,))
            conn.execute("DELETE FROM learning_steps WHERE project_id = ?", (project_id,))
            conn.execute("DELETE FROM modules WHERE project_id = ?", (project_id,))
            conn.execute("DELETE FROM repo_files WHERE project_id = ?", (project_id,))
            conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        if not row:
            return None
        return self._project_from_row(row)

    def get_bundle(self, project_id: str) -> dict[str, Any] | None:
        project = self.get_project(project_id)
        if not project:
            return None

        with self.connect() as conn:
            file_rows = conn.execute(
                "SELECT * FROM repo_files WHERE project_id = ? ORDER BY importance DESC, path",
                (project_id,),
            ).fetchall()
            module_rows = conn.execute(
                "SELECT * FROM modules WHERE project_id = ? ORDER BY name",
                (project_id,),
            ).fetchall()
            step_rows = conn.execute(
                "SELECT * FROM learning_steps WHERE project_id = ? ORDER BY step_order",
                (project_id,),
            ).fetchall()
            chat_rows = conn.execute(
                "SELECT * FROM chat_answers WHERE project_id = ? ORDER BY id DESC LIMIT 20",
                (project_id,),
            ).fetchall()
            code_chunk_rows = conn.execute(
                """
                SELECT * FROM code_chunks
                WHERE project_id = ?
                ORDER BY path, start_line, qualified_name
                """,
                (project_id,),
            ).fetchall()

        return {
            "project": project,
            "files": [self._file_from_row(row) for row in file_rows],
            "modules": [self._module_from_row(row) for row in module_rows],
            "learning_steps": [self._step_from_row(row) for row in step_rows],
            "chat_answers": [self._chat_from_row(row) for row in chat_rows],
            "code_chunks": [self._code_chunk_from_row(row) for row in code_chunk_rows],
            "analysis": project.get("analysis", {}),
        }

    def save_chat_answer(
        self,
        project_id: str,
        question: str,
        answer: str,
        citations: list[dict[str, Any]],
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_answers (project_id, question, answer, citations_json)
                VALUES (?, ?, ?, ?)
                """,
                (project_id, question, answer, json.dumps(citations, ensure_ascii=False)),
            )

    @staticmethod
    def _json(value: str, fallback: Any) -> Any:
        try:
            return json.loads(value or "")
        except json.JSONDecodeError:
            return fallback

    def _project_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "repo_url": row["repo_url"],
            "owner": row["owner"],
            "repo": row["repo"],
            "default_branch": row["default_branch"],
            "status": row["status"],
            "primary_language": row["primary_language"],
            "frameworks": self._json(row["frameworks_json"], []),
            "analysis": self._json(row["analysis_json"], {}),
            "error_message": row["error_message"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _file_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "path": row["path"],
            "extension": row["extension"],
            "language": row["language"],
            "size": row["size"],
            "content": row["content"],
            "summary": row["summary"],
            "importance": row["importance"],
            "is_core": bool(row["is_core"]),
            "imports": self._json(row["imports_json"], []),
            "exports": self._json(row["exports_json"], []),
            "symbols": self._json(row["symbols_json"], []),
        }

    def _module_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "name": row["name"],
            "responsibility": row["responsibility"],
            "files": self._json(row["files_json"], []),
            "depends_on": self._json(row["depends_on_json"], []),
        }

    def _step_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "order": row["step_order"],
            "title": row["title"],
            "goal": row["goal"],
            "files": self._json(row["files_json"], []),
            "tasks": self._json(row["tasks_json"], []),
            "quiz": self._json(row["quiz_json"], []),
        }

    def _chat_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "question": row["question"],
            "answer": row["answer"],
            "citations": self._json(row["citations_json"], []),
            "created_at": row["created_at"],
        }

    def _insert_code_chunk(
        self,
        conn: sqlite3.Connection,
        project_id: str,
        chunk: dict[str, Any],
    ) -> None:
        self._insert_prepared_code_chunk(conn, self._prepare_code_chunk(project_id, chunk))

    def _replace_code_chunks_in_scope(
        self,
        conn: sqlite3.Connection,
        project_id: str,
        code_chunks: list[dict[str, Any]],
        path: str | None = None,
        repository_revision: str | None = None,
    ) -> None:
        prepared = [self._prepare_code_chunk(project_id, chunk) for chunk in code_chunks]
        if path is not None:
            for chunk in prepared:
                if chunk["path"] != path:
                    raise ValueError(f"code chunk path {chunk['path']} does not match {path}")
        if repository_revision is not None:
            for chunk in prepared:
                if chunk["repository_revision"] != repository_revision:
                    raise ValueError(
                        "code chunk repository revision does not match replacement scope"
                    )

        conditions = ["project_id = ?"]
        params: list[Any] = [project_id]
        if path is not None:
            conditions.append("path = ?")
            params.append(path)
        if repository_revision is not None:
            conditions.append("repository_revision = ?")
            params.append(repository_revision)
        existing_rows = conn.execute(
            f"SELECT * FROM code_chunks WHERE {' AND '.join(conditions)}",
            params,
        ).fetchall()

        counts: dict[tuple[str, str, str, str], int] = {}
        for row in existing_rows:
            key = self._code_chunk_match_key(row)
            counts[key] = counts.get(key, 0) + 1
        existing_by_key = {
            self._code_chunk_match_key(row): row
            for row in existing_rows
            if counts[self._code_chunk_match_key(row)] == 1
        }

        kept_ids: set[int] = set()
        for chunk in prepared:
            existing = existing_by_key.get(self._code_chunk_match_key(chunk))
            if existing is not None and int(existing["id"]) not in kept_ids:
                chunk_id = int(existing["id"])
                self._update_code_chunk(conn, chunk_id, chunk)
                kept_ids.add(chunk_id)
            else:
                cursor = self._insert_prepared_code_chunk(conn, chunk)
                kept_ids.add(int(cursor.lastrowid))

        stale_ids = [int(row["id"]) for row in existing_rows if int(row["id"]) not in kept_ids]
        if stale_ids:
            placeholders = ",".join("?" for _ in stale_ids)
            conn.execute(
                f"DELETE FROM code_chunks WHERE id IN ({placeholders})",
                stale_ids,
            )

    def _prepare_code_chunk(
        self,
        project_id: str,
        chunk: dict[str, Any],
    ) -> dict[str, Any]:
        content = chunk["content"]
        if not isinstance(content, str):
            raise ValueError("code chunk content must be a string")
        content_hash = chunk["content_hash"]
        expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if content_hash != expected_hash:
            raise ValueError(
                f"code chunk hash mismatch for {chunk.get('path', '')} "
                f"{chunk.get('qualified_name', '')}"
            )
        start_line = int(chunk["start_line"])
        end_line = int(chunk["end_line"])
        if start_line < 1 or end_line < start_line:
            raise ValueError(
                f"invalid code chunk line range for {chunk.get('path', '')} "
                f"{chunk.get('qualified_name', '')}: {start_line}-{end_line}"
            )
        return {
            "project_id": project_id,
            "repository_revision": chunk.get("repository_revision") or "",
            "language": chunk.get("language") or "python",
            "path": self._normalize_repo_path(chunk["path"]),
            "chunk_type": chunk["chunk_type"],
            "symbol_name": chunk["symbol_name"],
            "qualified_name": chunk["qualified_name"],
            "parent_symbol": chunk.get("parent_symbol") or "",
            "start_line": start_line,
            "end_line": end_line,
            "content": content,
            "content_hash": content_hash,
        }

    def _insert_prepared_code_chunk(
        self,
        conn: sqlite3.Connection,
        chunk: dict[str, Any],
    ) -> sqlite3.Cursor:
        return conn.execute(
            """
            INSERT INTO code_chunks (
                project_id, repository_revision, language, path, chunk_type,
                symbol_name, qualified_name, parent_symbol, start_line,
                end_line, content, content_hash
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk["project_id"],
                chunk["repository_revision"],
                chunk["language"],
                chunk["path"],
                chunk["chunk_type"],
                chunk["symbol_name"],
                chunk["qualified_name"],
                chunk["parent_symbol"],
                chunk["start_line"],
                chunk["end_line"],
                chunk["content"],
                chunk["content_hash"],
            ),
        )

    def _update_code_chunk(
        self,
        conn: sqlite3.Connection,
        chunk_id: int,
        chunk: dict[str, Any],
    ) -> None:
        conn.execute(
            """
            UPDATE code_chunks
            SET repository_revision = ?,
                language = ?,
                path = ?,
                chunk_type = ?,
                symbol_name = ?,
                qualified_name = ?,
                parent_symbol = ?,
                start_line = ?,
                end_line = ?,
                content = ?,
                content_hash = ?
            WHERE id = ?
            """,
            (
                chunk["repository_revision"],
                chunk["language"],
                chunk["path"],
                chunk["chunk_type"],
                chunk["symbol_name"],
                chunk["qualified_name"],
                chunk["parent_symbol"],
                chunk["start_line"],
                chunk["end_line"],
                chunk["content"],
                chunk["content_hash"],
                chunk_id,
            ),
        )

    @staticmethod
    def _code_chunk_match_key(row: sqlite3.Row | dict[str, Any]) -> tuple[str, str, str, str]:
        return (
            row["repository_revision"],
            row["path"],
            row["chunk_type"],
            row["qualified_name"],
        )

    def _embedding_values(self, record: dict[str, Any]) -> tuple[Any, ...]:
        vector = list(record["vector"])
        dimension = int(record.get("embedding_dimension") or len(vector))
        if dimension < 1:
            raise ValueError("embedding dimension must be positive")
        if len(vector) != dimension:
            raise ValueError(
                f"embedding vector length {len(vector)} does not match dimension {dimension}"
            )
        dtype = record.get("embedding_dtype") or "float32"
        if dtype != "float32":
            raise ValueError(f"unsupported embedding dtype: {dtype}")
        normalized = bool(record.get("normalized", True))
        if normalized:
            _validate_normalized_vector(vector)
        return (
            int(record["code_chunk_id"]),
            record["content_hash"],
            _require_hash(record["embedding_input_hash"], "embedding_input_hash"),
            record["model_name"],
            record.get("model_revision") or "",
            record["text_format_version"],
            _require_hash(record["embedding_config_hash"], "embedding_config_hash"),
            dimension,
            dtype,
            1 if normalized else 0,
            pack_float32_vector(vector),
        )

    def _delete_stale_embeddings_for_record(
        self,
        conn: sqlite3.Connection,
        record: dict[str, Any],
    ) -> None:
        conn.execute(
            """
            DELETE FROM code_chunk_embeddings
            WHERE code_chunk_id = ?
              AND model_name = ?
              AND model_revision = ?
              AND text_format_version = ?
              AND embedding_config_hash = ?
              AND normalized = ?
              AND embedding_input_hash != ?
            """,
            (
                int(record["code_chunk_id"]),
                record["model_name"],
                record.get("model_revision") or "",
                record["text_format_version"],
                _require_hash(record["embedding_config_hash"], "embedding_config_hash"),
                1 if record.get("normalized", True) else 0,
                _require_hash(record["embedding_input_hash"], "embedding_input_hash"),
            ),
        )

    def _code_chunk_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "project_id": row["project_id"],
            "repository_revision": row["repository_revision"],
            "language": row["language"],
            "path": row["path"],
            "chunk_type": row["chunk_type"],
            "symbol_name": row["symbol_name"],
            "qualified_name": row["qualified_name"],
            "parent_symbol": row["parent_symbol"],
            "start_line": row["start_line"],
            "end_line": row["end_line"],
            "content": row["content"],
            "content_hash": row["content_hash"],
            "created_at": row["created_at"],
        }

    @staticmethod
    def _normalize_repo_path(path: str) -> str:
        return path.replace("\\", "/").lstrip("/")


def pack_float32_vector(vector: Sequence[float]) -> bytes:
    values = [_as_float32(value) for value in vector]
    if not values:
        raise ValueError("embedding vector must not be empty")
    return struct.pack(f"<{len(values)}f", *values)


def unpack_float32_vector(blob: bytes, dimension: int) -> list[float]:
    if dimension < 1:
        raise ValueError("embedding dimension must be positive")
    expected_length = dimension * 4
    if len(blob) != expected_length:
        raise ValueError(
            f"embedding vector byte length {len(blob)} does not match "
            f"dimension {dimension}"
        )
    values = list(struct.unpack(f"<{dimension}f", blob))
    if any(not math.isfinite(value) for value in values):
        raise ValueError("embedding vectors must contain only finite numbers")
    return values


def _as_float32(value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("embedding vectors must contain only finite numbers")
    return struct.unpack("<f", struct.pack("<f", number))[0]


def _validate_normalized_vector(vector: Sequence[float]) -> None:
    values = [_as_float32(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isclose(norm, 1.0, rel_tol=1e-3, abs_tol=1e-3):
        raise ValueError("embedding vector is marked normalized but has non-unit norm")


def _require_hash(value: Any, field_name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{field_name} must be a lowercase sha256 hex digest")
    return text
