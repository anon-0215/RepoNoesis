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

from app.learning_schema import LEARNING_SCHEMA_STATEMENTS


DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "gitlearn.sqlite"
SCHEMA_VERSION = 11
CODE_CHUNKER_VERSION = "python_ast_chunks_v1@1"
LEARNING_CONTINUITY_CONFIG_IDENTITY = "learning-continuity-v1@1"
LOCAL_CONTINUITY_LEARNER_ID = "learner-local-single-user-v1"
UPDATE_RUN_PHASES = (
    "revision_resolution",
    "manifest_diff",
    "source_analysis",
    "chunk_update",
    "relation_update",
    "embedding_update",
    "snapshot_validation",
    "activation",
)


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
    repository_revision TEXT NOT NULL DEFAULT '',
    source_type TEXT NOT NULL DEFAULT 'legacy_github',
    source_location TEXT NOT NULL DEFAULT '',
    source_identity TEXT NOT NULL DEFAULT '',
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
    identity_schema_version TEXT NOT NULL DEFAULT 'legacy',
    wrapper_model_identity TEXT NOT NULL DEFAULT '',
    resolved_revision TEXT NOT NULL DEFAULT '',
    identity_eligible INTEGER NOT NULL DEFAULT 0,
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

CREATE TABLE IF NOT EXISTS relation_nodes (
    node_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    repository_revision TEXT NOT NULL,
    language TEXT NOT NULL,
    node_type TEXT NOT NULL,
    path TEXT NOT NULL,
    code_chunk_id INTEGER,
    symbol_name TEXT NOT NULL DEFAULT '',
    qualified_name TEXT NOT NULL DEFAULT '',
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE,
    FOREIGN KEY(code_chunk_id) REFERENCES code_chunks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS code_relations (
    edge_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    repository_revision TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    source_node_id TEXT NOT NULL,
    source_path TEXT NOT NULL,
    source_chunk_id INTEGER,
    source_symbol TEXT NOT NULL DEFAULT '',
    source_start_line INTEGER NOT NULL,
    source_end_line INTEGER NOT NULL,
    target_node_id TEXT,
    target_path TEXT,
    target_chunk_id INTEGER,
    target_symbol TEXT,
    target_start_line INTEGER,
    target_end_line INTEGER,
    raw_target_name TEXT NOT NULL DEFAULT '',
    resolution_status TEXT NOT NULL,
    resolution_rule TEXT NOT NULL,
    language TEXT NOT NULL,
    source_content_hash TEXT NOT NULL,
    target_content_hash TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE,
    FOREIGN KEY(source_node_id) REFERENCES relation_nodes(node_id) ON DELETE CASCADE,
    FOREIGN KEY(target_node_id) REFERENCES relation_nodes(node_id) ON DELETE CASCADE,
    FOREIGN KEY(source_chunk_id) REFERENCES code_chunks(id) ON DELETE CASCADE,
    FOREIGN KEY(target_chunk_id) REFERENCES code_chunks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS relation_index_runs (
    project_id TEXT NOT NULL,
    repository_revision TEXT NOT NULL,
    status TEXT NOT NULL,
    parsed_files INTEGER NOT NULL DEFAULT 0,
    failed_files INTEGER NOT NULL DEFAULT 0,
    unsupported_files INTEGER NOT NULL DEFAULT 0,
    node_count INTEGER NOT NULL DEFAULT 0,
    edge_count INTEGER NOT NULL DEFAULT 0,
    warnings_json TEXT NOT NULL DEFAULT '[]',
    indexed_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(project_id, repository_revision),
    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
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

CREATE INDEX IF NOT EXISTS idx_relation_nodes_revision_path
ON relation_nodes (project_id, repository_revision, path);

CREATE INDEX IF NOT EXISTS idx_relation_nodes_revision_symbol
ON relation_nodes (project_id, repository_revision, qualified_name);

CREATE INDEX IF NOT EXISTS idx_code_relations_revision_source
ON code_relations (project_id, repository_revision, source_node_id);

CREATE INDEX IF NOT EXISTS idx_code_relations_revision_target
ON code_relations (project_id, repository_revision, target_node_id);

CREATE INDEX IF NOT EXISTS idx_code_relations_revision_type
ON code_relations (project_id, repository_revision, relation_type);

CREATE INDEX IF NOT EXISTS idx_code_relations_revision_symbol
ON code_relations (
    project_id, repository_revision, source_symbol, target_symbol
);

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
        self._activation_test_hook: Any | None = None
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
            try:
                conn.execute("BEGIN IMMEDIATE")
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
            except Exception:
                conn.rollback()
                raise
            else:
                conn.commit()

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        project_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(projects)").fetchall()
        }
        if "repository_revision" not in project_columns:
            conn.execute(
                """
                ALTER TABLE projects
                ADD COLUMN repository_revision TEXT NOT NULL DEFAULT ''
                """
            )
        for column, default_sql in (
            ("source_type", "'legacy_github'"),
            ("source_location", "''"),
            ("source_identity", "''"),
        ):
            if column not in project_columns:
                conn.execute(
                    f"ALTER TABLE projects ADD COLUMN {column} TEXT NOT NULL DEFAULT {default_sql}"
                )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_projects_source_identity
            ON projects (source_identity)
            WHERE source_identity != ''
            """
        )
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
        if "identity_schema_version" not in embedding_columns:
            conn.execute(
                "ALTER TABLE code_chunk_embeddings "
                "ADD COLUMN identity_schema_version TEXT NOT NULL DEFAULT 'legacy'"
            )
        if "wrapper_model_identity" not in embedding_columns:
            conn.execute(
                "ALTER TABLE code_chunk_embeddings "
                "ADD COLUMN wrapper_model_identity TEXT NOT NULL DEFAULT ''"
            )
        if "resolved_revision" not in embedding_columns:
            conn.execute(
                "ALTER TABLE code_chunk_embeddings "
                "ADD COLUMN resolved_revision TEXT NOT NULL DEFAULT ''"
            )
        if "identity_eligible" not in embedding_columns:
            conn.execute(
                "ALTER TABLE code_chunk_embeddings "
                "ADD COLUMN identity_eligible INTEGER NOT NULL DEFAULT 0"
            )
        conn.execute("DROP INDEX IF EXISTS idx_code_chunk_embeddings_unique")
        conn.execute("DROP INDEX IF EXISTS idx_code_chunk_embeddings_lookup")
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_code_chunk_embeddings_unique
            ON code_chunk_embeddings (
                code_chunk_id, content_hash, embedding_input_hash,
                identity_schema_version, wrapper_model_identity,
                resolved_revision, embedding_dimension, normalized,
                text_format_version, embedding_config_hash
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_code_chunk_embeddings_lookup
            ON code_chunk_embeddings (
                identity_eligible, identity_schema_version,
                wrapper_model_identity, resolved_revision,
                embedding_dimension, normalized, text_format_version,
                embedding_config_hash, content_hash, embedding_input_hash
            )
            """
        )
        for statement in LEARNING_SCHEMA_STATEMENTS:
            conn.execute(statement)
        learning_event_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(learning_events)").fetchall()
        }
        if "event_order" not in learning_event_columns:
            conn.execute(
                "ALTER TABLE learning_events ADD COLUMN event_order INTEGER NOT NULL DEFAULT 0"
            )
            conn.execute(
                "UPDATE learning_events SET event_order = rowid WHERE event_order = 0"
            )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_learning_events_target_order
            ON learning_events (learner_id, project_id, target_id, event_order)
            """
        )
        self._migrate_workspace_schema(conn)

    def _migrate_workspace_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS repository_workspaces (
                id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL CHECK(length(display_name) BETWEEN 1 AND 300),
                source_type TEXT NOT NULL,
                source_location TEXT NOT NULL DEFAULT '',
                active_project_id TEXT NOT NULL UNIQUE,
                activation_version INTEGER NOT NULL DEFAULT 1,
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
            CREATE TABLE IF NOT EXISTS workspace_revisions (
                workspace_id TEXT NOT NULL,
                project_id TEXT NOT NULL UNIQUE,
                repository_revision TEXT NOT NULL DEFAULT '',
                parent_project_id TEXT,
                manifest_hash TEXT NOT NULL DEFAULT '',
                chunker_version TEXT NOT NULL DEFAULT '',
                embedding_identity TEXT NOT NULL DEFAULT '',
                activation_status TEXT NOT NULL DEFAULT 'active',
                activated_at TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(workspace_id, project_id),
                FOREIGN KEY(workspace_id) REFERENCES repository_workspaces(id)
                    ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED,
                FOREIGN KEY(project_id) REFERENCES projects(id)
                    ON DELETE RESTRICT DEFERRABLE INITIALLY DEFERRED,
                FOREIGN KEY(parent_project_id) REFERENCES projects(id)
                    ON DELETE RESTRICT DEFERRABLE INITIALLY DEFERRED
            )
            """
        )
        self._backfill_legacy_workspaces(conn)
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_repository_workspaces_order
            ON repository_workspaces (updated_at DESC, created_at DESC, id ASC)
            """
        )
        self._migrate_workspace_update_schema(conn)
        self._migrate_learning_continuity_schema(conn)
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_workspace_revisions_project
            ON workspace_revisions (project_id)
            """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_projects_delete_workspace
            BEFORE DELETE ON projects
            BEGIN
                DELETE FROM repository_workspaces WHERE active_project_id = OLD.id;
            END
            """
        )

    def _backfill_legacy_workspaces(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            """
            SELECT p.id, p.repo, p.source_type, p.source_location,
                   p.repository_revision, p.created_at, p.updated_at
            FROM projects AS p
            LEFT JOIN workspace_revisions AS wr ON wr.project_id = p.id
            WHERE wr.project_id IS NULL
            ORDER BY p.created_at, p.id
            """
        ).fetchall()
        for row in rows:
            workspace_id = self._workspace_id_for_project(row["id"])
            conn.execute(
                """
                INSERT INTO repository_workspaces (
                    id, display_name, source_type, source_location,
                    active_project_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    workspace_id,
                    row["repo"] or "Unnamed repository",
                    row["source_type"] or "legacy_github",
                    row["source_location"] or "",
                    row["id"],
                    row["created_at"],
                    row["updated_at"],
                ),
            )
            conn.execute(
                """
                INSERT INTO workspace_revisions (
                    workspace_id, project_id, repository_revision, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    workspace_id,
                    row["id"],
                    row["repository_revision"] or "",
                    row["created_at"],
                ),
            )

    def _migrate_workspace_update_schema(self, conn: sqlite3.Connection) -> None:
        workspace_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(repository_workspaces)")
        }
        if "activation_version" not in workspace_columns:
            conn.execute(
                "ALTER TABLE repository_workspaces "
                "ADD COLUMN activation_version INTEGER NOT NULL DEFAULT 1"
            )

        revision_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(workspace_revisions)")
        }
        additions = (
            ("parent_project_id", "TEXT REFERENCES projects(id) ON DELETE RESTRICT"),
            ("manifest_hash", "TEXT NOT NULL DEFAULT ''"),
            ("chunker_version", "TEXT NOT NULL DEFAULT ''"),
            ("embedding_identity", "TEXT NOT NULL DEFAULT ''"),
            ("activation_status", "TEXT NOT NULL DEFAULT 'active'"),
            ("activated_at", "TEXT"),
        )
        for column, declaration in additions:
            if column not in revision_columns:
                conn.execute(
                    f"ALTER TABLE workspace_revisions ADD COLUMN {column} {declaration}"
                )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS repository_update_runs (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                base_project_id TEXT NOT NULL,
                project_id TEXT,
                target_revision TEXT NOT NULL,
                config_identity TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('pending', 'running', 'succeeded', 'failed')),
                phase TEXT NOT NULL,
                result TEXT NOT NULL DEFAULT '' CHECK(result IN ('', 'unchanged', 'activated')),
                stats_json TEXT NOT NULL DEFAULT '{}',
                error_code TEXT NOT NULL DEFAULT '',
                error_message TEXT NOT NULL DEFAULT '',
                retryable INTEGER NOT NULL DEFAULT 0,
                retry_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                started_at TEXT,
                finished_at TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(workspace_id, target_revision, config_identity),
                FOREIGN KEY(workspace_id) REFERENCES repository_workspaces(id) ON DELETE CASCADE,
                FOREIGN KEY(base_project_id) REFERENCES projects(id) ON DELETE RESTRICT,
                FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE RESTRICT
            )
            """
        )
        self._backfill_workspace_update_metadata(conn)
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_workspace_revisions_revision
            ON workspace_revisions(workspace_id, repository_revision)
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_workspace_revisions_active
            ON workspace_revisions(workspace_id)
            WHERE activation_status = 'active'
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_repository_update_runs_status
            ON repository_update_runs(status, updated_at, id)
            """
        )

    def _migrate_learning_continuity_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            INSERT OR IGNORE INTO learner_profiles (
                learner_id, profile_type, status, schema_version
            ) VALUES (?, 'local_single_user', 'active', 1)
            """,
            (LOCAL_CONTINUITY_LEARNER_ID,),
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS learning_continuity_transitions (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                source_project_id TEXT NOT NULL,
                target_project_id TEXT NOT NULL,
                source_revision TEXT NOT NULL,
                target_revision TEXT NOT NULL,
                activation_version INTEGER NOT NULL CHECK(activation_version >= 2),
                learner_id TEXT NOT NULL,
                mapping_config_identity TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('pending', 'running', 'succeeded', 'failed')),
                stats_json TEXT NOT NULL DEFAULT '{}',
                error_code TEXT NOT NULL DEFAULT '',
                error_message TEXT NOT NULL DEFAULT '',
                retryable INTEGER NOT NULL DEFAULT 0,
                retry_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                started_at TEXT,
                finished_at TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(workspace_id, source_project_id, target_project_id, learner_id, mapping_config_identity),
                UNIQUE(workspace_id, activation_version, learner_id),
                FOREIGN KEY(workspace_id) REFERENCES repository_workspaces(id) ON DELETE CASCADE,
                FOREIGN KEY(source_project_id) REFERENCES projects(id) ON DELETE RESTRICT,
                FOREIGN KEY(target_project_id) REFERENCES projects(id) ON DELETE RESTRICT,
                FOREIGN KEY(learner_id) REFERENCES learner_profiles(learner_id) ON DELETE RESTRICT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS learning_continuity_mappings (
                transition_id TEXT NOT NULL,
                source_target_id TEXT NOT NULL,
                target_target_id TEXT,
                mapping_status TEXT NOT NULL CHECK(mapping_status IN (
                    'unchanged_exact', 'renamed_exact', 'modified', 'deleted',
                    'ambiguous', 'unmapped', 'incompatible'
                )),
                mapping_rule TEXT NOT NULL,
                source_mastery_status TEXT NOT NULL DEFAULT 'unseen',
                derived_mastery_status TEXT NOT NULL DEFAULT '',
                source_content_hash TEXT NOT NULL DEFAULT '',
                target_content_hash TEXT NOT NULL DEFAULT '',
                source_path TEXT NOT NULL DEFAULT '',
                target_path TEXT NOT NULL DEFAULT '',
                source_qualified_name TEXT NOT NULL DEFAULT '',
                target_qualified_name TEXT NOT NULL DEFAULT '',
                review_reason TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(transition_id, source_target_id),
                FOREIGN KEY(transition_id) REFERENCES learning_continuity_transitions(id) ON DELETE CASCADE,
                FOREIGN KEY(source_target_id) REFERENCES learning_targets(target_id) ON DELETE RESTRICT,
                FOREIGN KEY(target_target_id) REFERENCES learning_targets(target_id) ON DELETE RESTRICT
            )
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_learning_continuity_target_once
            ON learning_continuity_mappings(transition_id, target_target_id)
            WHERE target_target_id IS NOT NULL
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_learning_continuity_transition_status
            ON learning_continuity_transitions(workspace_id, status, activation_version DESC)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_learning_continuity_mapping_status
            ON learning_continuity_mappings(transition_id, mapping_status, source_target_id)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS learning_continuity_goal_lineage (
                transition_id TEXT NOT NULL,
                source_goal_id TEXT NOT NULL,
                target_goal_id TEXT NOT NULL,
                source_plan_id TEXT,
                target_plan_id TEXT,
                lineage_status TEXT NOT NULL CHECK(lineage_status IN ('carried', 'needs_review', 'history_only')),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(transition_id, source_goal_id),
                FOREIGN KEY(transition_id) REFERENCES learning_continuity_transitions(id) ON DELETE CASCADE,
                FOREIGN KEY(source_goal_id) REFERENCES learning_goals(goal_id) ON DELETE RESTRICT,
                FOREIGN KEY(target_goal_id) REFERENCES learning_goals(goal_id) ON DELETE RESTRICT,
                FOREIGN KEY(source_plan_id) REFERENCES learning_plans(plan_id) ON DELETE RESTRICT,
                FOREIGN KEY(target_plan_id) REFERENCES learning_plans(plan_id) ON DELETE RESTRICT
            )
            """
        )

    def _backfill_workspace_update_metadata(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            """
            SELECT wr.workspace_id, wr.project_id, wr.manifest_hash,
                   wr.chunker_version, w.active_project_id
            FROM workspace_revisions AS wr
            JOIN repository_workspaces AS w ON w.id = wr.workspace_id
            """
        ).fetchall()
        for row in rows:
            manifest_hash = row["manifest_hash"] or self._manifest_hash_for_project(
                conn, row["project_id"]
            )
            status = "active" if row["project_id"] == row["active_project_id"] else "superseded"
            conn.execute(
                """
                UPDATE workspace_revisions
                SET manifest_hash = ?,
                    chunker_version = CASE WHEN chunker_version = '' THEN ? ELSE chunker_version END,
                    activation_status = ?,
                    activated_at = CASE
                        WHEN ? = 'active' THEN COALESCE(activated_at, created_at)
                        ELSE activated_at
                    END
                WHERE workspace_id = ? AND project_id = ?
                """,
                (
                    manifest_hash,
                    CODE_CHUNKER_VERSION,
                    status,
                    status,
                    row["workspace_id"],
                    row["project_id"],
                ),
            )

    @staticmethod
    def _manifest_hash_for_project(conn: sqlite3.Connection, project_id: str) -> str:
        rows = conn.execute(
            "SELECT path, content, size FROM repo_files WHERE project_id=? ORDER BY path",
            (project_id,),
        ).fetchall()
        payload = [
            {
                "path": str(row["path"]).replace("\\", "/").lstrip("/"),
                "content_hash": hashlib.sha256(str(row["content"]).encode("utf-8")).hexdigest(),
                "size": int(row["size"]),
            }
            for row in rows
        ]
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return f"manifest-sha256:{digest}"

    @staticmethod
    def _workspace_id_for_project(project_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"reponoesis:workspace:{project_id}"))

    def create_project(self, snapshot: dict[str, Any]) -> str:
        project_id = str(uuid.uuid4())
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO projects (
                    id, repo_url, owner, repo, default_branch,
                    repository_revision, source_type, source_location,
                    source_identity, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'analyzing')
                """,
                (
                    project_id,
                    snapshot["repo_url"],
                    snapshot["owner"],
                    snapshot["repo"],
                    snapshot["default_branch"],
                    snapshot.get("repository_revision", ""),
                    snapshot.get("source_type", "legacy_github"),
                    snapshot.get("source_location", ""),
                    snapshot.get("source_identity", ""),
                ),
            )
            workspace_id = self._workspace_id_for_project(project_id)
            conn.execute(
                """
                INSERT INTO repository_workspaces (
                    id, display_name, source_type, source_location,
                    active_project_id
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    workspace_id,
                    snapshot.get("repo") or "Unnamed repository",
                    snapshot.get("source_type", "legacy_github"),
                    snapshot.get("source_location", ""),
                    project_id,
                ),
            )
            conn.execute(
                """
                INSERT INTO workspace_revisions (
                    workspace_id, project_id, repository_revision
                ) VALUES (?, ?, ?)
                """,
                (
                    workspace_id,
                    project_id,
                    snapshot.get("repository_revision", ""),
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

    def begin_reanalysis(self, project_id: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE projects
                SET status = 'analyzing', error_message = '', updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (project_id,),
            )

    def save_analysis(
        self,
        project_id: str,
        analysis: dict[str, Any],
        files: list[dict[str, Any]],
        learning_steps: list[dict[str, Any]],
        code_chunks: list[dict[str, Any]] | None = None,
    ) -> None:
        prepared_chunks = (
            self._prepare_code_chunk_replacement(project_id, code_chunks)
            if code_chunks is not None
            else None
        )
        with self.connect() as conn:
            conn.execute("DELETE FROM code_relations WHERE project_id = ?", (project_id,))
            conn.execute("DELETE FROM relation_nodes WHERE project_id = ?", (project_id,))
            conn.execute(
                "DELETE FROM relation_index_runs WHERE project_id = ?", (project_id,)
            )
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

            if prepared_chunks is not None:
                self._replace_code_chunks_in_scope(conn, project_id, prepared_chunks)

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
            conn.execute(
                """
                UPDATE workspace_revisions
                SET manifest_hash = ?,
                    chunker_version = CASE WHEN chunker_version = '' THEN ? ELSE chunker_version END
                WHERE project_id = ?
                """,
                (
                    self._manifest_hash_for_project(conn, project_id),
                    CODE_CHUNKER_VERSION,
                    project_id,
                ),
            )

    def save_code_chunks_for_project(
        self,
        project_id: str,
        code_chunks: list[dict[str, Any]],
    ) -> None:
        prepared = self._prepare_code_chunk_replacement(project_id, code_chunks)
        with self.connect() as conn:
            self._invalidate_relation_index(conn, project_id)
            self._replace_code_chunks_in_scope(conn, project_id, prepared)

    def replace_code_chunks_for_file(
        self,
        project_id: str,
        path: str,
        code_chunks: list[dict[str, Any]],
        repository_revision: str | None = None,
    ) -> None:
        normalized_path = self._normalize_repo_path(path)
        prepared = self._prepare_code_chunk_replacement(
            project_id,
            code_chunks,
            path=normalized_path,
            repository_revision=repository_revision,
        )
        with self.connect() as conn:
            self._invalidate_relation_index(conn, project_id)
            self._replace_code_chunks_in_scope(
                conn,
                project_id,
                prepared,
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

    def get_code_chunks_for_hierarchy(
        self,
        project_id: str,
        repository_revision: str,
        path: str,
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Read one exact hierarchy scope with a caller-owned hard row limit."""
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or limit < 1
            or limit > 2_049
        ):
            raise ValueError("hierarchy row limit must be between 1 and 2049")
        normalized_path = self._normalize_repo_path(path)
        if not project_id or not repository_revision or not normalized_path:
            raise ValueError("hierarchy lookup requires project, revision, and path")
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM code_chunks
                WHERE project_id = ?
                  AND repository_revision = ?
                  AND path = ?
                ORDER BY start_line, end_line DESC, qualified_name, chunk_type, id
                LIMIT ?
                """,
                (
                    project_id,
                    repository_revision,
                    normalized_path,
                    limit,
                ),
            ).fetchall()
        return [self._code_chunk_from_row(row) for row in rows]

    def get_code_chunks_by_ids_bounded(
        self,
        project_id: str,
        repository_revision: str,
        code_chunk_ids: list[int],
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Resolve exact relation node targets without scanning the chunk index."""
        if not project_id or not repository_revision:
            raise ValueError("relation chunk lookup requires project and revision")
        if not code_chunk_ids:
            return []
        if len(code_chunk_ids) > 64:
            raise ValueError("relation chunk lookup accepts at most 64 identities")
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or limit < 1
            or limit > 65
        ):
            raise ValueError("relation chunk row limit must be between 1 and 65")
        normalized_ids = sorted({int(value) for value in code_chunk_ids})
        placeholders = ",".join("?" for _ in normalized_ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM code_chunks
                WHERE project_id = ?
                  AND repository_revision = ?
                  AND id IN ({placeholders})
                ORDER BY id, path, start_line, end_line, qualified_name
                LIMIT ?
                """,
                [project_id, repository_revision, *normalized_ids, limit],
            ).fetchall()
        return [self._code_chunk_from_row(row) for row in rows]

    def replace_relation_index(
        self,
        project_id: str,
        repository_revision: str,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        *,
        status: str,
        parsed_files: int,
        failed_files: int,
        unsupported_files: int,
        warnings: list[str],
    ) -> None:
        if status not in {"complete", "partial"}:
            raise ValueError("relation index status must be complete or partial")
        prepared_nodes = [
            self._prepare_relation_node(project_id, repository_revision, item)
            for item in nodes
        ]
        node_ids = {item["node_id"] for item in prepared_nodes}
        if len(node_ids) != len(prepared_nodes):
            raise ValueError("relation node IDs must be unique")
        prepared_edges = [
            self._prepare_relation_edge(
                project_id, repository_revision, item, node_ids
            )
            for item in edges
        ]
        if len({item["edge_id"] for item in prepared_edges}) != len(prepared_edges):
            raise ValueError("relation edge IDs must be unique")
        with self.connect() as conn:
            revisions = {
                str(row["repository_revision"])
                for row in conn.execute(
                    "SELECT repository_revision FROM code_chunks WHERE project_id = ?",
                    (project_id,),
                ).fetchall()
            }
            if revisions and revisions != {repository_revision}:
                raise ValueError("relation revision does not match code chunk revision")
            conn.execute(
                "DELETE FROM code_relations WHERE project_id = ? AND repository_revision = ?",
                (project_id, repository_revision),
            )
            conn.execute(
                "DELETE FROM relation_nodes WHERE project_id = ? AND repository_revision = ?",
                (project_id, repository_revision),
            )
            for item in prepared_nodes:
                conn.execute(
                    """
                    INSERT INTO relation_nodes (
                        node_id, project_id, repository_revision, language,
                        node_type, path, code_chunk_id, symbol_name,
                        qualified_name, start_line, end_line, content_hash
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item["node_id"],
                        project_id,
                        repository_revision,
                        item["language"],
                        item["node_type"],
                        item["path"],
                        item["code_chunk_id"],
                        item["symbol_name"],
                        item["qualified_name"],
                        item["start_line"],
                        item["end_line"],
                        item["content_hash"],
                    ),
                )
            for item in prepared_edges:
                conn.execute(
                    """
                    INSERT INTO code_relations (
                        edge_id, project_id, repository_revision, relation_type,
                        source_node_id, source_path, source_chunk_id, source_symbol,
                        source_start_line, source_end_line, target_node_id,
                        target_path, target_chunk_id, target_symbol,
                        target_start_line, target_end_line, raw_target_name,
                        resolution_status, resolution_rule, language,
                        source_content_hash, target_content_hash
                    )
                    VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?
                    )
                    """,
                    (
                        item["edge_id"],
                        project_id,
                        repository_revision,
                        item["relation_type"],
                        item["source_node_id"],
                        item["source_path"],
                        item["source_chunk_id"],
                        item["source_symbol"],
                        item["source_start_line"],
                        item["source_end_line"],
                        item["target_node_id"],
                        item["target_path"],
                        item["target_chunk_id"],
                        item["target_symbol"],
                        item["target_start_line"],
                        item["target_end_line"],
                        item["raw_target_name"],
                        item["resolution_status"],
                        item["resolution_rule"],
                        item["language"],
                        item["source_content_hash"],
                        item["target_content_hash"],
                    ),
                )
            conn.execute(
                """
                INSERT INTO relation_index_runs (
                    project_id, repository_revision, status, parsed_files,
                    failed_files, unsupported_files, node_count, edge_count,
                    warnings_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id, repository_revision)
                DO UPDATE SET
                    status = excluded.status,
                    parsed_files = excluded.parsed_files,
                    failed_files = excluded.failed_files,
                    unsupported_files = excluded.unsupported_files,
                    node_count = excluded.node_count,
                    edge_count = excluded.edge_count,
                    warnings_json = excluded.warnings_json,
                    indexed_at = CURRENT_TIMESTAMP
                """,
                (
                    project_id,
                    repository_revision,
                    status,
                    int(parsed_files),
                    int(failed_files),
                    int(unsupported_files),
                    len(prepared_nodes),
                    len(prepared_edges),
                    json.dumps(warnings[:100], ensure_ascii=False),
                ),
            )

    def get_relation_index_status(
        self, project_id: str, repository_revision: str
    ) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM relation_index_runs
                WHERE project_id = ? AND repository_revision = ?
                """,
                (project_id, repository_revision),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["warnings"] = self._json(result.pop("warnings_json"), [])
        return result

    def get_relation_nodes(
        self,
        project_id: str,
        repository_revision: str,
        *,
        node_ids: list[str] | None = None,
        code_chunk_ids: list[int] | None = None,
        path: str | None = None,
        qualified_name: str | None = None,
    ) -> list[dict[str, Any]]:
        conditions = ["project_id = ?", "repository_revision = ?"]
        params: list[Any] = [project_id, repository_revision]
        if node_ids is not None:
            if not node_ids:
                return []
            placeholders = ",".join("?" for _ in node_ids)
            conditions.append(f"node_id IN ({placeholders})")
            params.extend(node_ids)
        if code_chunk_ids is not None:
            if not code_chunk_ids:
                return []
            placeholders = ",".join("?" for _ in code_chunk_ids)
            conditions.append(f"code_chunk_id IN ({placeholders})")
            params.extend(int(value) for value in code_chunk_ids)
        if path is not None:
            conditions.append("path = ?")
            params.append(self._normalize_repo_path(path))
        if qualified_name is not None:
            conditions.append("qualified_name = ?")
            params.append(qualified_name)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM relation_nodes
                WHERE {' AND '.join(conditions)}
                ORDER BY path, start_line, qualified_name, node_id
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def get_relation_nodes_bounded(
        self,
        project_id: str,
        repository_revision: str,
        *,
        node_ids: list[str] | None = None,
        code_chunk_ids: list[int] | None = None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Read exact relation nodes with a mandatory identity filter and LIMIT."""
        if not project_id or not repository_revision:
            raise ValueError("bounded relation node lookup requires project and revision")
        if node_ids is None and code_chunk_ids is None:
            raise ValueError("bounded relation node lookup requires exact identities")
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or limit < 1
            or limit > 65
        ):
            raise ValueError("relation node row limit must be between 1 and 65")
        conditions = ["project_id = ?", "repository_revision = ?"]
        params: list[Any] = [project_id, repository_revision]
        if node_ids is not None:
            normalized_node_ids = sorted(set(node_ids))
            if not normalized_node_ids:
                return []
            if len(normalized_node_ids) > 64:
                raise ValueError("relation node lookup accepts at most 64 node IDs")
            placeholders = ",".join("?" for _ in normalized_node_ids)
            conditions.append(f"node_id IN ({placeholders})")
            params.extend(normalized_node_ids)
        if code_chunk_ids is not None:
            normalized_chunk_ids = sorted({int(value) for value in code_chunk_ids})
            if not normalized_chunk_ids:
                return []
            if len(normalized_chunk_ids) > 64:
                raise ValueError("relation node lookup accepts at most 64 chunk IDs")
            placeholders = ",".join("?" for _ in normalized_chunk_ids)
            conditions.append(f"code_chunk_id IN ({placeholders})")
            params.extend(normalized_chunk_ids)
        params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM relation_nodes
                WHERE {' AND '.join(conditions)}
                ORDER BY path, start_line, end_line, qualified_name, node_id
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def get_relation_neighbors_bounded(
        self,
        project_id: str,
        repository_revision: str,
        *,
        seed_node_ids: list[str],
        relation_types: list[str],
        direction: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Read one deterministic, scoped batch of one-hop relation edges."""
        if not project_id or not repository_revision:
            raise ValueError("relation neighbor lookup requires project and revision")
        if direction not in {"outgoing", "incoming", "both"}:
            raise ValueError("invalid bounded relation direction")
        node_ids = sorted(set(seed_node_ids))
        types = sorted(set(relation_types))
        if not node_ids or not types:
            return []
        if len(node_ids) > 12:
            raise ValueError("relation neighbor lookup accepts at most 12 seeds")
        if not set(types).issubset({"imports", "calls", "references", "defines"}):
            raise ValueError("unknown relation type")
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or limit < 1
            or limit > 97
        ):
            raise ValueError("relation row limit must be between 1 and 97")
        type_placeholders = ",".join("?" for _ in types)
        node_placeholders = ",".join("?" for _ in node_ids)
        params: list[Any] = [project_id, repository_revision, *types]
        if direction == "outgoing":
            endpoint_clause = f"source_node_id IN ({node_placeholders})"
            params.extend(node_ids)
        elif direction == "incoming":
            endpoint_clause = f"target_node_id IN ({node_placeholders})"
            params.extend(node_ids)
        else:
            endpoint_clause = (
                f"(source_node_id IN ({node_placeholders}) "
                f"OR target_node_id IN ({node_placeholders}))"
            )
            params.extend([*node_ids, *node_ids])
        params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM code_relations
                WHERE project_id = ?
                  AND repository_revision = ?
                  AND relation_type IN ({type_placeholders})
                  AND {endpoint_clause}
                ORDER BY source_path, source_start_line, relation_type,
                         raw_target_name, COALESCE(target_path, ''),
                         COALESCE(target_start_line, 0), edge_id
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def get_relations(
        self,
        project_id: str,
        repository_revision: str,
        *,
        relation_types: list[str] | None = None,
        source_node_ids: list[str] | None = None,
        target_node_ids: list[str] | None = None,
        resolution_statuses: list[str] | None = None,
        edge_ids: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        conditions = ["project_id = ?", "repository_revision = ?"]
        params: list[Any] = [project_id, repository_revision]
        for column, values in (
            ("edge_id", edge_ids),
            ("relation_type", relation_types),
            ("source_node_id", source_node_ids),
            ("target_node_id", target_node_ids),
            ("resolution_status", resolution_statuses),
        ):
            if values is not None:
                if not values:
                    return []
                placeholders = ",".join("?" for _ in values)
                conditions.append(f"{column} IN ({placeholders})")
                params.extend(values)
        limit_clause = ""
        if limit is not None:
            if (
                not isinstance(limit, int)
                or isinstance(limit, bool)
                or limit < 1
                or limit > 129
            ):
                raise ValueError("relation query limit must be between 1 and 129")
            limit_clause = "LIMIT ?"
            params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM code_relations
                WHERE {' AND '.join(conditions)}
                ORDER BY source_path, source_start_line, relation_type,
                         raw_target_name, COALESCE(target_path, ''),
                         COALESCE(target_start_line, 0), edge_id
                {limit_clause}
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def get_code_chunks_missing_embeddings(
        self,
        project_id: str,
        model_name: str,
        model_revision: str,
        text_format_version: str,
        embedding_config_hash: str = "",
        normalized: bool = True,
        embedding_input_hashes: dict[int, str] | None = None,
        effective_identity: Any | None = None,
    ) -> list[dict[str, Any]]:
        identity_clause, identity_params = _embedding_identity_predicate(
            "e", effective_identity
        )
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
                          {identity_clause}
                      )
                    ORDER BY c.path, c.start_line, c.id
                    """.format(identity_clause=identity_clause),
                    (
                        project_id,
                        model_name,
                        model_revision,
                        text_format_version,
                        embedding_config_hash,
                        1 if normalized else 0,
                        *identity_params,
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
                      {identity_clause}
                    LIMIT 1
                    """.format(identity_clause=identity_clause),
                    (
                        chunk_id,
                        chunk["content_hash"],
                        expected_input_hash,
                        model_name,
                        model_revision,
                        text_format_version,
                        embedding_config_hash,
                        1 if normalized else 0,
                        *identity_params,
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
        effective_identity: Any | None = None,
    ) -> list[int]:
        identity_clause, identity_params = _embedding_identity_predicate(
            "e", effective_identity
        )
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
                  {identity_clause}
                ORDER BY e.embedding_dimension
                """.format(identity_clause=identity_clause),
                (
                    project_id,
                    model_name,
                    model_revision,
                    text_format_version,
                    embedding_config_hash,
                    1 if normalized else 0,
                    *identity_params,
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
                        model_name, model_revision, identity_schema_version,
                        wrapper_model_identity, resolved_revision, identity_eligible,
                        text_format_version,
                        embedding_config_hash, embedding_dimension,
                        embedding_dtype, normalized, vector_blob
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (
                        code_chunk_id, content_hash, embedding_input_hash,
                        identity_schema_version, wrapper_model_identity,
                        resolved_revision, embedding_dimension, normalized,
                        text_format_version, embedding_config_hash
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
        effective_identity: Any | None = None,
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
        identity_clause, identity_params = _embedding_identity_predicate(
            "e", effective_identity
        )
        if identity_clause:
            conditions.append(identity_clause.removeprefix("AND "))
            params.extend(identity_params)
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
                    e.identity_schema_version,
                    e.wrapper_model_identity,
                    e.resolved_revision,
                    e.identity_eligible,
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
            vector = unpack_float32_vector(row["vector_blob"], dimension)
            if bool(row["normalized"]):
                _validate_normalized_vector(vector)
            item.update(
                {
                    "model_name": row["embedding_model_name"],
                    "model_revision": row["embedding_model_revision"],
                    "identity_schema_version": row["identity_schema_version"],
                    "wrapper_model_identity": row["wrapper_model_identity"],
                    "resolved_revision": row["resolved_revision"],
                    "identity_eligible": bool(row["identity_eligible"]),
                    "text_format_version": row["embedding_text_format_version"],
                    "embedding_input_hash": row["embedding_input_hash"],
                    "embedding_config_hash": row["embedding_config_hash"],
                    "embedding_dimension": dimension,
                    "embedding_dtype": row["embedding_dtype"],
                    "normalized": bool(row["normalized"]),
                    "vector": vector,
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
            workspace_row = conn.execute(
                "SELECT workspace_id FROM workspace_revisions WHERE project_id = ?",
                (project_id,),
            ).fetchone()
            if workspace_row is not None:
                conn.execute(
                    "DELETE FROM repository_workspaces WHERE id = ?",
                    (workspace_row["workspace_id"],),
                )
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

    def get_project_by_source_identity(
        self, source_identity: str
    ) -> dict[str, Any] | None:
        if not source_identity:
            return None
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM projects WHERE source_identity = ?",
                (source_identity,),
            ).fetchone()
        return self._project_from_row(row) if row else None

    def get_workspace_for_project(self, project_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT w.*
                FROM repository_workspaces AS w
                JOIN workspace_revisions AS wr
                  ON wr.workspace_id = w.id AND wr.project_id = w.active_project_id
                WHERE wr.project_id = ? AND wr.activation_status = 'active'
                """,
                (project_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def list_workspace_records(self, limit: int, offset: int) -> dict[str, Any]:
        with self.connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM repository_workspaces"
            ).fetchone()[0]
            rows = conn.execute(
                """
                SELECT
                    w.id AS workspace_id,
                    w.display_name,
                    w.source_type,
                    w.source_location,
                    w.active_project_id,
                    w.activation_version,
                    w.created_at,
                    w.updated_at,
                    p.repository_revision,
                    p.status AS project_status,
                    p.primary_language,
                    p.frameworks_json,
                    p.updated_at AS project_updated_at,
                    wr.workspace_id AS revision_workspace_id,
                    wr.project_id AS revision_project_id,
                    wr.repository_revision AS linked_revision
                    , wr.activation_status AS linked_activation_status
                    , wr.manifest_hash AS linked_manifest_hash
                FROM repository_workspaces AS w
                LEFT JOIN projects AS p ON p.id = w.active_project_id
                LEFT JOIN workspace_revisions AS wr
                  ON wr.workspace_id = w.id AND wr.project_id = w.active_project_id
                ORDER BY p.updated_at DESC, w.updated_at DESC, w.created_at DESC, w.id ASC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return {"items": [dict(row) for row in rows], "total": int(total)}

    def get_workspace_record(self, workspace_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    w.id AS workspace_id,
                    w.display_name,
                    w.source_type,
                    w.source_location,
                    w.active_project_id,
                    w.activation_version,
                    w.created_at,
                    w.updated_at,
                    p.repository_revision,
                    p.status AS project_status,
                    p.primary_language,
                    p.frameworks_json,
                    p.updated_at AS project_updated_at,
                    wr.workspace_id AS revision_workspace_id,
                    wr.project_id AS revision_project_id,
                    wr.repository_revision AS linked_revision
                    , wr.activation_status AS linked_activation_status
                    , wr.manifest_hash AS linked_manifest_hash
                FROM repository_workspaces AS w
                LEFT JOIN projects AS p ON p.id = w.active_project_id
                LEFT JOIN workspace_revisions AS wr
                  ON wr.workspace_id = w.id AND wr.project_id = w.active_project_id
                WHERE w.id = ?
                """,
                (workspace_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def get_workspace_revision(
        self, workspace_id: str, repository_revision: str
    ) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM workspace_revisions
                WHERE workspace_id=? AND repository_revision=?
                """,
                (workspace_id, repository_revision),
            ).fetchone()
        return dict(row) if row is not None else None

    def count_workspace_revisions(self, workspace_id: str) -> int:
        with self.connect() as conn:
            return int(
                conn.execute(
                    "SELECT COUNT(*) FROM workspace_revisions WHERE workspace_id=?",
                    (workspace_id,),
                ).fetchone()[0]
            )

    def create_or_get_update_run(
        self,
        workspace_id: str,
        target_revision: str,
        config_identity: str,
    ) -> dict[str, Any]:
        if len(target_revision) != 40:
            raise ValueError("target revision must be a 40-character Git commit")
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            workspace = conn.execute(
                """
                SELECT w.active_project_id, p.repository_revision
                FROM repository_workspaces AS w
                JOIN projects AS p ON p.id=w.active_project_id
                WHERE w.id=?
                """,
                (workspace_id,),
            ).fetchone()
            if workspace is None:
                raise LookupError("workspace not found")
            existing = conn.execute(
                """
                SELECT * FROM repository_update_runs
                WHERE workspace_id=? AND target_revision=? AND config_identity=?
                """,
                (workspace_id, target_revision, config_identity),
            ).fetchone()
            if existing is not None:
                return self._update_run_from_row(existing)
            run_id = str(uuid.uuid4())
            unchanged = workspace["repository_revision"] == target_revision
            conn.execute(
                """
                INSERT INTO repository_update_runs (
                    id, workspace_id, base_project_id, target_revision,
                    config_identity, status, phase, result, finished_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'revision_resolution', ?,
                          CASE WHEN ? THEN CURRENT_TIMESTAMP ELSE NULL END)
                """,
                (
                    run_id,
                    workspace_id,
                    workspace["active_project_id"],
                    target_revision,
                    config_identity,
                    "succeeded" if unchanged else "pending",
                    "unchanged" if unchanged else "",
                    1 if unchanged else 0,
                ),
            )
            row = conn.execute(
                "SELECT * FROM repository_update_runs WHERE id=?", (run_id,)
            ).fetchone()
        return self._update_run_from_row(row)

    def get_update_run(self, workspace_id: str, run_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM repository_update_runs WHERE workspace_id=? AND id=?",
                (workspace_id, run_id),
            ).fetchone()
        return self._update_run_from_row(row) if row is not None else None

    def get_latest_update_run(self, workspace_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM repository_update_runs
                WHERE workspace_id=?
                ORDER BY created_at DESC, id DESC LIMIT 1
                """,
                (workspace_id,),
            ).fetchone()
        return self._update_run_from_row(row) if row is not None else None

    def claim_update_run(self, workspace_id: str, run_id: str) -> bool:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            changed = conn.execute(
                """
                UPDATE repository_update_runs
                SET status='running', phase='revision_resolution',
                    started_at=CURRENT_TIMESTAMP, finished_at=NULL,
                    error_code='', error_message='', retryable=0,
                    updated_at=CURRENT_TIMESTAMP
                WHERE workspace_id=? AND id=? AND status='pending'
                """,
                (workspace_id, run_id),
            ).rowcount
        return changed == 1

    def update_run_phase(
        self,
        workspace_id: str,
        run_id: str,
        phase: str,
        stats: dict[str, Any] | None = None,
    ) -> None:
        if phase not in UPDATE_RUN_PHASES:
            raise ValueError("unknown update run phase")
        with self.connect() as conn:
            current = conn.execute(
                "SELECT phase, status FROM repository_update_runs WHERE workspace_id=? AND id=?",
                (workspace_id, run_id),
            ).fetchone()
            if current is None or current["status"] != "running":
                raise RuntimeError("update run is not running")
            if UPDATE_RUN_PHASES.index(phase) < UPDATE_RUN_PHASES.index(current["phase"]):
                raise RuntimeError("update run phases cannot move backwards")
            conn.execute(
                """
                UPDATE repository_update_runs
                SET phase=?, stats_json=COALESCE(?, stats_json), updated_at=CURRENT_TIMESTAMP
                WHERE workspace_id=? AND id=? AND status='running'
                """,
                (
                    phase,
                    json.dumps(stats, ensure_ascii=False, sort_keys=True) if stats is not None else None,
                    workspace_id,
                    run_id,
                ),
            )

    def create_staging_project(
        self,
        workspace_id: str,
        snapshot: dict[str, Any],
        parent_project_id: str,
        manifest_hash: str,
        chunker_version: str,
        run_id: str | None = None,
    ) -> str:
        revision = str(snapshot.get("repository_revision", ""))
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            workspace = conn.execute(
                "SELECT active_project_id FROM repository_workspaces WHERE id=?",
                (workspace_id,),
            ).fetchone()
            if workspace is None:
                raise LookupError("workspace not found")
            if workspace["active_project_id"] != parent_project_id:
                raise RuntimeError("active snapshot changed")
            existing = conn.execute(
                """
                SELECT project_id FROM workspace_revisions
                WHERE workspace_id=? AND repository_revision=?
                """,
                (workspace_id, revision),
            ).fetchone()
            if existing is not None:
                project_id = str(existing["project_id"])
                if run_id:
                    self._attach_update_project_in_connection(
                        conn, workspace_id, run_id, project_id
                    )
                return project_id
            project_id = str(uuid.uuid4())
            staging_identity = "staging-sha256:" + hashlib.sha256(
                f"{workspace_id}\0{snapshot.get('source_identity', '')}".encode("utf-8")
            ).hexdigest()
            conn.execute(
                """
                INSERT INTO projects (
                    id, repo_url, owner, repo, default_branch,
                    repository_revision, source_type, source_location,
                    source_identity, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'analyzing')
                """,
                (
                    project_id,
                    snapshot["repo_url"],
                    snapshot["owner"],
                    snapshot["repo"],
                    snapshot["default_branch"],
                    revision,
                    snapshot.get("source_type", "legacy_github"),
                    snapshot.get("source_location", ""),
                    staging_identity,
                ),
            )
            conn.execute(
                """
                INSERT INTO workspace_revisions (
                    workspace_id, project_id, repository_revision,
                    parent_project_id, manifest_hash, chunker_version,
                    activation_status
                ) VALUES (?, ?, ?, ?, ?, ?, 'staging')
                """,
                (
                    workspace_id,
                    project_id,
                    revision,
                    parent_project_id,
                    manifest_hash,
                    chunker_version,
                ),
            )
            if run_id:
                self._attach_update_project_in_connection(
                    conn, workspace_id, run_id, project_id
                )
        return project_id

    @staticmethod
    def _attach_update_project_in_connection(
        conn: sqlite3.Connection,
        workspace_id: str,
        run_id: str,
        project_id: str,
    ) -> None:
        changed = conn.execute(
            """
            UPDATE repository_update_runs SET project_id=?, updated_at=CURRENT_TIMESTAMP
            WHERE workspace_id=? AND id=? AND status='running'
              AND (project_id IS NULL OR project_id=?)
            """,
            (project_id, workspace_id, run_id, project_id),
        ).rowcount
        if changed != 1:
            raise RuntimeError("update run project association changed")

    def set_workspace_revision_embedding_identity(
        self, workspace_id: str, project_id: str, identity: str
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE workspace_revisions SET embedding_identity=?
                WHERE workspace_id=? AND project_id=? AND activation_status='staging'
                """,
                (identity, workspace_id, project_id),
            )

    def fail_update_run(
        self,
        workspace_id: str,
        run_id: str,
        *,
        code: str,
        message: str,
        retryable: bool,
    ) -> None:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT project_id FROM repository_update_runs WHERE workspace_id=? AND id=?",
                (workspace_id, run_id),
            ).fetchone()
            if row is None:
                return
            if row["project_id"]:
                conn.execute(
                    """
                    UPDATE workspace_revisions SET activation_status='failed'
                    WHERE workspace_id=? AND project_id=? AND activation_status='staging'
                    """,
                    (workspace_id, row["project_id"]),
                )
                conn.execute(
                    """
                    UPDATE projects SET status='failed', error_message=?, updated_at=CURRENT_TIMESTAMP
                    WHERE id=?
                    """,
                    (message[:500], row["project_id"]),
                )
            conn.execute(
                """
                UPDATE repository_update_runs
                SET status='failed', error_code=?, error_message=?, retryable=?,
                    finished_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP
                WHERE workspace_id=? AND id=? AND status IN ('pending', 'running')
                """,
                (code, message[:500], 1 if retryable else 0, workspace_id, run_id),
            )

    def retry_update_run(self, workspace_id: str, run_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            changed = conn.execute(
                """
                UPDATE repository_update_runs
                SET status='pending', phase='revision_resolution', result='',
                    error_code='', error_message='', retryable=0,
                    retry_count=retry_count+1, started_at=NULL, finished_at=NULL,
                    updated_at=CURRENT_TIMESTAMP
                WHERE workspace_id=? AND id=? AND status='failed' AND retryable=1
                """,
                (workspace_id, run_id),
            ).rowcount
            if changed == 1:
                row = conn.execute(
                    "SELECT project_id FROM repository_update_runs WHERE id=?", (run_id,)
                ).fetchone()
                if row["project_id"]:
                    conn.execute(
                        "UPDATE projects SET status='analyzing', error_message='' WHERE id=?",
                        (row["project_id"],),
                    )
                    conn.execute(
                        """
                        UPDATE workspace_revisions SET activation_status='staging'
                        WHERE workspace_id=? AND project_id=? AND activation_status='failed'
                        """,
                        (workspace_id, row["project_id"]),
                    )
            result = conn.execute(
                "SELECT * FROM repository_update_runs WHERE workspace_id=? AND id=?",
                (workspace_id, run_id),
            ).fetchone()
        return self._update_run_from_row(result) if result is not None else None

    def activate_workspace_snapshot(
        self,
        workspace_id: str,
        run_id: str,
        *,
        project_id: str,
        expected_active_project_id: str,
        source_identity: str,
        stats: dict[str, Any],
    ) -> None:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            workspace = conn.execute(
                "SELECT active_project_id, activation_version FROM repository_workspaces WHERE id=?",
                (workspace_id,),
            ).fetchone()
            project = conn.execute(
                "SELECT status, repository_revision FROM projects WHERE id=?", (project_id,)
            ).fetchone()
            source_project = conn.execute(
                "SELECT repository_revision FROM projects WHERE id=?",
                (expected_active_project_id,),
            ).fetchone()
            revision = conn.execute(
                """
                SELECT activation_status FROM workspace_revisions
                WHERE workspace_id=? AND project_id=?
                """,
                (workspace_id, project_id),
            ).fetchone()
            if (
                workspace is None
                or workspace["active_project_id"] != expected_active_project_id
                or project is None
                or project["status"] != "done"
                or source_project is None
                or revision is None
                or revision["activation_status"] != "staging"
            ):
                raise RuntimeError("snapshot activation precondition failed")
            conn.execute(
                """
                UPDATE workspace_revisions SET activation_status='superseded'
                WHERE workspace_id=? AND project_id=? AND activation_status='active'
                """,
                (workspace_id, expected_active_project_id),
            )
            if self._activation_test_hook is not None:
                self._activation_test_hook()
            conn.execute(
                """
                UPDATE workspace_revisions
                SET activation_status='active', activated_at=CURRENT_TIMESTAMP
                WHERE workspace_id=? AND project_id=? AND activation_status='staging'
                """,
                (workspace_id, project_id),
            )
            conn.execute(
                """
                UPDATE repository_workspaces
                SET active_project_id=?, activation_version=activation_version+1,
                    updated_at=CURRENT_TIMESTAMP
                WHERE id=? AND active_project_id=?
                """,
                (project_id, workspace_id, expected_active_project_id),
            )
            activation_version = int(workspace["activation_version"]) + 1
            transition_id = str(uuid.uuid5(
                uuid.NAMESPACE_URL,
                "\0".join((
                    "reponoesis:learning-continuity",
                    workspace_id,
                    expected_active_project_id,
                    project_id,
                    LOCAL_CONTINUITY_LEARNER_ID,
                    LEARNING_CONTINUITY_CONFIG_IDENTITY,
                )),
            ))
            conn.execute(
                """
                INSERT INTO learning_continuity_transitions (
                    id, workspace_id, source_project_id, target_project_id,
                    source_revision, target_revision, activation_version,
                    learner_id, mapping_config_identity, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
                """,
                (
                    transition_id,
                    workspace_id,
                    expected_active_project_id,
                    project_id,
                    source_project["repository_revision"],
                    project["repository_revision"],
                    activation_version,
                    LOCAL_CONTINUITY_LEARNER_ID,
                    LEARNING_CONTINUITY_CONFIG_IDENTITY,
                ),
            )
            conn.execute(
                "UPDATE projects SET source_identity=? WHERE id=?",
                (source_identity, project_id),
            )
            changed = conn.execute(
                """
                UPDATE repository_update_runs
                SET status='succeeded', phase='activation', result='activated',
                    stats_json=?, finished_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP
                WHERE workspace_id=? AND id=? AND status='running' AND project_id=?
                """,
                (json.dumps(stats, ensure_ascii=False, sort_keys=True), workspace_id, run_id, project_id),
            ).rowcount
            if changed != 1:
                raise RuntimeError("update run activation state changed")

    def recover_interrupted_update_runs(self) -> int:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT id, workspace_id, project_id FROM repository_update_runs WHERE status IN ('pending', 'running')"
            ).fetchall()
            for row in rows:
                if row["project_id"]:
                    conn.execute(
                        """
                        UPDATE workspace_revisions SET activation_status='failed'
                        WHERE workspace_id=? AND project_id=? AND activation_status='staging'
                        """,
                        (row["workspace_id"], row["project_id"]),
                    )
                    conn.execute(
                        "UPDATE projects SET status='failed', error_message='Update interrupted.' WHERE id=?",
                        (row["project_id"],),
                    )
            conn.execute(
                """
                UPDATE repository_update_runs
                SET status='failed', error_code='update_interrupted',
                    error_message='The previous update was interrupted and can be retried safely.',
                    retryable=1, finished_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP
                WHERE status IN ('pending', 'running')
                """
            )
        return len(rows)

    def learning_record_counts(self, project_id: str) -> dict[str, int]:
        direct_tables = (
            "learning_goals",
            "learning_targets",
            "learning_plans",
            "learning_tasks",
            "learning_attempts",
            "learning_events",
            "learner_target_states",
        )
        with self.connect() as conn:
            counts = {
                table: int(
                    conn.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE project_id=?", (project_id,)
                    ).fetchone()[0]
                )
                for table in direct_tables
            }
            counts["learning_evaluations"] = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM learning_evaluations AS e
                    JOIN learning_attempts AS a ON a.attempt_id=e.attempt_id
                    WHERE a.project_id=?
                    """,
                    (project_id,),
                ).fetchone()[0]
            )
            return counts

    @staticmethod
    def _update_run_from_row(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        value = dict(row)
        try:
            stats = json.loads(value.get("stats_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            stats = {}
        return {
            "run_id": value["id"],
            "workspace_id": value["workspace_id"],
            "base_project_id": value["base_project_id"],
            "project_id": value.get("project_id"),
            "target_revision": value["target_revision"],
            "status": value["status"],
            "phase": value["phase"],
            "result": value["result"],
            "stats": stats,
            "error_code": value["error_code"],
            "error_message": value["error_message"],
            "retryable": bool(value["retryable"]),
            "retry_count": int(value["retry_count"]),
            "created_at": value["created_at"],
            "started_at": value["started_at"],
            "finished_at": value["finished_at"],
            "updated_at": value["updated_at"],
        }

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
            "repository_revision": row["repository_revision"],
            "source_type": row["source_type"],
            "source_location": row["source_location"],
            "source_identity": row["source_identity"],
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

    @staticmethod
    def _invalidate_relation_index(
        conn: sqlite3.Connection, project_id: str
    ) -> None:
        conn.execute("DELETE FROM code_relations WHERE project_id = ?", (project_id,))
        conn.execute("DELETE FROM relation_nodes WHERE project_id = ?", (project_id,))
        conn.execute(
            "DELETE FROM relation_index_runs WHERE project_id = ?", (project_id,)
        )

    def _prepare_code_chunk_replacement(
        self,
        project_id: str,
        code_chunks: list[dict[str, Any]],
        *,
        path: str | None = None,
        repository_revision: str | None = None,
    ) -> list[dict[str, Any]]:
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

        target_keys: set[tuple[str, str, str, str, int, int]] = set()
        for chunk in prepared:
            key = self._code_chunk_match_key(chunk)
            if key in target_keys:
                raise ValueError(
                    "duplicate code chunk persistence identity in replacement input: "
                    f"{chunk['path']} {chunk['qualified_name']} "
                    f"{chunk['start_line']}-{chunk['end_line']}"
                )
            target_keys.add(key)
        return prepared

    def _replace_code_chunks_in_scope(
        self,
        conn: sqlite3.Connection,
        project_id: str,
        prepared: list[dict[str, Any]],
        path: str | None = None,
        repository_revision: str | None = None,
    ) -> None:
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

        existing_by_key = {
            self._code_chunk_match_key(row): row for row in existing_rows
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
    def _code_chunk_match_key(
        row: sqlite3.Row | dict[str, Any],
    ) -> tuple[str, str, str, str, int, int]:
        return (
            row["repository_revision"],
            row["path"],
            row["chunk_type"],
            row["qualified_name"],
            int(row["start_line"]),
            int(row["end_line"]),
        )

    def _prepare_relation_node(
        self,
        project_id: str,
        repository_revision: str,
        node: dict[str, Any],
    ) -> dict[str, Any]:
        if node.get("project_id") != project_id:
            raise ValueError("relation node project mismatch")
        if node.get("repository_revision") != repository_revision:
            raise ValueError("relation node revision mismatch")
        node_id = str(node.get("node_id", ""))
        if len(node_id) != 65 or not node_id.startswith("N"):
            raise ValueError("invalid relation node identity")
        start_line = int(node.get("start_line", 0))
        end_line = int(node.get("end_line", 0))
        if start_line < 1 or end_line < start_line:
            raise ValueError("invalid relation node line range")
        content_hash = str(node.get("content_hash", ""))
        _require_hash(content_hash, "relation node content_hash")
        return {
            **node,
            "node_id": node_id,
            "path": self._normalize_repo_path(str(node.get("path", ""))),
            "code_chunk_id": (
                int(node["code_chunk_id"])
                if node.get("code_chunk_id") is not None
                else None
            ),
            "language": str(node.get("language", ""))[:80],
            "node_type": str(node.get("node_type", ""))[:80],
            "symbol_name": str(node.get("symbol_name", ""))[:500],
            "qualified_name": str(node.get("qualified_name", ""))[:500],
            "start_line": start_line,
            "end_line": end_line,
            "content_hash": content_hash,
        }

    def _prepare_relation_edge(
        self,
        project_id: str,
        repository_revision: str,
        edge: dict[str, Any],
        node_ids: set[str],
    ) -> dict[str, Any]:
        if edge.get("project_id") != project_id:
            raise ValueError("relation edge project mismatch")
        if edge.get("repository_revision") != repository_revision:
            raise ValueError("relation edge revision mismatch")
        edge_id = str(edge.get("edge_id", ""))
        if len(edge_id) != 65 or not edge_id.startswith("R"):
            raise ValueError("invalid relation edge identity")
        relation_type = str(edge.get("relation_type", ""))
        if relation_type not in {"imports", "calls", "references", "defines"}:
            raise ValueError("unknown relation type")
        status = str(edge.get("resolution_status", ""))
        if status not in {
            "resolved",
            "ambiguous",
            "unresolved",
            "external",
            "unsupported",
        }:
            raise ValueError("unknown relation resolution status")
        source_node_id = str(edge.get("source_node_id", ""))
        target_node_id = edge.get("target_node_id")
        if source_node_id not in node_ids:
            raise ValueError("relation source node is missing")
        if target_node_id is not None and str(target_node_id) not in node_ids:
            raise ValueError("relation target node is missing")
        if status == "resolved" and target_node_id is None:
            raise ValueError("resolved relation requires a target node")
        source_hash = _require_hash(
            str(edge.get("source_content_hash", "")),
            "relation source_content_hash",
        )
        target_hash = edge.get("target_content_hash")
        if target_hash is not None:
            target_hash = _require_hash(
                str(target_hash), "relation target_content_hash"
            )
        source_start = int(edge.get("source_start_line", 0))
        source_end = int(edge.get("source_end_line", 0))
        if source_start < 1 or source_end < source_start:
            raise ValueError("invalid relation source line range")
        return {
            **edge,
            "edge_id": edge_id,
            "relation_type": relation_type,
            "source_node_id": source_node_id,
            "source_path": self._normalize_repo_path(str(edge.get("source_path", ""))),
            "source_chunk_id": (
                int(edge["source_chunk_id"])
                if edge.get("source_chunk_id") is not None
                else None
            ),
            "source_symbol": str(edge.get("source_symbol", ""))[:500],
            "source_start_line": source_start,
            "source_end_line": source_end,
            "target_node_id": str(target_node_id) if target_node_id is not None else None,
            "target_path": (
                self._normalize_repo_path(str(edge["target_path"]))
                if edge.get("target_path") is not None
                else None
            ),
            "target_chunk_id": (
                int(edge["target_chunk_id"])
                if edge.get("target_chunk_id") is not None
                else None
            ),
            "target_symbol": (
                str(edge["target_symbol"])[:500]
                if edge.get("target_symbol") is not None
                else None
            ),
            "target_start_line": (
                int(edge["target_start_line"])
                if edge.get("target_start_line") is not None
                else None
            ),
            "target_end_line": (
                int(edge["target_end_line"])
                if edge.get("target_end_line") is not None
                else None
            ),
            "raw_target_name": str(edge.get("raw_target_name", ""))[:500],
            "resolution_status": status,
            "resolution_rule": str(edge.get("resolution_rule", ""))[:120],
            "language": str(edge.get("language", ""))[:80],
            "source_content_hash": source_hash,
            "target_content_hash": target_hash,
        }

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
        identity_schema_version = str(
            record.get("identity_schema_version") or "legacy"
        )
        wrapper_model_identity = str(record.get("wrapper_model_identity") or "")
        resolved_revision = str(record.get("resolved_revision") or "")
        identity_eligible = bool(record.get("identity_eligible", False))
        if identity_eligible:
            if identity_schema_version == "legacy":
                raise ValueError("eligible embedding identity cannot use legacy schema")
            if not wrapper_model_identity.startswith("embedding-sha256:"):
                raise ValueError("wrapper model identity must use embedding-sha256")
        return (
            int(record["code_chunk_id"]),
            record["content_hash"],
            _require_hash(record["embedding_input_hash"], "embedding_input_hash"),
            record["model_name"],
            record.get("model_revision") or "",
            identity_schema_version,
            wrapper_model_identity,
            resolved_revision,
            1 if identity_eligible else 0,
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
              AND identity_schema_version = ?
              AND wrapper_model_identity = ?
              AND resolved_revision = ?
              AND embedding_dimension = ?
              AND text_format_version = ?
              AND embedding_config_hash = ?
              AND normalized = ?
              AND embedding_input_hash != ?
            """,
            (
                int(record["code_chunk_id"]),
                record.get("identity_schema_version") or "legacy",
                record.get("wrapper_model_identity") or "",
                record.get("resolved_revision") or "",
                int(record.get("embedding_dimension") or len(record["vector"])),
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


def _embedding_identity_predicate(
    alias: str,
    effective_identity: Any | None,
) -> tuple[str, tuple[Any, ...]]:
    if effective_identity is None:
        return f"AND {alias}.identity_eligible = 0", ()
    data = (
        effective_identity.to_dict()
        if hasattr(effective_identity, "to_dict")
        else dict(effective_identity)
    )
    schema_version = str(data.get("identity_schema_version") or "")
    wrapper_identity = str(data.get("model_identity") or "")
    resolved_revision = str(data.get("resolved_revision") or "")
    dimension = int(data.get("dimension") or 0)
    if not schema_version or schema_version == "legacy":
        raise ValueError("effective embedding identity schema is invalid")
    if not wrapper_identity.startswith("embedding-sha256:"):
        raise ValueError("effective wrapper model identity is invalid")
    if dimension < 1:
        raise ValueError("effective embedding dimension is invalid")
    return (
        f"AND {alias}.identity_eligible = 1 "
        f"AND {alias}.identity_schema_version = ? "
        f"AND {alias}.wrapper_model_identity = ? "
        f"AND {alias}.resolved_revision = ? "
        f"AND {alias}.embedding_dimension = ?",
        (schema_version, wrapper_identity, resolved_revision, dimension),
    )
