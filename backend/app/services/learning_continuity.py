from __future__ import annotations

import ast
import hashlib
import json
import sqlite3
import uuid
from collections import Counter, defaultdict
from typing import Any

from app.database import (
    LEARNING_CONTINUITY_CONFIG_IDENTITY,
    Database,
)
from app.services.learning_contracts import LOCAL_LEARNER_ID, TargetSpec
from app.services.learning_service import LearningService


CONTINUITY_MAPPING_CONFIG_IDENTITY = LEARNING_CONTINUITY_CONFIG_IDENTITY


class LearningContinuityError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 409,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable


class LearningContinuityService:
    """Publish conservative, revision-bound learning projections.

    Source M4 rows remain immutable. A successful transition creates only new
    target identities, explicit system-derived events, a revision plan projection,
    and additive lineage rows in one transaction.
    """

    def __init__(self, database: Database) -> None:
        self.database = database
        self.learning = LearningService(database)
        self.fail_before_publish = False

    def recover_interrupted(self) -> int:
        with self.database.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            changed = conn.execute(
                """
                UPDATE learning_continuity_transitions
                SET status='failed', error_code='continuity_interrupted',
                    error_message='The previous continuity transition was interrupted.',
                    retryable=1, finished_at=CURRENT_TIMESTAMP,
                    updated_at=CURRENT_TIMESTAMP
                WHERE status IN ('pending', 'running')
                """
            ).rowcount
        return int(changed)

    def get_current(self, workspace_id: str) -> dict[str, Any]:
        with self.database.connect() as conn:
            workspace = conn.execute(
                "SELECT active_project_id, activation_version FROM repository_workspaces WHERE id=?",
                (workspace_id,),
            ).fetchone()
            if workspace is None:
                raise LearningContinuityError(
                    "workspace_not_found", "The requested workspace does not exist.",
                    status_code=404,
                )
            row = conn.execute(
                """
                SELECT * FROM learning_continuity_transitions
                WHERE workspace_id=? AND activation_version=? AND target_project_id=?
                  AND learner_id=?
                ORDER BY created_at DESC, id DESC LIMIT 1
                """,
                (
                    workspace_id,
                    workspace["activation_version"],
                    workspace["active_project_id"],
                    LOCAL_LEARNER_ID,
                ),
            ).fetchone()
        if row is None:
            return {
                "transition_id": None,
                "workspace_id": workspace_id,
                "status": "not_required",
                "activation_version": int(workspace["activation_version"]),
                "stats": self._empty_stats(),
                "retryable": False,
            }
        return self._public_transition(row)

    def get_transition(self, workspace_id: str, transition_id: str) -> dict[str, Any]:
        self._validate_transition_id(transition_id)
        row = self._transition_row(workspace_id, transition_id)
        if row is None:
            raise LearningContinuityError(
                "continuity_transition_not_found",
                "The requested learning continuity transition does not exist.",
                status_code=404,
            )
        return self._public_transition(row)

    def get_impacts(self, workspace_id: str, transition_id: str) -> dict[str, Any]:
        self._validate_transition_id(transition_id)
        transition = self._transition_row(workspace_id, transition_id)
        if transition is None:
            raise LearningContinuityError(
                "continuity_transition_not_found",
                "The requested learning continuity transition does not exist.",
                status_code=404,
            )
        with self.database.connect() as conn:
            rows = conn.execute(
                """
                SELECT source_target_id, target_target_id, mapping_status,
                       mapping_rule, source_mastery_status, derived_mastery_status,
                       source_path, target_path, source_qualified_name,
                       target_qualified_name, review_reason
                FROM learning_continuity_mappings
                WHERE transition_id=?
                ORDER BY source_path, source_qualified_name, source_target_id
                """,
                (transition_id,),
            ).fetchall()
        return {
            "transition_id": transition_id,
            "workspace_id": workspace_id,
            "status": transition["status"],
            "items": [dict(row) for row in rows],
        }

    def retry(self, workspace_id: str, transition_id: str) -> dict[str, Any]:
        self._validate_transition_id(transition_id)
        with self.database.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            exists = conn.execute(
                "SELECT * FROM learning_continuity_transitions WHERE workspace_id=? AND id=?",
                (workspace_id, transition_id),
            ).fetchone()
            if exists is None:
                raise LearningContinuityError(
                    "continuity_transition_not_found",
                    "The requested learning continuity transition does not exist.",
                    status_code=404,
                )
            changed = conn.execute(
                """
                UPDATE learning_continuity_transitions
                SET status='pending', error_code='', error_message='', retryable=0,
                    retry_count=retry_count+1, started_at=NULL, finished_at=NULL,
                    updated_at=CURRENT_TIMESTAMP
                WHERE workspace_id=? AND id=? AND status='failed' AND retryable=1
                """,
                (workspace_id, transition_id),
            ).rowcount
            if changed != 1:
                raise LearningContinuityError(
                    "continuity_not_retryable",
                    "The learning continuity transition is not retryable.",
                )
            row = conn.execute(
                "SELECT * FROM learning_continuity_transitions WHERE id=?",
                (transition_id,),
            ).fetchone()
        return self._public_transition(row)

    def execute(self, workspace_id: str, transition_id: str) -> dict[str, Any]:
        self._validate_transition_id(transition_id)
        row = self._transition_row(workspace_id, transition_id)
        if row is None:
            raise LearningContinuityError(
                "continuity_transition_not_found",
                "The requested learning continuity transition does not exist.",
                status_code=404,
            )
        if row["status"] == "succeeded":
            return self._public_transition(row)
        if row["status"] == "failed":
            return self._public_transition(row)
        if not self._claim(workspace_id, transition_id):
            current = self._transition_row(workspace_id, transition_id)
            return self._public_transition(current)
        try:
            claimed = self._transition_row(workspace_id, transition_id)
            mappings = self._build_mappings(dict(claimed))
            if self.fail_before_publish:
                raise RuntimeError("forced continuity publish failure")
            self._publish(dict(claimed), mappings)
        except Exception:
            self._fail(workspace_id, transition_id)
        return self.get_transition(workspace_id, transition_id)

    def _claim(self, workspace_id: str, transition_id: str) -> bool:
        with self.database.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            changed = conn.execute(
                """
                UPDATE learning_continuity_transitions
                SET status='running', started_at=CURRENT_TIMESTAMP,
                    finished_at=NULL, error_code='', error_message='', retryable=0,
                    updated_at=CURRENT_TIMESTAMP
                WHERE workspace_id=? AND id=? AND status='pending'
                """,
                (workspace_id, transition_id),
            ).rowcount
        return changed == 1

    def _fail(self, workspace_id: str, transition_id: str) -> None:
        with self.database.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                UPDATE learning_continuity_transitions
                SET status='failed', error_code='continuity_publish_failed',
                    error_message='Learning continuity could not be published safely.',
                    retryable=1, finished_at=CURRENT_TIMESTAMP,
                    updated_at=CURRENT_TIMESTAMP
                WHERE workspace_id=? AND id=? AND status='running'
                """,
                (workspace_id, transition_id),
            )

    def _build_mappings(self, transition: dict[str, Any]) -> list[dict[str, Any]]:
        with self.database.connect() as conn:
            source_targets = conn.execute(
                """
                SELECT t.*, s.mastery_status, s.verified_pass_count,
                       s.qualifying_pass_count, s.last_validated_revision,
                       s.last_validated_content_hash
                FROM learning_targets AS t
                LEFT JOIN learner_target_states AS s
                  ON s.learner_id=t.learner_id AND s.project_id=t.project_id
                 AND s.target_id=t.target_id
                WHERE t.learner_id=? AND t.project_id=?
                ORDER BY t.target_id
                """,
                (transition["learner_id"], transition["source_project_id"]),
            ).fetchall()
            versions = {
                row["project_id"]: str(row["chunker_version"] or "")
                for row in conn.execute(
                    """
                    SELECT project_id, chunker_version FROM workspace_revisions
                    WHERE workspace_id=? AND project_id IN (?, ?)
                    """,
                    (
                        transition["workspace_id"],
                        transition["source_project_id"],
                        transition["target_project_id"],
                    ),
                )
            }
            mappings = [
                self._classify_target(
                    conn,
                    dict(target),
                    transition["target_project_id"],
                    compatible=(
                        versions.get(transition["source_project_id"])
                        == versions.get(transition["target_project_id"])
                        and bool(versions.get(transition["source_project_id"]))
                    ),
                )
                for target in source_targets
            ]
        by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in mappings:
            if item.get("candidate_key"):
                by_candidate[item["candidate_key"]].append(item)
        for collision in by_candidate.values():
            if len(collision) > 1:
                for item in collision:
                    item.update({
                        "mapping_status": "ambiguous",
                        "mapping_rule": "multiple_source_targets_share_candidate",
                        "candidate": None,
                        "candidate_key": "",
                    })
        return sorted(mappings, key=lambda item: item["source"]["target_id"])

    def _classify_target(
        self,
        conn: sqlite3.Connection,
        source: dict[str, Any],
        target_project_id: str,
        *,
        compatible: bool,
    ) -> dict[str, Any]:
        base = {
            "source": source,
            "candidate": None,
            "candidate_key": "",
            "mapping_status": "unmapped",
            "mapping_rule": "insufficient_identity",
        }
        target_type = source["target_type"]
        if target_type in {"file", "module", "symbol"} and not compatible:
            return {**base, "mapping_status": "incompatible", "mapping_rule": "chunker_identity_changed"}
        if target_type == "bounded_concept":
            return base
        if target_type == "repository":
            candidate = {
                "target_type": "repository", "path": "", "qualified_name": "",
                "concept": "", "content_hash": "", "candidate_key": "repository",
            }
            return {**base, "candidate": candidate, "candidate_key": "repository",
                    "mapping_status": "modified", "mapping_rule": "repository_revision_changed"}
        if target_type == "file":
            rows = [dict(row) for row in conn.execute(
                "SELECT path, content FROM repo_files WHERE project_id=? ORDER BY path",
                (target_project_id,),
            ).fetchall()]
            for row in rows:
                row["content_hash"] = hashlib.sha256(str(row["content"]).encode("utf-8")).hexdigest()
            exact_path = [row for row in rows if row["path"] == source["normalized_path"]]
            if len(exact_path) == 1:
                row = exact_path[0]
                status = "unchanged_exact" if row["content_hash"] == source["observed_content_hash"] else "modified"
                return self._candidate_result(base, source, row, status, "file_path_and_content" if status == "unchanged_exact" else "file_path_content_changed")
            same_hash = [row for row in rows if row["content_hash"] == source["observed_content_hash"]]
            if len(same_hash) == 1:
                return self._candidate_result(base, source, same_hash[0], "renamed_exact", "unique_file_content_rename")
            if len(same_hash) > 1:
                return {**base, "mapping_status": "ambiguous", "mapping_rule": "multiple_exact_file_candidates"}
            return {**base, "mapping_status": "deleted", "mapping_rule": "file_missing"}
        if target_type == "module":
            prefix = str(source["normalized_path"]).rstrip("/")
            rows = conn.execute(
                "SELECT path, content FROM repo_files WHERE project_id=? AND (path=? OR path LIKE ?) ORDER BY path",
                (target_project_id, prefix, prefix + "/%"),
            ).fetchall()
            if not rows:
                return {**base, "mapping_status": "deleted", "mapping_rule": "module_missing"}
            content_hash = hashlib.sha256("\n".join(
                hashlib.sha256(str(row["content"]).encode("utf-8")).hexdigest()
                for row in rows
            ).encode("utf-8")).hexdigest()
            row = {"path": prefix, "qualified_name": "", "content_hash": content_hash}
            status = "unchanged_exact" if content_hash == source["observed_content_hash"] else "modified"
            return self._candidate_result(base, source, row, status, "module_content_identity" if status == "unchanged_exact" else "module_content_changed")
        source_chunk = conn.execute(
            "SELECT chunk_type, content FROM code_chunks WHERE id=? AND project_id=?",
            (source["code_chunk_id"], source["project_id"]),
        ).fetchone()
        exact = [dict(row) for row in conn.execute(
            """
            SELECT id, path, qualified_name, content_hash, chunk_type, content
            FROM code_chunks WHERE project_id=? AND path=? AND qualified_name=?
            ORDER BY id
            """,
            (target_project_id, source["normalized_path"], source["qualified_name"]),
        ).fetchall()]
        if len(exact) == 1:
            status = "unchanged_exact" if exact[0]["content_hash"] == source["observed_content_hash"] else "modified"
            return self._candidate_result(base, source, exact[0], status, "symbol_identity_and_content" if status == "unchanged_exact" else "symbol_identity_content_changed")
        if len(exact) > 1:
            return {**base, "mapping_status": "ambiguous", "mapping_rule": "duplicate_symbol_identity"}
        same_hash = [dict(row) for row in conn.execute(
            """
            SELECT id, path, qualified_name, content_hash, chunk_type, content
            FROM code_chunks WHERE project_id=? AND content_hash=? ORDER BY id
            """,
            (target_project_id, source["observed_content_hash"]),
        ).fetchall()]
        if len(same_hash) == 1:
            return self._candidate_result(
                base, source, same_hash[0], "modified",
                "unique_symbol_content_moved_path_requires_review",
            )
        if len(same_hash) > 1:
            return {**base, "mapping_status": "ambiguous", "mapping_rule": "multiple_exact_symbol_candidates"}
        if source_chunk is not None:
            structural = [dict(row) for row in conn.execute(
                """
                SELECT id, path, qualified_name, content_hash, chunk_type, content
                FROM code_chunks WHERE project_id=? AND path=? AND chunk_type=? ORDER BY id
                """,
                (target_project_id, source["normalized_path"], source_chunk["chunk_type"]),
            ).fetchall()]
            source_structure = _structure_identity(str(source_chunk["content"]))
            structural = [row for row in structural if _structure_identity(str(row["content"])) == source_structure]
            if len(structural) == 1:
                return self._candidate_result(base, source, structural[0], "modified", "unique_symbol_structure_changed")
            if len(structural) > 1:
                return {**base, "mapping_status": "ambiguous", "mapping_rule": "multiple_structural_symbol_candidates"}
        return {**base, "mapping_status": "deleted", "mapping_rule": "symbol_missing"}

    @staticmethod
    def _candidate_result(
        base: dict[str, Any], source: dict[str, Any], row: dict[str, Any],
        status: str, rule: str,
    ) -> dict[str, Any]:
        target_type = source["target_type"]
        candidate = {
            "target_type": target_type,
            "path": str(row.get("path") or ""),
            "qualified_name": str(row.get("qualified_name") or ""),
            "concept": str(source.get("bounded_concept") or ""),
            "content_hash": str(row.get("content_hash") or ""),
            "candidate_key": (
                f"chunk:{row.get('id')}" if row.get("id") is not None
                else f"{target_type}:{row.get('path', '')}"
            ),
        }
        return {**base, "candidate": candidate, "candidate_key": candidate["candidate_key"],
                "mapping_status": status, "mapping_rule": rule}

    def _publish(self, transition: dict[str, Any], mappings: list[dict[str, Any]]) -> None:
        with self.database.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute(
                "SELECT * FROM learning_continuity_transitions WHERE id=? AND status='running'",
                (transition["id"],),
            ).fetchone()
            workspace = conn.execute(
                "SELECT active_project_id, activation_version FROM repository_workspaces WHERE id=?",
                (transition["workspace_id"],),
            ).fetchone()
            if (
                current is None or workspace is None
                or workspace["active_project_id"] != transition["target_project_id"]
                or int(workspace["activation_version"]) != int(transition["activation_version"])
            ):
                raise RuntimeError("continuity activation binding is stale")
            target_project = self.learning._project(conn, transition["target_project_id"])
            target_ids: dict[str, str] = {}
            counts: Counter[str] = Counter()
            for item in mappings:
                source = item["source"]
                status = item["mapping_status"]
                counts[status] += 1
                target_id: str | None = None
                candidate = item.get("candidate")
                source_state = str(source.get("mastery_status") or "unseen")
                derived = ""
                review_reason = ""
                if candidate is not None and status in {"unchanged_exact", "renamed_exact", "modified"}:
                    target = self.learning._resolve_or_create_target(
                        conn,
                        transition["learner_id"],
                        target_project,
                        TargetSpec(
                            target_type=candidate["target_type"],
                            path=candidate["path"],
                            qualified_name=candidate["qualified_name"],
                            concept=candidate["concept"],
                        ),
                    )
                    target_id = target["target_id"]
                    target_ids[source["target_id"]] = target_id
                    if source.get("mastery_status") is not None:
                        if status in {"unchanged_exact", "renamed_exact"}:
                            derived = source_state
                        else:
                            derived = "needs_review"
                            review_reason = "revision_modified"
                        event_id = _stable_id("E", "continuity", transition["id"], source["target_id"])
                        self.learning._append_event(
                            conn,
                            event_id=event_id,
                            idempotency_key=f"continuity:{transition['id']}:{source['target_id']}",
                            learner_id=transition["learner_id"],
                            project=target_project,
                            target_id=target_id,
                            event_type="continuity_state_derived",
                            provenance="revision_continuity",
                            outcome={
                                "transition_id": transition["id"],
                                "source_project_id": transition["source_project_id"],
                                "source_target_id": source["target_id"],
                                "mapping_status": status,
                                "source_mastery_status": source_state,
                                "source_verified_pass_count": int(source.get("verified_pass_count") or 0),
                                "source_qualifying_pass_count": int(source.get("qualifying_pass_count") or 0),
                                "availability": "current" if status in {"unchanged_exact", "renamed_exact"} else "changed",
                                "content_hash": candidate["content_hash"],
                            },
                        )
                        self.learning._rebuild_target_state_tx(
                            conn, transition["learner_id"], transition["target_project_id"], target_id
                        )
                if status == "deleted":
                    review_reason = "revision_deleted"
                elif status in {"ambiguous", "unmapped", "incompatible"}:
                    review_reason = f"revision_{status}"
                conn.execute(
                    """
                    INSERT INTO learning_continuity_mappings (
                        transition_id, source_target_id, target_target_id,
                        mapping_status, mapping_rule, source_mastery_status,
                        derived_mastery_status, source_content_hash,
                        target_content_hash, source_path, target_path,
                        source_qualified_name, target_qualified_name, review_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        transition["id"], source["target_id"], target_id,
                        status, item["mapping_rule"], source_state, derived,
                        source.get("observed_content_hash") or "",
                        candidate["content_hash"] if candidate else "",
                        source.get("normalized_path") or "",
                        candidate["path"] if candidate else "",
                        source.get("qualified_name") or "",
                        candidate["qualified_name"] if candidate else "",
                        review_reason,
                    ),
                )
            self._publish_goal_lineage(conn, transition, target_project, mappings, target_ids)
            stats = self._empty_stats()
            stats.update({key: counts[key] for key in (
                "unchanged_exact", "renamed_exact", "modified", "deleted",
                "ambiguous", "unmapped", "incompatible",
            )})
            stats["total"] = len(mappings)
            stats["retained"] = counts["unchanged_exact"] + counts["renamed_exact"]
            stats["needs_review"] = counts["modified"]
            stats["history_only"] = counts["deleted"]
            stats["not_inherited"] = (
                counts["ambiguous"] + counts["unmapped"] + counts["incompatible"]
            )
            changed = conn.execute(
                """
                UPDATE learning_continuity_transitions
                SET status='succeeded', stats_json=?, error_code='', error_message='',
                    retryable=0, finished_at=CURRENT_TIMESTAMP,
                    updated_at=CURRENT_TIMESTAMP
                WHERE id=? AND status='running'
                """,
                (json.dumps(stats, sort_keys=True), transition["id"]),
            ).rowcount
            if changed != 1:
                raise RuntimeError("continuity transition state changed")

    def _publish_goal_lineage(
        self,
        conn: sqlite3.Connection,
        transition: dict[str, Any],
        target_project: dict[str, Any],
        mappings: list[dict[str, Any]],
        target_ids: dict[str, str],
    ) -> None:
        mapping_by_source = {item["source"]["target_id"]: item for item in mappings}
        goals = conn.execute(
            """
            SELECT * FROM learning_goals
            WHERE learner_id=? AND project_id=? AND status IN ('active', 'completed')
            ORDER BY created_at, goal_id
            """,
            (transition["learner_id"], transition["source_project_id"]),
        ).fetchall()
        for goal in goals:
            target_goal_id = _stable_id("G", "continuity", transition["id"], goal["goal_id"])
            conn.execute(
                """
                INSERT OR IGNORE INTO learning_goals (
                    goal_id, learner_id, project_id, repository_id,
                    created_revision, goal_text, goal_type, status,
                    idempotency_key, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    target_goal_id, transition["learner_id"], transition["target_project_id"],
                    target_project["repository_id"], transition["target_revision"],
                    goal["goal_text"], goal["goal_type"], goal["status"],
                    f"continuity:{transition['id']}:{goal['goal_id']}",
                ),
            )
            plan = conn.execute(
                """
                SELECT * FROM learning_plans WHERE goal_id=?
                  AND status IN ('active', 'completed')
                ORDER BY plan_version DESC LIMIT 1
                """,
                (goal["goal_id"],),
            ).fetchone()
            target_plan_id: str | None = None
            lineage_status = "carried"
            if plan is not None:
                source_steps = conn.execute(
                    "SELECT * FROM learning_plan_steps WHERE plan_id=? ORDER BY step_order",
                    (plan["plan_id"],),
                ).fetchall()
                retained = [step for step in source_steps if step["target_id"] in target_ids]
                affected = [
                    mapping_by_source[step["target_id"]]["mapping_status"]
                    for step in source_steps if step["target_id"] in mapping_by_source
                ]
                if any(status not in {"unchanged_exact", "renamed_exact"} for status in affected):
                    lineage_status = "needs_review"
                if retained:
                    target_plan_id = _stable_id("P", "continuity", transition["id"], plan["plan_id"])
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO learning_plans (
                            plan_id, goal_id, learner_id, project_id, repository_id,
                            source_revision, plan_version, status, adapted,
                            adaptation_reason, idempotency_key, schema_version
                        ) VALUES (?, ?, ?, ?, ?, ?, 1, 'active', 1,
                                  'revision_continuity', ?, 1)
                        """,
                        (
                            target_plan_id, target_goal_id, transition["learner_id"],
                            transition["target_project_id"], target_project["repository_id"],
                            transition["target_revision"],
                            f"continuity:{transition['id']}:{plan['plan_id']}",
                        ),
                    )
                    step_map: dict[str, str] = {}
                    for order, step in enumerate(retained, start=1):
                        mapping = mapping_by_source[step["target_id"]]
                        step_id = _stable_id("S", target_plan_id, str(order))
                        step_map[step["step_id"]] = step_id
                        status = step["status"]
                        if mapping["mapping_status"] == "modified":
                            status = "needs_review"
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO learning_plan_steps (
                                step_id, plan_id, step_order, target_id, objective,
                                action_type, completion_requirement, status, schema_version
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
                            """,
                            (
                                step_id, target_plan_id, order, target_ids[step["target_id"]],
                                step["objective"], step["action_type"],
                                step["completion_requirement"], status,
                            ),
                        )
                    prerequisites = conn.execute(
                        "SELECT * FROM learning_step_prerequisites WHERE plan_id=?",
                        (plan["plan_id"],),
                    ).fetchall()
                    for edge in prerequisites:
                        if edge["step_id"] in step_map and edge["prerequisite_step_id"] in step_map:
                            conn.execute(
                                """
                                INSERT OR IGNORE INTO learning_step_prerequisites
                                    (plan_id, step_id, prerequisite_step_id)
                                VALUES (?, ?, ?)
                                """,
                                (
                                    target_plan_id, step_map[edge["step_id"]],
                                    step_map[edge["prerequisite_step_id"]],
                                ),
                            )
                else:
                    lineage_status = "history_only"
            conn.execute(
                """
                INSERT INTO learning_continuity_goal_lineage (
                    transition_id, source_goal_id, target_goal_id,
                    source_plan_id, target_plan_id, lineage_status
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    transition["id"], goal["goal_id"], target_goal_id,
                    plan["plan_id"] if plan is not None else None,
                    target_plan_id, lineage_status,
                ),
            )

    def _transition_row(self, workspace_id: str, transition_id: str) -> sqlite3.Row | None:
        with self.database.connect() as conn:
            return conn.execute(
                "SELECT * FROM learning_continuity_transitions WHERE workspace_id=? AND id=?",
                (workspace_id, transition_id),
            ).fetchone()

    @staticmethod
    def _validate_transition_id(transition_id: str) -> None:
        try:
            parsed = uuid.UUID(transition_id)
        except (ValueError, AttributeError) as exc:
            raise LearningContinuityError(
                "invalid_continuity_transition_id",
                "The learning continuity transition ID is invalid.",
                status_code=422,
            ) from exc
        if str(parsed) != transition_id.lower():
            raise LearningContinuityError(
                "invalid_continuity_transition_id",
                "The learning continuity transition ID is invalid.",
                status_code=422,
            )

    @classmethod
    def _public_transition(cls, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "transition_id": row["id"],
            "workspace_id": row["workspace_id"],
            "source_project_id": row["source_project_id"],
            "target_project_id": row["target_project_id"],
            "source_revision": row["source_revision"],
            "target_revision": row["target_revision"],
            "activation_version": int(row["activation_version"]),
            "learner_id": row["learner_id"],
            "mapping_config_identity": row["mapping_config_identity"],
            "status": row["status"],
            "stats": _json(row["stats_json"], cls._empty_stats()),
            "error_code": row["error_code"],
            "error_message": row["error_message"],
            "retryable": bool(row["retryable"]),
            "retry_count": int(row["retry_count"]),
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _empty_stats() -> dict[str, int]:
        return {
            "total": 0, "unchanged_exact": 0, "renamed_exact": 0,
            "modified": 0, "deleted": 0, "ambiguous": 0,
            "unmapped": 0, "incompatible": 0, "retained": 0,
            "needs_review": 0, "history_only": 0, "not_inherited": 0,
        }


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:32]
    return f"{prefix}-{digest}"


def _json(value: str, fallback: Any) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def _structure_identity(content: str) -> str:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            node.name = "<symbol>"
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode("utf-8")).hexdigest()
