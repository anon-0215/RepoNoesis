from __future__ import annotations

import json
from typing import Any

from app.database import Database


DEFAULT_WORKSPACE_LIMIT = 20
MAX_WORKSPACE_LIMIT = 100


class WorkspaceNotFound(LookupError):
    pass


class WorkspaceCorrupt(RuntimeError):
    pass


class WorkspaceUnavailable(RuntimeError):
    pass


class WorkspaceService:
    def __init__(self, database: Database) -> None:
        self.database = database

    def list_workspaces(self, *, limit: int, offset: int) -> dict[str, Any]:
        records = self.database.list_workspace_records(limit, offset)
        return {
            "items": [self._summary(row) for row in records["items"]],
            "total": records["total"],
            "limit": limit,
            "offset": offset,
        }

    def get_workspace(self, workspace_id: str) -> dict[str, Any]:
        row = self.database.get_workspace_record(workspace_id)
        if row is None:
            raise WorkspaceNotFound(workspace_id)
        if not self._association_is_valid(row):
            raise WorkspaceCorrupt(workspace_id)
        if row["project_status"] != "done":
            raise WorkspaceUnavailable(workspace_id)
        return {
            **self._summary(row),
            "openable": True,
            "active_snapshot": {
                "project_id": row["active_project_id"],
                "repository_revision": row["repository_revision"],
                "status": row["project_status"],
                "primary_language": row["primary_language"] or "",
                "frameworks": self._json_list(row["frameworks_json"]),
                "updated_at": row["project_updated_at"],
            },
        }

    def _summary(self, row: dict[str, Any]) -> dict[str, Any]:
        valid = self._association_is_valid(row)
        openable = valid and row.get("project_status") == "done"
        return {
            "workspace_id": row["workspace_id"],
            "display_name": row["display_name"],
            "source_type": row["source_type"],
            "project_status": row["project_status"] if valid else "unavailable",
            "repository_revision": row["repository_revision"] if valid else "",
            "openable": openable,
            "project_id": row.get("active_project_id") if valid else None,
            "total_chunks": int(row.get("total_chunks") or 0) if valid else 0,
            "embedding_count": int(row.get("embedding_count") or 0) if valid else 0,
            "created_at": row["created_at"],
            "updated_at": row["project_updated_at"] or row["updated_at"] or row["created_at"],
        }

    @staticmethod
    def _association_is_valid(row: dict[str, Any]) -> bool:
        return bool(
            row.get("active_project_id")
            and row.get("revision_workspace_id") == row.get("workspace_id")
            and row.get("revision_project_id") == row.get("active_project_id")
            and row.get("linked_revision") == row.get("repository_revision")
            and row.get("linked_activation_status") == "active"
        )

    @staticmethod
    def _json_list(value: str | None) -> list[str]:
        try:
            parsed = json.loads(value or "[]")
        except (TypeError, json.JSONDecodeError):
            return []
        return [str(item) for item in parsed] if isinstance(parsed, list) else []
