"""Saved answer projections. These bindings are historical data, not live Evidence."""
from __future__ import annotations

import base64
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.services.agent_contracts import normalize_repository_relative_path


class BaseRetrievalSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    attempted: bool
    status: str
    new_evidence_count: int = Field(ge=0)
    retrieval_detail: dict[str, int | str] | None = None


class ExecutionSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
    answer_mode: str
    evidence_count: int = Field(ge=0)
    citation_count: int = Field(ge=0)
    base_retrieval: BaseRetrievalSummaryResponse | None = None
    planner_requests_attempted: int | None = Field(default=None, ge=0)
    planner_repair_attempts: int | None = Field(default=None, ge=0)
    provider_logical_calls: int | None = Field(default=None, ge=0)
    provider_http_attempt_count: int | None = Field(default=None, ge=0)
    tool_calls_attempted: int | None = Field(default=None, ge=0)
    tool_calls_succeeded: int | None = Field(default=None, ge=0)
    tool_calls_failed: int | None = Field(default=None, ge=0)
    steps_used: int | None = Field(default=None, ge=0)
    tool_calls_used: int | None = Field(default=None, ge=0)
    agent_elapsed_ms: int | None = Field(default=None, ge=0)
    planner_duration_ms: int | None = Field(default=None, ge=0)
    tool_duration_ms: int | None = Field(default=None, ge=0)
    finalization_duration_ms: int | None = Field(default=None, ge=0)
    planner_termination_reason: str | None = None
    citation_validation_passed: bool | None = None
    relation_validation_passed: bool | None = None
    post_generation_validation_passed: bool | None = None
    diagnostics_truncated: bool | None = None


class SavedCitation(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    evidence_id: str | None = None
    path: str | None = None
    summary: str | None = None
    snippet: str | None = None
    qualified_name: str | None = None
    start_line: int | None = None
    end_line: int | None = None


class SavedCitationBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    evidence_id: str = Field(pattern=r"^E[1-9]\d*$", max_length=32)
    project_id: str = Field(min_length=1, max_length=128)
    repository_revision: str = Field(min_length=1, max_length=256)
    path: str = Field(min_length=1, max_length=4096)
    qualified_name: str = Field(max_length=4096)
    symbol_name: str = Field(max_length=4096)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


class RecoveryMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    format_version: Literal[1]
    citation_bindings: list[SavedCitationBinding] = Field(max_length=64)
    execution_mode: Literal["rag", "agent"] | None = None
    execution_summary: ExecutionSummaryResponse | None = None
    answer_mode: Literal["llm_grounded", "deterministic"] | None = None
    grounding_status: Literal["grounded", "insufficient_evidence", "degraded"] | None = None
    warnings: list[str] | None = Field(default=None, max_length=32)
    metadata_truncated: bool = False


class ChatHistoryRecord(BaseModel):
    record_id: str
    project_id: str
    repository_revision: str | None
    revision_source: Literal["saved", "snapshot_link", "unavailable"]
    identity_status: Literal["reliable", "missing", "conflict"]
    created_at: str | None
    question: str
    answer: str
    citations: list[SavedCitation]
    citations_status: Literal["available", "partial", "invalid"]
    recovery_metadata: RecoveryMetadata | None
    metadata_status: Literal["available", "missing", "invalid", "binding_conflict"]
    snapshot_status: Literal["active", "superseded"]
    is_current_snapshot: bool
    full_file_available: bool
    full_file_unavailable_reason: str | None


class ChatHistoryPage(BaseModel):
    workspace_id: str
    active_project_id: str
    repository_revision: str
    activation_version: int
    items: list[ChatHistoryRecord]
    next_cursor: str | None


class ChatHistoryError(ValueError):
    def __init__(self, code: str, status_code: int) -> None:
        super().__init__(code)
        self.code = code
        self.status_code = status_code


class HistoryCursor(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    format_version: Literal[1] = 1
    workspace_id: str
    active_project_id: str
    repository_revision: str
    activation_version: int = Field(ge=1)
    anchor_id: str = Field(pattern=r"^[1-9]\d{0,18}$")
    before_id: str = Field(pattern=r"^[1-9]\d{0,18}$")

    def encode(self) -> str:
        return base64.urlsafe_b64encode(self.model_dump_json().encode()).decode().rstrip("=")

    @classmethod
    def decode(cls, value: str) -> "HistoryCursor":
        try:
            if not value or len(value) > 2048:
                raise ValueError()
            raw = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
            result = cls.model_validate_json(raw)
            if not 0 < int(result.before_id) <= int(result.anchor_id) <= 9_223_372_036_854_775_807:
                raise ValueError()
            return result
        except (ValueError, UnicodeError):
            raise ChatHistoryError("history_cursor_invalid", 422) from None


def bindings_match(metadata: RecoveryMetadata, citations: list[dict[str, Any]],
                   project_id: str, revision: str | None) -> bool:
    """Check saved binding consistency only; no source read or Evidence promotion."""
    by_id = {item.evidence_id: item for item in metadata.citation_bindings}
    if len(by_id) != len(metadata.citation_bindings):
        return False
    for item in metadata.citation_bindings:
        try:
            normalize_repository_relative_path(item.path)
        except ValueError:
            return False
        if (item.project_id != project_id or item.repository_revision != revision
                or item.end_line < item.start_line):
            return False
    for citation in citations:
        evidence_id = citation.get("evidence_id")
        if evidence_id is None:
            continue
        item = by_id.get(evidence_id)
        start, end = citation.get("start_line"), citation.get("end_line")
        if (item is None or citation.get("path") != item.path
                or citation.get("qualified_name") != (item.qualified_name or item.symbol_name)
                or type(start) is not int or type(end) is not int
                or start < item.start_line or end > item.end_line or end < start):
            return False
    return True


def prepare_recovery_metadata(payload: dict[str, Any], project_id: str,
                              revision: str) -> dict[str, Any]:
    """White-list the already validated public response before persistence."""
    keys = tuple(SavedCitationBinding.model_fields)
    original_warnings = payload.get("warnings")
    warnings = ([item[:512] for item in original_warnings[:32]]
                if isinstance(original_warnings, list) else None)
    candidate = {
        "format_version": 1,
        "citation_bindings": [{key: item[key] for key in keys} for item in payload["evidence"]],
        **{key: payload.get(key) for key in (
            "execution_mode", "execution_summary", "answer_mode", "grounding_status")},
        "warnings": warnings,
        "metadata_truncated": warnings != original_warnings,
    }
    metadata = RecoveryMetadata.model_validate(candidate)
    if not bindings_match(metadata, payload["citations"], project_id, revision):
        raise ValueError("saved citation binding is invalid")
    # The execution summary has already passed the bounded public projection.
    return json.loads(metadata.model_dump_json())


def project_history_record(row: dict[str, Any], active_project_id: str) -> dict[str, Any]:
    saved = row["repository_revision"]
    linked = row["linked_revision"]
    association_valid = bool(linked and linked == row["project_revision"])
    revision = saved if saved is not None else (linked if association_valid else None)
    source = "saved" if saved is not None else "snapshot_link" if revision else "unavailable"
    identity = ("reliable" if association_valid and revision == linked else
                "conflict" if saved is not None or linked != row["project_revision"] else "missing")
    current = row["project_id"] == active_project_id and row["activation_status"] == "active"
    if row["activation_status"] == "active" and not current:
        identity = "conflict"
    citations, citations_status = [], "available"
    try:
        raw = json.loads(row["citations_json"] or "[]")
        if not isinstance(raw, list):
            raise ValueError()
        for item in raw:
            try:
                citations.append(SavedCitation.model_validate(item).model_dump())
            except ValueError:
                citations_status = "partial"
    except (ValueError, TypeError):
        citations_status = "invalid"
    metadata, metadata_status = None, "missing"
    if row["recovery_metadata_json"] is not None:
        try:
            metadata = RecoveryMetadata.model_validate_json(row["recovery_metadata_json"])
            metadata_status = "available"
            if citations_status != "available" or not bindings_match(metadata, citations, row["project_id"], revision):
                metadata, metadata_status = None, "binding_conflict"
        except (ValueError, TypeError):
            metadata_status = "invalid"
    reason = ("history_identity_unreliable" if identity != "reliable" else
              "history_metadata_invalid" if metadata_status in {"invalid", "binding_conflict"} else
              "history_snapshot_old" if not current else None)
    return {
        "record_id": str(row["id"]), "project_id": row["project_id"],
        "repository_revision": revision, "revision_source": source, "identity_status": identity,
        "created_at": row["created_at"], "question": row["question"], "answer": row["answer"],
        "citations": citations, "citations_status": citations_status,
        "recovery_metadata": metadata.model_dump() if metadata else None, "metadata_status": metadata_status,
        "snapshot_status": row["activation_status"], "is_current_snapshot": current,
        "full_file_available": reason is None, "full_file_unavailable_reason": reason,
    }
