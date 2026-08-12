from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import PurePosixPath, PureWindowsPath
from threading import Event
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


AGENT_SCHEMA_VERSION = 1
_PUBLIC_AGENT_ACTIONS = frozenset(
    {
        "answer",
        "expand_relations",
        "get_learning_context",
        "insufficient_evidence",
        "lookup_symbol",
        "read_source",
        "search_code",
        "unknown_tool",
        "validate_evidence",
    }
)


def normalize_repository_relative_path(path: str) -> str:
    """Return one safe POSIX repository-relative path without broadening scope."""

    if not isinstance(path, str) or not path.strip() or "\0" in path:
        raise ValueError("unsafe repository path")
    windows = PureWindowsPath(path)
    posix = PurePosixPath(path)
    if (
        windows.is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
        or posix.is_absolute()
    ):
        raise ValueError("unsafe repository path")
    normalized = path.replace("\\", "/")
    normalized_path = PurePosixPath(normalized)
    if (
        any(part in {"", ".", ".."} for part in normalized.split("/"))
        or any(part in {"", ".", ".."} for part in normalized_path.parts)
        or str(normalized_path) != normalized
    ):
        raise ValueError("unsafe repository path")
    return normalized


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class AgentLimits:
    max_agent_steps: int = 5
    max_tool_calls: int = 8
    max_calls_per_step: int = 1
    max_same_tool_calls: int = 3
    max_no_progress_steps: int = 2
    total_deadline_ms: int = 60_000
    default_tool_timeout_ms: int = 40_000
    min_final_answer_budget_ms: int = 5_000
    max_search_results: int = 20
    max_observation_bytes: int = 65_536
    max_source_read_lines: int = 200
    max_source_read_bytes: int = 32_768
    max_accumulated_evidence_context_bytes: int = 49_152
    max_planner_output_tokens_per_step: int = 512
    max_total_planner_output_tokens: int = 2_048
    max_final_answer_tokens: int = 1_600
    default_relation_depth: int = 1
    max_relation_depth: int = 2
    max_relation_seed_nodes: int = 8
    max_relation_neighbors_per_node: int = 20
    max_relation_nodes: int = 64
    max_relation_edges: int = 128
    max_relation_paths: int = 24
    max_relation_observation_bytes: int = 65_536
    max_relation_evidence_items: int = 16
    max_learning_state_items: int = 16
    max_recent_learning_events: int = 8
    max_plan_steps_in_learning_context: int = 12
    max_learning_context_bytes: int = 16_384


BudgetFailureReason = Literal[
    "deadline_exceeded",
    "planner_budget_exhausted",
    "final_answer_not_attempted",
    "tool_timeout",
]


@dataclass(frozen=True)
class StageDeadline:
    deadline_monotonic: float
    reason: BudgetFailureReason


@dataclass(frozen=True)
class RequestBudget:
    """One request-owned absolute deadline and its derived work cutoff.

    The final-answer reserve is a start gate, not a second request deadline and
    not a guarantee that synchronous finalization will finish in time.
    """

    request_started_at: float
    request_deadline_at: float
    final_answer_reserve_ms: int

    @classmethod
    def create(cls, *, started_at: float, limits: AgentLimits) -> "RequestBudget":
        return cls(
            request_started_at=started_at,
            request_deadline_at=started_at + max(0, limits.total_deadline_ms) / 1000,
            final_answer_reserve_ms=max(0, limits.min_final_answer_budget_ms),
        )

    @classmethod
    def from_deadline(
        cls,
        *,
        started_at: float,
        deadline_at: float,
        final_answer_reserve_ms: int,
    ) -> "RequestBudget":
        return cls(
            request_started_at=started_at,
            request_deadline_at=deadline_at,
            final_answer_reserve_ms=max(0, final_answer_reserve_ms),
        )

    @property
    def work_cutoff_at(self) -> float:
        return self.request_deadline_at - self.final_answer_reserve_ms / 1000

    @property
    def total_budget_ms(self) -> int:
        return max(0, int((self.request_deadline_at - self.request_started_at) * 1000))

    def request_remaining_ms(self, now: float) -> int:
        return max(0, int((self.request_deadline_at - now) * 1000))

    def work_remaining_ms(self, now: float) -> int:
        return max(0, int((self.work_cutoff_at - now) * 1000))

    def request_expired(self, now: float) -> bool:
        return now >= self.request_deadline_at

    def work_expired(self, now: float) -> bool:
        return now >= self.work_cutoff_at

    def work_failure_reason(self, now: float) -> BudgetFailureReason:
        return (
            "deadline_exceeded"
            if self.request_expired(now)
            else "planner_budget_exhausted"
        )

    def tool_deadline(self, *, started_at: float, timeout_ms: int) -> StageDeadline:
        return self.derive_tool_deadline(
            request_deadline_at=self.request_deadline_at,
            work_cutoff_at=self.work_cutoff_at,
            started_at=started_at,
            timeout_ms=timeout_ms,
        )

    @staticmethod
    def derive_tool_deadline(
        *,
        request_deadline_at: float,
        work_cutoff_at: float,
        started_at: float,
        timeout_ms: int,
    ) -> StageDeadline:
        tool_timeout_at = started_at + max(0, timeout_ms) / 1000
        earliest = min(
            request_deadline_at,
            work_cutoff_at,
            tool_timeout_at,
        )
        if request_deadline_at <= earliest:
            reason: BudgetFailureReason = "deadline_exceeded"
        elif work_cutoff_at <= earliest:
            reason = "final_answer_not_attempted"
        else:
            reason = "tool_timeout"
        return StageDeadline(deadline_monotonic=earliest, reason=reason)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class SearchCodeInput(StrictModel):
    query: str = Field(min_length=1, max_length=2_000)
    path: str | None = Field(default=None, max_length=500)
    language: str | None = Field(default=None, max_length=80)
    symbol: str | None = Field(default=None, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20, strict=True)

    @field_validator("path")
    @classmethod
    def normalize_path(cls, value: str | None) -> str | None:
        return _normalize_optional_tool_path(value)


class LookupSymbolInput(StrictModel):
    symbol: str = Field(min_length=1, max_length=500)
    match_mode: Literal["exact", "prefix", "fuzzy"] = "exact"
    path: str | None = Field(default=None, max_length=500)
    language: str | None = Field(default=None, max_length=80)
    top_k: int = Field(default=10, ge=1, le=20, strict=True)

    @field_validator("path")
    @classmethod
    def normalize_path(cls, value: str | None) -> str | None:
        return _normalize_optional_tool_path(value)


class ReadSourceInput(StrictModel):
    path: str = Field(min_length=1, max_length=500)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)

    @field_validator("path")
    @classmethod
    def normalize_path(cls, value: str) -> str:
        return normalize_repository_relative_path(value)


def _normalize_optional_tool_path(value: str | None) -> str | None:
    if value is None:
        return None
    return normalize_repository_relative_path(value)


class ValidateEvidenceInput(StrictModel):
    evidence_ids: list[str] = Field(min_length=1, max_length=20)


class ExpandRelationsInput(StrictModel):
    seed_evidence_ids: list[str] = Field(default_factory=list, max_length=8)
    seed_symbol_ids: list[str] = Field(default_factory=list, max_length=8)
    relation_types: list[
        Literal["imports", "calls", "references", "defines"]
    ] = Field(default_factory=lambda: ["imports", "calls", "references"], min_length=1)
    direction: Literal["outbound", "inbound", "both"] = "outbound"
    max_depth: int = Field(default=1, ge=1, le=2)
    per_node_limit: int = Field(default=20, ge=1, le=20)

    @model_validator(mode="after")
    def require_seed(self) -> "ExpandRelationsInput":
        if not self.seed_evidence_ids and not self.seed_symbol_ids:
            raise ValueError("at least one relation seed is required")
        return self


class GetLearningContextInput(StrictModel):
    pass


class PlannerDecision(StrictModel):
    status: Literal["continue", "answer", "insufficient_evidence"]
    action: str | None = Field(default=None, max_length=80)
    arguments: dict[str, Any] = Field(default_factory=dict)
    decision_summary: str = Field(min_length=1, max_length=240)


PlannerValidationStage = Literal["adapter", "parser", "schema", "semantic"]
PlannerValidationCode = Literal[
    "valid",
    "invalid_json",
    "wrong_top_level_type",
    "schema_missing_field",
    "schema_extra_field",
    "schema_invalid_type",
    "schema_invalid_literal",
    "semantic_invalid_decision",
    "semantic_invalid_tool_contract",
    "provider_output_truncated",
    "provider_empty_content",
    "provider_invalid_response",
    "provider_unavailable",
    "provider_authentication_failed",
    "provider_rate_limited",
    "provider_request_rejected",
    "deadline_exceeded",
]


@dataclass(frozen=True)
class PlannerValidationFailure:
    """Content-free, stable details for one rejected Planner response."""

    stage: PlannerValidationStage
    stable_code: PlannerValidationCode
    field_path: tuple[str | int, ...] = ()
    output_chars: int = 0
    output_sha256: str | None = None
    finish_reason_present: bool = False
    finish_reason_value: str | None = None
    content_present: bool = False
    reasoning_content_present: bool = False
    markdown_fence_detected: bool = False
    repair_attempt: bool = False

    def to_safe_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "stage": self.stage,
            "stable_code": self.stable_code,
            "field_path": list(self.field_path),
            "output_chars": max(0, self.output_chars),
            "finish_reason_present": self.finish_reason_present,
            "content_present": self.content_present,
            "reasoning_content_present": self.reasoning_content_present,
            "markdown_fence_detected": self.markdown_fence_detected,
            "repair_attempt": self.repair_attempt,
        }
        if self.output_sha256 is not None:
            result["output_sha256"] = self.output_sha256
        if self.finish_reason_value is not None:
            result["finish_reason_value"] = self.finish_reason_value
        return result


@dataclass(frozen=True)
class PlannerValidationResult:
    decision: PlannerDecision | None = None
    failure: PlannerValidationFailure | None = None

    @property
    def valid(self) -> bool:
        return self.decision is not None and self.failure is None


@dataclass
class ToolCall:
    call_id: str
    step_id: str
    tool_name: str
    tool_version: str | None
    parameters: dict[str, Any]
    timeout_ms: int
    budget: dict[str, int]
    started_at: str | None = None
    ended_at: str | None = None
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolObservation:
    call_id: str
    status: str
    structured_results: Any = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    truncated: bool = False
    error: dict[str, str] | None = None
    metrics: dict[str, int | float | str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentStep:
    step_id: str
    user_goal: str
    action: str
    tool_calls: list[ToolCall]
    observations: list[ToolObservation]
    decision_summary: str
    completion_status: str
    remaining_budget: dict[str, int]

    def to_public_dict(self) -> dict[str, Any]:
        calls = []
        for call, observation in zip(self.tool_calls, self.observations):
            calls.append(
                {
                    "call_id": call.call_id,
                    "tool_name": _public_tool_name(call.tool_name),
                    "tool_version": call.tool_version,
                    "status": observation.status,
                    "result_count": observation.metrics.get("result_count", 0),
                    "duration_ms": observation.metrics.get("duration_ms", 0),
                    "truncated": observation.truncated,
                }
            )
        return {
            "step_id": self.step_id,
            "action": _public_tool_name(self.action),
            "tool_calls": calls,
            "decision_summary": self.decision_summary[:240],
            "completion_status": self.completion_status,
            "remaining_budget": dict(self.remaining_budget),
        }


def _public_tool_name(value: str) -> str:
    return value if value in _PUBLIC_AGENT_ACTIONS else "unknown_tool"


class CancellationToken:
    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()
