from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from threading import Event
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


AGENT_SCHEMA_VERSION = 1


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
    default_tool_timeout_ms: int = 15_000
    max_search_results: int = 20
    max_observation_bytes: int = 65_536
    max_source_read_lines: int = 200
    max_source_read_bytes: int = 32_768
    max_accumulated_evidence_context_bytes: int = 49_152
    max_planner_output_tokens_per_step: int = 512
    max_total_planner_output_tokens: int = 2_048
    max_final_answer_tokens: int = 1_600


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchCodeInput(StrictModel):
    query: str = Field(min_length=1, max_length=2_000)
    path: str | None = Field(default=None, max_length=500)
    language: str | None = Field(default=None, max_length=80)
    symbol: str | None = Field(default=None, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)


class LookupSymbolInput(StrictModel):
    symbol: str = Field(min_length=1, max_length=500)
    match_mode: Literal["exact", "prefix", "fuzzy"] = "exact"
    path: str | None = Field(default=None, max_length=500)
    language: str | None = Field(default=None, max_length=80)
    top_k: int = Field(default=10, ge=1, le=20)


class ReadSourceInput(StrictModel):
    path: str = Field(min_length=1, max_length=500)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


class ValidateEvidenceInput(StrictModel):
    evidence_ids: list[str] = Field(min_length=1, max_length=20)


class PlannerDecision(StrictModel):
    status: Literal["continue", "answer", "insufficient_evidence"]
    action: str | None = Field(default=None, max_length=80)
    arguments: dict[str, Any] = Field(default_factory=dict)
    decision_summary: str = Field(min_length=1, max_length=240)


@dataclass
class ToolCall:
    call_id: str
    step_id: str
    tool_name: str
    tool_version: str
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
                    "tool_name": call.tool_name,
                    "tool_version": call.tool_version,
                    "status": observation.status,
                    "result_count": observation.metrics.get("result_count", 0),
                    "duration_ms": observation.metrics.get("duration_ms", 0),
                    "truncated": observation.truncated,
                }
            )
        return {
            "step_id": self.step_id,
            "action": self.action,
            "tool_calls": calls,
            "decision_summary": self.decision_summary[:240],
            "completion_status": self.completion_status,
            "remaining_budget": dict(self.remaining_budget),
        }


class CancellationToken:
    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()
