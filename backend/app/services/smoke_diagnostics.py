from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal


MAX_SMOKE_DIAGNOSTICS_BYTES = 8_192
MAX_PROVIDER_CALLS = 16
MAX_TOOL_ENTRIES = 16

SmokeStage = Literal[
    "fixture_creation",
    "repository_import",
    "provider_preflight",
    "embedding_preflight",
    "repository_persistence",
    "source_analysis",
    "chunk_extraction",
    "relation_index",
    "embedding_index",
    "agent_setup",
    "agent_planner",
    "agent_tools",
    "final_answer",
    "citation_validation",
    "relation_validation",
    "post_generation_validation",
    "gate_assertion",
    "report_build",
]
ProviderPurpose = Literal["planner", "final_answer"]
FallbackReasonCode = Literal[
    "llm_unavailable",
    "context_binding_failed",
    "planner_validation_failed",
]
FinalAnswerFailureReasonCode = Literal[
    "response_empty",
    "answer_token_budget_exceeded",
    "citation_missing",
    "citation_format_invalid",
    "citation_unknown",
    "citation_validation_failed",
    "citation_location_missing",
    "citation_path_mismatch",
    "citation_line_range_mismatch",
    "citation_evidence_binding_failed",
    "relation_validation_failed",
    "post_generation_validation_failed",
    "provider_failed",
    "deadline_exhausted",
    "unknown_safe_failure",
]

SMOKE_STAGES = frozenset(SmokeStage.__args__)
PROVIDER_PURPOSES = frozenset(ProviderPurpose.__args__)
FALLBACK_REASON_CODES = frozenset(FallbackReasonCode.__args__)
FINAL_ANSWER_FAILURE_REASON_CODES = frozenset(
    FinalAnswerFailureReasonCode.__args__
)
SMOKE_ERROR_MESSAGES = {
    "smoke_embedding_configuration_incomplete": (
        "The smoke gate embedding configuration is incomplete."
    ),
    "smoke_no_python_chunks": "The smoke gate did not produce Python code chunks.",
    "smoke_validated_evidence_missing": (
        "The smoke gate did not produce validated Evidence and citations."
    ),
    "smoke_provider_grounding_failed": (
        "The smoke gate provider answer did not satisfy the bounded grounding requirements."
    ),
    "smoke_stage_failed": "The smoke gate failed during a recorded stage.",
}
_TYPE_VALUES = frozenset({"null", "boolean", "number", "string", "array", "object", "other"})
_FINISH_REASONS = frozenset(
    {"stop", "length", "tool_calls", "content_filter", "function_call", "other"}
)
_AGENT_MODES = frozenset({"bounded", "deterministic_fallback"})
_ANSWER_MODES = frozenset({"llm_grounded", "deterministic"})
_AGENT_STATUSES = frozenset(
    {
        "completed",
        "degraded",
        "insufficient_evidence",
        "tool_budget_exhausted",
        "final_answer_failed",
        "budget_exhausted",
        "cancelled",
    }
)
_TOOL_STATUSES = frozenset({"succeeded", "failed", "rejected", "timed_out", "cancelled"})


@dataclass(frozen=True)
class SmokeGateError(RuntimeError):
    code: str
    gate: str
    stage: str
    exception_type: str
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        if self.code not in SMOKE_ERROR_MESSAGES:
            raise ValueError("unsupported smoke error code")
        if self.gate not in {"A", "B", "C"}:
            raise ValueError("unsupported smoke gate")
        if self.stage not in SMOKE_STAGES:
            raise ValueError("unsupported smoke stage")
        exception_type = (
            self.exception_type
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", self.exception_type)
            else "Exception"
        )
        object.__setattr__(self, "exception_type", exception_type)
        object.__setattr__(self, "diagnostics", _safe_smoke_diagnostics(self.diagnostics))

    def __str__(self) -> str:
        return SMOKE_ERROR_MESSAGES[self.code]

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": SMOKE_ERROR_MESSAGES[self.code],
            "gate": self.gate,
            "stage": self.stage,
            "exception_type": self.exception_type,
            "diagnostics": dict(self.diagnostics),
        }


class SmokeDiagnosticsRecorder:
    """Request-local, bounded recorder used only when explicitly injected by smoke."""

    def __init__(self) -> None:
        self.stage: str | None = None
        self._provider_calls: list[dict[str, Any]] = []
        self._agent: dict[str, Any] = {}
        self._registered_tools: set[str] = set()
        self._tool_calls: dict[str, dict[str, int | str]] = {}
        self._truncated = False

    def enter_stage(self, stage: SmokeStage) -> None:
        if stage not in SMOKE_STAGES:
            raise ValueError("unsupported smoke stage")
        self.stage = stage

    def start_provider_call(self, purpose: ProviderPurpose) -> int | None:
        if purpose not in PROVIDER_PURPOSES:
            raise ValueError("unsupported provider diagnostics purpose")
        if len(self._provider_calls) >= MAX_PROVIDER_CALLS:
            self._truncated = True
            return None
        self._provider_calls.append(
            {
                "purpose": purpose,
                "request_started": True,
                "response_received": False,
            }
        )
        return len(self._provider_calls) - 1

    def record_provider_response(
        self, call_id: int | None, metadata: dict[str, Any]
    ) -> None:
        if call_id is None or call_id < 0 or call_id >= len(self._provider_calls):
            return
        self._provider_calls[call_id].update(_safe_provider_metadata(metadata))

    def begin_agent(self, registered_tools: list[str]) -> None:
        self._registered_tools = {
            item for item in registered_tools[:MAX_TOOL_ENTRIES] if _safe_tool_name(item)
        }
        self._agent.update(
            {
                "planner_requests_attempted": 0,
                "planner_response_received": False,
                "planner_repair_attempts": 0,
                "tool_calls_attempted": 0,
                "tool_calls_succeeded": 0,
                "tool_calls_failed": 0,
                "final_answer_attempted": False,
                "final_answer_response_received": False,
            }
        )

    def record_planner_request(self, *, repair: bool) -> None:
        self.enter_stage("agent_planner")
        self._agent["planner_requests_attempted"] = int(
            self._agent.get("planner_requests_attempted", 0)
        ) + 1
        if repair:
            self._agent["planner_repair_attempts"] = int(
                self._agent.get("planner_repair_attempts", 0)
            ) + 1

    def record_planner_response(self, received: bool) -> None:
        if received:
            self._agent["planner_response_received"] = True

    def record_planner_validation(self, valid: bool) -> None:
        self._agent["planner_json_valid"] = bool(valid)

    def record_fallback(self, reason: FallbackReasonCode) -> None:
        if reason not in FALLBACK_REASON_CODES:
            raise ValueError("unsupported fallback reason")
        self._agent["fallback_reason_code"] = reason

    def record_tool_attempt(self, tool_name: str) -> None:
        self.enter_stage("agent_tools")
        self._agent["tool_calls_attempted"] = int(
            self._agent.get("tool_calls_attempted", 0)
        ) + 1
        if tool_name in self._registered_tools:
            item = self._tool_calls.setdefault(
                tool_name,
                {"name": tool_name, "attempted": 0, "succeeded": 0, "failed": 0},
            )
            item["attempted"] = int(item["attempted"]) + 1

    def record_tool_result(self, tool_name: str, status: str) -> None:
        succeeded = status == "succeeded"
        key = "tool_calls_succeeded" if succeeded else "tool_calls_failed"
        self._agent[key] = int(self._agent.get(key, 0)) + 1
        if tool_name in self._tool_calls and status in _TOOL_STATUSES:
            item = self._tool_calls[tool_name]
            item["succeeded" if succeeded else "failed"] = int(
                item["succeeded" if succeeded else "failed"]
            ) + 1

    def record_tool_budget_exhausted(self) -> None:
        self._agent["tool_budget_exhausted"] = True

    def record_final_answer_attempt(self) -> None:
        self.enter_stage("final_answer")
        self._agent["final_answer_attempted"] = True

    def record_final_answer_response(self) -> None:
        self._agent["final_answer_response_received"] = True

    def mark_citation_validation_completed(self, *, passed: bool) -> None:
        self._record_validation("citation_validation", passed)

    def mark_relation_validation_completed(self, *, passed: bool) -> None:
        self._record_validation("relation_validation", passed)

    def mark_post_generation_validation_completed(self, *, passed: bool) -> None:
        self._record_validation("post_generation_validation", passed)

    def mark_grounded_reference_validation_completed(self, *, passed: bool) -> None:
        self._record_validation("grounded_reference_validation", passed)

    def _record_validation(self, prefix: str, passed: bool) -> None:
        completed_key = f"{prefix}_completed"
        passed_key = f"{prefix}_passed"
        self._agent[completed_key] = True
        previous = self._agent.get(passed_key)
        self._agent[passed_key] = bool(passed) and (
            bool(previous) if isinstance(previous, bool) else True
        )

    def record_grounded_answer_candidate(
        self, *, received: bool, citation_count: int | None = None
    ) -> None:
        self._agent["grounded_answer_candidate_received"] = bool(received)
        if (
            received
            and isinstance(citation_count, int)
            and not isinstance(citation_count, bool)
            and 0 <= citation_count <= 1_000
        ):
            self._agent["grounded_candidate_citation_count"] = citation_count

    def record_grounded_answer_accepted(self, accepted: bool) -> None:
        self._agent["grounded_answer_accepted"] = bool(accepted)

    def record_final_answer_failure(
        self, reason: FinalAnswerFailureReasonCode
    ) -> None:
        if reason not in FINAL_ANSWER_FAILURE_REASON_CODES:
            raise ValueError("unsupported final answer failure reason")
        self._agent["final_answer_failure_reason_code"] = reason

    def record_agent_result(self, result: dict[str, Any]) -> None:
        for key, allowed in (
            ("agent_mode", _AGENT_MODES),
            ("answer_mode", _ANSWER_MODES),
            ("agent_status", _AGENT_STATUSES),
        ):
            value = result.get(key)
            if isinstance(value, str) and value in allowed:
                self._agent[key] = value
        evidence = result.get("evidence")
        citations = result.get("citations")
        if isinstance(evidence, list):
            self._agent["evidence_count"] = len(evidence)
        if isinstance(citations, list):
            self._agent["citation_count"] = len(citations)

    def snapshot(self) -> dict[str, Any]:
        payload: dict[str, Any] = dict(self._agent)
        if self._provider_calls:
            payload["provider_calls"] = [dict(item) for item in self._provider_calls]
        if self._tool_calls:
            payload["tool_calls"] = [
                dict(self._tool_calls[name]) for name in sorted(self._tool_calls)
            ][:MAX_TOOL_ENTRIES]
        if self._truncated:
            payload["diagnostics_truncated"] = True
        payload = _safe_smoke_diagnostics(payload)
        while (
            len(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
            > MAX_SMOKE_DIAGNOSTICS_BYTES
            and payload.get("provider_calls")
        ):
            payload["provider_calls"].pop(0)
            payload["diagnostics_truncated"] = True
        if (
            len(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
            > MAX_SMOKE_DIAGNOSTICS_BYTES
        ):
            payload.pop("tool_calls", None)
            payload["diagnostics_truncated"] = True
        return payload


def _safe_smoke_diagnostics(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    integer_keys = {
        "planner_requests_attempted",
        "planner_repair_attempts",
        "tool_calls_attempted",
        "tool_calls_succeeded",
        "tool_calls_failed",
        "evidence_count",
        "citation_count",
        "grounded_candidate_citation_count",
    }
    boolean_keys = {
        "planner_response_received",
        "planner_json_valid",
        "final_answer_attempted",
        "final_answer_response_received",
        "citation_validation_completed",
        "relation_validation_completed",
        "post_generation_validation_completed",
        "tool_budget_exhausted",
        "citation_validation_passed",
        "relation_validation_passed",
        "post_generation_validation_passed",
        "grounded_answer_candidate_received",
        "grounded_answer_accepted",
        "grounded_reference_validation_completed",
        "grounded_reference_validation_passed",
        "diagnostics_truncated",
    }
    for key in integer_keys:
        item = value.get(key)
        if isinstance(item, int) and not isinstance(item, bool) and 0 <= item <= 1_000_000:
            result[key] = item
    for key in boolean_keys:
        item = value.get(key)
        if isinstance(item, bool):
            result[key] = item
    for key, allowed in (
        ("fallback_reason_code", FALLBACK_REASON_CODES),
        ("agent_mode", _AGENT_MODES),
        ("answer_mode", _ANSWER_MODES),
        ("agent_status", _AGENT_STATUSES),
        ("final_answer_failure_reason_code", FINAL_ANSWER_FAILURE_REASON_CODES),
    ):
        item = value.get(key)
        if isinstance(item, str) and item in allowed:
            result[key] = item
    provider_calls = value.get("provider_calls")
    if isinstance(provider_calls, list):
        safe_calls = []
        for item in provider_calls[:MAX_PROVIDER_CALLS]:
            safe = _safe_provider_metadata(item)
            purpose = item.get("purpose") if isinstance(item, dict) else None
            request_started = item.get("request_started") if isinstance(item, dict) else None
            if purpose in PROVIDER_PURPOSES:
                safe["purpose"] = purpose
            if isinstance(request_started, bool):
                safe["request_started"] = request_started
            if safe:
                safe_calls.append(safe)
        if safe_calls:
            result["provider_calls"] = safe_calls
    tool_calls = value.get("tool_calls")
    if isinstance(tool_calls, list):
        safe_tools = []
        for item in tool_calls[:MAX_TOOL_ENTRIES]:
            if not isinstance(item, dict) or not _safe_tool_name(item.get("name")):
                continue
            safe_item: dict[str, Any] = {"name": item["name"]}
            for key in ("attempted", "succeeded", "failed"):
                count = item.get(key)
                if isinstance(count, int) and not isinstance(count, bool) and 0 <= count <= 1_000:
                    safe_item[key] = count
            safe_tools.append(safe_item)
        if safe_tools:
            result["tool_calls"] = safe_tools
    return result


def _safe_provider_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    purpose = value.get("purpose")
    if purpose in PROVIDER_PURPOSES:
        result["purpose"] = purpose
    for key in (
        "request_started",
        "response_received",
        "response_json_valid",
        "choices_present",
        "content_present",
        "content_empty",
        "reasoning_content_present",
    ):
        item = value.get(key)
        if isinstance(item, bool):
            result[key] = item
    status = value.get("http_status")
    if isinstance(status, int) and not isinstance(status, bool) and 100 <= status <= 599:
        result["http_status"] = status
    count = value.get("choices_count")
    if isinstance(count, int) and not isinstance(count, bool) and 0 <= count <= 10_000:
        result["choices_count"] = count
    finish_reason = value.get("finish_reason")
    if isinstance(finish_reason, str) and finish_reason in _FINISH_REASONS:
        result["finish_reason"] = finish_reason
    for key in ("content_type", "reasoning_content_type"):
        item = value.get(key)
        if isinstance(item, str) and item in _TYPE_VALUES:
            result[key] = item
    usage = _safe_usage(value.get("usage"))
    if usage:
        result["usage"] = usage
    return result


def _safe_usage(value: Any) -> dict[str, int | float]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int | float] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens"):
        item = value.get(key)
        if (
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and 0 <= item <= 1_000_000_000
        ):
            result[key] = item
    return result


def _safe_tool_name(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value))
