from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal


MAX_SMOKE_DIAGNOSTICS_BYTES = 4_096
MAX_PROVIDER_CALLS = 16
MAX_LOGICAL_PROVIDER_CALLS = 1_000_000
MAX_TOOL_ENTRIES = 16
MAX_PROVIDER_ATTEMPT_DETAILS = 8
MAX_PLANNER_ATTEMPTS = 10
PUBLIC_DEADLINE_REASON_CODE = "deadline_exceeded"
INTERNAL_DEADLINE_REASON_CODE = "deadline_exhausted"

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
_VALIDATOR_CITATION_FAILURE_PRIORITY = {
    "citation_path_mismatch": 0,
    "citation_line_range_mismatch": 1,
    "citation_evidence_binding_failed": 2,
}
AGENT_FAILURE_REASON_CODES = frozenset(
    {
        "planner_budget_exhausted",
        "planner_invalid",
        "planner_repair_failed",
        PUBLIC_DEADLINE_REASON_CODE,
        "final_answer_not_attempted",
        "tool_timeout",
    }
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
        "failed",
    }
)
_TOOL_STATUSES = frozenset({"succeeded", "failed", "rejected", "timed_out", "cancelled"})
_TOOL_PHASES = frozenset({"seed", "planner"})
_TOOL_REASON_CODES = frozenset(
    {
        "cancelled",
        "deadline_exceeded",
        "final_answer_not_attempted",
        "invalid_parameters",
        "repeat_call",
        "tool_error",
        "tool_failed",
        "tool_timeout",
        "unknown_tool",
    }
)
_PROVIDER_ATTEMPT_OUTCOMES = frozenset(
    {"success", "http_error", "timeout", "network_error", "invalid_response", "deadline"}
)
_PLANNER_VALIDATION_STAGES = frozenset({"adapter", "parser", "schema", "semantic"})
_PLANNER_VALIDATION_CODES = frozenset(
    {
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
    }
)
_FINAL_ANSWER_PROTOCOL_CODES = frozenset(
    {
        "final_answer_invalid_json",
        "final_answer_schema_invalid",
        "citation_alias_missing",
        "citation_alias_unknown",
        "citation_alias_invalid_type",
        "citation_alias_limit_exceeded",
        "canonical_render_failed",
        "citation_format_invalid",
        "citation_binding_failed",
    }
)


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
        self._provider_logical_calls = 0
        self._provider_http_attempt_count = 0
        self._provider_attempt_outcomes: list[str] = []
        self._provider_attempt_durations_ms: list[int] = []
        self._provider_attempt_timeouts_ms: list[int] = []
        self._backoff_total_ms = 0
        self._agent: dict[str, Any] = {}
        self._registered_tools: set[str] = set()
        self._tool_calls: dict[str, dict[str, int | str]] = {}
        self._tool_executions: list[dict[str, Any]] = []
        self._planner_attempts: list[dict[str, Any]] = []
        self._final_answer_protocol_failure: dict[str, Any] | None = None
        self._truncated = False

    def begin_request(self, *, deadline_budget_ms: int, remaining_ms: int) -> None:
        self._agent["deadline_budget_ms"] = _bounded_duration(deadline_budget_ms)
        self._agent["deadline_remaining_ms"] = _bounded_duration(remaining_ms)

    def record_route_elapsed(self, elapsed_ms: int) -> None:
        self._agent["route_elapsed_ms"] = _bounded_duration(elapsed_ms)

    def record_agent_elapsed(self, elapsed_ms: int) -> None:
        self._agent["agent_elapsed_ms"] = _bounded_duration(elapsed_ms)

    def record_deadline_state(self, *, remaining_ms: int, overrun_ms: int) -> None:
        self._agent["deadline_remaining_ms"] = _bounded_duration(remaining_ms)
        self._agent["deadline_overrun_ms"] = _bounded_duration(overrun_ms)

    def record_request_deadline_reached(self, reached: bool) -> None:
        self._agent["request_deadline_reached"] = bool(
            self._agent.get("request_deadline_reached", False) or reached
        )

    def record_stage_duration(self, stage: str, duration_ms: int) -> None:
        key = {
            "planner": "planner_duration_ms",
            "tool": "tool_duration_ms",
            "finalization": "finalization_duration_ms",
        }.get(stage)
        if key is None:
            raise ValueError("unsupported timed stage")
        self._agent[key] = _bounded_duration(
            int(self._agent.get(key, 0)) + _bounded_duration(duration_ms)
        )

    def record_tool_deadline_overrun(
        self, occurred: bool, *, overrun_ms: int = 0
    ) -> None:
        self._agent["tool_deadline_overrun"] = bool(
            self._agent.get("tool_deadline_overrun", False) or occurred
        )
        self._agent["tool_deadline_overrun_ms"] = max(
            int(self._agent.get("tool_deadline_overrun_ms", 0)),
            _bounded_duration(overrun_ms),
        )

    def record_embedding_stage(self, stage: str) -> None:
        key = {
            "model_load": "model_load_attempted",
            "query_encode": "query_encode_attempted",
        }.get(stage)
        if key is None:
            raise ValueError("unsupported embedding stage")
        self._agent[key] = True

    def enter_stage(self, stage: SmokeStage) -> None:
        if stage not in SMOKE_STAGES:
            raise ValueError("unsupported smoke stage")
        self.stage = stage

    def start_provider_call(self, purpose: ProviderPurpose) -> int | None:
        if purpose not in PROVIDER_PURPOSES:
            raise ValueError("unsupported provider diagnostics purpose")
        self._provider_logical_calls = min(
            MAX_LOGICAL_PROVIDER_CALLS, self._provider_logical_calls + 1
        )
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

    def record_provider_attempt(
        self,
        call_id: int | None,
        *,
        outcome: str,
        duration_ms: int,
        timeout_ms: int,
    ) -> None:
        if outcome not in _PROVIDER_ATTEMPT_OUTCOMES:
            raise ValueError("unsupported provider attempt outcome")
        self._provider_http_attempt_count = min(
            MAX_LOGICAL_PROVIDER_CALLS, self._provider_http_attempt_count + 1
        )
        if len(self._provider_attempt_outcomes) < MAX_PROVIDER_ATTEMPT_DETAILS:
            self._provider_attempt_outcomes.append(outcome)
            self._provider_attempt_durations_ms.append(_bounded_duration(duration_ms))
            self._provider_attempt_timeouts_ms.append(_bounded_duration(timeout_ms))
        else:
            self._truncated = True
        if call_id is not None and 0 <= call_id < len(self._provider_calls):
            self._provider_calls[call_id]["http_attempt_count"] = min(
                MAX_PROVIDER_ATTEMPT_DETAILS,
                int(self._provider_calls[call_id].get("http_attempt_count", 0)) + 1,
            )

    def record_backoff(self, duration_ms: int) -> None:
        self._backoff_total_ms = _bounded_duration(
            self._backoff_total_ms + _bounded_duration(duration_ms)
        )

    def begin_agent(
        self, registered_tools: list[str], *, request_id: str | None = None
    ) -> None:
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
        if isinstance(request_id, str) and re.fullmatch(r"[A-Za-z0-9-]{1,64}", request_id):
            self._agent["request_id"] = request_id

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

    def record_planner_attempt(
        self, value: dict[str, Any], *, duration_ms: int
    ) -> None:
        """Keep one bounded, content-free Planner validation result."""

        safe = _safe_planner_attempt({**value, "duration_ms": duration_ms})
        if not safe:
            self._truncated = True
            return
        if len(self._planner_attempts) >= MAX_PLANNER_ATTEMPTS:
            self._truncated = True
            return
        self._planner_attempts.append(safe)

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

    def record_tool_execution(
        self,
        *,
        phase: str,
        tool_name: str,
        status: str,
        result_count: int,
        evidence_added: int,
        reason_code: str | None,
    ) -> None:
        """Record one fixed-shape, content-free summary of a real tool execution."""

        if len(self._tool_executions) >= MAX_TOOL_ENTRIES:
            self._truncated = True
            return
        if (
            phase not in _TOOL_PHASES
            or tool_name not in self._registered_tools
            or status not in _TOOL_STATUSES
        ):
            raise ValueError("unsupported tool execution diagnostics")
        self._tool_executions.append(
            {
                "phase": phase,
                "tool_name": tool_name,
                "status": status,
                "result_count": _bounded_count(result_count),
                "evidence_added": _bounded_count(evidence_added),
                "reason_code": (
                    reason_code if reason_code in _TOOL_REASON_CODES else None
                ),
            }
        )

    def record_unknown_tool_rejection(self) -> None:
        """Record a content-free sentinel for an unregistered Planner tool."""

        if len(self._tool_executions) >= MAX_TOOL_ENTRIES:
            self._truncated = True
            return
        self._tool_executions.append(
            {
                "phase": "planner",
                "tool_name": "unknown_tool",
                "status": "rejected",
                "result_count": 0,
                "evidence_added": 0,
                "reason_code": "unknown_tool",
            }
        )

    def clear_agent_failure(self, reason: str) -> None:
        if self._agent.get("agent_failure_reason_code") == reason:
            self._agent.pop("agent_failure_reason_code", None)

    def record_tool_budget_exhausted(self) -> None:
        self._agent["tool_budget_exhausted"] = True

    def record_agent_progress(
        self, *, steps_used: int, tool_calls_used: int, elapsed_ms: int
    ) -> None:
        for key, value in (
            ("steps_used", steps_used),
            ("tool_calls_used", tool_calls_used),
            ("elapsed_ms", elapsed_ms),
        ):
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                self._agent[key] = min(value, 86_400_000 if key == "elapsed_ms" else 1_000_000)

    def record_evidence_count(self, count: int) -> None:
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            self._agent["evidence_count"] = max(
                int(self._agent.get("evidence_count", 0)), min(count, 1_000_000)
            )

    def record_agent_failure(self, reason: str) -> None:
        if reason not in AGENT_FAILURE_REASON_CODES:
            raise ValueError("unsupported agent failure reason")
        self._agent["agent_failure_reason_code"] = reason

    def record_final_answer_attempt(self) -> None:
        self.enter_stage("final_answer")
        self._agent["final_answer_attempted"] = True

    def record_final_answer_protocol_failure(self, value: dict[str, Any]) -> None:
        """Record one content-free structured-answer rejection."""

        safe = _safe_final_answer_protocol_failure(value)
        if not safe:
            self._truncated = True
            return
        self._final_answer_protocol_failure = safe

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
        existing_citation = self._agent.get("citation_failure_reason_code")
        if reason.startswith("citation_"):
            selected = reason
            if isinstance(existing_citation, str):
                if reason in _VALIDATOR_CITATION_FAILURE_PRIORITY:
                    if existing_citation in _VALIDATOR_CITATION_FAILURE_PRIORITY:
                        selected = min(
                            (existing_citation, reason),
                            key=_VALIDATOR_CITATION_FAILURE_PRIORITY.__getitem__,
                        )
                else:
                    selected = existing_citation
            self._agent["citation_failure_reason_code"] = selected
            self._agent["final_answer_failure_reason_code"] = selected
        elif not isinstance(existing_citation, str):
            self._agent["final_answer_failure_reason_code"] = reason
        if reason == "relation_validation_failed":
            self._agent["relation_failure_reason_code"] = reason

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
            self.record_evidence_count(len(evidence))
        if isinstance(citations, list):
            self._agent["citation_count"] = len(citations)

    def snapshot(self) -> dict[str, Any]:
        payload: dict[str, Any] = dict(self._agent)
        if self._provider_logical_calls:
            payload["provider_logical_calls"] = self._provider_logical_calls
        if self._provider_http_attempt_count:
            payload["provider_http_attempt_count"] = self._provider_http_attempt_count
            payload["provider_attempt_outcomes"] = list(self._provider_attempt_outcomes)
            payload["provider_attempt_durations_ms"] = list(
                self._provider_attempt_durations_ms
            )
            payload["provider_attempt_timeouts_ms"] = list(
                self._provider_attempt_timeouts_ms
            )
        if self._backoff_total_ms:
            payload["backoff_total_ms"] = self._backoff_total_ms
        if self._provider_calls:
            payload["provider_calls"] = [dict(item) for item in self._provider_calls]
        if self._tool_calls:
            payload["tool_calls"] = [
                dict(self._tool_calls[name]) for name in sorted(self._tool_calls)
            ][:MAX_TOOL_ENTRIES]
        if self._tool_executions:
            payload["tool_executions"] = [dict(item) for item in self._tool_executions]
        if self._planner_attempts:
            payload["planner_attempts"] = [dict(item) for item in self._planner_attempts]
        if self._final_answer_protocol_failure is not None:
            payload["final_answer_protocol_failure"] = dict(
                self._final_answer_protocol_failure
            )
        if self._truncated:
            payload["diagnostics_truncated"] = True
        payload = _safe_smoke_diagnostics(payload)
        while (
            len(json.dumps(payload, sort_keys=True).encode("utf-8"))
            > MAX_SMOKE_DIAGNOSTICS_BYTES
            and payload.get("provider_calls")
        ):
            payload["provider_calls"].pop(0)
            payload["diagnostics_truncated"] = True
        if (
            len(json.dumps(payload, sort_keys=True).encode("utf-8"))
            > MAX_SMOKE_DIAGNOSTICS_BYTES
        ):
            payload.pop("tool_calls", None)
            payload["diagnostics_truncated"] = True
        while (
            len(json.dumps(payload, sort_keys=True).encode("utf-8"))
            > MAX_SMOKE_DIAGNOSTICS_BYTES
            and payload.get("tool_executions")
        ):
            payload["tool_executions"].pop(0)
            payload["diagnostics_truncated"] = True
        while (
            len(json.dumps(payload, sort_keys=True).encode("utf-8"))
            > MAX_SMOKE_DIAGNOSTICS_BYTES
            and payload.get("provider_attempt_outcomes")
        ):
            payload["provider_attempt_outcomes"].pop(0)
            if payload.get("provider_attempt_durations_ms"):
                payload["provider_attempt_durations_ms"].pop(0)
            if payload.get("provider_attempt_timeouts_ms"):
                payload["provider_attempt_timeouts_ms"].pop(0)
            payload["diagnostics_truncated"] = True
        while (
            len(json.dumps(payload, sort_keys=True).encode("utf-8"))
            > MAX_SMOKE_DIAGNOSTICS_BYTES
            and len(payload.get("planner_attempts", [])) > 2
        ):
            payload["planner_attempts"].pop(0)
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
        "steps_used",
        "tool_calls_used",
        "elapsed_ms",
        "provider_logical_calls",
        "provider_http_attempt_count",
        "route_elapsed_ms",
        "agent_elapsed_ms",
        "deadline_budget_ms",
        "deadline_remaining_ms",
        "deadline_overrun_ms",
        "tool_deadline_overrun_ms",
        "planner_duration_ms",
        "tool_duration_ms",
        "finalization_duration_ms",
        "backoff_total_ms",
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
        "tool_deadline_overrun",
        "request_deadline_reached",
        "model_load_attempted",
        "query_encode_attempted",
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
        ("citation_failure_reason_code", FINAL_ANSWER_FAILURE_REASON_CODES),
        ("relation_failure_reason_code", FINAL_ANSWER_FAILURE_REASON_CODES),
        ("agent_failure_reason_code", AGENT_FAILURE_REASON_CODES),
    ):
        item = value.get(key)
        if isinstance(item, str) and item in allowed:
            result[key] = item
    request_id = value.get("request_id")
    if isinstance(request_id, str) and re.fullmatch(r"[A-Za-z0-9-]{1,64}", request_id):
        result["request_id"] = request_id
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
    outcomes = value.get("provider_attempt_outcomes")
    durations = value.get("provider_attempt_durations_ms")
    timeouts = value.get("provider_attempt_timeouts_ms")
    if isinstance(outcomes, list):
        safe_outcomes = [
            item
            for item in outcomes[:MAX_PROVIDER_ATTEMPT_DETAILS]
            if isinstance(item, str) and item in _PROVIDER_ATTEMPT_OUTCOMES
        ]
        if safe_outcomes:
            result["provider_attempt_outcomes"] = safe_outcomes
    for key, items in (
        ("provider_attempt_durations_ms", durations),
        ("provider_attempt_timeouts_ms", timeouts),
    ):
        if isinstance(items, list):
            safe_values = [
                _bounded_duration(item)
                for item in items[:MAX_PROVIDER_ATTEMPT_DETAILS]
                if isinstance(item, int) and not isinstance(item, bool) and item >= 0
            ]
            if safe_values:
                result[key] = safe_values
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
    tool_executions = value.get("tool_executions")
    if isinstance(tool_executions, list):
        safe_executions = []
        for item in tool_executions[:MAX_TOOL_ENTRIES]:
            if not isinstance(item, dict):
                continue
            phase = item.get("phase")
            tool_name = item.get("tool_name")
            status = item.get("status")
            reason_code = item.get("reason_code")
            if (
                phase not in _TOOL_PHASES
                or not _safe_tool_name(tool_name)
                or status not in _TOOL_STATUSES
            ):
                continue
            safe_executions.append(
                {
                    "phase": phase,
                    "tool_name": tool_name,
                    "status": status,
                    "result_count": _bounded_count(item.get("result_count")),
                    "evidence_added": _bounded_count(item.get("evidence_added")),
                    "reason_code": (
                        reason_code if reason_code in _TOOL_REASON_CODES else None
                    ),
                }
            )
        if safe_executions:
            result["tool_executions"] = safe_executions
    planner_attempts = value.get("planner_attempts")
    if isinstance(planner_attempts, list):
        safe_attempts = [
            safe
            for item in planner_attempts[:MAX_PLANNER_ATTEMPTS]
            if (safe := _safe_planner_attempt(item))
        ]
        if safe_attempts:
            result["planner_attempts"] = safe_attempts
    protocol_failure = _safe_final_answer_protocol_failure(
        value.get("final_answer_protocol_failure")
    )
    if protocol_failure:
        result["final_answer_protocol_failure"] = protocol_failure
    return result


def _bounded_duration(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        return 0
    return min(86_400_000, max(0, value))


def _bounded_count(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        return 0
    return min(1_000_000, max(0, value))


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
        "finish_reason_present",
        "reasoning_content_present",
        "markdown_fence_detected",
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
    output_chars = value.get("output_chars")
    if isinstance(output_chars, int) and not isinstance(output_chars, bool):
        result["output_chars"] = _bounded_count(output_chars)
    output_sha256 = value.get("output_sha256")
    if isinstance(output_sha256, str) and re.fullmatch(
        r"[0-9a-f]{64}", output_sha256
    ):
        result["output_sha256"] = output_sha256
    usage = _safe_usage(value.get("usage"))
    if usage:
        result["usage"] = usage
    return result


def _safe_planner_attempt(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    stage = value.get("stage")
    code = value.get("stable_code")
    if stage not in _PLANNER_VALIDATION_STAGES or code not in _PLANNER_VALIDATION_CODES:
        return {}
    result: dict[str, Any] = {
        "stage": stage,
        "stable_code": code,
        "field_path": [],
        "output_chars": _bounded_count(value.get("output_chars")),
        "duration_ms": _bounded_duration(value.get("duration_ms")),
        "finish_reason_present": value.get("finish_reason_present") is True,
        "content_present": value.get("content_present") is True,
        "reasoning_content_present": value.get("reasoning_content_present") is True,
        "markdown_fence_detected": value.get("markdown_fence_detected") is True,
        "repair_attempt": value.get("repair_attempt") is True,
    }
    path = value.get("field_path")
    if isinstance(path, list):
        result["field_path"] = [
            item
            for item in path[:16]
            if (
                isinstance(item, int)
                and not isinstance(item, bool)
                and item >= 0
            )
            or (
                isinstance(item, str)
                and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", item)
            )
        ]
    finish_reason = value.get("finish_reason_value")
    if finish_reason in _FINISH_REASONS:
        result["finish_reason_value"] = finish_reason
    output_sha256 = value.get("output_sha256")
    if isinstance(output_sha256, str) and re.fullmatch(
        r"[0-9a-f]{64}", output_sha256
    ):
        result["output_sha256"] = output_sha256
    return result


def _safe_final_answer_protocol_failure(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    code = value.get("stable_code")
    if code not in _FINAL_ANSWER_PROTOCOL_CODES:
        return {}
    result: dict[str, Any] = {
        "stage": "final_answer",
        "stable_code": code,
        "field_path": [],
        "output_chars": _bounded_count(value.get("output_chars")),
        "part_count": _bounded_count(value.get("part_count")),
        "alias_count": _bounded_count(value.get("alias_count")),
        "markdown_fence_detected": value.get("markdown_fence_detected") is True,
    }
    path = value.get("field_path")
    if isinstance(path, list):
        result["field_path"] = [
            item
            for item in path[:16]
            if (
                isinstance(item, int)
                and not isinstance(item, bool)
                and item >= 0
            )
            or (
                isinstance(item, str)
                and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", item)
            )
        ]
    output_sha256 = value.get("output_sha256")
    if isinstance(output_sha256, str) and re.fullmatch(
        r"[0-9a-f]{64}", output_sha256
    ):
        result["output_sha256"] = output_sha256
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


def normalize_public_failure_reason(value: Any) -> Any:
    if value == INTERNAL_DEADLINE_REASON_CODE:
        return PUBLIC_DEADLINE_REASON_CODE
    return value
