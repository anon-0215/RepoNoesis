from __future__ import annotations

import json
import re
from typing import Any

from app.services.smoke_diagnostics import normalize_public_failure_reason


MAX_ASK_DIAGNOSTICS_BYTES = 4_096
MAX_COUNTER = 1_000_000
MAX_ELAPSED_MS = 86_400_000

FAILURE_REASON_CODES = frozenset(
    {
        "planner_invalid",
        "planner_repair_failed",
        "planner_budget_exhausted",
        "deadline_exceeded",
        "tool_timeout",
        "evidence_insufficient",
        "final_answer_not_attempted",
        "citation_missing",
        "citation_format_invalid",
        "citation_unknown",
        "citation_location_missing",
        "citation_path_mismatch",
        "citation_line_range_mismatch",
        "citation_evidence_binding_failed",
        "relation_validation_failed",
        "response_contract_invalid",
        "unsupported_provider",
        "provider_not_configured",
        "provider_authentication_failed",
        "provider_rate_limited",
        "provider_unavailable",
        "provider_output_truncated",
        "provider_empty_content",
        "provider_invalid_response",
        "provider_request_rejected",
        "provider_error",
        "persistence_failed",
    }
)
PROVIDER_FAILURE_CODES = frozenset(
    {
        "unsupported_provider",
        "provider_not_configured",
        "provider_authentication_failed",
        "provider_rate_limited",
        "provider_unavailable",
        "provider_output_truncated",
        "provider_empty_content",
        "provider_invalid_response",
        "provider_request_rejected",
        "provider_error",
    }
)
CITATION_FAILURE_REASON_CODES = frozenset(
    code for code in FAILURE_REASON_CODES if code.startswith("citation_")
)
RELATION_FAILURE_REASON_CODES = frozenset({"relation_validation_failed"})
FAILURE_STAGES = frozenset(
    {
        "planner",
        "budget",
        "deadline",
        "retrieval",
        "final_answer",
        "citation_validation",
        "relation_validation",
        "response",
        "provider",
        "tool",
        "persistence",
    }
)
AGENT_MODES = frozenset({"bounded", "deterministic_fallback", "unknown"})
AGENT_STATUSES = frozenset(
    {
        "completed",
        "degraded",
        "insufficient_evidence",
        "tool_budget_exhausted",
        "final_answer_failed",
        "budget_exhausted",
        "cancelled",
        "failed",
        "unknown",
    }
)
ANSWER_MODES = frozenset({"llm_grounded", "deterministic", "not_available"})
RETRIEVAL_VERSIONS = frozenset({"v1", "v2"})
HIERARCHY_MODES = frozenset({"off", "normalize_v1"})
RELATION_MODES = frozenset({"off", "expand_v1"})


def build_ask_success_diagnostics(
    *,
    result: dict[str, Any],
    recorder_snapshot: dict[str, Any],
    retrieval_version: str,
    hierarchy_mode: str,
    relation_mode: str,
) -> dict[str, Any]:
    """Project one successful run into the same bounded, content-free boundary."""

    budget = result.get("budget_usage") if isinstance(result.get("budget_usage"), dict) else {}
    planner_requests = _count(recorder_snapshot.get("planner_requests_attempted"))
    planner_repairs = _count(recorder_snapshot.get("planner_repair_attempts"))
    provider_calls = _count(recorder_snapshot.get("provider_logical_calls"))
    final_attempted = recorder_snapshot.get("final_answer_attempted") is True
    if provider_calls == 0:
        provider_calls = planner_requests + int(final_attempted)
    diagnostics = {
        "request_id": _request_id(
            result.get("request_id") or recorder_snapshot.get("request_id")
        ),
        "agent_mode": _enum(
            result.get("agent_mode") or recorder_snapshot.get("agent_mode"),
            AGENT_MODES,
            "unknown",
        ),
        "agent_status": _enum(
            result.get("agent_status") or recorder_snapshot.get("agent_status"),
            AGENT_STATUSES,
            "unknown",
        ),
        "answer_mode": _enum(
            result.get("answer_mode") or recorder_snapshot.get("answer_mode"),
            ANSWER_MODES,
            "not_available",
        ),
        "retrieval_version": _enum(retrieval_version, RETRIEVAL_VERSIONS, "v1"),
        "hierarchy_mode": _enum(hierarchy_mode, HIERARCHY_MODES, "off"),
        "relation_mode": _enum(relation_mode, RELATION_MODES, "off"),
        "steps_used": _count(
            budget.get("steps_used", recorder_snapshot.get("steps_used"))
        ),
        "tool_calls_used": _count(
            budget.get("tool_calls_used", recorder_snapshot.get("tool_calls_used"))
        ),
        "planner_logical_calls": max(0, planner_requests - planner_repairs),
        "planner_repair_calls": planner_repairs,
        "final_answer_attempted": final_attempted,
        "provider_logical_calls": _count(provider_calls),
        "provider_http_attempt_count": _count(
            recorder_snapshot.get("provider_http_attempt_count")
        ),
        "provider_attempt_outcomes": _safe_attempt_outcomes(
            recorder_snapshot.get("provider_attempt_outcomes")
        ),
        "provider_attempt_durations_ms": _safe_durations(
            recorder_snapshot.get("provider_attempt_durations_ms")
        ),
        "provider_attempt_timeouts_ms": _safe_durations(
            recorder_snapshot.get("provider_attempt_timeouts_ms")
        ),
        "backoff_total_ms": _elapsed(recorder_snapshot.get("backoff_total_ms")),
        "evidence_count": max(
            _bounded_list_count(result.get("evidence")),
            _count(recorder_snapshot.get("evidence_count")),
        ),
        "citation_count": _bounded_list_count(result.get("citations")),
        "citation_validation_passed": (
            recorder_snapshot.get("citation_validation_passed") is True
        ),
        "relation_validation_passed": (
            recorder_snapshot.get("relation_validation_passed") is True
        ),
        "post_generation_validation_passed": (
            recorder_snapshot.get("post_generation_validation_passed") is True
        ),
        "elapsed_ms": _elapsed(
            budget.get("elapsed_ms", recorder_snapshot.get("elapsed_ms"))
        ),
        "route_elapsed_ms": _elapsed(recorder_snapshot.get("route_elapsed_ms")),
        "agent_elapsed_ms": _elapsed(recorder_snapshot.get("agent_elapsed_ms")),
        "deadline_budget_ms": _elapsed(
            recorder_snapshot.get("deadline_budget_ms")
        ),
        "deadline_remaining_ms": _elapsed(
            recorder_snapshot.get("deadline_remaining_ms")
        ),
        "deadline_overrun_ms": _elapsed(
            recorder_snapshot.get("deadline_overrun_ms")
        ),
        "request_deadline_reached": (
            recorder_snapshot.get("request_deadline_reached") is True
        ),
        "planner_duration_ms": _elapsed(
            recorder_snapshot.get("planner_duration_ms")
        ),
        "tool_duration_ms": _elapsed(recorder_snapshot.get("tool_duration_ms")),
        "finalization_duration_ms": _elapsed(
            recorder_snapshot.get("finalization_duration_ms")
        ),
        "tool_executions": _safe_tool_executions(
            recorder_snapshot.get("tool_executions")
        ),
        "planner_attempts": _safe_planner_attempts(
            recorder_snapshot.get("planner_attempts")
        ),
        "final_answer_protocol_failure": _safe_final_answer_protocol_failure(
            recorder_snapshot.get("final_answer_protocol_failure")
        ),
    }
    return _bounded_payload(diagnostics)


def format_ask_success_log(diagnostics: dict[str, Any]) -> str:
    safe = _bounded_payload(dict(diagnostics))
    return json.dumps(
        {
            "event": "ask_succeeded",
            "request_id": _request_id(safe.get("request_id")),
            "diagnostics": safe,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def build_ask_failure_detail(
    *,
    result: dict[str, Any] | None,
    recorder_snapshot: dict[str, Any],
    retrieval_version: str,
    hierarchy_mode: str,
    relation_mode: str,
    provider_error: bool = False,
    retryable: bool = False,
    terminal_reason: str | None = None,
) -> dict[str, Any]:
    """Build the only product-safe representation of a failed /ask run."""

    result = result or {}
    reason = _failure_reason(
        result,
        recorder_snapshot,
        provider_error=provider_error,
        terminal_reason=terminal_reason,
    )
    stage = _failure_stage(reason, recorder_snapshot)
    budget = result.get("budget_usage") if isinstance(result.get("budget_usage"), dict) else {}
    provider_calls = recorder_snapshot.get("provider_calls")
    provider_detail_count = (
        sum(
            1
            for item in provider_calls
            if isinstance(item, dict) and item.get("request_started") is True
        )
        if isinstance(provider_calls, list)
        else 0
    )
    planner_requests = _count(recorder_snapshot.get("planner_requests_attempted"))
    planner_repairs = _count(recorder_snapshot.get("planner_repair_attempts"))
    final_attempted = recorder_snapshot.get("final_answer_attempted") is True
    provider_call_count = _count(recorder_snapshot.get("provider_logical_calls"))
    if provider_call_count == 0:
        provider_call_count = provider_detail_count
    if provider_call_count == 0:
        # Fake LLMs do not know about the recorder. These logical call sites are
        # still exact and deliberately exclude provider transport retries.
        provider_call_count = planner_requests + int(final_attempted)

    candidate_citations = recorder_snapshot.get("grounded_candidate_citation_count")
    citation_count = (
        _count(candidate_citations)
        if isinstance(candidate_citations, int) and not isinstance(candidate_citations, bool)
        else _bounded_list_count(result.get("citations"))
    )
    diagnostics = {
        "request_id": _request_id(result.get("request_id") or recorder_snapshot.get("request_id")),
        "agent_mode": _enum(
            result.get("agent_mode") or recorder_snapshot.get("agent_mode"),
            AGENT_MODES,
            "unknown",
        ),
        "agent_status": _enum(
            result.get("agent_status") or recorder_snapshot.get("agent_status"),
            AGENT_STATUSES,
            "unknown",
        ),
        "answer_mode": _enum(
            result.get("answer_mode") or recorder_snapshot.get("answer_mode"),
            ANSWER_MODES,
            "not_available",
        ),
        "failure_stage": stage,
        "failure_reason_code": reason,
        "retrieval_version": _enum(retrieval_version, RETRIEVAL_VERSIONS, "v1"),
        "hierarchy_mode": _enum(hierarchy_mode, HIERARCHY_MODES, "off"),
        "relation_mode": _enum(relation_mode, RELATION_MODES, "off"),
        "steps_used": _count(
            budget.get("steps_used", recorder_snapshot.get("steps_used"))
        ),
        "tool_calls_used": _count(
            budget.get("tool_calls_used", recorder_snapshot.get("tool_calls_used"))
        ),
        "planner_logical_calls": max(0, planner_requests - planner_repairs),
        "planner_repair_calls": planner_repairs,
        "final_answer_attempted": final_attempted,
        "provider_logical_calls": _count(provider_call_count),
        "evidence_count": max(
            _bounded_list_count(result.get("evidence")),
            _count(recorder_snapshot.get("evidence_count")),
        ),
        "citation_count": citation_count,
        "citation_failure_reason_code": _enum_or_none(
            recorder_snapshot.get("citation_failure_reason_code"),
            CITATION_FAILURE_REASON_CODES,
        ),
        "relation_failure_reason_code": _enum_or_none(
            recorder_snapshot.get("relation_failure_reason_code"),
            RELATION_FAILURE_REASON_CODES,
        ),
        "elapsed_ms": _elapsed(
            budget.get("elapsed_ms", recorder_snapshot.get("elapsed_ms"))
        ),
        "route_elapsed_ms": _elapsed(recorder_snapshot.get("route_elapsed_ms")),
        "agent_elapsed_ms": _elapsed(recorder_snapshot.get("agent_elapsed_ms")),
        "deadline_budget_ms": _elapsed(recorder_snapshot.get("deadline_budget_ms")),
        "deadline_remaining_ms": _elapsed(
            recorder_snapshot.get("deadline_remaining_ms")
        ),
        "deadline_overrun_ms": _elapsed(recorder_snapshot.get("deadline_overrun_ms")),
        "request_deadline_reached": recorder_snapshot.get("request_deadline_reached")
        is True,
        "planner_duration_ms": _elapsed(recorder_snapshot.get("planner_duration_ms")),
        "tool_duration_ms": _elapsed(recorder_snapshot.get("tool_duration_ms")),
        "tool_deadline_overrun": recorder_snapshot.get("tool_deadline_overrun") is True,
        "tool_deadline_overrun_ms": _elapsed(
            recorder_snapshot.get("tool_deadline_overrun_ms")
        ),
        "finalization_duration_ms": _elapsed(
            recorder_snapshot.get("finalization_duration_ms")
        ),
        "provider_http_attempt_count": _count(
            recorder_snapshot.get("provider_http_attempt_count")
        ),
        "provider_attempt_outcomes": _safe_attempt_outcomes(
            recorder_snapshot.get("provider_attempt_outcomes")
        ),
        "provider_attempt_durations_ms": _safe_durations(
            recorder_snapshot.get("provider_attempt_durations_ms")
        ),
        "provider_attempt_timeouts_ms": _safe_durations(
            recorder_snapshot.get("provider_attempt_timeouts_ms")
        ),
        "backoff_total_ms": _elapsed(recorder_snapshot.get("backoff_total_ms")),
        "model_load_attempted": recorder_snapshot.get("model_load_attempted") is True,
        "query_encode_attempted": recorder_snapshot.get("query_encode_attempted") is True,
        "tool_executions": _safe_tool_executions(
            recorder_snapshot.get("tool_executions")
        ),
        "planner_attempts": _safe_planner_attempts(
            recorder_snapshot.get("planner_attempts")
        ),
        "final_answer_protocol_failure": _safe_final_answer_protocol_failure(
            recorder_snapshot.get("final_answer_protocol_failure")
        ),
    }
    diagnostics = _bounded_payload(diagnostics)
    message = (
        "The server rejected an invalid answer response before persistence."
        if reason == "response_contract_invalid"
        else "The grounded answer request failed a server-enforced stage. No unvalidated model answer was saved or returned."
    )
    return {
        "code": reason,
        "message": message,
        "retryable": bool(retryable),
        "diagnostics": diagnostics,
    }


def _failure_reason(
    result: dict[str, Any],
    snapshot: dict[str, Any],
    *,
    provider_error: bool,
    terminal_reason: str | None,
) -> str:
    agent_reason = normalize_public_failure_reason(
        snapshot.get("agent_failure_reason_code")
    )
    if (
        terminal_reason == "deadline_exceeded"
        or agent_reason == "deadline_exceeded"
        or snapshot.get("request_deadline_reached") is True
    ):
        return "deadline_exceeded"
    if terminal_reason in FAILURE_REASON_CODES:
        return terminal_reason
    if provider_error:
        return "provider_error"
    if agent_reason in {
        "tool_timeout",
        "planner_budget_exhausted",
        "final_answer_not_attempted",
    }:
        return agent_reason
    citation_reason = snapshot.get("citation_failure_reason_code")
    if citation_reason in CITATION_FAILURE_REASON_CODES:
        return citation_reason
    relation_reason = snapshot.get("relation_failure_reason_code")
    if relation_reason in RELATION_FAILURE_REASON_CODES:
        return relation_reason
    if agent_reason in FAILURE_REASON_CODES:
        return agent_reason
    final_reason = normalize_public_failure_reason(
        snapshot.get("final_answer_failure_reason_code")
    )
    if final_reason in FAILURE_REASON_CODES:
        return final_reason
    fallback = snapshot.get("fallback_reason_code")
    if fallback == "planner_validation_failed":
        return (
            "planner_repair_failed"
            if _count(snapshot.get("planner_repair_attempts")) > 0
            else "planner_invalid"
        )
    status = result.get("agent_status")
    if status in {"budget_exhausted", "tool_budget_exhausted"}:
        return "planner_budget_exhausted"
    if status == "insufficient_evidence" or not result.get("evidence"):
        return "evidence_insufficient"
    if snapshot.get("final_answer_attempted") is not True:
        return "final_answer_not_attempted"
    return "provider_error"


def _failure_stage(reason: str, snapshot: dict[str, Any]) -> str:
    if reason.startswith("planner_"):
        return "planner" if reason != "planner_budget_exhausted" else "budget"
    if reason == "deadline_exceeded":
        return "deadline"
    if reason == "tool_timeout":
        return "tool"
    if reason == "persistence_failed":
        return "persistence"
    if reason == "response_contract_invalid":
        return "response"
    if reason == "evidence_insufficient":
        return "retrieval"
    if reason == "final_answer_not_attempted":
        return "final_answer"
    if reason in CITATION_FAILURE_REASON_CODES:
        return "citation_validation"
    if reason in RELATION_FAILURE_REASON_CODES:
        return "relation_validation"
    if reason in PROVIDER_FAILURE_CODES:
        return "provider"
    provider_calls = snapshot.get("provider_calls")
    if isinstance(provider_calls, list) and provider_calls:
        purpose = provider_calls[-1].get("purpose") if isinstance(provider_calls[-1], dict) else None
        if purpose == "planner":
            return "planner"
        if purpose == "final_answer":
            return "final_answer"
    return "provider"


def _request_id(value: Any) -> str:
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9-]{1,64}", value):
        return value
    return "unknown"


def normalize_provider_failure_code(value: Any) -> str:
    """Return the only public code allowed for a ProviderError."""

    return (
        value
        if isinstance(value, str) and value in PROVIDER_FAILURE_CODES
        else "provider_error"
    )


def _enum(value: Any, allowed: frozenset[str], default: str) -> str:
    return value if isinstance(value, str) and value in allowed else default


def _enum_or_none(value: Any, allowed: frozenset[str]) -> str | None:
    return value if isinstance(value, str) and value in allowed else None


def _count(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return min(MAX_COUNTER, max(0, value))
    return 0


def _elapsed(value: Any) -> int:
    return min(MAX_ELAPSED_MS, _count(value))


def _bounded_list_count(value: Any, fallback: Any = None) -> int:
    if isinstance(value, list):
        return min(MAX_COUNTER, len(value))
    return _count(fallback)


def _safe_attempt_outcomes(value: Any) -> list[str]:
    allowed = {"success", "http_error", "timeout", "network_error", "invalid_response", "deadline"}
    if not isinstance(value, list):
        return []
    return [item for item in value[:8] if isinstance(item, str) and item in allowed]


def _safe_durations(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return [_elapsed(item) for item in value[:8] if isinstance(item, int) and not isinstance(item, bool)]


def _safe_tool_executions(value: Any) -> list[dict[str, Any]]:
    phases = frozenset({"seed", "planner"})
    statuses = frozenset({"succeeded", "failed", "rejected", "timed_out", "cancelled"})
    reasons = frozenset(
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
    if not isinstance(value, list):
        return []
    safe = []
    for item in value[:16]:
        if not isinstance(item, dict):
            continue
        phase = item.get("phase")
        tool_name = item.get("tool_name")
        status = item.get("status")
        reason = item.get("reason_code")
        if (
            phase not in phases
            or status not in statuses
            or not isinstance(tool_name, str)
            or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", tool_name) is None
        ):
            continue
        safe.append(
            {
                "phase": phase,
                "tool_name": tool_name,
                "status": status,
                "result_count": _count(item.get("result_count")),
                "evidence_added": _count(item.get("evidence_added")),
                "reason_code": reason if reason in reasons else None,
            }
        )
    return safe


def _safe_planner_attempts(value: Any) -> list[dict[str, Any]]:
    stages = frozenset({"adapter", "parser", "schema", "semantic"})
    codes = frozenset(
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
    finish_reasons = frozenset(
        {"stop", "length", "tool_calls", "content_filter", "function_call", "other"}
    )
    if not isinstance(value, list):
        return []
    safe_attempts: list[dict[str, Any]] = []
    for item in value[:10]:
        if not isinstance(item, dict):
            continue
        stage = item.get("stage")
        code = item.get("stable_code")
        if stage not in stages or code not in codes:
            continue
        path = item.get("field_path")
        safe_path = []
        if isinstance(path, list):
            safe_path = [
                part
                for part in path[:16]
                if (
                    isinstance(part, int)
                    and not isinstance(part, bool)
                    and part >= 0
                )
                or (
                    isinstance(part, str)
                    and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", part)
                )
            ]
        safe_item: dict[str, Any] = {
            "stage": stage,
            "stable_code": code,
            "field_path": safe_path,
            "output_chars": _count(item.get("output_chars")),
            "duration_ms": _elapsed(item.get("duration_ms")),
            "finish_reason_present": item.get("finish_reason_present") is True,
            "content_present": item.get("content_present") is True,
            "reasoning_content_present": item.get("reasoning_content_present") is True,
            "markdown_fence_detected": item.get("markdown_fence_detected") is True,
            "repair_attempt": item.get("repair_attempt") is True,
        }
        finish_reason = item.get("finish_reason_value")
        if finish_reason in finish_reasons:
            safe_item["finish_reason_value"] = finish_reason
        output_sha256 = item.get("output_sha256")
        if isinstance(output_sha256, str) and re.fullmatch(
            r"[0-9a-f]{64}", output_sha256
        ):
            safe_item["output_sha256"] = output_sha256
        safe_attempts.append(safe_item)
    return safe_attempts


def _safe_final_answer_protocol_failure(value: Any) -> dict[str, Any] | None:
    codes = frozenset(
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
    if not isinstance(value, dict) or value.get("stable_code") not in codes:
        return None
    path = value.get("field_path")
    safe_path: list[str | int] = []
    if isinstance(path, list):
        safe_path = [
            part
            for part in path[:16]
            if (
                isinstance(part, int)
                and not isinstance(part, bool)
                and part >= 0
            )
            or (
                isinstance(part, str)
                and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", part)
            )
        ]
    safe: dict[str, Any] = {
        "stage": "final_answer",
        "stable_code": value["stable_code"],
        "field_path": safe_path,
        "output_chars": _count(value.get("output_chars")),
        "part_count": _count(value.get("part_count")),
        "alias_count": _count(value.get("alias_count")),
        "markdown_fence_detected": value.get("markdown_fence_detected") is True,
    }
    output_sha256 = value.get("output_sha256")
    if isinstance(output_sha256, str) and re.fullmatch(
        r"[0-9a-f]{64}", output_sha256
    ):
        safe["output_sha256"] = output_sha256
    return safe


def _bounded_payload(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)

    def oversized() -> bool:
        return len(
            json.dumps(result, sort_keys=True).encode("utf-8")
        ) > MAX_ASK_DIAGNOSTICS_BYTES

    for key in (
        "tool_executions",
        "provider_attempt_outcomes",
        "provider_attempt_durations_ms",
        "provider_attempt_timeouts_ms",
    ):
        while oversized() and isinstance(result.get(key), list) and result[key]:
            result[key].pop(0)
            result["diagnostics_truncated"] = True
    while (
        oversized()
        and isinstance(result.get("planner_attempts"), list)
        and len(result["planner_attempts"]) > 2
    ):
        result["planner_attempts"].pop(0)
        result["diagnostics_truncated"] = True
    core = {
        "request_id",
        "agent_mode",
        "agent_status",
        "answer_mode",
        "failure_stage",
        "failure_reason_code",
        "retrieval_version",
        "hierarchy_mode",
        "relation_mode",
        "steps_used",
        "tool_calls_used",
        "planner_logical_calls",
        "planner_repair_calls",
        "final_answer_attempted",
        "provider_logical_calls",
        "evidence_count",
        "citation_count",
        "citation_failure_reason_code",
        "relation_failure_reason_code",
        "elapsed_ms",
        "planner_attempts",
        "final_answer_protocol_failure",
        "diagnostics_truncated",
    }
    for key in list(result):
        if not oversized():
            break
        if key not in core:
            result.pop(key, None)
            result["diagnostics_truncated"] = True
    return result


def format_ask_failure_log(detail: dict[str, Any]) -> str:
    """Return a compact, formatter-visible log event from safe diagnostics only."""

    diagnostics = detail.get("diagnostics")
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    code = detail.get("code")
    payload = {
        "code": code
        if isinstance(code, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code)
        else "ask_failed",
        "event": "ask_failed",
        "failure_reason_code": _enum(
            diagnostics.get("failure_reason_code"), FAILURE_REASON_CODES, "provider_error"
        ),
        "failure_stage": _enum(
            diagnostics.get("failure_stage"), FAILURE_STAGES, "provider"
        ),
        "request_id": _request_id(diagnostics.get("request_id")),
        "diagnostics": diagnostics,
    }
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    )


def ask_failure_http_status(detail: dict[str, Any]) -> int:
    return {
        "deadline_exceeded": 504,
        "tool_timeout": 503,
        "planner_budget_exhausted": 503,
        "final_answer_not_attempted": 503,
        "evidence_insufficient": 422,
        "response_contract_invalid": 500,
        "persistence_failed": 500,
    }.get(detail.get("code"), 502)


def ask_result_is_failure(
    result: dict[str, Any],
    recorder_snapshot: dict[str, Any],
    *,
    product_project: bool,
) -> bool:
    """Reject explicit Agent/validation failures before every persistence path."""

    if result.get("agent_status") in {
        "insufficient_evidence",
        "budget_exhausted",
        "tool_budget_exhausted",
        "final_answer_failed",
        "cancelled",
        "failed",
    }:
        return True
    if result.get("grounding_status") in {
        "insufficient_evidence",
        "budget_exhausted",
    }:
        return True
    if (
        recorder_snapshot.get("agent_failure_reason_code")
        or recorder_snapshot.get("citation_failure_reason_code")
        or recorder_snapshot.get("relation_failure_reason_code")
    ):
        return True
    if product_project and (
        result.get("answer_mode") != "llm_grounded"
        or result.get("agent_mode") != "bounded"
    ):
        return True
    return False
