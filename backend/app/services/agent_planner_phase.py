"""Optional Planner enhancement phase for one request-owned agent state.

This module owns the complete Planner decision and tool-execution control flow.
It deliberately receives the request's existing state and operations: it never
creates an EvidenceStore, ToolContext, recorder, or secondary agent state.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Literal

from app.services.agent_contracts import AgentStep, PlannerDecision, ToolCall, ToolObservation
from app.services.agent_tools import ToolRegistry, ToolSpec
from app.services.llm_client import LLMClient, ProviderError
from app.services.smoke_diagnostics import SmokeDiagnosticsRecorder


PlannerPhaseStatus = Literal[
    "completed",
    "fallback_required",
    "planner_failure_response_required",
]


@dataclass(frozen=True)
class PlannerPhaseResult:
    """Finite result of optional enhancement, before finalization begins."""

    status: PlannerPhaseStatus
    decision_error: str = ""
    deadline_recovery_authorized: bool = False


@dataclass(frozen=True)
class PlannerPhaseOperations:
    """Core-owned deterministic operations used by the enhancement phase.

    Keeping these as injected operations avoids a reverse import from this
    phase owner into ``agent_core`` while preserving the established production
    validation, tool execution, Candidate/Evidence, and diagnostics contracts.
    """

    get_valid_decision: Callable[[Any, Any, ToolRegistry, SmokeDiagnosticsRecorder | None], tuple[PlannerDecision | None, str]]
    pre_step_stop_status: Callable[[Any], str | None]
    planner_deadline_is_recoverable: Callable[..., bool]
    resolve_tool_spec: Callable[[ToolRegistry, str], ToolSpec | None]
    merge_server_bound_constraints: Callable[[ToolSpec | None, dict[str, Any], Any], dict[str, Any]]
    apply_request_top_k_limit: Callable[..., dict[str, Any]]
    execute_agent_tool: Callable[..., tuple[ToolCall, ToolObservation]]


def run_planner_enhancement(
    *,
    planner: Any,
    state: Any,
    registry: ToolRegistry,
    llm: LLMClient | None,
    question: str,
    server_constraints: Any,
    evidence_count: int,
    allow_planner_failure_fallback: bool,
    diagnostics_recorder: SmokeDiagnosticsRecorder | None,
    started: float,
    operations: PlannerPhaseOperations,
) -> PlannerPhaseResult:
    """Run the bounded optional Planner phase against the one request state.

    An ``answer`` decision is enhancement-only: it terminates this phase and
    leaves final answer generation to the caller's existing finalization stage.
    Provider errors remain hard failures and intentionally propagate.
    """

    planner_deadline_recovery_authorized = False
    for ordinal in range(1, state.limits.max_agent_steps + 1):
        if state.failure_reason is not None or state.completion_status == "cancelled":
            break
        stop_status = operations.pre_step_stop_status(state)
        if stop_status:
            state.completion_status = stop_status
            if stop_status == "budget_exhausted":
                now = time.monotonic()
                stop_reason = (
                    "deadline_exceeded"
                    if state.budget.request_expired(now)
                    else "final_answer_not_attempted"
                    if state.budget.work_expired(now)
                    else "planner_budget_exhausted"
                )
                if stop_reason == "deadline_exceeded" or not state.context.evidence_store.all(
                    state.request_id
                ):
                    state.failure_reason = stop_reason
                else:
                    state.enhancement_termination_reason = stop_reason
                    state.warnings.append(
                        "Planner enhancement ended; retained canonical Evidence for finalization."
                    )
                    if diagnostics_recorder is not None:
                        diagnostics_recorder.record_planner_enhancement_termination(
                            "planner_budget_exhausted"
                        )
            if diagnostics_recorder is not None and state.failure_reason is not None:
                diagnostics_recorder.record_agent_failure(state.failure_reason)
            break
        planner_started = time.monotonic()
        try:
            decision, decision_error = operations.get_valid_decision(
                planner, state, registry, diagnostics_recorder
            )
        except ProviderError as exc:
            now = time.monotonic()
            if operations.planner_deadline_is_recoverable(
                exc=exc,
                state=state,
                llm=llm,
                now=now,
            ):
                state.completion_status = "budget_exhausted"
                state.failure_reason = "planner_budget_exhausted"
                planner_deadline_recovery_authorized = True
                state.warnings.append(
                    "Planner reached its work cutoff after valid Evidence was collected."
                )
                if diagnostics_recorder is not None:
                    diagnostics_recorder.record_agent_failure("planner_budget_exhausted")
                break
            if diagnostics_recorder is not None:
                diagnostics_recorder.record_evidence_count(
                    len(state.context.evidence_store.all(state.request_id))
                )
                diagnostics_recorder.record_agent_progress(
                    steps_used=len(state.steps),
                    tool_calls_used=state.tool_call_count,
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                )
                diagnostics_recorder.record_agent_result(
                    {"agent_mode": "bounded", "agent_status": "failed"}
                )
            raise
        finally:
            if diagnostics_recorder is not None:
                diagnostics_recorder.record_stage_duration(
                    "planner", int((time.monotonic() - planner_started) * 1000)
                )
        if decision is None:
            if "budget exhausted" in decision_error:
                state.completion_status = "budget_exhausted"
                if state.context.evidence_store.all(state.request_id):
                    state.enhancement_termination_reason = "planner_budget_exhausted"
                else:
                    state.failure_reason = "planner_budget_exhausted"
                state.warnings.append(decision_error)
                if diagnostics_recorder is not None:
                    if state.failure_reason is not None:
                        diagnostics_recorder.record_agent_failure("planner_budget_exhausted")
                    else:
                        diagnostics_recorder.record_planner_enhancement_termination(
                            "planner_budget_exhausted"
                        )
                break
            if state.context.evidence_store.all(state.request_id):
                state.completion_status = "completed"
                state.enhancement_termination_reason = "planner_repair_failed"
                state.warnings.append(
                    "Planner decision failed validation; retained canonical Evidence for finalization."
                )
                if diagnostics_recorder is not None:
                    diagnostics_recorder.record_planner_enhancement_termination(
                        "planner_repair_failed"
                    )
                break
            if not allow_planner_failure_fallback:
                if diagnostics_recorder is not None:
                    diagnostics_recorder.record_agent_failure("planner_repair_failed")
                return PlannerPhaseResult(
                    status="planner_failure_response_required",
                    decision_error=decision_error,
                )
            if diagnostics_recorder is not None:
                diagnostics_recorder.record_fallback("planner_validation_failed")
            return PlannerPhaseResult(
                status="fallback_required", decision_error=decision_error
            )

        if state.context.cancellation.cancelled:
            state.completion_status = "cancelled"
            break
        now = time.monotonic()
        if state.budget.request_expired(now) or state.budget.work_expired(now):
            state.completion_status = "budget_exhausted"
            stop_reason = (
                "deadline_exceeded"
                if state.budget.request_expired(now)
                else "planner_budget_exhausted"
            )
            if stop_reason == "deadline_exceeded" or not state.context.evidence_store.all(
                state.request_id
            ):
                state.failure_reason = stop_reason
            else:
                state.enhancement_termination_reason = stop_reason
            if diagnostics_recorder is not None:
                if state.failure_reason is not None:
                    diagnostics_recorder.record_agent_failure(state.failure_reason)
                else:
                    diagnostics_recorder.record_planner_enhancement_termination(
                        "planner_budget_exhausted"
                    )
            state.warnings.append(
                "The request or work cutoff was reached after planning; no tool was started."
            )
            break

        step_id = f"S{ordinal}"
        if decision.status != "continue":
            state.completion_status = (
                "completed" if decision.status == "answer" else "insufficient_evidence"
            )
            state.steps.append(
                AgentStep(
                    step_id=step_id,
                    user_goal=question,
                    action=decision.status,
                    tool_calls=[],
                    observations=[],
                    decision_summary=decision.decision_summary,
                    completion_status=state.completion_status,
                    remaining_budget=state.remaining_budget(),
                )
            )
            if diagnostics_recorder is not None:
                diagnostics_recorder.record_planner_enhancement_termination(
                    "planner_answer" if decision.status == "answer" else "no_progress"
                )
            break

        if state.remaining_budget()["tool_calls"] <= 0:
            state.completion_status = "tool_budget_exhausted"
            state.warnings.append(
                "Agent tool budget was exhausted; continuing to bounded finalization."
            )
            if diagnostics_recorder is not None:
                diagnostics_recorder.record_tool_budget_exhausted()
                diagnostics_recorder.record_planner_enhancement_termination(
                    "tool_budget_exhausted"
                )
            break

        action = decision.action or ""
        arguments = dict(decision.arguments)
        if action == "search_code":
            if not isinstance(arguments.get("query"), str) or not arguments["query"].strip():
                arguments["query"] = question
        tool_spec = operations.resolve_tool_spec(registry, action)
        public_action = action if tool_spec is not None else "unknown_tool"
        arguments = operations.merge_server_bound_constraints(
            tool_spec, arguments, server_constraints
        )
        arguments = operations.apply_request_top_k_limit(
            tool_spec, arguments, evidence_count=evidence_count
        )
        tool_budget_ms = state.budget.work_remaining_ms(time.monotonic())
        if tool_budget_ms <= 0:
            state.completion_status = "budget_exhausted"
            state.warnings.append(
                "The next tool was not started because the final-answer reserve would be consumed."
            )
            if diagnostics_recorder is not None:
                diagnostics_recorder.record_agent_failure("final_answer_not_attempted")
            state.failure_reason = "final_answer_not_attempted"
            break
        call, observation = operations.execute_agent_tool(
            state=state,
            registry=registry,
            action=public_action,
            tool_spec=tool_spec,
            arguments=arguments,
            step_id=step_id,
            phase="planner",
            diagnostics_recorder=diagnostics_recorder,
        )
        state.steps.append(
            AgentStep(
                step_id=step_id,
                user_goal=question,
                action=call.tool_name,
                tool_calls=[call],
                observations=[observation],
                decision_summary=(
                    "Planner requested an unregistered tool; the call was rejected."
                    if call.tool_name == "unknown_tool"
                    else decision.decision_summary
                ),
                completion_status="running",
                remaining_budget=state.remaining_budget(),
            )
        )
        now = time.monotonic()
        observation_code = (observation.error or {}).get("code")
        if state.budget.request_expired(now) or observation_code == "deadline_exceeded":
            state.completion_status = "budget_exhausted"
            state.failure_reason = "deadline_exceeded"
            if diagnostics_recorder is not None:
                diagnostics_recorder.record_agent_failure("deadline_exceeded")
            break
        if observation_code in {"tool_timeout", "final_answer_not_attempted"}:
            state.completion_status = "failed"
            state.failure_reason = observation_code
            if diagnostics_recorder is not None:
                diagnostics_recorder.record_agent_failure(observation_code)
            break
        if observation.status == "cancelled":
            state.completion_status = "cancelled"
            break
        if observation.status == "rejected" and diagnostics_recorder is not None:
            rejection_code = (observation.error or {}).get("code")
            if rejection_code in {"repeat_call", "unknown_tool", "invalid_parameters"}:
                diagnostics_recorder.record_planner_enhancement_termination(rejection_code)
        if state.remaining_budget()["tool_calls"] <= 0:
            state.completion_status = "tool_budget_exhausted"
            state.warnings.append(
                "Agent tool budget was exhausted; continuing to bounded finalization."
            )
            if diagnostics_recorder is not None:
                diagnostics_recorder.record_tool_budget_exhausted()
                diagnostics_recorder.record_planner_enhancement_termination(
                    "tool_budget_exhausted"
                )
            break
        if state.no_progress_count >= state.limits.max_no_progress_steps:
            state.completion_status = (
                "completed"
                if state.context.evidence_store.all(state.request_id)
                else "insufficient_evidence"
            )
            state.warnings.append("Agent stopped after consecutive no-progress steps.")
            if diagnostics_recorder is not None:
                diagnostics_recorder.record_planner_enhancement_termination("no_progress")
            break
    else:
        state.completion_status = "budget_exhausted"

    if state.completion_status == "running":
        state.completion_status = "budget_exhausted"
        state.failure_reason = "planner_budget_exhausted"
    return PlannerPhaseResult(
        status="completed",
        deadline_recovery_authorized=planner_deadline_recovery_authorized,
    )
