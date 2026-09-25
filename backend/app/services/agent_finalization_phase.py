"""Single owner of bounded Evidence finalization for one agent request."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Literal

from app.services.llm_client import LLMClient, ProviderError
from app.services.qa_agent import INSUFFICIENT_ANSWER
from app.services.smoke_diagnostics import SmokeDiagnosticsRecorder


FinalizationPhaseStatus = Literal["completed", "fixed_failure"]
FinalizationMode = Literal["bounded", "deterministic_fallback"]


@dataclass(frozen=True)
class FinalizationPhaseResult:
    """The only finalization projection returned to the agent coordinator."""
    status: FinalizationPhaseStatus
    response: dict[str, Any]


@dataclass(frozen=True)
class FinalizationPhaseOperations:
    budget_failure: Callable[..., dict[str, Any]]
    assert_evidence_capacity: Callable[[Any, list[Any]], None]
    validated_relation_context: Callable[..., tuple[list[Any], list[Any], list[str], str | None]]
    bounded_evidence_context: Callable[[list[Any], int], tuple[list[Any], bool]]
    relation_answer_context: Callable[[Any, list[Any]], dict[str, Any] | None]
    finalization_rejection_response: Callable[..., dict[str, Any]]
    answer_from_evidence: Callable[..., dict[str, Any]]
    attach_agent_fields: Callable[..., dict[str, Any]]
    attach_relation_fields: Callable[..., dict[str, Any]]
    attach_learning_fields: Callable[..., dict[str, Any]]


def run_finalization_phase(
    *,
    state: Any,
    question: str,
    llm: LLMClient | None,
    database: Any,
    started: float,
    planner_deadline_recovery_authorized: bool,
    diagnostics_recorder: SmokeDiagnosticsRecorder | None,
    operations: FinalizationPhaseOperations,
    mode: FinalizationMode = "bounded",
) -> FinalizationPhaseResult:
    """Run eligibility, validation, bounded generation/repair, and postchecks.

    The received state owns the request's EvidenceStore, ToolContext and
    recorder.  This phase neither persists nor creates alternate state.
    """
    if mode not in {"bounded", "deterministic_fallback"}:
        raise ValueError("unsupported finalization mode")

    deterministic_fallback = mode == "deterministic_fallback"
    evidence_store = state.context.evidence_store
    evidence = evidence_store.all(state.request_id)
    operations.assert_evidence_capacity(state.context, evidence)
    if diagnostics_recorder is not None:
        diagnostics_recorder.record_evidence_count(len(evidence))
    if (not deterministic_fallback and evidence and planner_deadline_recovery_authorized and
            state.failure_reason == "planner_budget_exhausted" and
            not state.budget.request_expired(time.monotonic()) and
            not state.context.cancellation.cancelled and
            not state.final_answer_attempted and state.limits.max_final_answer_tokens > 0):
        recovered_reason = state.failure_reason
        state.failure_reason = None
        state.completion_status = "budget_exhausted"
        state.warnings.append("Planner work ended after valid Evidence was collected; continued to bounded finalization.")
        if diagnostics_recorder is not None and recovered_reason is not None:
            diagnostics_recorder.clear_agent_failure(recovered_reason)
    if (not deterministic_fallback and evidence and
            state.completion_status == "insufficient_evidence"):
        state.completion_status = "completed"
        state.warnings.append("Planner stopped while valid Evidence was available; continued to bounded finalization.")

    def budget_failure(reason: str) -> FinalizationPhaseResult:
        return FinalizationPhaseResult("fixed_failure", operations.budget_failure(
            request_id=state.request_id, mode=mode, started=started,
            limits=state.limits, steps=state.steps, tool_calls=state.tool_call_count,
            planner_tokens=state.planner_token_usage,
            retrieval_mode=state.retrieval_mode,
            learning_context=state.context.learning_context,
            diagnostics_recorder=diagnostics_recorder,
            deadline_at=state.context.deadline_monotonic,
            evidence_count=len(evidence), reason=reason))

    if state.failure_reason is not None:
        return budget_failure(state.failure_reason)
    if time.monotonic() >= state.context.deadline_monotonic:
        return budget_failure("deadline_exceeded")
    if deterministic_fallback and state.budget.work_expired(time.monotonic()):
        return budget_failure("final_answer_not_attempted")
    if state.budget.work_expired(time.monotonic()) and not evidence:
        return budget_failure("final_answer_not_attempted")

    finalization_started = time.monotonic()
    relation_enabled = (
        not deterministic_fallback
        or state.context.relation_mode == "expand_v1"
    )
    if deterministic_fallback:
        evidence, evidence_truncated = operations.bounded_evidence_context(
            evidence, state.limits.max_accumulated_evidence_context_bytes
        )
    else:
        evidence_truncated = False
    valid_chains: list[Any] = []
    if relation_enabled:
        evidence, valid_chains, relation_warnings, citation_failure = (
            operations.validated_relation_context(
                state, evidence, diagnostics_recorder, checkpoint="initial_evidence"
            )
        )
        state.warnings.extend(relation_warnings)
        if citation_failure is not None:
            return FinalizationPhaseResult(
                "fixed_failure",
                operations.finalization_rejection_response(
                    state=state,
                    evidence=evidence,
                    chains=valid_chains,
                    reason=citation_failure,
                    mode=mode,
                    started=started,
                    tool_calls=state.tool_call_count,
                    planner_tokens=state.planner_token_usage,
                    planner_usage_mode=state.planner_usage_mode,
                    diagnostics_recorder=diagnostics_recorder,
                    finalization_started=finalization_started,
                ),
            )
    if not deterministic_fallback:
        evidence, evidence_truncated = operations.bounded_evidence_context(
            evidence, state.limits.max_accumulated_evidence_context_bytes
        )
    if evidence_truncated:
        state.warnings.append("Accumulated Evidence context was truncated at the server byte limit.")
    if state.context.cancellation.cancelled:
        state.completion_status = "cancelled"
    final_llm = (llm if llm and llm.available and state.completion_status not in {"insufficient_evidence", "cancelled"} and state.budget.request_remaining_ms(time.monotonic()) > 0 else None)
    if (diagnostics_recorder is not None and evidence and llm and llm.available and
            state.completion_status not in {"insufficient_evidence", "cancelled"} and
            state.budget.request_remaining_ms(time.monotonic()) <= 0):
        diagnostics_recorder.record_final_answer_failure("deadline_exhausted")
    try:
        state.final_answer_attempted = True
        final = operations.answer_from_evidence(question, evidence, final_llm, database,
            retrieval_mode=state.retrieval_mode, warnings=state.warnings,
            max_answer_tokens=state.limits.max_final_answer_tokens,
            answer_timeout_seconds=max(0.1, state.budget.request_remaining_ms(time.monotonic()) / 1000),
            relation_context=operations.relation_answer_context(state, valid_chains),
            learning_context=state.context.learning_context,
            diagnostics_recorder=diagnostics_recorder,
            request_deadline_at=state.context.deadline_monotonic)
    except ProviderError as exc:
        if exc.code == "deadline_exceeded":
            if diagnostics_recorder is not None:
                diagnostics_recorder.record_final_answer_failure("deadline_exhausted")
                diagnostics_recorder.record_stage_duration("finalization", int((time.monotonic() - finalization_started) * 1000))
            return budget_failure("deadline_exceeded")
        if diagnostics_recorder is not None:
            if isinstance((exc.diagnostics or {}).get("http_status"), int):
                diagnostics_recorder.record_final_answer_response()
            diagnostics_recorder.record_final_answer_failure("response_empty" if exc.code == "provider_empty_content" else "provider_failed")
            diagnostics_recorder.record_agent_result({"agent_mode": "bounded", "agent_status": "final_answer_failed", "evidence": evidence})
            diagnostics_recorder.record_agent_progress(steps_used=len(state.steps), tool_calls_used=state.tool_call_count, elapsed_ms=int((time.monotonic() - started) * 1000))
        raise
    if time.monotonic() >= state.context.deadline_monotonic:
        if diagnostics_recorder is not None:
            diagnostics_recorder.record_stage_duration("finalization", int((time.monotonic() - finalization_started) * 1000))
        return budget_failure("deadline_exceeded")
    if relation_enabled:
        post_evidence, post_chains, post_warnings, post_failure = (
            operations.validated_relation_context(
                state, evidence_store.all(state.request_id), diagnostics_recorder,
                checkpoint="post_answer_evidence"
            )
        )
        state.warnings.extend(post_warnings)
        if post_failure is not None:
            return FinalizationPhaseResult(
                "fixed_failure",
                operations.finalization_rejection_response(
                    state=state,
                    evidence=post_evidence,
                    chains=post_chains,
                    reason=post_failure,
                    mode=mode,
                    started=started,
                    tool_calls=state.tool_call_count,
                    planner_tokens=state.planner_token_usage,
                    planner_usage_mode=state.planner_usage_mode,
                    diagnostics_recorder=diagnostics_recorder,
                    finalization_started=finalization_started,
                ),
            )
        if {item.chain_id for item in post_chains} != {
            item.chain_id for item in valid_chains
        }:
            state.warnings.append(
                "Relation data changed during answer generation; relation-dependent generated text was discarded."
            )
            if diagnostics_recorder is not None:
                diagnostics_recorder.record_grounded_answer_accepted(False)
                diagnostics_recorder.record_final_answer_failure(
                    "relation_validation_failed"
                )
            return FinalizationPhaseResult(
                "fixed_failure",
                operations.finalization_rejection_response(
                    state=state,
                    evidence=post_evidence,
                    chains=post_chains,
                    reason="relation_validation_failed",
                    mode=mode,
                    started=started,
                    tool_calls=state.tool_call_count,
                    planner_tokens=state.planner_token_usage,
                    planner_usage_mode=state.planner_usage_mode,
                    diagnostics_recorder=diagnostics_recorder,
                    finalization_started=finalization_started,
                ),
            )
        valid_chains = post_chains
    if state.completion_status == "cancelled":
        pass
    elif deterministic_fallback:
        if state.context.cancellation.cancelled:
            state.completion_status = "cancelled"
        elif (
            time.monotonic() >= state.context.deadline_monotonic
            or state.tool_call_count >= state.limits.max_tool_calls
        ):
            state.completion_status = "budget_exhausted"
        else:
            state.completion_status = "degraded"
    elif state.completion_status == "insufficient_evidence":
        final["answer"] = INSUFFICIENT_ANSWER
        final["citations"] = []
        final["evidence"] = []
        final["grounding_status"] = "insufficient_evidence"
    elif not final["evidence"] and not (state.completion_status == "budget_exhausted" and state.remaining_budget()["time_ms"] <= 0):
        state.completion_status = "insufficient_evidence"
    elif final.get("answer_mode") == "llm_grounded":
        state.completion_status = "completed"
    elif final_llm is not None:
        state.completion_status = "final_answer_failed"
    response = operations.attach_agent_fields(final, request_id=state.request_id, mode=mode, status=state.completion_status, steps=state.steps, started=started, tool_calls=state.tool_call_count, planner_tokens=state.planner_token_usage, planner_usage_mode=state.planner_usage_mode, limits=state.limits)
    if relation_enabled:
        response = operations.attach_relation_fields(response, state, valid_chains)
    response = operations.attach_learning_fields(response, state.context.learning_context)
    operations.assert_evidence_capacity(state.context, response["evidence"])
    if diagnostics_recorder is not None:
        if not deterministic_fallback:
            diagnostics_recorder.record_final_answer_repair_result(succeeded=(state.completion_status == "completed" and final.get("answer_mode") == "llm_grounded"))
        diagnostics_recorder.record_stage_duration("finalization", int((time.monotonic() - finalization_started) * 1000))
        diagnostics_recorder.record_agent_elapsed(int((time.monotonic() - started) * 1000))
        diagnostics_recorder.record_deadline_state(remaining_ms=max(0, int((state.context.deadline_monotonic - time.monotonic()) * 1000)), overrun_ms=max(0, int((time.monotonic() - state.context.deadline_monotonic) * 1000)))
        diagnostics_recorder.record_agent_result(response)
    return FinalizationPhaseResult("completed", response)
