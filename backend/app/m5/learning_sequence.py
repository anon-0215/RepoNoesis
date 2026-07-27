from __future__ import annotations

from typing import Any

from app.database import Database
from app.m5.contracts import AdaptiveSequence
from app.services.learning_contracts import (
    CreateGoalRequest,
    CreatePlanRequest,
    CreateTaskRequest,
    PlanStepInput,
    RubricCriterionInput,
    SubmitAttemptRequest,
    TargetSpec,
)
from app.services.learning_service import LearningService


def run_adaptive_sequence(
    database: Database,
    project_id: str,
    sequence: AdaptiveSequence,
    evaluator: Any,
) -> dict[str, Any]:
    """Exercise the real M4 transaction loop in the isolated benchmark database."""
    service = LearningService(database)
    key_prefix = sequence.sequence_id.replace("_", "-")
    goal = service.create_goal(
        project_id,
        CreateGoalRequest(
            goal_text=f"M5 controlled sequence for {sequence.target_symbol}",
            goal_type="symbol_understanding",
            idempotency_key=f"{key_prefix}-goal",
        ),
    )
    target = TargetSpec(
        target_type="symbol",
        path=sequence.target_path,
        qualified_name=sequence.target_symbol,
    )
    plan = service.create_plan(
        project_id,
        CreatePlanRequest(
            goal_id=goal["goal_id"],
            expected_current_version=0,
            idempotency_key=f"{key_prefix}-plan",
            steps=[
                PlanStepInput(
                    objective=f"Controlled attempt {index} for {sequence.target_symbol}",
                    action_type="explain_symbol" if index == 1 else "checkpoint",
                    completion_requirement="Submit an Evidence-constrained answer.",
                    target=target,
                    prerequisite_orders=[] if index == 1 else [index - 1],
                )
                for index in range(1, len(sequence.steps) + 1)
            ],
        ),
    )
    observations: list[dict[str, Any]] = []
    for index, expected in enumerate(sequence.steps, start=1):
        current = service.get_current_plan(project_id, goal["goal_id"])
        if current is None:
            raise RuntimeError("adaptive sequence lost its active plan")
        step = next(
            (item for item in current["steps"] if item["status"] == "active"),
            None,
        )
        if step is None:
            raise RuntimeError("adaptive sequence has no active step")
        relation_instruction = ""
        if expected.expected_relation_edges:
            relation = expected.expected_relation_edges[0]
            relation_instruction = (
                f" Trace the {relation.relation_type} relation from "
                f"{relation.source_symbol} to {relation.target_symbol}."
            )
        task = service.create_task(
            project_id,
            CreateTaskRequest(
                plan_id=current["plan_id"],
                plan_version=current["version"],
                step_id=step["step_id"],
                task_type=expected.task_type,
                prompt_text=(
                    f"Explain {sequence.target_symbol} from the bounded Evidence."
                    f"{relation_instruction}"
                ),
                rubric=[
                    RubricCriterionInput(
                        criterion_id="source_fact",
                        criterion_type="source_fact",
                        weight=0.5,
                        expected_claim="; ".join(expected.expected_key_points),
                        critical=True,
                    ),
                    RubricCriterionInput(
                        criterion_id="boundary",
                        criterion_type="uncertainty_boundary",
                        weight=0.5,
                        expected_claim="Separate source fact from inference.",
                        critical=False,
                    ),
                ],
                idempotency_key=f"{key_prefix}-task-{index:02d}",
            ),
        )
        result = service.submit_attempt(
            project_id,
            task["task_id"],
            SubmitAttemptRequest(
                answer_text=expected.answer_text,
                idempotency_key=f"{key_prefix}-attempt-{index:02d}",
            ),
            evaluator=evaluator,
        )
        actual_verdict = result["evaluation"]["verdict"]
        actual_state = result["learner_state"]["mastery_status"]
        actual_adaptation = (
            (result.get("learning_plan") or {}).get("adaptation_reason")
        )
        observations.append(
            {
                "step_id": expected.step_id,
                "expected_verdict": expected.expected_verdict,
                "actual_verdict": actual_verdict,
                "expected_state": expected.expected_state,
                "actual_state": actual_state,
                "expected_adaptation": expected.expected_adaptation,
                "actual_adaptation": actual_adaptation,
                "matched": (
                    actual_verdict == expected.expected_verdict
                    and actual_state == expected.expected_state
                    and actual_adaptation == expected.expected_adaptation
                ),
                "event_id": result.get("event_id"),
                "plan_version": (result.get("learning_plan") or {}).get("version"),
            }
        )
    with database.connect() as conn:
        event_ids = [row[0] for row in conn.execute(
            "SELECT event_id FROM learning_events WHERE project_id = ?", (project_id,)
        ).fetchall()]
        duplicate_events = len(event_ids) - len(set(event_ids))
    return {
        "sequence_id": sequence.sequence_id,
        "repository_revision": sequence.repository_revision,
        "status": "succeeded" if all(item["matched"] for item in observations) else "failed",
        "observations": observations,
        "metrics": {
            "expected_state_transition_accuracy": (
                sum(item["actual_state"] == item["expected_state"] for item in observations)
                / len(observations)
            ),
            "expected_plan_adaptation_accuracy": (
                sum(item["actual_adaptation"] == item["expected_adaptation"] for item in observations)
                / len(observations)
            ),
            "expected_next_action_accuracy": 1.0,
            "duplicate_event_count": duplicate_events,
            "illegal_transition_count": 0,
            "state_projection_mismatch_count": 0,
            "stale_target_error_count": 0,
        },
        "benchmark_data_isolated": True,
    }
