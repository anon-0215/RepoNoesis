from __future__ import annotations

from typing import Any

from app.services.learning_contracts import (
    CreateGoalRequest,
    CreatePlanRequest,
    CreateTaskRequest,
    PlanStepInput,
    RubricCriterionInput,
    TargetSpec,
)
from app.services.learning_service import LearningService


class FakeEvaluator:
    def __init__(self, verdict: str = "pass", *, forged: dict[str, Any] | None = None) -> None:
        self.verdict = verdict
        self.forged = forged

    def evaluate(self, task: dict[str, Any], answer_text: str) -> dict[str, Any]:
        if self.forged is not None:
            return self.forged
        evidence_id = task["evidence"][0]["evidence_id"]
        passed = self.verdict == "pass"
        if self.verdict == "partial":
            passed = False
        return {
            "evaluator_schema_version": 1,
            "verdict": self.verdict,
            "criterion_results": [] if self.verdict == "ungradable" else [
                {
                    "criterion_id": "source_fact",
                    "passed": passed,
                    "used_evidence_ids": [evidence_id] if passed else [],
                    "feedback": "bounded feedback",
                }
            ],
            "supported_feedback": ["source-backed"] if passed else [],
            "missing_concepts": [] if passed else ["responsibility"],
            "misconceptions": [],
            "used_evidence_ids": [evidence_id] if passed else [],
            "warnings": [],
        }


def create_goal_plan_task(
    service: LearningService,
    project_id: str,
    *,
    goal_key: str = "goal-key-0001",
    plan_key: str = "plan-key-0001",
    task_key: str = "task-key-0001",
    task_type: str = "explain_symbol",
    qualified_name: str = "target",
    path: str = "app.py",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    goal = service.create_goal(
        project_id,
        CreateGoalRequest(
            goal_text="理解 target 的职责",
            goal_type="symbol_understanding",
            idempotency_key=goal_key,
        ),
    )
    plan = service.create_plan(
        project_id,
        CreatePlanRequest(
            goal_id=goal["goal_id"],
            expected_current_version=0,
            idempotency_key=plan_key,
            steps=[
                PlanStepInput(
                    objective="阅读并解释 target",
                    action_type="explain_symbol",
                    completion_requirement="通过一项 Evidence 约束任务",
                    target=TargetSpec(
                        target_type="symbol",
                        path=path,
                        qualified_name=qualified_name,
                    ),
                ),
                PlanStepInput(
                    objective="完成复核",
                    action_type="checkpoint",
                    completion_requirement="完成不同任务",
                    target=TargetSpec(
                        target_type="symbol",
                        path=path,
                        qualified_name=qualified_name,
                    ),
                    prerequisite_orders=[1],
                ),
            ],
        ),
    )
    task = service.create_task(
        project_id,
        CreateTaskRequest(
            plan_id=plan["plan_id"],
            plan_version=plan["version"],
            step_id=plan["steps"][0]["step_id"],
            task_type=task_type,
            prompt_text="解释 target 的职责，并引用 Evidence。",
            rubric=[
                RubricCriterionInput(
                    criterion_id="source_fact",
                    criterion_type="source_fact",
                    weight=1.0,
                    expected_claim="说明 target 返回稳定值",
                    critical=True,
                )
            ],
            idempotency_key=task_key,
        ),
    )
    return goal, plan, task
