from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


LEARNING_API_SCHEMA_VERSION = 1
STATE_UPDATE_RULE_VERSION = 1
LOCAL_LEARNER_ID = "learner-local-single-user-v1"

MAX_LEARNING_STATE_ITEMS = 16
DEFAULT_LEARNING_STATE_ITEMS = 8
MAX_RECENT_LEARNING_EVENTS = 8
MAX_PLAN_STEPS_IN_CONTEXT = 12
MAX_LEARNING_CONTEXT_BYTES = 16_384
MAX_RUBRIC_CRITERIA = 8
MAX_USER_ANSWER_CHARS = 12_000
MAX_PLAN_STEPS = 20
MAX_PLAN_PREREQUISITE_EDGES = 40


GoalType = Literal[
    "repository_onboarding",
    "module_understanding",
    "symbol_understanding",
    "change_impact",
    "architecture_understanding",
    "custom_bounded",
]
TargetType = Literal["repository", "module", "file", "symbol", "bounded_concept"]
StepAction = Literal[
    "read_evidence",
    "explain_symbol",
    "trace_static_relation",
    "answer_question",
    "predict_static_behavior",
    "analyze_change_impact",
    "review",
    "checkpoint",
]
TaskType = Literal[
    "explain_symbol",
    "trace_static_relation",
    "locate_symbol",
    "explain_static_relationship",
    "analyze_change_impact",
    "separate_fact_inference_unknown",
]
CriterionType = Literal[
    "source_fact",
    "static_relation",
    "location",
    "change_impact",
    "uncertainty_boundary",
]


class StrictLearningModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateGoalRequest(StrictLearningModel):
    goal_text: str = Field(min_length=1, max_length=2_000)
    goal_type: GoalType
    idempotency_key: str = Field(min_length=8, max_length=120)


class GoalStatusRequest(StrictLearningModel):
    status: Literal["active", "completed", "cancelled"]


class TargetSpec(StrictLearningModel):
    target_type: TargetType
    path: str = Field(default="", max_length=500)
    qualified_name: str = Field(default="", max_length=500)
    concept: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def validate_identity(self) -> "TargetSpec":
        if self.target_type in {"file", "module", "symbol"} and not self.path:
            raise ValueError("source targets require a repository-relative path")
        if self.target_type == "symbol" and not self.qualified_name:
            raise ValueError("symbol targets require qualified_name")
        if self.target_type == "bounded_concept" and not self.concept:
            raise ValueError("bounded concept targets require concept")
        if self.target_type == "repository" and (
            self.path or self.qualified_name or self.concept
        ):
            raise ValueError("repository target does not accept path, symbol, or concept")
        return self


class PlanStepInput(StrictLearningModel):
    objective: str = Field(min_length=1, max_length=1_000)
    action_type: StepAction
    completion_requirement: str = Field(min_length=1, max_length=1_000)
    target: TargetSpec
    prerequisite_orders: list[int] = Field(default_factory=list, max_length=20)


class CreatePlanRequest(StrictLearningModel):
    goal_id: str = Field(min_length=2, max_length=80)
    expected_current_version: int = Field(default=0, ge=0)
    steps: list[PlanStepInput] = Field(min_length=1, max_length=MAX_PLAN_STEPS)
    idempotency_key: str = Field(min_length=8, max_length=120)


class RubricCriterionInput(StrictLearningModel):
    criterion_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,40}$")
    criterion_type: CriterionType
    weight: float = Field(gt=0, le=1)
    expected_claim: str = Field(min_length=1, max_length=1_000)
    critical: bool = False
    supporting_evidence_ids: list[str] = Field(default_factory=list, max_length=16)


class CreateTaskRequest(StrictLearningModel):
    plan_id: str = Field(min_length=2, max_length=80)
    plan_version: int = Field(ge=1)
    step_id: str = Field(min_length=2, max_length=80)
    task_type: TaskType
    prompt_text: str = Field(min_length=1, max_length=2_000)
    rubric: list[RubricCriterionInput] = Field(
        min_length=1, max_length=MAX_RUBRIC_CRITERIA
    )
    idempotency_key: str = Field(min_length=8, max_length=120)

    @model_validator(mode="after")
    def unique_criteria(self) -> "CreateTaskRequest":
        ids = [item.criterion_id for item in self.rubric]
        if len(ids) != len(set(ids)):
            raise ValueError("rubric criterion IDs must be unique")
        if sum(item.weight for item in self.rubric) > 1.000001:
            raise ValueError("rubric weights must sum to at most 1")
        return self


class SubmitAttemptRequest(StrictLearningModel):
    answer_text: str = Field(min_length=1, max_length=MAX_USER_ANSWER_CHARS)
    idempotency_key: str = Field(min_length=8, max_length=120)


class SelfReportRequest(StrictLearningModel):
    target: TargetSpec
    report_text: str = Field(min_length=1, max_length=1_000)
    idempotency_key: str = Field(min_length=8, max_length=120)


class EvaluationCorrectionRequest(StrictLearningModel):
    corrected_verdict: Literal["pass", "partial", "fail", "ungradable"]
    reason: str = Field(min_length=1, max_length=500)
    idempotency_key: str = Field(min_length=8, max_length=120)


class CriterionEvaluation(StrictLearningModel):
    criterion_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,40}$")
    passed: bool
    used_evidence_ids: list[str] = Field(default_factory=list, max_length=16)
    feedback: str = Field(default="", max_length=500)


class EvaluationOutput(StrictLearningModel):
    evaluator_schema_version: Literal[1] = 1
    verdict: Literal["pass", "partial", "fail", "ungradable"]
    criterion_results: list[CriterionEvaluation] = Field(
        default_factory=list, max_length=MAX_RUBRIC_CRITERIA
    )
    supported_feedback: list[str] = Field(default_factory=list, max_length=8)
    missing_concepts: list[str] = Field(default_factory=list, max_length=8)
    misconceptions: list[str] = Field(default_factory=list, max_length=8)
    used_evidence_ids: list[str] = Field(default_factory=list, max_length=16)
    warnings: list[str] = Field(default_factory=list, max_length=8)


class GetLearningContextInput(StrictLearningModel):
    pass
