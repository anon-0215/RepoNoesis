from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.m5 import BENCHMARK_SCHEMA_VERSION, METRIC_SCHEMA_VERSION


EXPERIMENT_MODES = (
    "fixed_lexical_rag",
    "fixed_dense_rag",
    "m1_hybrid_rag",
    "m2_bounded_agent",
    "m3_relation_agent",
    "m4_profiled_agent",
    "m4_adaptive_sequence",
)
SCENARIO_CATEGORIES = (
    "locate",
    "explain",
    "relation",
    "impact",
    "unanswerable",
)
AnnotationProvenance = Literal["agent_assisted_developer_curation", "user_confirmed"]
AnnotationStatus = Literal["agent_curated_pending_human_review", "human_reviewed"]
AnnotationReviewMethod = Literal["codex_conversation"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnalysisConfiguration(StrictModel):
    include_globs: list[str] = Field(default_factory=lambda: ["**/*.py"])
    maximum_file_bytes: int = Field(default=250_000, ge=1, le=1_000_000)
    maximum_files: int = Field(default=1_500, ge=1, le=5_000)


class RepositorySpec(StrictModel):
    repo_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$")
    display_name: str = Field(min_length=1, max_length=120)
    source_url: str = Field(min_length=1, max_length=500)
    license: str = Field(min_length=1, max_length=120)
    language: Literal["python"]
    exact_commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    default_branch: str = Field(min_length=1, max_length=200)
    acquisition_method: Literal["shallow_clone", "local_existing"]
    checkout_name: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    content_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    analysis_configuration: AnalysisConfiguration
    excluded_paths: list[str] = Field(default_factory=list, max_length=100)
    annotation_status: Literal["agent_curated_pending_human_review"]


class SourceSpan(StrictModel):
    path: str = Field(min_length=1, max_length=500)
    qualified_symbol: str = Field(min_length=1, max_length=500)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def valid_range(self) -> "SourceSpan":
        if self.end_line < self.start_line:
            raise ValueError("source span end_line precedes start_line")
        return self


class RelationIdentity(StrictModel):
    relation_type: Literal["imports", "calls", "references", "defines"]
    source_path: str = Field(min_length=1, max_length=500)
    source_symbol: str = Field(min_length=1, max_length=500)
    target_path: str | None = Field(default=None, max_length=500)
    target_symbol: str = Field(min_length=1, max_length=500)


class AllowedEvidenceScope(StrictModel):
    paths: list[str] = Field(default_factory=list, max_length=50)
    repository_only: bool = True


class Scenario(StrictModel):
    scenario_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,95}$")
    dataset_version: str = Field(min_length=1, max_length=80)
    repo_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$")
    repository_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    language: Literal["python"]
    question: str = Field(min_length=1, max_length=2_000)
    category: Literal["locate", "explain", "relation", "impact", "unanswerable"]
    difficulty: Literal["easy", "medium", "hard"]
    expected_target_type: Literal["file", "symbol", "span", "relation", "none"]
    expected_files: list[str] = Field(default_factory=list, max_length=30)
    expected_symbols: list[str] = Field(default_factory=list, max_length=30)
    expected_source_spans: list[SourceSpan] = Field(default_factory=list, max_length=30)
    expected_content_hashes: list[str] = Field(default_factory=list, max_length=30)
    expected_relation_edges: list[RelationIdentity] = Field(default_factory=list, max_length=30)
    expected_key_points: list[str] = Field(default_factory=list, max_length=20)
    unanswerable: bool
    allowed_evidence_scope: AllowedEvidenceScope
    maximum_steps: int = Field(ge=1, le=8)
    maximum_tool_calls: int = Field(ge=1, le=12)
    annotation_provenance: AnnotationProvenance
    annotation_status: AnnotationStatus
    annotation_reviewed_at: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    annotation_review_method: AnnotationReviewMethod | None = None
    annotation_note: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def consistent_answerability(self) -> "Scenario":
        if self.unanswerable:
            if self.expected_target_type != "none":
                raise ValueError("unanswerable scenario must use expected_target_type=none")
            if any(
                (
                    self.expected_files,
                    self.expected_symbols,
                    self.expected_source_spans,
                    self.expected_content_hashes,
                    self.expected_relation_edges,
                )
            ):
                raise ValueError("unanswerable scenario cannot declare source gold")
        elif not self.expected_files and not self.expected_symbols:
            raise ValueError("answerable scenario requires file or symbol gold")
        _validate_annotation_review(
            self.annotation_provenance,
            self.annotation_status,
            self.annotation_reviewed_at,
            self.annotation_review_method,
        )
        return self


class SequenceStep(StrictModel):
    step_id: str = Field(min_length=1, max_length=80)
    task_type: Literal[
        "explain_symbol", "trace_static_relation", "locate_symbol",
        "explain_static_relationship", "analyze_change_impact",
        "separate_fact_inference_unknown",
    ]
    answer_text: str = Field(min_length=1, max_length=12_000)
    expected_key_points: list[str] = Field(min_length=1, max_length=20)
    expected_source_spans: list[SourceSpan] = Field(min_length=1, max_length=10)
    expected_relation_edges: list[RelationIdentity] = Field(default_factory=list, max_length=10)
    expected_verdict: Literal["fail", "partial", "pass"]
    expected_state: Literal[
        "unseen", "introduced", "practicing", "demonstrated", "mastered", "needs_review"
    ]
    expected_adaptation: str = Field(min_length=1, max_length=120)

    @model_validator(mode="after")
    def consistent_relation_gold(self) -> "SequenceStep":
        if self.task_type == "trace_static_relation" and not self.expected_relation_edges:
            raise ValueError("trace_static_relation step requires relation gold")
        if self.task_type != "trace_static_relation" and self.expected_relation_edges:
            raise ValueError("non-relation sequence step cannot declare relation gold")
        return self


class AdaptiveSequence(StrictModel):
    sequence_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,95}$")
    dataset_version: str
    repo_id: str
    repository_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    target_path: str = Field(min_length=1, max_length=500)
    target_symbol: str = Field(min_length=1, max_length=500)
    steps: list[SequenceStep] = Field(min_length=1, max_length=12)
    annotation_provenance: AnnotationProvenance
    annotation_status: AnnotationStatus
    annotation_reviewed_at: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    annotation_review_method: AnnotationReviewMethod | None = None
    annotation_note: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def consistent_review(self) -> "AdaptiveSequence":
        _validate_annotation_review(
            self.annotation_provenance,
            self.annotation_status,
            self.annotation_reviewed_at,
            self.annotation_review_method,
        )
        return self


class DatasetManifest(StrictModel):
    benchmark_schema_version: Literal[BENCHMARK_SCHEMA_VERSION] = BENCHMARK_SCHEMA_VERSION
    metric_schema_version: Literal[METRIC_SCHEMA_VERSION] = METRIC_SCHEMA_VERSION
    dataset_version: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=200)
    repositories_file: Literal["repositories.json"] = "repositories.json"
    scenarios_file: Literal["scenarios.jsonl"] = "scenarios.jsonl"
    sequences_file: Literal["sequences.jsonl"] = "sequences.jsonl"
    annotation_provenance: AnnotationProvenance
    annotation_status: AnnotationStatus
    annotation_reviewed_at: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    annotation_review_method: AnnotationReviewMethod | None = None
    minimum_scenarios: int = Field(default=36, ge=36)

    @model_validator(mode="after")
    def consistent_review(self) -> "DatasetManifest":
        _validate_annotation_review(
            self.annotation_provenance,
            self.annotation_status,
            self.annotation_reviewed_at,
            self.annotation_review_method,
        )
        return self


def _validate_annotation_review(
    provenance: AnnotationProvenance,
    status: AnnotationStatus,
    reviewed_at: str | None,
    method: AnnotationReviewMethod | None,
) -> None:
    if status == "human_reviewed":
        if provenance != "user_confirmed" or reviewed_at is None or method is None:
            raise ValueError("human-reviewed annotation requires user-confirmed provenance, date, and method")
    elif provenance != "agent_assisted_developer_curation" or reviewed_at is not None or method is not None:
        raise ValueError("pending annotation cannot declare completed human-review provenance")


@dataclass(frozen=True)
class ProviderIdentity:
    provider: str
    model: str
    model_revision: str
    capability: str
    is_real: bool
    endpoint_identity: str = "local"
    pricing_identity: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProviderUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None
    cost_status: Literal["known_zero", "calculated", "unknown"] = "unknown"
    cost_currency: str | None = None
    cost_unknown_reason: str | None = "pricing_not_configured"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["estimated_cost_usd"] = (
            "unknown" if self.estimated_cost_usd is None else self.estimated_cost_usd
        )
        return value


@dataclass(frozen=True)
class ProviderResult:
    status: Literal["succeeded", "failed", "cancelled", "timed_out"]
    content: str | None
    identity: ProviderIdentity
    usage: ProviderUsage
    latency_ms: int
    actual_model: str | None = None
    error_type: str | None = None
    attempt_count: int = 1
    seed_supported: bool | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def safe_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "identity": self.identity.to_dict(),
            "usage": self.usage.to_dict(),
            "latency_ms": self.latency_ms,
            "actual_model": self.actual_model,
            "error_type": self.error_type,
            "attempt_count": self.attempt_count,
            "seed_supported": self.seed_supported,
            "raw_metadata": dict(self.raw_metadata),
        }


class BenchmarkConfig(StrictModel):
    dataset_directory: str
    repository_root: str
    artifacts_directory: str
    modes: list[Literal[*EXPERIMENT_MODES]] = Field(min_length=1)
    scenario_ids: list[str] = Field(default_factory=list)
    repo_ids: list[str] = Field(default_factory=list)
    cells: list[str] = Field(default_factory=list)
    batch_index: int = Field(default=0, ge=0)
    batch_count: int = Field(default=1, ge=1, le=216)
    run_purpose: Literal["smoke", "pilot", "full"] = "smoke"
    top_k: int = Field(default=5, ge=1, le=8)
    maximum_steps: int = Field(default=5, ge=1, le=8)
    maximum_tool_calls: int = Field(default=8, ge=1, le=12)
    maximum_output_tokens: int = Field(default=1_600, ge=128, le=1_600)
    timeout_seconds: float = Field(default=60.0, ge=1.0, le=60.0)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    random_seed: int = Field(default=20260726, ge=0, le=2_147_483_647)
    prompt_version: str = "m5-answer-v1"
    evaluator_version: str = "m5-evaluator-v1"
    metric_version: str = "m5-metrics-v1"
    parallelism: int = Field(default=1, ge=1, le=4)
    maximum_answer_requests: int = Field(default=24, ge=1, le=1_000)
    maximum_evaluator_requests: int = Field(default=24, ge=1, le=250)
    maximum_answer_input_tokens: int = Field(default=1_000_000, ge=1, le=2_000_000)
    maximum_answer_output_tokens: int = Field(default=250_000, ge=1, le=500_000)
    maximum_evaluator_input_tokens: int = Field(default=100_000, ge=1, le=500_000)
    maximum_evaluator_output_tokens: int = Field(default=25_000, ge=1, le=100_000)
    maximum_answer_cost_usd: float | None = Field(default=None, gt=0.0, le=10_000.0)
    maximum_evaluator_cost_usd: float | None = Field(default=None, gt=0.0, le=10_000.0)
    maximum_wall_clock_seconds: float = Field(default=3_600.0, ge=1.0, le=86_400.0)
    maximum_provider_attempts: int = Field(default=2, ge=1, le=3)
    dry_run: bool = False

    @model_validator(mode="after")
    def consistent_execution_plan(self) -> "BenchmarkConfig":
        if self.batch_index >= self.batch_count:
            raise ValueError("batch_index must be smaller than batch_count")
        seen: set[str] = set()
        for cell in self.cells:
            parts = cell.rsplit("::", 1)
            if len(parts) != 2 or not parts[0] or parts[1] not in EXPERIMENT_MODES or parts[1] == "m4_adaptive_sequence":
                raise ValueError(f"invalid scenario-mode cell: {cell}")
            if cell in seen:
                raise ValueError(f"duplicate scenario-mode cell: {cell}")
            if parts[1] not in self.modes:
                raise ValueError(f"cell mode is absent from configured modes: {cell}")
            seen.add(cell)
        if self.cells and (self.scenario_ids or self.repo_ids):
            raise ValueError("explicit cells cannot be combined with scenario or repository filters")
        return self
