from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


LIVE_DENSE_PROTOCOL_ID = "m5-live-dense-acceptance"
LIVE_DENSE_PROTOCOL_VERSION = 1
DEFAULT_LIVE_DENSE_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "m5"
    / "protocols"
    / "live-dense-acceptance-v1.json"
)

_REQUIRED_FORBIDDEN_COMPONENTS = {
    "bm25",
    "weighted_rrf",
    "hybrid",
    "relation_expansion",
    "planner",
    "agent_loop",
    "llm",
    "evaluator",
}
_REQUIRED_TRACEABILITY_FIELDS = {
    "repository_id",
    "repository_revision",
    "repository_content_identity",
    "path",
    "qualified_symbol",
    "start_line",
    "end_line",
    "content_hash",
    "chunk_identity",
    "full_index_membership",
}
_REQUIRED_REPORTED_METRICS = {
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "mrr_at_10",
    "ndcg_at_10",
    "evidence_precision",
    "evidence_recall",
    "evidence_f1",
    "expected_file_recall",
    "gold_rank",
}


class LiveDenseProtocolError(ValueError):
    pass


class ProtocolStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ProtocolApplicability(ProtocolStrictModel):
    repository_rule: Literal["all_validated_m5_python_repositories"]
    languages: list[Literal["python"]] = Field(min_length=1, max_length=1)

    @model_validator(mode="after")
    def exact_languages(self) -> "ProtocolApplicability":
        if self.languages != ["python"]:
            raise ValueError("live dense protocol languages must be exactly ['python']")
        return self


class StableOrderingProtocol(ProtocolStrictModel):
    identity_builder: Literal["app.m5.dense_artifact.build_chunk_inventory"]
    manifest_field: Literal["chunk_identities"]
    identity_format: Literal["chunk-sha256:<64-lowercase-hex>"]
    encoding: Literal["utf-8-bytes"]
    direction: Literal["ascending"]
    uniqueness_scope: Literal["repository-revision-content-identity"]
    persistent_identity_contract: Literal["existing-code-chunk-contract"]


class StageExpectation(ProtocolStrictModel):
    generated_formula: str = Field(min_length=1)
    cached_formula: str = Field(min_length=1)
    document_encode_calls_formula: str = Field(min_length=1)
    document_encode_batches_formula: str = Field(min_length=1)
    document_encode_items_formula: str = Field(min_length=1)


class PartitionProtocol(ProtocolStrictModel):
    a_count_formula: Literal["(N + 1) // 2"]
    b_count_formula: Literal["N - A"]
    a_slice: Literal["sorted_chunks[0:A]"]
    b_slice: Literal["sorted_chunks[A:N]"]
    full_formula: Literal["A union B"]
    c_target: Literal["FULL"]
    stage_a: StageExpectation
    stage_b: StageExpectation
    stage_c: StageExpectation

    @model_validator(mode="after")
    def exact_stage_formulas(self) -> "PartitionProtocol":
        expected = {
            "stage_a": ("A", "0", "implementation-recorded", "implementation-recorded", "A"),
            "stage_b": ("B", "A", "implementation-recorded", "implementation-recorded", "B"),
            "stage_c": ("0", "N", "0", "0", "0"),
        }
        for name, values in expected.items():
            stage = getattr(self, name)
            observed = (
                stage.generated_formula,
                stage.cached_formula,
                stage.document_encode_calls_formula,
                stage.document_encode_batches_formula,
                stage.document_encode_items_formula,
            )
            if observed != values:
                raise ValueError(f"{name} formulas do not match the frozen A/B/C contract")
        return self


class RetrievalProtocol(ProtocolStrictModel):
    mode: Literal["dense"]
    top_k: Literal[3]
    answerable_query_encode_count: Literal[1]
    unanswerable_query_behavior: Literal["skip"]
    forbidden_components: list[
        Literal[
            "bm25",
            "weighted_rrf",
            "hybrid",
            "relation_expansion",
            "planner",
            "agent_loop",
            "llm",
            "evaluator",
        ]
    ] = Field(min_length=8, max_length=8)

    @model_validator(mode="after")
    def exact_forbidden_components(self) -> "RetrievalProtocol":
        if len(set(self.forbidden_components)) != len(self.forbidden_components):
            raise ValueError("forbidden_components contains duplicates")
        if set(self.forbidden_components) != _REQUIRED_FORBIDDEN_COMPONENTS:
            raise ValueError("forbidden_components does not freeze the dense-only boundary")
        return self


class GoldAcceptanceProtocol(ProtocolStrictModel):
    scenario_gold_policy: Literal["any-complete-valid-gold"]
    required_traceability_fields: list[
        Literal[
            "repository_id",
            "repository_revision",
            "repository_content_identity",
            "path",
            "qualified_symbol",
            "start_line",
            "end_line",
            "content_hash",
            "chunk_identity",
            "full_index_membership",
        ]
    ] = Field(min_length=10, max_length=10)
    per_scenario_pass_condition: Literal["any-complete-valid-gold-in-top-3"]
    path_match: Literal["exact-posix-path"]
    symbol_span_hash_match: Literal["exact-source-span-identity"]

    @model_validator(mode="after")
    def exact_traceability(self) -> "GoldAcceptanceProtocol":
        if len(set(self.required_traceability_fields)) != len(
            self.required_traceability_fields
        ):
            raise ValueError("required_traceability_fields contains duplicates")
        if set(self.required_traceability_fields) != _REQUIRED_TRACEABILITY_FIELDS:
            raise ValueError("required_traceability_fields is incomplete")
        return self


class OverallAcceptanceProtocol(ProtocolStrictModel):
    pass_condition: Literal["all-answerable-scenarios-pass"]
    answerable_pass_rate: Literal[1.0]
    unanswerable_affects_pass_rate: Literal[False]


class ReportingProtocol(ProtocolStrictModel):
    reported_metrics: list[
        Literal[
            "hit_at_1",
            "hit_at_3",
            "hit_at_5",
            "mrr_at_10",
            "ndcg_at_10",
            "evidence_precision",
            "evidence_recall",
            "evidence_f1",
            "expected_file_recall",
            "gold_rank",
        ]
    ] = Field(min_length=10, max_length=10)
    hit_at_5_disclosure: Literal["computed-from-at-most-top-3-not-five-retrieved"]
    fake_provider_metrics_can_set_acceptance_threshold: Literal[False]

    @model_validator(mode="after")
    def exact_metrics(self) -> "ReportingProtocol":
        if len(set(self.reported_metrics)) != len(self.reported_metrics):
            raise ValueError("reported_metrics contains duplicates")
        if set(self.reported_metrics) != _REQUIRED_REPORTED_METRICS:
            raise ValueError("reported_metrics does not match the frozen reporting contract")
        return self


class LiveDenseAcceptanceProtocol(ProtocolStrictModel):
    protocol_id: Literal[LIVE_DENSE_PROTOCOL_ID]
    protocol_version: Literal[LIVE_DENSE_PROTOCOL_VERSION]
    applicability: ProtocolApplicability
    stable_ordering: StableOrderingProtocol
    partition: PartitionProtocol
    retrieval: RetrievalProtocol
    gold_acceptance: GoldAcceptanceProtocol
    overall_acceptance: OverallAcceptanceProtocol
    reporting: ReportingProtocol


class PersistentChunkIdentity(ProtocolStrictModel):
    repository_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    path: str = Field(min_length=1, max_length=500)
    chunk_type: str = Field(min_length=1, max_length=80)
    qualified_name: str = Field(min_length=1, max_length=500)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def valid_span(self) -> "PersistentChunkIdentity":
        if self.end_line < self.start_line:
            raise ValueError("persistent identity end_line precedes start_line")
        if "\\" in self.path or self.path.startswith("/") or ".." in Path(self.path).parts:
            raise ValueError("persistent identity path must be a repository-relative POSIX path")
        return self


class ProtocolChunkRecord(ProtocolStrictModel):
    chunk_identity: str = Field(pattern=r"^chunk-sha256:[0-9a-f]{64}$")
    repository_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$")
    repository_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    repository_content_identity: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    persistent_identity: PersistentChunkIdentity

    @model_validator(mode="after")
    def consistent_revision(self) -> "ProtocolChunkRecord":
        if self.repository_revision != self.persistent_identity.repository_revision:
            raise ValueError("chunk and persistent identity revisions differ")
        return self


class TraceableGold(ProtocolStrictModel):
    repository_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$")
    repository_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    repository_content_identity: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    path: str = Field(min_length=1, max_length=500)
    qualified_symbol: str = Field(min_length=1, max_length=500)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunk_identity: str = Field(pattern=r"^chunk-sha256:[0-9a-f]{64}$")
    full_index_membership: Literal[True]

    @model_validator(mode="after")
    def valid_span(self) -> "TraceableGold":
        if self.end_line < self.start_line:
            raise ValueError("gold end_line precedes start_line")
        if "\\" in self.path or self.path.startswith("/") or ".." in Path(self.path).parts:
            raise ValueError("gold path must be a repository-relative POSIX path")
        return self


class TraceableCandidate(TraceableGold):
    validation_status: Literal["valid"]


class PhysicalFileState(ProtocolStrictModel):
    byte_length: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mtime_ns: int = Field(ge=0)


class PhysicalArtifactState(ProtocolStrictModel):
    manifest: PhysicalFileState
    checkpoint: PhysicalFileState


@dataclass(frozen=True)
class ChunkPartition:
    ordered: tuple[ProtocolChunkRecord, ...]
    stage_a: tuple[ProtocolChunkRecord, ...]
    stage_b: tuple[ProtocolChunkRecord, ...]
    full: tuple[ProtocolChunkRecord, ...]


@dataclass(frozen=True)
class ScenarioAcceptance:
    scenario_id: str
    answerable: bool
    passed: bool
    gold_rank: int | None
    query_encode_count: int
    skipped: bool


def load_live_dense_protocol(
    path: Path = DEFAULT_LIVE_DENSE_PROTOCOL_PATH,
) -> LiveDenseAcceptanceProtocol:
    resolved = path.resolve()
    if not resolved.is_file():
        raise LiveDenseProtocolError(f"live dense protocol is missing: {resolved}")
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
        return LiveDenseAcceptanceProtocol.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise LiveDenseProtocolError(
            f"live dense protocol validation failed: {type(exc).__name__}: {exc}"
        ) from exc


def validate_protocol_repository_coverage(
    protocol: LiveDenseAcceptanceProtocol,
    repositories: Sequence[Any],
) -> None:
    if not repositories:
        raise LiveDenseProtocolError("live dense protocol has no repositories to cover")
    for repository in repositories:
        repo_id = _read_value(repository, "repo_id")
        language = _read_value(repository, "language")
        if not repo_id:
            raise LiveDenseProtocolError("repository without repo_id has no protocol association")
        if language not in protocol.applicability.languages:
            raise LiveDenseProtocolError(
                f"repository {repo_id} has no associated live dense protocol"
            )


def partition_chunk_records(
    records: Sequence[ProtocolChunkRecord | dict[str, Any]],
    *,
    repository_id: str,
    repository_revision: str,
    repository_content_identity: str,
) -> ChunkPartition:
    parsed = tuple(
        value
        if isinstance(value, ProtocolChunkRecord)
        else ProtocolChunkRecord.model_validate(value)
        for value in records
    )
    if not parsed:
        raise LiveDenseProtocolError("complete chunk inventory must not be empty")
    identities = [item.chunk_identity for item in parsed]
    if len(identities) != len(set(identities)):
        raise LiveDenseProtocolError("stable chunk identities must be globally unique")
    for item in parsed:
        if (
            item.repository_id != repository_id
            or item.repository_revision != repository_revision
            or item.repository_content_identity != repository_content_identity
        ):
            raise LiveDenseProtocolError(
                "chunk inventory crosses repository, revision, or content identity"
            )
        _validate_existing_chunk_identity(item)
    ordered = tuple(sorted(parsed, key=lambda item: item.chunk_identity.encode("utf-8")))
    a_count = (len(ordered) + 1) // 2
    stage_a = ordered[:a_count]
    stage_b = ordered[a_count:]
    if set(item.chunk_identity for item in stage_a).intersection(
        item.chunk_identity for item in stage_b
    ):
        raise LiveDenseProtocolError("A and B chunk partitions overlap")
    if tuple(stage_a + stage_b) != ordered:
        raise LiveDenseProtocolError("A and B do not cover the complete ordered inventory")
    return ChunkPartition(ordered=ordered, stage_a=stage_a, stage_b=stage_b, full=ordered)


def validate_stage_statistics(
    partition: ChunkPartition,
    *,
    stage: Literal["A", "B", "C"],
    generated: int,
    cached: int,
    document_encode_calls: int,
    document_encode_batches: int,
    document_encode_items: int,
) -> None:
    n = len(partition.full)
    expected = {
        "A": (len(partition.stage_a), 0, len(partition.stage_a)),
        "B": (len(partition.stage_b), len(partition.stage_a), len(partition.stage_b)),
        "C": (0, n, 0),
    }[stage]
    if (generated, cached, document_encode_items) != expected:
        raise LiveDenseProtocolError(f"stage {stage} generated/cached/item counts are invalid")
    if stage == "C":
        valid_activity = document_encode_calls == 0 and document_encode_batches == 0
    elif document_encode_items:
        valid_activity = document_encode_calls > 0 and document_encode_batches > 0
    else:
        valid_activity = document_encode_calls == 0 and document_encode_batches == 0
    if not valid_activity:
        raise LiveDenseProtocolError(f"stage {stage} document encode call/batch counts are invalid")


def validate_stage_target(
    partition: ChunkPartition,
    *,
    stage: Literal["A", "B", "C"],
    target_chunk_identities: Sequence[str],
) -> None:
    expected = {
        "A": tuple(item.chunk_identity for item in partition.stage_a),
        "B": tuple(item.chunk_identity for item in partition.stage_b),
        "C": tuple(item.chunk_identity for item in partition.full),
    }[stage]
    if tuple(target_chunk_identities) != expected:
        raise LiveDenseProtocolError(f"stage {stage} target does not match the frozen partition")


def validate_stage_c_physical_noop(
    before: PhysicalArtifactState | dict[str, Any],
    after: PhysicalArtifactState | dict[str, Any],
) -> None:
    before_state = (
        before if isinstance(before, PhysicalArtifactState) else PhysicalArtifactState.model_validate(before)
    )
    after_state = (
        after if isinstance(after, PhysicalArtifactState) else PhysicalArtifactState.model_validate(after)
    )
    if before_state != after_state:
        raise LiveDenseProtocolError(
            "stage C changed manifest/checkpoint bytes, hashes, lengths, or mtimes"
        )


def evaluate_answerable_scenario(
    protocol: LiveDenseAcceptanceProtocol,
    *,
    scenario_id: str,
    query_encode_count: int,
    gold: Sequence[TraceableGold | dict[str, Any]],
    candidates: Sequence[TraceableCandidate | dict[str, Any]],
) -> ScenarioAcceptance:
    if query_encode_count != protocol.retrieval.answerable_query_encode_count:
        raise LiveDenseProtocolError("answerable scenario query must be encoded exactly once")
    parsed_gold = tuple(
        item if isinstance(item, TraceableGold) else TraceableGold.model_validate(item)
        for item in gold
    )
    if not parsed_gold:
        raise LiveDenseProtocolError("answerable scenario requires complete gold")
    parsed_candidates = tuple(
        item
        if isinstance(item, TraceableCandidate)
        else TraceableCandidate.model_validate(item)
        for item in candidates
    )
    gold_rank = next(
        (
            rank
            for rank, candidate in enumerate(parsed_candidates[: protocol.retrieval.top_k], 1)
            if any(_candidate_matches_gold(candidate, expected) for expected in parsed_gold)
        ),
        None,
    )
    return ScenarioAcceptance(
        scenario_id=scenario_id,
        answerable=True,
        passed=gold_rank is not None,
        gold_rank=gold_rank,
        query_encode_count=query_encode_count,
        skipped=False,
    )


def evaluate_unanswerable_scenario(
    protocol: LiveDenseAcceptanceProtocol,
    *,
    scenario_id: str,
    query_encode_count: int,
    gold: Sequence[Any],
    candidates: Sequence[Any],
    gold_rank: int | None,
) -> ScenarioAcceptance:
    if protocol.retrieval.unanswerable_query_behavior != "skip":
        raise LiveDenseProtocolError("unsupported unanswerable query behavior")
    if query_encode_count != 0 or gold or candidates or gold_rank is not None:
        raise LiveDenseProtocolError(
            "unanswerable scenario must be skipped without query, gold, candidates, or rank"
        )
    return ScenarioAcceptance(
        scenario_id=scenario_id,
        answerable=False,
        passed=True,
        gold_rank=None,
        query_encode_count=0,
        skipped=True,
    )


def evaluate_overall_acceptance(
    protocol: LiveDenseAcceptanceProtocol,
    scenarios: Sequence[ScenarioAcceptance],
    *,
    required_answerable_scenario_ids: Sequence[str],
) -> dict[str, Any]:
    required = tuple(required_answerable_scenario_ids)
    if not required or any(not value for value in required) or len(required) != len(set(required)):
        raise LiveDenseProtocolError(
            "required answerable scenario identities must be nonempty and unique"
        )
    answerable = [item for item in scenarios if item.answerable]
    observed = [item.scenario_id for item in answerable]
    if len(observed) != len(set(observed)) or set(observed) != set(required):
        raise LiveDenseProtocolError(
            "overall acceptance does not cover exactly the required answerable scenarios"
        )
    passed = sum(item.passed for item in answerable)
    pass_rate = passed / len(answerable)
    return {
        "pass_condition": protocol.overall_acceptance.pass_condition,
        "answerable_count": len(answerable),
        "passed_answerable_count": passed,
        "skipped_unanswerable_count": sum(item.skipped for item in scenarios),
        "answerable_pass_rate": pass_rate,
        "passed": passed == len(answerable),
        "hit_at_5_disclosure": protocol.reporting.hit_at_5_disclosure,
    }


def _validate_existing_chunk_identity(item: ProtocolChunkRecord) -> None:
    # Import lazily so protocol/schema validation remains independent of embedding/model imports.
    from app.m5.dense_artifact import build_chunk_inventory

    expected = build_chunk_inventory([item.persistent_identity.model_dump()])[0]
    if item.chunk_identity != expected:
        raise LiveDenseProtocolError(
            "stable chunk identity does not match the existing build_chunk_inventory contract"
        )


def _candidate_matches_gold(candidate: TraceableCandidate, gold: TraceableGold) -> bool:
    return all(
        (
            candidate.repository_id == gold.repository_id,
            candidate.repository_revision == gold.repository_revision,
            candidate.repository_content_identity == gold.repository_content_identity,
            candidate.path == gold.path,
            candidate.qualified_symbol == gold.qualified_symbol,
            candidate.start_line == gold.start_line,
            candidate.end_line == gold.end_line,
            candidate.content_hash == gold.content_hash,
            candidate.chunk_identity == gold.chunk_identity,
            candidate.full_index_membership,
            gold.full_index_membership,
        )
    )


def _read_value(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)
