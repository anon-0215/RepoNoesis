from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Literal

from app.database import Database
from app.services.embedding_service import EmbeddingService
from app.services.hierarchy_normalization import (
    HIERARCHY_MODE_OFF,
    HIERARCHY_MODE_NORMALIZE_V1,
    HierarchyLimits,
    HierarchyResolver,
    normalize_hierarchy_candidates,
    validate_hierarchy_mode,
)
from app.services.hybrid_retriever import HybridRetriever, HybridSearchResult
from app.services.lexical_retriever import LexicalRetriever, LexicalSearchResult
from app.services.query_analyzer import QueryAnalysis, QueryAnalyzer
from app.services.semantic_retriever import SemanticRetriever, SemanticSearchResult
from app.services.symbol_retriever import SymbolRetriever, SymbolSearchResult


RetrievalVersion = Literal["v1", "v2"]
CandidateSource = Literal["dense", "lexical", "symbol"]

RETRIEVAL_VERSION_V1 = "v1"
RETRIEVAL_VERSION_V2 = "v2"
SUPPORTED_RETRIEVAL_VERSIONS = frozenset(
    {RETRIEVAL_VERSION_V1, RETRIEVAL_VERSION_V2}
)
V1_FUSION_VERSION = "weighted-rrf-v1"
V2_FUSION_VERSION = "weighted_rrf_v2@1"
SOURCE_ORDER: tuple[CandidateSource, ...] = ("dense", "lexical", "symbol")


class RetrievalContractError(ValueError):
    """A candidate violated the server-owned retrieval identity contract."""


@dataclass(frozen=True)
class RetrievalV2Config:
    fusion_version: str = V2_FUSION_VERSION
    rrf_k: int = 60
    min_source_weight: float = 0.05
    max_source_weight: float = 10.0
    default_source_pool: int = 24
    max_source_pool: int = 50
    max_symbol_hints: int = 8
    max_merged_pool: int = 72
    max_final_top_k: int = 8

    def __post_init__(self) -> None:
        if self.fusion_version != V2_FUSION_VERSION:
            raise ValueError("unknown Retrieval v2 fusion version")
        _require_bounded_int("rrf_k", self.rrf_k, 1, 10_000)
        _require_finite_number(
            "min_source_weight", self.min_source_weight, minimum=0.000_001
        )
        _require_finite_number(
            "max_source_weight",
            self.max_source_weight,
            minimum=self.min_source_weight,
        )
        _require_bounded_int(
            "default_source_pool",
            self.default_source_pool,
            1,
            self.max_source_pool,
        )
        _require_bounded_int("max_source_pool", self.max_source_pool, 1, 50)
        _require_bounded_int("max_symbol_hints", self.max_symbol_hints, 1, 16)
        _require_bounded_int("max_merged_pool", self.max_merged_pool, 1, 150)
        _require_bounded_int("max_final_top_k", self.max_final_top_k, 1, 8)


@dataclass(frozen=True)
class RetrievalExecutionOutcome:
    results: list[HybridSearchResult]
    retrieval_mode: str
    warnings: list[str]
    retrieval_version: RetrievalVersion
    retrieval_strategy_version: str
    audit: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalSourceCandidate:
    chunk_identity: str
    project_id: str
    repository_revision: str
    code_chunk_id: int
    language: str
    path: str
    chunk_type: str
    symbol_name: str
    qualified_name: str
    start_line: int
    end_line: int
    content: str
    content_hash: str
    source: CandidateSource
    source_rank: int
    source_raw_score: float
    source_reasons: tuple[str, ...]
    source_metadata: dict[str, Any]


@dataclass
class _SourceRecord:
    rank: int
    raw_score: float
    reasons: list[str]
    metadata: dict[str, Any]

    def merge(self, candidate: RetrievalSourceCandidate) -> None:
        if (candidate.source_rank, -candidate.source_raw_score) < (
            self.rank,
            -self.raw_score,
        ):
            self.rank = candidate.source_rank
            self.raw_score = candidate.source_raw_score
        self.reasons = _deduplicate([*self.reasons, *candidate.source_reasons])
        for key, value in candidate.source_metadata.items():
            if key in {"matched_hints", "match_types"}:
                self.metadata[key] = _deduplicate(
                    [*list(self.metadata.get(key, [])), *list(value or [])]
                )
            elif key not in self.metadata:
                self.metadata[key] = value
            elif self.metadata[key] != value:
                existing = self.metadata[key]
                values = existing if isinstance(existing, list) else [existing]
                self.metadata[key] = _stable_values([*values, value])


@dataclass
class _MergedCandidate:
    chunk_identity: str
    project_id: str
    repository_revision: str
    code_chunk_id: int
    language: str
    path: str
    chunk_type: str
    symbol_name: str
    qualified_name: str
    start_line: int
    end_line: int
    content: str
    content_hash: str
    source_records: dict[CandidateSource, _SourceRecord] = field(default_factory=dict)
    fusion_contributions: dict[CandidateSource, float] = field(default_factory=dict)
    fused_score: float = 0.0
    fusion_rank: int = 0

    @classmethod
    def from_source(cls, candidate: RetrievalSourceCandidate) -> "_MergedCandidate":
        item = cls(
            chunk_identity=candidate.chunk_identity,
            project_id=candidate.project_id,
            repository_revision=candidate.repository_revision,
            code_chunk_id=candidate.code_chunk_id,
            language=candidate.language,
            path=candidate.path,
            chunk_type=candidate.chunk_type,
            symbol_name=candidate.symbol_name,
            qualified_name=candidate.qualified_name,
            start_line=candidate.start_line,
            end_line=candidate.end_line,
            content=candidate.content,
            content_hash=candidate.content_hash,
        )
        item.add_source(candidate)
        return item

    def add_source(self, candidate: RetrievalSourceCandidate) -> None:
        if _candidate_metadata(self) != _candidate_metadata(candidate):
            raise RetrievalContractError(
                "candidate identity metadata conflict for the same exact chunk"
            )
        existing = self.source_records.get(candidate.source)
        if existing is None:
            self.source_records[candidate.source] = _SourceRecord(
                rank=candidate.source_rank,
                raw_score=candidate.source_raw_score,
                reasons=list(candidate.source_reasons),
                metadata=dict(candidate.source_metadata),
            )
        else:
            existing.merge(candidate)

    def to_hybrid_result(self) -> HybridSearchResult:
        lexical = self.source_records.get("lexical")
        dense = self.source_records.get("dense")
        return HybridSearchResult(
            project_id=self.project_id,
            repository_revision=self.repository_revision,
            code_chunk_id=self.code_chunk_id,
            language=self.language,
            path=self.path,
            chunk_type=self.chunk_type,
            symbol_name=self.symbol_name,
            qualified_name=self.qualified_name,
            start_line=self.start_line,
            end_line=self.end_line,
            content=self.content,
            content_hash=self.content_hash,
            retrieval_sources=[
                source for source in SOURCE_ORDER if source in self.source_records
            ],
            lexical_score=lexical.raw_score if lexical else None,
            lexical_rank=lexical.rank if lexical else None,
            semantic_score=dense.raw_score if dense else None,
            semantic_rank=dense.rank if dense else None,
            fusion_score=self.fused_score,
            fusion_rank=self.fusion_rank,
        )

    def to_audit_dict(self, query_summary: dict[str, Any]) -> dict[str, Any]:
        return {
            "chunk_identity": self.chunk_identity,
            "code_chunk_id": self.code_chunk_id,
            "path": self.path,
            "symbol_name": self.symbol_name,
            "qualified_name": self.qualified_name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "retrieval_version": RETRIEVAL_VERSION_V2,
            "fusion_version": V2_FUSION_VERSION,
            "query_analysis_summary": query_summary,
            "merged_sources": [
                source for source in SOURCE_ORDER if source in self.source_records
            ],
            "source_records": {
                source: {
                    "rank": self.source_records[source].rank,
                    "raw_score": self.source_records[source].raw_score,
                    "reasons": list(self.source_records[source].reasons),
                    "metadata": dict(self.source_records[source].metadata),
                }
                for source in SOURCE_ORDER
                if source in self.source_records
            },
            "fusion_contributions": {
                source: self.fusion_contributions.get(source, 0.0)
                for source in SOURCE_ORDER
            },
            "fused_score": self.fused_score,
            "fusion_rank": self.fusion_rank,
            "tie_break": {
                "best_source_rank": min(
                    record.rank for record in self.source_records.values()
                ),
                "source_count": len(self.source_records),
                "qualified_name": self.qualified_name,
                "normalized_path": _normalize_path(self.path),
                "start_line": self.start_line,
                "end_line": self.end_line,
                "chunk_identity": self.chunk_identity,
            },
        }


class RetrievalV2Orchestrator:
    """Deterministic three-source retrieval without hierarchy or relation expansion."""

    def __init__(
        self,
        database: Database,
        embedding_service: EmbeddingService,
        *,
        query_analyzer: QueryAnalyzer | None = None,
        lexical_retriever: LexicalRetriever | None = None,
        semantic_retriever: SemanticRetriever | None = None,
        symbol_retriever: SymbolRetriever | None = None,
        config: RetrievalV2Config | None = None,
        hierarchy_limits: HierarchyLimits | None = None,
    ) -> None:
        self.database = database
        self.embedding_service = embedding_service
        self.query_analyzer = query_analyzer or QueryAnalyzer()
        self.lexical_retriever = lexical_retriever or LexicalRetriever(database)
        self.semantic_retriever = semantic_retriever or SemanticRetriever(
            database, embedding_service
        )
        self.symbol_retriever = symbol_retriever or SymbolRetriever(database)
        self.config = config or RetrievalV2Config()
        self.hierarchy_limits = hierarchy_limits or HierarchyLimits()

    def search(
        self,
        project_id: str,
        query: str,
        evidence_count: int = 5,
        path: str | None = None,
        language: str | None = None,
        symbol: str | None = None,
        hierarchy_mode: str = HIERARCHY_MODE_OFF,
    ) -> RetrievalExecutionOutcome:
        if not str(query).strip():
            raise ValueError("retrieval query must not be empty")
        hierarchy_mode = validate_hierarchy_mode(
            hierarchy_mode,
            retrieval_version=RETRIEVAL_VERSION_V2,
        )
        analysis = self.query_analyzer.analyze(query)
        weights, budgets, policy_warnings = _validated_policy(analysis, self.config)
        source_candidates: list[RetrievalSourceCandidate] = []
        warnings = list(policy_warnings)
        source_audit: dict[str, dict[str, Any]] = {}
        all_symbol_hints = _symbol_hints(analysis, symbol)
        symbol_hints = all_symbol_hints[: self.config.max_symbol_hints]
        if len(symbol_hints) < len(all_symbol_hints):
            warnings.append(
                "Symbol hints exceeded the v2 request limit; later hints were "
                "deterministically truncated."
            )

        lexical = self.lexical_retriever.search(
            project_id,
            query,
            top_k=budgets["lexical"],
            path=path,
            language=language,
            symbol=symbol,
        )
        lexical = sorted(lexical, key=_lexical_sort_key)[: budgets["lexical"]]
        source_candidates.extend(_adapt_lexical(item) for item in lexical)
        source_audit["lexical"] = {
            "status": "ok" if lexical else "empty",
            "budget": budgets["lexical"],
            "candidate_count": len(lexical),
        }

        dense: list[SemanticSearchResult] = []
        if self.embedding_service.settings.enabled:
            try:
                dense_outcome = self.semantic_retriever.search(
                    project_id,
                    query,
                    top_k=budgets["dense"],
                    path=path,
                    language=language,
                    symbol=symbol,
                    local_files_only=True,
                )
                dense = list(dense_outcome.results)[: budgets["dense"]]
                warnings.extend(dense_outcome.warnings)
                source_audit["dense"] = {
                    "status": dense_outcome.status if not dense else "ok",
                    "budget": budgets["dense"],
                    "candidate_count": len(dense),
                    "model_name": dense_outcome.model_name,
                }
            except Exception as exc:
                warnings.append(
                    "Dense retrieval unavailable; continuing with the other v2 "
                    f"sources: {type(exc).__name__}."
                )
                source_audit["dense"] = {
                    "status": "unavailable",
                    "budget": budgets["dense"],
                    "candidate_count": 0,
                    "error_type": type(exc).__name__,
                }
        else:
            warnings.append(
                "Embeddings are disabled; the dense v2 source is controlled-unavailable."
            )
            source_audit["dense"] = {
                "status": "disabled",
                "budget": budgets["dense"],
                "candidate_count": 0,
            }
        source_candidates.extend(
            _adapt_dense(item, rank)
            for rank, item in enumerate(dense, start=1)
        )

        symbol_candidates, symbol_status = self._symbol_candidates(
            project_id=project_id,
            query=query,
            hints=symbol_hints,
            budget=budgets["symbol"],
            path=path,
            language=language,
        )
        source_candidates.extend(symbol_candidates)
        source_audit["symbol"] = {
            "status": symbol_status,
            "budget": budgets["symbol"],
            "candidate_count": len(symbol_candidates),
            "hint_count": len(all_symbol_hints),
            "executed_hint_count": len(symbol_hints),
            "hints_truncated": len(symbol_hints) < len(all_symbol_hints),
        }

        merged = _merge_and_fuse(
            source_candidates,
            weights=weights,
            config=self.config,
        )
        limit = min(
            self.config.max_final_top_k,
            max(1, int(evidence_count)),
        )
        selected = merged[:limit]
        normalized_results: list[HybridSearchResult] | None = None
        hierarchy_audit: dict[str, Any] | None = None
        if hierarchy_mode == HIERARCHY_MODE_NORMALIZE_V1:
            resolution = HierarchyResolver(
                self.database,
                self.hierarchy_limits,
            ).resolve(merged)
            normalized = normalize_hierarchy_candidates(
                merged,
                resolution,
                final_top_k=limit,
                limits=self.hierarchy_limits,
            )
            normalized_results = normalized.results
            hierarchy_audit = normalized.audit
            warnings.extend(normalized.warnings)
        query_summary = _query_summary(analysis)
        audit = {
            "retrieval_version": RETRIEVAL_VERSION_V2,
            "fusion_version": self.config.fusion_version,
            "rrf_k": self.config.rrf_k,
            "query_analysis": _query_audit(analysis),
            "effective_policy": {
                "source_weights": dict(weights),
                "source_budgets": dict(budgets),
            },
            "sources": {
                source: source_audit[source] for source in SOURCE_ORDER
            },
            "limits": {
                "max_source_pool": self.config.max_source_pool,
                "max_symbol_hints": self.config.max_symbol_hints,
                "max_merged_pool": self.config.max_merged_pool,
                "final_top_k": limit,
            },
            "candidates": [
                item.to_audit_dict(query_summary) for item in selected
            ],
        }
        if hierarchy_audit is not None:
            audit["hierarchy"] = hierarchy_audit
        return RetrievalExecutionOutcome(
            results=(
                normalized_results
                if normalized_results is not None
                else [item.to_hybrid_result() for item in selected]
            ),
            retrieval_mode=(
                "hybrid"
                if any("dense" in item.source_records for item in selected)
                else "lexical"
            ),
            warnings=_deduplicate(warnings),
            retrieval_version=RETRIEVAL_VERSION_V2,
            retrieval_strategy_version=self.config.fusion_version,
            audit=audit,
        )

    def _symbol_candidates(
        self,
        *,
        project_id: str,
        query: str,
        hints: list[str],
        budget: int,
        path: str | None,
        language: str | None,
    ) -> tuple[list[RetrievalSourceCandidate], str]:
        raw: list[tuple[SymbolSearchResult, str | None]] = []
        repository_revision = _project_revision(self.database, project_id)
        if hints:
            per_hint = max(1, math.ceil(budget / len(hints)))
            for hint in hints:
                ranked = self.symbol_retriever.search(
                    project_id,
                    hint,
                    top_k=per_hint,
                    path=path,
                    language=language,
                    match_mode="auto",
                    explicit_symbol=True,
                    repository_revision=repository_revision,
                )
                raw.extend((item, hint) for item in ranked[:per_hint])
        else:
            ranked = self.symbol_retriever.search(
                project_id,
                query,
                top_k=budget,
                path=path,
                language=language,
                match_mode="auto",
                explicit_symbol=False,
                repository_revision=repository_revision,
            )
            raw.extend((item, None) for item in ranked[:budget])
        aggregated = _aggregate_symbol(raw, budget)
        status = "ok" if aggregated else ("empty" if hints else "no_reliable_hint")
        return aggregated, status


def retrieve_code(
    database: Database,
    embedding_service: EmbeddingService,
    project_id: str,
    query: str,
    *,
    retrieval_version: str = RETRIEVAL_VERSION_V1,
    evidence_count: int = 5,
    path: str | None = None,
    language: str | None = None,
    symbol: str | None = None,
    hierarchy_mode: str = HIERARCHY_MODE_OFF,
) -> RetrievalExecutionOutcome:
    version = validate_retrieval_version(retrieval_version)
    hierarchy_mode = validate_hierarchy_mode(
        hierarchy_mode,
        retrieval_version=version,
    )
    if version == RETRIEVAL_VERSION_V1:
        outcome = HybridRetriever(database, embedding_service).search(
            project_id,
            query,
            evidence_count=evidence_count,
            path=path,
            language=language,
            symbol=symbol,
        )
        return RetrievalExecutionOutcome(
            results=outcome.results,
            retrieval_mode=outcome.retrieval_mode,
            warnings=outcome.warnings,
            retrieval_version=RETRIEVAL_VERSION_V1,
            retrieval_strategy_version=V1_FUSION_VERSION,
        )
    return RetrievalV2Orchestrator(database, embedding_service).search(
        project_id,
        query,
        evidence_count=evidence_count,
        path=path,
        language=language,
        symbol=symbol,
        hierarchy_mode=hierarchy_mode,
    )


def validate_retrieval_version(value: str) -> RetrievalVersion:
    if not isinstance(value, str) or value not in SUPPORTED_RETRIEVAL_VERSIONS:
        raise ValueError("retrieval_version must be exactly 'v1' or 'v2'")
    return value  # type: ignore[return-value]


def _validated_policy(
    analysis: QueryAnalysis,
    config: RetrievalV2Config,
) -> tuple[dict[CandidateSource, float], dict[CandidateSource, int], list[str]]:
    proposed = {
        "dense": analysis.routing_hint.dense_weight,
        "lexical": analysis.routing_hint.lexical_weight,
        "symbol": analysis.routing_hint.symbol_weight,
    }
    pool = analysis.routing_hint.candidate_pool
    valid_weights = all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and config.min_source_weight <= float(value) <= config.max_source_weight
        for value in proposed.values()
    )
    valid_pool = (
        isinstance(pool, int)
        and not isinstance(pool, bool)
        and 1 <= pool <= config.max_source_pool
    )
    if valid_weights and valid_pool:
        weights = {source: float(proposed[source]) for source in SOURCE_ORDER}
        effective_pool = min(pool, config.max_source_pool)
        return (
            weights,
            {source: effective_pool for source in SOURCE_ORDER},
            [],
        )
    return (
        {source: 1.0 for source in SOURCE_ORDER},
        {source: config.default_source_pool for source in SOURCE_ORDER},
        [
            "Query analysis routing values failed validation; used the neutral "
            "three-source v2 policy."
        ],
    )


def _adapt_lexical(item: LexicalSearchResult) -> RetrievalSourceCandidate:
    return _source_candidate(
        item,
        source="lexical",
        rank=item.lexical_rank,
        raw_score=item.lexical_score,
        reasons=("bm25_match",),
        metadata={"scoring": "bm25"},
    )


def _adapt_dense(item: SemanticSearchResult, rank: int) -> RetrievalSourceCandidate:
    return _source_candidate(
        item,
        source="dense",
        rank=rank,
        raw_score=item.semantic_score,
        reasons=("semantic_similarity",),
        metadata={"model_name": item.model_name},
    )


def _adapt_symbol(
    item: SymbolSearchResult,
    *,
    rank: int,
    matched_hints: list[str],
    match_types: list[str],
    reasons: list[str],
) -> RetrievalSourceCandidate:
    expected_identity = _exact_chunk_identity(item)
    if item.chunk_identity != expected_identity:
        raise RetrievalContractError("symbol candidate chunk identity mismatch")
    return _source_candidate(
        item,
        source="symbol",
        rank=rank,
        raw_score=item.symbol_score,
        reasons=tuple(reasons),
        metadata={
            "matched_hints": matched_hints,
            "match_types": match_types,
            "candidate_source": item.candidate_source,
        },
    )


def _source_candidate(
    item: Any,
    *,
    source: CandidateSource,
    rank: int,
    raw_score: float,
    reasons: tuple[str, ...],
    metadata: dict[str, Any],
) -> RetrievalSourceCandidate:
    if int(item.code_chunk_id) < 1:
        raise RetrievalContractError("candidate is missing an authoritative code chunk ID")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
        raise RetrievalContractError("source rank must start at one")
    if not math.isfinite(float(raw_score)):
        raise RetrievalContractError("source raw score must be finite")
    return RetrievalSourceCandidate(
        chunk_identity=_exact_chunk_identity(item),
        project_id=str(item.project_id),
        repository_revision=str(item.repository_revision),
        code_chunk_id=int(item.code_chunk_id),
        language=str(item.language),
        path=str(item.path),
        chunk_type=str(item.chunk_type),
        symbol_name=str(item.symbol_name),
        qualified_name=str(item.qualified_name),
        start_line=int(item.start_line),
        end_line=int(item.end_line),
        content=str(item.content),
        content_hash=str(item.content_hash),
        source=source,
        source_rank=rank,
        source_raw_score=float(raw_score),
        source_reasons=tuple(_deduplicate(list(reasons))),
        source_metadata=dict(metadata),
    )


def _aggregate_symbol(
    raw: list[tuple[SymbolSearchResult, str | None]],
    budget: int,
) -> list[RetrievalSourceCandidate]:
    aggregated: dict[str, dict[str, Any]] = {}
    chunk_keys: dict[tuple[str, str, int], str] = {}
    for item, hint in raw:
        identity = _exact_chunk_identity(item)
        chunk_key = (item.project_id, item.repository_revision, item.code_chunk_id)
        other_identity = chunk_keys.get(chunk_key)
        if other_identity is not None and other_identity != identity:
            raise RetrievalContractError(
                "the same database chunk ID carried conflicting identity metadata"
            )
        chunk_keys[chunk_key] = identity
        value = aggregated.get(identity)
        if value is None:
            value = {
                "item": item,
                "best_rank": item.symbol_rank,
                "best_score": item.symbol_score,
                "matched_hints": [],
                "match_types": [],
                "reasons": [],
            }
            aggregated[identity] = value
        elif _candidate_metadata(value["item"]) != _candidate_metadata(item):
            raise RetrievalContractError("symbol candidate metadata conflict")
        if (item.symbol_rank, -item.symbol_score) < (
            value["best_rank"],
            -value["best_score"],
        ):
            value["item"] = item
            value["best_rank"] = item.symbol_rank
            value["best_score"] = item.symbol_score
        if hint is not None:
            value["matched_hints"] = _deduplicate([*value["matched_hints"], hint])
        value["match_types"] = _deduplicate(
            [*value["match_types"], item.symbol_match_type]
        )
        value["reasons"] = _deduplicate(
            [*value["reasons"], *item.match_reasons]
        )
    ordered = sorted(
        aggregated.values(),
        key=lambda value: (
            value["best_rank"],
            -value["best_score"],
            value["item"].qualified_name.casefold(),
            _normalize_path(value["item"].path),
            value["item"].start_line,
            value["item"].end_line,
            value["item"].code_chunk_id,
        ),
    )[:budget]
    return [
        _adapt_symbol(
            value["item"],
            rank=rank,
            matched_hints=list(value["matched_hints"]),
            match_types=list(value["match_types"]),
            reasons=list(value["reasons"]),
        )
        for rank, value in enumerate(ordered, start=1)
    ]


def _merge_and_fuse(
    candidates: list[RetrievalSourceCandidate],
    *,
    weights: dict[CandidateSource, float],
    config: RetrievalV2Config,
) -> list[_MergedCandidate]:
    by_identity: dict[str, _MergedCandidate] = {}
    identity_by_chunk: dict[tuple[str, str, int], str] = {}
    for candidate in candidates:
        chunk_key = (
            candidate.project_id,
            candidate.repository_revision,
            candidate.code_chunk_id,
        )
        previous_identity = identity_by_chunk.get(chunk_key)
        if previous_identity is not None and previous_identity != candidate.chunk_identity:
            raise RetrievalContractError(
                "the same database chunk ID carried conflicting identity metadata"
            )
        identity_by_chunk[chunk_key] = candidate.chunk_identity
        merged = by_identity.get(candidate.chunk_identity)
        if merged is None:
            merged = _MergedCandidate.from_source(candidate)
            by_identity[candidate.chunk_identity] = merged
        else:
            merged.add_source(candidate)

    results = list(by_identity.values())
    for item in results:
        item.fusion_contributions = {
            source: (
                weights[source] / (config.rrf_k + item.source_records[source].rank)
                if source in item.source_records
                else 0.0
            )
            for source in SOURCE_ORDER
        }
        item.fused_score = math.fsum(item.fusion_contributions.values())
    results.sort(key=_fusion_sort_key)
    results = results[: config.max_merged_pool]
    for rank, item in enumerate(results, start=1):
        item.fusion_rank = rank
    return results


def _fusion_sort_key(item: _MergedCandidate) -> tuple[Any, ...]:
    return (
        -item.fused_score,
        min(record.rank for record in item.source_records.values()),
        -len(item.source_records),
        item.qualified_name.casefold(),
        _normalize_path(item.path),
        item.start_line,
        item.end_line,
        item.chunk_identity,
    )


def _lexical_sort_key(item: LexicalSearchResult) -> tuple[Any, ...]:
    return (
        item.lexical_rank,
        -item.lexical_score,
        item.qualified_name.casefold(),
        _normalize_path(item.path),
        item.start_line,
        item.end_line,
        item.code_chunk_id,
    )


def _symbol_hints(analysis: QueryAnalysis, explicit_symbol: str | None) -> list[str]:
    values = []
    if explicit_symbol and explicit_symbol.strip():
        values.append(explicit_symbol.strip())
    values.extend(analysis.symbol_hints)
    return _deduplicate(values)


def _project_revision(database: Database, project_id: str) -> str | None:
    revisions = {
        str(chunk["repository_revision"])
        for chunk in database.get_code_chunks(project_id)
    }
    revisions.discard("")
    if len(revisions) > 1:
        raise RetrievalContractError("project contains more than one active chunk revision")
    return next(iter(revisions), None)


def _exact_chunk_identity(item: Any) -> str:
    return "|".join(
        [
            str(item.project_id),
            str(item.repository_revision),
            str(item.path),
            str(item.start_line),
            str(item.end_line),
            str(item.content_hash),
            str(item.code_chunk_id),
        ]
    )


def _candidate_metadata(item: Any) -> tuple[Any, ...]:
    return (
        str(item.project_id),
        str(item.repository_revision),
        int(item.code_chunk_id),
        str(item.language),
        str(item.path),
        str(item.chunk_type),
        str(item.symbol_name),
        str(item.qualified_name),
        int(item.start_line),
        int(item.end_line),
        str(item.content),
        str(item.content_hash),
    )


def _query_summary(analysis: QueryAnalysis) -> dict[str, Any]:
    return {
        "primary_intent": analysis.primary_intent,
        "secondary_intents": list(analysis.secondary_intents),
        "confidence": _audit_number(analysis.confidence),
        "reason_codes": list(analysis.reason_codes),
        "symbol_hints": list(analysis.symbol_hints),
        "neutral_fallback": analysis.neutral_fallback,
    }


def _query_audit(analysis: QueryAnalysis) -> dict[str, Any]:
    return {
        **_query_summary(analysis),
        "proposed_routing_hint": {
            "dense_weight": _audit_number(analysis.routing_hint.dense_weight),
            "lexical_weight": _audit_number(analysis.routing_hint.lexical_weight),
            "symbol_weight": _audit_number(analysis.routing_hint.symbol_weight),
            "candidate_pool": analysis.routing_hint.candidate_pool,
            "relation_direction": analysis.routing_hint.relation_direction,
            "relation_budget": analysis.routing_hint.relation_budget,
        },
    }


def _audit_number(value: Any) -> float | str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
    return "invalid"


def _normalize_path(value: str) -> str:
    return str(value).replace("\\", "/").lstrip("/").casefold()


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _stable_values(values: list[Any]) -> list[Any]:
    return sorted(
        {repr(value): value for value in values}.values(),
        key=lambda value: repr(value),
    )


def _require_bounded_int(name: str, value: int, minimum: int, maximum: int) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
        or value > maximum
    ):
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")


def _require_finite_number(
    name: str,
    value: float,
    *,
    minimum: float,
) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise ValueError(f"{name} must be a finite number of at least {minimum}")
