from __future__ import annotations

from dataclasses import asdict, dataclass, field
from concurrent.futures import TimeoutError as FutureTimeout
import time
from typing import Any, Callable

from app.database import Database
from app.services.embedding_service import EmbeddingError, EmbeddingService, SemanticCapacityUnavailable
from app.services.lexical_retriever import LexicalRetriever
from app.services.semantic_retriever import SemanticDataError, SemanticRetriever
from app.services.smoke_diagnostics import SmokeDiagnosticsRecorder


RRF_K = 60
LEXICAL_WEIGHT = 1.0
SEMANTIC_WEIGHT = 1.0
LEXICAL_CANDIDATE_COUNT = 20
SEMANTIC_CANDIDATE_COUNT = 20
DEFAULT_EVIDENCE_COUNT = 5
MAXIMUM_EVIDENCE_COUNT = 8


@dataclass
class HybridSearchResult:
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
    retrieval_sources: list[str] = field(default_factory=list)
    lexical_score: float | None = None
    lexical_rank: int | None = None
    semantic_score: float | None = None
    semantic_rank: int | None = None
    fusion_score: float = 0.0
    fusion_rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HybridSearchOutcome:
    results: list[HybridSearchResult]
    retrieval_mode: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [result.to_dict() for result in self.results],
            "retrieval_mode": self.retrieval_mode,
            "warnings": list(self.warnings),
        }


class HybridRetriever:
    def __init__(
        self,
        database: Database,
        embedding_service: EmbeddingService,
        lexical_retriever: LexicalRetriever | None = None,
        semantic_retriever: SemanticRetriever | None = None,
    ) -> None:
        self.database = database
        self.embedding_service = embedding_service
        self.lexical_retriever = lexical_retriever or LexicalRetriever(database)
        self.semantic_retriever = semantic_retriever or SemanticRetriever(
            database,
            embedding_service,
        )

    def search(
        self,
        project_id: str,
        query: str,
        evidence_count: int = DEFAULT_EVIDENCE_COUNT,
        path: str | None = None,
        language: str | None = None,
        symbol: str | None = None,
        check_active: Callable[[], None] | None = None,
        diagnostics_recorder: SmokeDiagnosticsRecorder | None = None,
        semantic_deadline_at: float | None = None,
    ) -> HybridSearchOutcome:
        _check(check_active)
        limit = min(MAXIMUM_EVIDENCE_COUNT, max(1, int(evidence_count)))
        lexical_started = time.monotonic()
        lexical = self.lexical_retriever.search(
            project_id,
            query,
            top_k=LEXICAL_CANDIDATE_COUNT,
            path=path,
            language=language,
            symbol=symbol,
        )
        _check(check_active)
        detail: dict[str, Any] = {
            "lexical_ms": int((time.monotonic() - lexical_started) * 1000),
            "lexical_hit_count": len(lexical),
            "model_state": getattr(self.embedding_service, "model_load_state", "unknown"),
        }
        detail["model_state_at_start"] = detail["model_state"]
        warnings: list[str] = []
        semantic = []
        if self.embedding_service.settings.enabled:
            # Keep part of the existing tool window for fusion, canonical
            # reread and EvidenceStore admission. The request deadline itself
            # is never extended.
            remaining = None if semantic_deadline_at is None else semantic_deadline_at - time.monotonic()
            if remaining is not None:
                detail["semantic_remaining_ms"] = max(0, int(remaining * 1000))
            phase_timings: dict[str, int] = {}

            def semantic_work() -> Any:
                return self.semantic_retriever.search(
                    project_id,
                    query,
                    top_k=SEMANTIC_CANDIDATE_COUNT,
                    path=path,
                    language=language,
                    symbol=symbol,
                    local_files_only=True,
                    # Native model work cannot be interrupted reliably. It
                    # must never mutate the request recorder or EvidenceStore.
                    check_active=None if semantic_deadline_at is not None else check_active,
                    diagnostics_recorder=None if semantic_deadline_at is not None else diagnostics_recorder,
                    phase_timings=phase_timings,
                )

            try:
                if semantic_deadline_at is None:
                    semantic_outcome = semantic_work()
                    semantic = semantic_outcome.results
                    warnings.extend(semantic_outcome.warnings)
                    detail["semantic_status"] = "completed"
                else:
                    status, semantic_outcome, wait_ms = run_bounded_semantic(
                        self.embedding_service, semantic_work,
                        deadline_at=semantic_deadline_at, check_active=check_active,
                    )
                    detail["semantic_status"] = status
                    if wait_ms is not None:
                        detail["semantic_wait_ms"] = wait_ms
                    if semantic_outcome is not None:
                        semantic = semantic_outcome.results
                        warnings.extend(semantic_outcome.warnings)
                _check(check_active)
            except SemanticCapacityUnavailable:
                detail["semantic_status"] = "capacity"
            except (EmbeddingError, SemanticDataError, OSError):
                _check(check_active)
                detail["semantic_status"] = "failed"
            detail.update(dict(phase_timings))
            if detail["model_state"] != "ready" and "model_identity_ms" in detail:
                detail["model_load_ms"] = detail["model_identity_ms"]
            detail["model_state"] = getattr(self.embedding_service, "model_load_state", "unknown")
            if detail.get("semantic_status") != "completed":
                warnings.append("Semantic retrieval did not complete; continuing with lexical code-chunk candidates.")
        else:
            detail["semantic_status"] = "disabled"
            warnings.append(
                "Embeddings are disabled; using lexical code-chunk search."
            )

        revision_read_started = time.monotonic()
        current_revisions = {
            str(chunk["repository_revision"])
            for chunk in self.database.get_code_chunks(project_id)
        }
        detail["revision_read_ms"] = int((time.monotonic() - revision_read_started) * 1000)
        _check(check_active)
        fusion_started = time.monotonic()
        valid_semantic = []
        for item in semantic:
            if (
                item.project_id != project_id
                or item.repository_revision not in current_revisions
            ):
                warnings.append(
                    "Ignored a semantic result from a different project or revision."
                )
                continue
            valid_semantic.append(item)

        by_identity: dict[tuple[Any, ...], HybridSearchResult] = {}
        for item in lexical:
            identity = _identity(item)
            result = HybridSearchResult(
                project_id=item.project_id,
                repository_revision=item.repository_revision,
                code_chunk_id=item.code_chunk_id,
                language=item.language,
                path=item.path,
                chunk_type=item.chunk_type,
                symbol_name=item.symbol_name,
                qualified_name=item.qualified_name,
                start_line=item.start_line,
                end_line=item.end_line,
                content=item.content,
                content_hash=item.content_hash,
                retrieval_sources=["lexical"],
                lexical_score=item.lexical_score,
                lexical_rank=item.lexical_rank,
            )
            by_identity[identity] = result

        for semantic_rank, item in enumerate(valid_semantic, start=1):
            identity = _identity(item)
            result = by_identity.get(identity)
            if result is None:
                result = HybridSearchResult(
                    project_id=item.project_id,
                    repository_revision=item.repository_revision,
                    code_chunk_id=item.code_chunk_id,
                    language=item.language,
                    path=item.path,
                    chunk_type=item.chunk_type,
                    symbol_name=item.symbol_name,
                    qualified_name=item.qualified_name,
                    start_line=item.start_line,
                    end_line=item.end_line,
                    content=item.content,
                    content_hash=item.content_hash,
                )
                by_identity[identity] = result
            result.retrieval_sources.append("semantic")
            result.semantic_score = item.semantic_score
            result.semantic_rank = semantic_rank

        results = list(by_identity.values())
        _check(check_active)
        for result in results:
            if result.lexical_rank is not None:
                result.fusion_score += LEXICAL_WEIGHT / (RRF_K + result.lexical_rank)
            if result.semantic_rank is not None:
                result.fusion_score += SEMANTIC_WEIGHT / (RRF_K + result.semantic_rank)
        results.sort(
            key=lambda item: (
                -item.fusion_score,
                min(
                    rank
                    for rank in (item.lexical_rank, item.semantic_rank)
                    if rank is not None
                ),
                item.path,
                item.start_line,
                item.end_line,
                item.code_chunk_id,
            )
        )
        for rank, result in enumerate(results, start=1):
            result.fusion_rank = rank
        mode = "hybrid" if valid_semantic else "lexical"
        detail["fusion_ms"] = int((time.monotonic() - fusion_started) * 1000)
        detail["retrieval_source"] = mode
        if diagnostics_recorder is not None:
            diagnostics_recorder.record_retrieval_detail(detail)
        return HybridSearchOutcome(
            results=results[:limit],
            retrieval_mode=mode,
            warnings=_deduplicate(warnings),
        )


def _check(callback: Callable[[], None] | None) -> None:
    if callback is not None:
        callback()


def run_bounded_semantic(
    embedding_service: EmbeddingService,
    work: Callable[[], Any],
    *,
    deadline_at: float,
    check_active: Callable[[], None] | None,
) -> tuple[str, Any | None, int | None]:
    """Stop waiting before the tool cutoff; native work may finish later."""
    _check(check_active)
    remaining = deadline_at - time.monotonic()
    reserve = min(1.0, max(0.02, remaining * 0.1))
    if remaining <= reserve:
        return "skipped_budget", None, None
    if not hasattr(embedding_service, "submit_semantic_work"):
        return "capacity", None, None
    try:
        future = embedding_service.submit_semantic_work(work)
    except SemanticCapacityUnavailable:
        return "capacity", None, None
    cutoff = deadline_at - reserve
    started = time.monotonic()
    while True:
        _check(check_active)
        left = cutoff - time.monotonic()
        if left <= 0:
            return "timed_out", None, int((time.monotonic() - started) * 1000)
        try:
            outcome = future.result(timeout=min(0.05, left))
        except FutureTimeout:
            if future.done():
                raise
            continue
        _check(check_active)
        waited = int((time.monotonic() - started) * 1000)
        if time.monotonic() >= cutoff:
            return "timed_out", None, waited
        return "completed", outcome, waited


def _identity(item: Any) -> tuple[Any, ...]:
    return (
        item.project_id,
        item.repository_revision,
        item.path,
        item.start_line,
        item.end_line,
        item.content_hash,
    )


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
