from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from app.database import Database
from app.services.embedding_service import EmbeddingService
from app.services.lexical_retriever import LexicalRetriever
from app.services.semantic_retriever import SemanticRetriever


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
    ) -> HybridSearchOutcome:
        limit = min(MAXIMUM_EVIDENCE_COUNT, max(1, int(evidence_count)))
        lexical = self.lexical_retriever.search(
            project_id,
            query,
            top_k=LEXICAL_CANDIDATE_COUNT,
            path=path,
            language=language,
            symbol=symbol,
        )
        warnings: list[str] = []
        semantic = []
        if self.embedding_service.settings.enabled:
            try:
                semantic_outcome = self.semantic_retriever.search(
                    project_id,
                    query,
                    top_k=SEMANTIC_CANDIDATE_COUNT,
                    path=path,
                    language=language,
                    symbol=symbol,
                    local_files_only=True,
                )
                semantic = semantic_outcome.results
                warnings.extend(semantic_outcome.warnings)
            except Exception as exc:
                warnings.append(
                    f"Semantic retrieval unavailable; using lexical code-chunk search: "
                    f"{type(exc).__name__}."
                )
        else:
            warnings.append(
                "Embeddings are disabled; using lexical code-chunk search."
            )

        current_revisions = {
            str(chunk["repository_revision"])
            for chunk in self.database.get_code_chunks(project_id)
        }
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
        return HybridSearchOutcome(
            results=results[:limit],
            retrieval_mode=mode,
            warnings=_deduplicate(warnings),
        )


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
