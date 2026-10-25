from __future__ import annotations

from dataclasses import dataclass, field, replace
import ast
import hashlib
import math
import textwrap
from typing import Literal

from app.database import Database
from app.services.evidence import Evidence, EvidenceBuilder
from app.services.hybrid_retriever import HybridSearchResult


CandidateProvenance = Literal[
    "deterministic_base_retrieval",
    "planner_search_code",
    "planner_lookup_symbol",
    "planner_read_source",
]
CandidateRejectionCode = Literal[
    "project_mismatch",
    "repository_revision_mismatch",
    "chunk_not_found",
    "chunk_identity_mismatch",
    "path_mismatch",
    "language_mismatch",
    "symbol_mismatch",
    "span_mismatch",
    "content_hash_mismatch",
    "retrieval_sources_invalid",
    "score_invalid",
]

CANDIDATE_REJECTION_CODES: tuple[CandidateRejectionCode, ...] = (
    "project_mismatch",
    "repository_revision_mismatch",
    "chunk_not_found",
    "chunk_identity_mismatch",
    "path_mismatch",
    "language_mismatch",
    "symbol_mismatch",
    "span_mismatch",
    "content_hash_mismatch",
    "retrieval_sources_invalid",
    "score_invalid",
)

_PROVENANCE_ORDER: tuple[CandidateProvenance, ...] = (
    "deterministic_base_retrieval",
    "planner_search_code",
    "planner_lookup_symbol",
    "planner_read_source",
)
_MAX_RETRIEVAL_SOURCES = 8
_RETRIEVAL_SOURCE_ORDER = (
    "dense",
    "lexical",
    "semantic",
    "symbol",
    "hierarchy",
    "relation",
)
_DERIVED_SENTINEL_SOURCE_SHAPES = (
    ("hierarchy",),
    ("relation",),
)


@dataclass(frozen=True)
class CandidateEvidence:
    """A pending source candidate, not formal Evidence and not citation-eligible.

    This request-owned object intentionally retains only bounded locator, identity,
    score, and provenance fields. Source content is re-read from the authoritative
    code-chunk store only during promotion.
    """

    project_id: str
    repository_revision: str
    code_chunk_id: int
    chunk_identity: str
    path: str
    language: str
    chunk_type: str
    symbol_name: str
    qualified_name: str
    start_line: int
    end_line: int
    content_hash: str
    provenance: tuple[CandidateProvenance, ...]
    retrieval_sources: tuple[str, ...]
    lexical_score: float | None
    lexical_rank: int | None
    semantic_score: float | None
    semantic_rank: int | None
    fusion_score: float
    fusion_rank: int
    normalization_rejection: CandidateRejectionCode | None
    derived_sentinel_source_shape: bool
    symbol_match_type: str | None = None
    match_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidatePromotion:
    evidence: list[Evidence]
    selected_candidates: list[CandidateEvidence]
    rejection_counts: dict[CandidateRejectionCode, int]
    candidate_count: int
    valid_candidate_count: int
    new_candidate_count: int
    accepted_candidate_count: int
    duplicate_candidate_count: int = 0
    capacity_rejected_count: int = 0
    admission_deferred_count: int = 0
    selection_audit: list[dict[str, object]] = field(default_factory=list)


@dataclass
class CandidateEvidencePool:
    """One request-local, bounded owner for CandidateEvidence normalization and promotion."""

    capacity: int
    _accepted_by_identity: dict[str, CandidateEvidence] = field(
        default_factory=dict,
        init=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.capacity, int) or isinstance(self.capacity, bool):
            raise ValueError("CandidateEvidencePool capacity must be an integer")
        if self.capacity < 0:
            raise ValueError("CandidateEvidencePool capacity must be non-negative")

    def normalize_retrieval_results(
        self,
        results: list[HybridSearchResult],
        *,
        provenance: CandidateProvenance,
    ) -> list[CandidateEvidence]:
        """Normalize bounded retrieval output without retaining its source content."""

        prepared = [
            _candidate_from_retrieval(item, provenance=provenance)
            for item in results
        ]
        # Normalization is intentionally stateless. Only a candidate that has
        # passed the bound canonical lookup may consume the request-level pool.
        return sorted(prepared, key=_candidate_sort_key)

    def normalize_symbol_results(
        self,
        results: list[object],
        *,
        provenance: CandidateProvenance,
    ) -> list[CandidateEvidence]:
        """Normalize symbol locators while retaining their claimed identity."""

        prepared = [
            _candidate_from_symbol_result(item, provenance=provenance)
            for item in results
        ]
        return sorted(prepared, key=_candidate_sort_key)

    def promote(
        self,
        candidates: list[CandidateEvidence],
        *,
        database: Database,
        project: dict[str, object],
        project_id: str,
        repository_revision: str,
        retrieval_strategy_version: str,
        accepted_limit: int | None = None,
        new_evidence_limit: int | None = None,
        reserved_slots: int = 0,
        held_chunk_identities: set[str] | None = None,
        directed_read_evidence: list[Evidence] | None = None,
        directed_read_scope_chunks: list[dict[str, object]] | None = None,
    ) -> CandidatePromotion:
        """Rebuild from canonical chunks before considering owned duplicates.

        Product callers supply the current request-owned EvidenceStore snapshot.
        Pool metadata alone is insufficient when another tool acquired Evidence.
        Callers without a snapshot retain the standalone pool contract.
        """

        rejection_counts: dict[CandidateRejectionCode, int] = {}
        valid_candidates: list[CandidateEvidence] = []
        for candidate in candidates:
            rejection = _bound_candidate_rejection(
                candidate,
                project_id=project_id,
                repository_revision=repository_revision,
            )
            if rejection is None:
                valid_candidates.append(candidate)
            else:
                _record_rejection(rejection_counts, rejection)

        candidate_ids = sorted({candidate.code_chunk_id for candidate in valid_candidates})
        canonical_by_id = (
            {
                int(chunk["id"]): chunk
                for chunk in database.get_code_chunks_by_ids_bounded(
                    project_id,
                    repository_revision,
                    candidate_ids,
                    limit=max(1, len(candidate_ids)),
                )
            }
            if candidate_ids
            else {}
        )
        canonical_valid: list[CandidateEvidence] = []
        for candidate in valid_candidates:
            chunk = canonical_by_id.get(candidate.code_chunk_id)
            rejection = _canonical_rejection(candidate, chunk)
            if rejection is not None:
                _record_rejection(rejection_counts, rejection)
                continue
            canonical_valid.append(candidate)

        merged_for_call: dict[str, CandidateEvidence] = {}
        for candidate in sorted(canonical_valid, key=_candidate_sort_key):
            identity = candidate.chunk_identity
            existing = merged_for_call.get(identity)
            merged_for_call[identity] = (
                candidate if existing is None else _merge_candidate_metadata(existing, candidate)
            )

        selected_candidates = sorted(merged_for_call.values(), key=_candidate_sort_key)
        priority_reasons: dict[str, tuple[str, ...]] = {}
        if directed_read_evidence is not None:
            priority_reasons = _directed_read_priorities(
                selected_candidates, canonical_by_id, directed_read_evidence,
                directed_read_scope_chunks or [], project_id, repository_revision,
            )
            # Retain the original scores/ranks. Static syntax changes only this
            # call's admission order, never retrieval or runtime relation facts.
            selected_candidates.sort(key=lambda c: (
                0 if c.chunk_identity in priority_reasons else 1,
                *_candidate_sort_key(c),
            ))
        if accepted_limit is not None:
            selected_candidates = selected_candidates[: max(0, accepted_limit)]

        if held_chunk_identities is None:
            occupied = set(self._accepted_by_identity)
        else:
            occupied = set(held_chunk_identities)
            # A promotion that never reached the Store must not occupy a later
            # call's capacity. The snapshot does not make a candidate valid:
            # all candidates above still underwent bound canonical validation.
            self._accepted_by_identity = {
                identity: candidate
                for identity, candidate in self._accepted_by_identity.items()
                if identity in occupied
            }

        newly_accepted: list[CandidateEvidence] = []
        duplicates = capacity_rejected = deferred = 0
        admission_capacity = max(0, self.capacity - max(0, reserved_slots))
        selection_audit: list[dict[str, object]] = []
        for candidate in selected_candidates:
            identity = candidate.chunk_identity
            audit = {"code_chunk_id": candidate.code_chunk_id,
                     "original_fusion_rank": candidate.fusion_rank,
                     "selection_position": len(selection_audit) + 1,
                     "selection_reason": "same_class_static_call_syntax" if identity in priority_reasons else "original_order_fallback",
                     "source_evidence_ids": list(priority_reasons.get(identity, ())),
                     "admission": "added"}
            if directed_read_evidence is not None:
                selection_audit.append(audit)
            existing = self._accepted_by_identity.get(identity)
            if identity in occupied:
                audit["admission"] = "duplicate"
                duplicates += 1
                self._accepted_by_identity[identity] = (
                    candidate if existing is None
                    else _merge_candidate_metadata(existing, candidate)
                )
                continue
            if len(occupied) >= self.capacity:
                audit["admission"] = "capacity_exhausted"
                capacity_rejected += 1
                continue
            if (len(occupied) >= admission_capacity
                or (new_evidence_limit is not None and len(newly_accepted) >= max(0, new_evidence_limit))):
                deferred += 1
                audit["admission"] = "admission_deferred"
                continue
            self._accepted_by_identity[identity] = candidate
            occupied.add(identity)
            newly_accepted.append(candidate)

        evidence: list[Evidence] = []
        for candidate in newly_accepted:
            chunk = canonical_by_id[candidate.code_chunk_id]
            canonical = _canonical_retrieval_result(candidate, chunk)
            evidence.extend(
                EvidenceBuilder().build(
                    [canonical],
                    project,
                    retrieval_strategy_version=retrieval_strategy_version,
                )
            )
            if candidate.chunk_identity in priority_reasons:
                evidence[-1].selection_reason += "; directed read priority: same-class static call syntax only, runtime target unresolved"
        return CandidatePromotion(
            evidence=evidence,
            selected_candidates=selected_candidates,
            rejection_counts=dict(sorted(rejection_counts.items())),
            candidate_count=len(candidates),
            valid_candidate_count=len(canonical_valid),
            new_candidate_count=len(newly_accepted),
            accepted_candidate_count=len(self._accepted_by_identity),
            duplicate_candidate_count=duplicates,
            capacity_rejected_count=capacity_rejected,
            admission_deferred_count=deferred,
            selection_audit=selection_audit,
        )

    def accepted_candidates(self) -> list[CandidateEvidence]:
        """Return a stable snapshot for bounded request-local auditing."""

        return sorted(self._accepted_by_identity.values(), key=_candidate_sort_key)


def _method_ast(content: str, symbol: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Conservative instance-method syntax; no aliases or nested scopes."""
    if len(content.encode("utf-8")) > 8192:
        return None
    try:
        tree = ast.parse(textwrap.dedent(content.replace("\r\n", "\n").replace("\r", "\n")))
    except (SyntaxError, ValueError, RecursionError):
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    method = tree.body[0]
    args = method.args.posonlyargs + method.args.args
    if method.name != symbol or not args or args[0].arg != "self" or method.decorator_list:
        return None
    # Rebinding self or introducing nested scopes makes lexical target matching
    # unsafe. Attribute writes are filtered separately for each call name.
    for statement in method.body:
        for node in ast.walk(statement):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                return None
            if isinstance(node, ast.Name) and node.id == "self" and isinstance(node.ctx, (ast.Store, ast.Del)):
                return None
            if (isinstance(node, ast.alias) and (node.asname or node.name.split(".", 1)[0]) == "self"
                or isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name == "self"
                or isinstance(node, ast.MatchMapping) and node.rest == "self"
                or isinstance(node, (ast.Global, ast.Nonlocal)) and "self" in node.names):
                return None
    return method


def _directed_read_priorities(
    candidates: list[CandidateEvidence], canonical: dict[int, dict[str, object]],
    held: list[Evidence], scope_chunks: list[dict[str, object]],
    project_id: str, revision: str,
) -> dict[str, tuple[str, ...]]:
    # A capped hierarchy cannot prove uniqueness. Scope metadata is used only
    # to veto ambiguity, never to validate or admit a candidate.
    if not scope_chunks or len(scope_chunks) >= 2049:
        return {}
    scopes: dict[tuple[str, str], set[int]] = {}
    for chunk in scope_chunks:
        key = (str(chunk.get("path", "")), str(chunk.get("qualified_name", "")))
        scopes.setdefault(key, set()).add(int(chunk["id"]))
    reasons: dict[str, set[str]] = {}
    for source in held:
        coverage = source.planner_excerpt_coverage()
        hints = source.planner_direct_call_hints()
        if (source.project_id != project_id or source.repository_revision != revision
            or source.chunk_type not in {"method", "async_method"}
            or coverage.get("status") != "known" or coverage.get("truncated") is not False
            or coverage.get("complete_end_line") != source.end_line
            or hints.get("status") != "available"
            or hashlib.sha256(source.excerpt.encode("utf-8")).hexdigest() != source.content_hash):
            continue
        owner, separator, _ = source.qualified_name.rpartition(".")
        if (not separator or len(scopes.get((source.path, source.qualified_name), ())) != 1
            or len(scopes.get((source.path, owner), ())) != 1):
            continue
        method = _method_ast(source.excerpt, source.symbol_name)
        if method is None:
            continue
        written_attributes = {n.attr for n in ast.walk(method) if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name) and n.value.id == "self" and isinstance(n.ctx, (ast.Store, ast.Del))}
        # Use the existing bounded trusted hint set, then require an exact call
        # in this method body. Do not resolve self dynamically or infer inheritance.
        expressions = {h["callee_expression"] for h in hints["calls"]}
        names = {n.func.attr for statement in method.body for n in ast.walk(statement)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name) and n.func.value.id == "self"
            and "self." + n.func.attr in expressions and n.func.attr not in written_attributes}
        for candidate in candidates:
            chunk = canonical[candidate.code_chunk_id]
            if (candidate.path != source.path or candidate.language.casefold() != "python"
                or "planner_read_source" not in candidate.provenance
                or str(chunk.get("chunk_type")) not in {"method", "async_method"}
                or candidate.symbol_name not in names
                or candidate.qualified_name != owner + "." + candidate.symbol_name
                or len(scopes.get((candidate.path, candidate.qualified_name), ())) != 1
                or _method_ast(str(chunk["content"]), candidate.symbol_name) is None):
                continue
            reasons.setdefault(candidate.chunk_identity, set()).add(source.evidence_id)
    return {identity: tuple(sorted(ids)) for identity, ids in reasons.items()}


def _candidate_from_retrieval(
    item: HybridSearchResult,
    *,
    provenance: CandidateProvenance,
) -> CandidateEvidence:
    (
        retrieval_sources,
        normalization_rejection,
        derived_sentinel_source_shape,
    ) = _normalize_retrieval_sources(item.retrieval_sources)
    return CandidateEvidence(
        project_id=item.project_id,
        repository_revision=item.repository_revision,
        code_chunk_id=item.code_chunk_id,
        chunk_identity=_chunk_identity(
            item.project_id,
            item.repository_revision,
            item.path,
            item.start_line,
            item.end_line,
            item.content_hash,
            item.code_chunk_id,
        ),
        path=item.path,
        language=item.language,
        chunk_type=item.chunk_type,
        symbol_name=item.symbol_name,
        qualified_name=item.qualified_name,
        start_line=item.start_line,
        end_line=item.end_line,
        content_hash=item.content_hash,
        provenance=(provenance,),
        retrieval_sources=retrieval_sources,
        lexical_score=item.lexical_score,
        lexical_rank=item.lexical_rank,
        semantic_score=item.semantic_score,
        semantic_rank=item.semantic_rank,
        fusion_score=item.fusion_score,
        fusion_rank=item.fusion_rank,
        normalization_rejection=normalization_rejection,
        derived_sentinel_source_shape=derived_sentinel_source_shape,
    )


def _candidate_from_symbol_result(
    item: object,
    *,
    provenance: CandidateProvenance,
) -> CandidateEvidence:
    raw_score = getattr(item, "symbol_score", None)
    raw_rank = getattr(item, "symbol_rank", None)
    # Symbol locators are untrusted until canonical promotion. Preserve the
    # original score/rank values so bools, strings, and non-finite numbers
    # cannot be converted into apparently valid metadata before rejection.
    ranking_rejection = (
        None
        if _valid_score(raw_score, optional=False)
        and _valid_rank(raw_rank, optional=False)
        else "score_invalid"
    )
    return CandidateEvidence(
        project_id=str(item.project_id),
        repository_revision=str(item.repository_revision),
        code_chunk_id=_raw_int_or_sentinel(getattr(item, "code_chunk_id", None)),
        chunk_identity=str(item.chunk_identity),
        path=str(item.path),
        language=str(item.language),
        chunk_type=str(item.chunk_type),
        symbol_name=str(item.symbol_name),
        qualified_name=str(item.qualified_name),
        start_line=_raw_int_or_sentinel(getattr(item, "start_line", None)),
        end_line=_raw_int_or_sentinel(getattr(item, "end_line", None)),
        content_hash=str(item.content_hash),
        provenance=(provenance,),
        retrieval_sources=("symbol",),
        lexical_score=None,
        lexical_rank=None,
        semantic_score=None,
        semantic_rank=None,
        fusion_score=raw_score,
        fusion_rank=raw_rank,
        normalization_rejection=ranking_rejection,
        derived_sentinel_source_shape=False,
        symbol_match_type=(
            item.symbol_match_type
            if isinstance(getattr(item, "symbol_match_type", None), str)
            else None
        ),
        match_reasons=tuple(
            reason
            for reason in getattr(item, "match_reasons", ())
            if isinstance(reason, str)
        )[:8],
    )


def _raw_int_or_sentinel(value: object) -> int:
    """Keep malformed locator integers out of conversion-driven acceptance."""

    return value if isinstance(value, int) and not isinstance(value, bool) else -1


def _normalize_retrieval_sources(
    value: object,
) -> tuple[tuple[str, ...], CandidateRejectionCode | None, bool]:
    """Return a bounded canonical source tuple and preserve exact sentinel shape."""

    if not isinstance(value, (list, tuple)) or not value:
        return (), "retrieval_sources_invalid", False
    if len(value) > _MAX_RETRIEVAL_SOURCES:
        return (), "retrieval_sources_invalid", False
    raw: list[str] = []
    for item in value:
        if not isinstance(item, str) or item not in _RETRIEVAL_SOURCE_ORDER:
            return (), "retrieval_sources_invalid", False
        raw.append(item)
    raw_shape = tuple(raw)
    normalized = tuple(
        source for source in _RETRIEVAL_SOURCE_ORDER if source in raw_shape
    )
    return (
        normalized,
        None,
        raw_shape in _DERIVED_SENTINEL_SOURCE_SHAPES,
    )


def _candidate_sort_key(candidate: CandidateEvidence) -> tuple[int, float, str]:
    rank = (
        candidate.fusion_rank
        if _valid_rank(candidate.fusion_rank, optional=False)
        else 2**31 - 1
    )
    score = candidate.fusion_score
    score_key = (
        -float(score)
        if _valid_score(score, optional=False)
        else math.inf
    )
    return (rank, score_key, str(candidate.chunk_identity))


def _candidate_metadata_sort_key(candidate: CandidateEvidence) -> tuple[object, ...]:
    """Choose one best metadata record independently of arrival order."""

    return (
        *_candidate_sort_key(candidate),
        _optional_rank_key(candidate.lexical_rank),
        _optional_score_key(candidate.lexical_score),
        _optional_rank_key(candidate.semantic_rank),
        _optional_score_key(candidate.semantic_score),
        candidate.retrieval_sources,
        candidate.provenance,
    )


def _merge_candidate_metadata(
    left: CandidateEvidence,
    right: CandidateEvidence,
) -> CandidateEvidence:
    if left.chunk_identity != right.chunk_identity:
        raise ValueError("candidate identities must match before metadata merge")
    best = min((left, right), key=_candidate_metadata_sort_key)
    provenance = tuple(
        value
        for value in _PROVENANCE_ORDER
        if value in {*left.provenance, *right.provenance}
    )
    if best.fusion_rank == 0:
        retrieval_sources = best.retrieval_sources
    else:
        merged_source_input = [
            source
            for source in _RETRIEVAL_SOURCE_ORDER
            if source in left.retrieval_sources or source in right.retrieval_sources
        ]
        retrieval_sources, rejection, _sentinel_shape = _normalize_retrieval_sources(
            merged_source_input
        )
        if rejection is not None:
            raise ValueError("accepted candidate retrieval sources must remain bounded")
    return replace(
        best,
        provenance=provenance,
        retrieval_sources=retrieval_sources,
    )


def _optional_rank_key(value: int | None) -> int:
    return value if _valid_rank(value, optional=True) and value is not None else 2**31 - 1


def _optional_score_key(value: float | None) -> float:
    return -float(value) if _valid_score(value, optional=True) and value is not None else math.inf


def _bound_candidate_rejection(
    candidate: CandidateEvidence,
    *,
    project_id: str,
    repository_revision: str,
) -> CandidateRejectionCode | None:
    if candidate.normalization_rejection is not None:
        return candidate.normalization_rejection
    if candidate.project_id != project_id:
        return "project_mismatch"
    if candidate.repository_revision != repository_revision:
        return "repository_revision_mismatch"
    if not _ranking_metadata_is_valid(candidate):
        return "score_invalid"
    return None


def _ranking_metadata_is_valid(candidate: CandidateEvidence) -> bool:
    # Retrieval v2's hierarchy/relation phases use one explicit, frozen
    # sentinel for server-derived candidates: no source score/rank and
    # fusion=(0.0, 0). Accept only that complete shape. Direct retrieval
    # candidates still cannot use a zero rank to bypass validation.
    if _is_zero_rank_sentinel(candidate.fusion_rank):
        return (
            candidate.derived_sentinel_source_shape
            and candidate.retrieval_sources in _DERIVED_SENTINEL_SOURCE_SHAPES
            and _valid_score(candidate.fusion_score, optional=False)
            and candidate.fusion_score == 0
            and candidate.lexical_score is None
            and candidate.lexical_rank is None
            and candidate.semantic_score is None
            and candidate.semantic_rank is None
        )
    return (
        _valid_score(candidate.fusion_score, optional=False)
        and _valid_rank(candidate.fusion_rank, optional=False)
        and _valid_score(candidate.lexical_score, optional=True)
        and _valid_rank(candidate.lexical_rank, optional=True)
        and _valid_score(candidate.semantic_score, optional=True)
        and _valid_rank(candidate.semantic_rank, optional=True)
    )


def _is_zero_rank_sentinel(value: object) -> bool:
    """Accept only the non-bool integer rank used by derived candidates."""
    return isinstance(value, int) and not isinstance(value, bool) and value == 0


def _valid_score(value: object, *, optional: bool) -> bool:
    if value is None:
        return optional
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _valid_rank(value: object, *, optional: bool) -> bool:
    if value is None:
        return optional
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _canonical_retrieval_result(
    candidate: CandidateEvidence,
    chunk: dict[str, object],
) -> HybridSearchResult:
    return HybridSearchResult(
        project_id=str(chunk["project_id"]),
        repository_revision=str(chunk["repository_revision"]),
        code_chunk_id=int(chunk["id"]),
        language=str(chunk["language"]),
        path=str(chunk["path"]),
        chunk_type=str(chunk["chunk_type"]),
        symbol_name=str(chunk["symbol_name"]),
        qualified_name=str(chunk["qualified_name"]),
        start_line=int(chunk["start_line"]),
        end_line=int(chunk["end_line"]),
        content=str(chunk["content"]),
        content_hash=str(chunk["content_hash"]),
        retrieval_sources=list(candidate.retrieval_sources),
        lexical_score=candidate.lexical_score,
        lexical_rank=candidate.lexical_rank,
        semantic_score=candidate.semantic_score,
        semantic_rank=candidate.semantic_rank,
        fusion_score=candidate.fusion_score,
        fusion_rank=candidate.fusion_rank,
    )


def _canonical_rejection(
    candidate: CandidateEvidence,
    chunk: dict[str, object] | None,
) -> CandidateRejectionCode | None:
    if chunk is None:
        return "chunk_not_found"
    if candidate.path != str(chunk["path"]):
        return "path_mismatch"
    if candidate.language.casefold() != str(chunk["language"]).casefold():
        return "language_mismatch"
    if candidate.symbol_name != str(chunk["symbol_name"]) or candidate.qualified_name != str(
        chunk["qualified_name"]
    ):
        return "symbol_mismatch"
    if candidate.start_line != int(chunk["start_line"]) or candidate.end_line != int(
        chunk["end_line"]
    ):
        return "span_mismatch"
    if candidate.content_hash != str(chunk["content_hash"]):
        return "content_hash_mismatch"
    expected_identity = _chunk_identity(
        str(chunk["project_id"]),
        str(chunk["repository_revision"]),
        str(chunk["path"]),
        int(chunk["start_line"]),
        int(chunk["end_line"]),
        str(chunk["content_hash"]),
        int(chunk["id"]),
    )
    if candidate.chunk_identity != expected_identity:
        return "chunk_identity_mismatch"
    return None


def _chunk_identity(
    project_id: str,
    repository_revision: str,
    path: str,
    start_line: int,
    end_line: int,
    content_hash: str,
    code_chunk_id: int,
) -> str:
    return "|".join(
        [
            project_id,
            repository_revision,
            path,
            str(start_line),
            str(end_line),
            content_hash,
            str(code_chunk_id),
        ]
    )


def _record_rejection(
    counts: dict[CandidateRejectionCode, int],
    code: CandidateRejectionCode,
) -> None:
    counts[code] = counts.get(code, 0) + 1
