from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import re
from typing import Any, Literal

from app.database import Database
from app.services.query_analyzer import SymbolHint, extract_symbol_query_hints


SymbolMatchMode = Literal["auto", "exact", "prefix", "fuzzy"]
DEFAULT_TOP_K = 20
MAX_TOP_K = 50


@dataclass(frozen=True)
class SymbolSearchResult:
    project_id: str
    repository_revision: str
    code_chunk_id: int
    chunk_identity: str
    language: str
    path: str
    chunk_type: str
    symbol_name: str
    qualified_name: str
    start_line: int
    end_line: int
    content: str
    content_hash: str
    symbol_match_type: str
    symbol_score: float
    symbol_rank: int
    match_reasons: tuple[str, ...]
    candidate_source: str = "symbol"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["match_reasons"] = list(self.match_reasons)
        return value


class SymbolRetriever:
    """Search the persisted AST chunk index without maintaining a second index."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def search(
        self,
        project_id: str,
        query: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        path: str | None = None,
        language: str | None = None,
        match_mode: SymbolMatchMode = "auto",
        explicit_symbol: bool = False,
        repository_revision: str | None = None,
    ) -> list[SymbolSearchResult]:
        if match_mode not in {"auto", "exact", "prefix", "fuzzy"}:
            raise ValueError("unknown symbol match mode")
        explicit_hint = _explicit_hint(query) if explicit_symbol else None
        hints = (
            (explicit_hint,)
            if explicit_hint is not None
            else extract_symbol_query_hints(query)
        )
        hints = tuple(hint for hint in hints if hint is not None)
        if not hints:
            return []

        chunks = self.database.get_code_chunks(
            project_id,
            path=path,
            language=language,
        )
        ranked: list[tuple[tuple[Any, ...], SymbolSearchResult]] = []
        for chunk in chunks:
            if repository_revision is not None and str(
                chunk["repository_revision"]
            ) != repository_revision:
                continue
            best: tuple[int, float, tuple[str, ...], str] | None = None
            for hint in hints:
                match = _match_chunk(chunk, hint, match_mode)
                if match is None:
                    continue
                if best is None or (match[0], -match[1], match[3]) < (
                    best[0],
                    -best[1],
                    best[3],
                ):
                    best = match
            if best is None:
                continue
            tier, score, reasons, match_type = best
            result = SymbolSearchResult(
                project_id=project_id,
                repository_revision=str(chunk["repository_revision"]),
                code_chunk_id=int(chunk["id"]),
                chunk_identity=_chunk_identity(project_id, chunk),
                language=str(chunk["language"]),
                path=str(chunk["path"]),
                chunk_type=str(chunk["chunk_type"]),
                symbol_name=str(chunk["symbol_name"]),
                qualified_name=str(chunk["qualified_name"]),
                start_line=int(chunk["start_line"]),
                end_line=int(chunk["end_line"]),
                content=str(chunk["content"]),
                content_hash=str(chunk["content_hash"]),
                symbol_match_type=match_type,
                symbol_score=score,
                symbol_rank=0,
                match_reasons=reasons,
            )
            sort_key = (
                tier,
                -score,
                result.qualified_name.casefold(),
                result.path,
                result.start_line,
                result.end_line,
                result.code_chunk_id,
            )
            ranked.append((sort_key, result))

        ranked.sort(key=lambda item: item[0])
        limit = min(MAX_TOP_K, max(1, int(top_k)))
        return [
            replace(result, symbol_rank=rank)
            for rank, (_sort_key, result) in enumerate(ranked[:limit], start=1)
        ]


def _explicit_hint(value: str) -> SymbolHint | None:
    cleaned = str(value).strip().strip("`\"'").removesuffix("()").strip()
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", cleaned) is None:
        return None
    reasons = ["explicit_symbol_input"]
    if "." in cleaned:
        reasons.append("qualified_symbol_detected")
    return SymbolHint(cleaned, reason_codes=tuple(reasons))


def _match_chunk(
    chunk: dict[str, Any],
    hint: SymbolHint,
    match_mode: SymbolMatchMode,
) -> tuple[int, float, tuple[str, ...], str] | None:
    if hint.path is not None and _normalize_path(str(chunk["path"])) != _normalize_path(
        hint.path
    ):
        return None
    needle = hint.value.casefold()
    symbol = str(chunk["symbol_name"]).casefold()
    qualified = str(chunk["qualified_name"]).casefold()
    leaf = needle.rsplit(".", 1)[-1]
    path_reason = ("path_context_exact",) if hint.path is not None else ()

    if needle == qualified:
        return 0, 1.0, ("qualified_symbol_exact", *path_reason), "exact_qualified"
    if needle == symbol:
        if hint.path is not None:
            return 1, 0.95, ("symbol_name_exact", *path_reason), "exact_symbol_context"
        return 2, 0.85, ("leaf_symbol_exact",), "exact_leaf"
    if match_mode == "auto" and leaf == symbol:
        return 2, 0.85, ("leaf_symbol_exact", *path_reason), "exact_leaf"

    normalized_needle = _normalize_identifier(needle)
    if normalized_needle and normalized_needle in {
        _normalize_identifier(qualified),
        _normalize_identifier(symbol),
    }:
        return 3, 0.75, ("normalized_identifier_match", *path_reason), "normalized_identifier"
    if match_mode in {"prefix", "fuzzy"} and any(
        candidate.startswith(needle) for candidate in (qualified, symbol)
    ):
        return 4, 0.55, ("symbol_prefix_match", *path_reason), "lexical_prefix"
    if match_mode == "fuzzy" and any(
        needle in candidate for candidate in (qualified, symbol)
    ):
        return 5, 0.35, ("symbol_contains_match", *path_reason), "lexical_contains"
    return None


def _chunk_identity(project_id: str, chunk: dict[str, Any]) -> str:
    return "|".join(
        [
            project_id,
            str(chunk["repository_revision"]),
            str(chunk["path"]),
            str(chunk["start_line"]),
            str(chunk["end_line"]),
            str(chunk["content_hash"]),
            str(chunk["id"]),
        ]
    )


def _normalize_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _normalize_path(value: str) -> str:
    return value.replace("\\", "/").lstrip("/").casefold()
