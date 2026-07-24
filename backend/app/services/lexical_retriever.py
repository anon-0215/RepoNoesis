from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
from pathlib import PurePosixPath
import re
from typing import Any, Iterable

from app.database import Database


BM25_K1 = 1.5
BM25_B = 0.75
DEFAULT_TOP_K = 20
MAX_TOP_K = 50
_RAW_TOKEN = re.compile(r"\w+", re.UNICODE)
_CAMEL_BOUNDARY = re.compile(
    r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"
)
_CJK_RUN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+"
)


@dataclass(frozen=True)
class LexicalSearchResult:
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
    lexical_score: float
    lexical_rank: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def tokenize_code_text(text: str) -> list[str]:
    """Tokenize source metadata and questions with one deterministic algorithm."""
    tokens: list[str] = []
    for match in _RAW_TOKEN.finditer(text):
        raw = match.group(0)
        variants = [raw]
        for underscore_part in raw.split("_"):
            if not underscore_part:
                continue
            variants.append(underscore_part)
            variants.extend(_CAMEL_BOUNDARY.split(underscore_part))

        seen: set[str] = set()
        for variant in variants:
            normalized = variant.casefold()
            if normalized and normalized not in seen:
                tokens.append(normalized)
                seen.add(normalized)
            for cjk_match in _CJK_RUN.finditer(normalized):
                run = cjk_match.group(0)
                for cjk_token in _cjk_tokens(run):
                    if cjk_token not in seen:
                        tokens.append(cjk_token)
                        seen.add(cjk_token)
    return tokens


class LexicalRetriever:
    def __init__(
        self,
        database: Database,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> None:
        self.database = database
        self.k1 = float(k1)
        self.b = float(b)

    def search(
        self,
        project_id: str,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        path: str | None = None,
        language: str | None = None,
        symbol: str | None = None,
    ) -> list[LexicalSearchResult]:
        query_terms = tokenize_code_text(query)
        if not query_terms:
            return []
        chunks = self.database.get_code_chunks(
            project_id,
            path=path,
            symbol=symbol,
        )
        if language:
            expected_language = language.casefold()
            chunks = [
                chunk
                for chunk in chunks
                if str(chunk.get("language", "")).casefold() == expected_language
            ]
        if not chunks:
            return []

        documents = [tokenize_code_text(_searchable_text(chunk)) for chunk in chunks]
        document_frequencies = _document_frequencies(documents)
        average_length = sum(len(document) for document in documents) / len(documents)
        query_counts = Counter(query_terms)
        scored: list[tuple[float, dict[str, Any]]] = []
        for chunk, document in zip(chunks, documents):
            frequencies = Counter(document)
            score = 0.0
            for term, query_frequency in query_counts.items():
                term_frequency = frequencies.get(term, 0)
                if term_frequency == 0:
                    continue
                inverse_document_frequency = math.log(
                    1.0
                    + (
                        len(documents)
                        - document_frequencies[term]
                        + 0.5
                    )
                    / (document_frequencies[term] + 0.5)
                )
                length_ratio = len(document) / average_length if average_length else 0.0
                denominator = term_frequency + self.k1 * (
                    1.0 - self.b + self.b * length_ratio
                )
                score += (
                    query_frequency
                    * inverse_document_frequency
                    * term_frequency
                    * (self.k1 + 1.0)
                    / denominator
                )
            if score > 0.0 and math.isfinite(score):
                scored.append((score, chunk))

        scored.sort(
            key=lambda item: (
                -item[0],
                item[1]["path"],
                int(item[1]["start_line"]),
                int(item[1]["end_line"]),
                int(item[1]["id"]),
            )
        )
        results: list[LexicalSearchResult] = []
        for rank, (score, chunk) in enumerate(
            scored[: min(MAX_TOP_K, max(1, int(top_k)))],
            start=1,
        ):
            results.append(
                LexicalSearchResult(
                    project_id=project_id,
                    repository_revision=chunk["repository_revision"],
                    code_chunk_id=int(chunk["id"]),
                    language=chunk["language"],
                    path=chunk["path"],
                    chunk_type=chunk["chunk_type"],
                    symbol_name=chunk["symbol_name"],
                    qualified_name=chunk["qualified_name"],
                    start_line=int(chunk["start_line"]),
                    end_line=int(chunk["end_line"]),
                    content=chunk["content"],
                    content_hash=chunk["content_hash"],
                    lexical_score=float(score),
                    lexical_rank=rank,
                )
            )
        return results


def _searchable_text(chunk: dict[str, Any]) -> str:
    path = str(chunk.get("path", ""))
    filename = PurePosixPath(path).name
    content = str(chunk.get("content", ""))
    signature = _signature_text(content)
    return "\n".join(
        [
            path,
            filename,
            str(chunk.get("symbol_name", "")),
            str(chunk.get("qualified_name", "")),
            signature,
            content,
        ]
    )


def _signature_text(content: str) -> str:
    lines = content.splitlines()
    if not lines:
        return ""
    signature_lines: list[str] = []
    for line in lines:
        signature_lines.append(line)
        if line.rstrip().endswith(":"):
            break
        if len(signature_lines) >= 20:
            break
    return "\n".join(signature_lines)


def _cjk_tokens(run: str) -> Iterable[str]:
    yield run
    yield from run
    yield from (run[index : index + 2] for index in range(len(run) - 1))


def _document_frequencies(documents: list[list[str]]) -> Counter[str]:
    frequencies: Counter[str] = Counter()
    for document in documents:
        frequencies.update(set(document))
    return frequencies
