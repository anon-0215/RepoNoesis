from __future__ import annotations

from app.source_lines import source_lines as physical_source_lines

import ast
from dataclasses import asdict, dataclass, field, replace
import hashlib
from pathlib import PurePosixPath, PureWindowsPath
import textwrap
from typing import Any

from app.database import Database
from app.services.hybrid_retriever import HybridSearchResult


EVIDENCE_SCHEMA_VERSION = 1
DEFAULT_EXCERPT_LIMIT = 2000
RETRIEVAL_STRATEGY_VERSION = "weighted-rrf-v1"
MAX_DIRECT_CALL_HINTS_PER_EVIDENCE = 4
MAX_DIRECT_CALL_HINT_SOURCE_BYTES = 8192
MAX_DIRECT_CALL_EXPRESSION_CHARS = 160
MAX_DIRECT_CALL_EXPRESSION_BYTES = 512


@dataclass(frozen=True)
class _ExcerptCoverage:
    """Request-local proof for one exact prefix; never a public/history field."""

    start_line: int
    end_line: int
    excerpt_sha256: str
    complete_end_line: int | None
    partial_end_line: int | None
    truncated: bool
    source_identity: tuple[Any, ...] | None = None


@dataclass
class Evidence:
    evidence_id: str
    project_id: str
    repository_id: str
    repository_url: str
    repository_revision: str
    path: str
    language: str
    code_chunk_id: int
    chunk_identity: str
    chunk_type: str
    symbol_name: str
    qualified_name: str
    start_line: int
    end_line: int
    content_hash: str
    excerpt: str
    retrieval_sources: list[str]
    lexical_score: float | None
    lexical_rank: int | None
    semantic_score: float | None
    semantic_rank: int | None
    fusion_score: float
    fusion_rank: int
    selection_reason: str
    validation_status: str = "unvalidated"
    invalid_reason: str | None = None
    retrieval_strategy_version: str = RETRIEVAL_STRATEGY_VERSION
    _excerpt_coverage: _ExcerptCoverage | None = field(
        default=None, repr=False, compare=False,
    )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("_excerpt_coverage")
        return value

    def planner_excerpt_coverage(self) -> dict[str, Any]:
        proof = self._excerpt_coverage
        if (
            proof is None
            or proof.start_line != self.start_line
            or proof.end_line != self.end_line
            or not isinstance(self.excerpt, str)
            or proof.excerpt_sha256 != hashlib.sha256(self.excerpt.encode("utf-8")).hexdigest()
        ):
            # Legacy Evidence or a changed excerpt cannot inherit an old proof.
            return {"status": "unknown"}
        return {
            "status": "known",
            "start_line": proof.start_line,
            "complete_end_line": proof.complete_end_line,
            "partial_end_line": proof.partial_end_line,
            "truncated": proof.truncated,
        }

    def _planner_source_identity(self) -> tuple[Any, ...]:
        # Store-assigned Evidence IDs may change; canonical source identity may not.
        return (self.project_id, self.repository_id, self.repository_url,
                self.repository_revision, self.path, self.language, self.code_chunk_id,
                self.chunk_identity, self.chunk_type, self.symbol_name, self.qualified_name,
                self.start_line, self.end_line, self.content_hash)

    def _bind_planner_source(self) -> None:
        if self._excerpt_coverage is not None:
            self._excerpt_coverage = replace(
                self._excerpt_coverage, source_identity=self._planner_source_identity(),
            )

    def planner_direct_call_hints(self, limit: int = MAX_DIRECT_CALL_HINTS_PER_EVIDENCE) -> dict[str, Any]:
        """One-hop syntax from complete acquired lines, never target Evidence.

        No resolution or execution is attempted. Recompute from the exact prefix;
        private source binding prevents a changed identity retaining an old proof.
        """
        def unknown(reason: str) -> dict[str, Any]:
            return {"status": "unknown", "reason": reason,
                    "calls": [], "may_be_incomplete": True}

        proof = self._excerpt_coverage
        if (self.planner_excerpt_coverage()["status"] != "known" or proof is None
            or proof.source_identity != self._planner_source_identity()):
            return unknown("source_proof_unavailable_or_changed")
        if self.language.casefold() != "python":
            return {"status": "unavailable", "reason": "unsupported_language",
                    "calls": [], "may_be_incomplete": True}
        if len(self.excerpt.encode("utf-8")) > MAX_DIRECT_CALL_HINT_SOURCE_BYTES:
            return unknown("analysis_input_limit")
        if proof.complete_end_line is None:
            return unknown("no_complete_source_lines")
        line_count = proof.complete_end_line - self.start_line + 1
        # Never parse a partial final line, even if its prefix looks like a call.
        # Normalize only the AST analysis copy. CRLF blank lines otherwise
        # contribute a zero-indent margin to textwrap.dedent. Original text,
        # identity/hash and complete-line proof remain unchanged.
        analysis_text = "".join(physical_source_lines(self.excerpt, keepends=True)[:line_count])
        source = textwrap.dedent(analysis_text.replace("\r\n", "\n").replace("\r", "\n"))
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError, RecursionError):
            return unknown("parse_unavailable")
        calls = []
        incomplete = proof.truncated
        limit = max(0, min(MAX_DIRECT_CALL_HINTS_PER_EVIDENCE, limit))
        nodes = sorted((n for n in ast.walk(tree) if isinstance(n, ast.Call)),
                       key=lambda n: (n.lineno, n.col_offset))
        for node in nodes:
            if node.end_lineno is None or node.end_lineno > line_count:
                incomplete = True
                continue
            expression = ast.get_source_segment(source, node.func)
            if (not expression or len(expression) > MAX_DIRECT_CALL_EXPRESSION_CHARS
                or len(expression.encode("utf-8")) > MAX_DIRECT_CALL_EXPRESSION_BYTES):
                incomplete = True
                continue
            if len(calls) >= limit:
                incomplete = True
                break
            calls.append({
                "callee_expression": expression,
                "start_line": self.start_line + node.lineno - 1,
                "end_line": self.start_line + node.end_lineno - 1,
                "resolution_status": "unresolved",
            })
        return {"status": "available", "calls": calls,
                "may_be_incomplete": incomplete}


def _excerpt_coverage(
    content: str, excerpt: str, start_line: int, end_line: int,
) -> _ExcerptCoverage | None:
    """A complete line includes its existing terminator (CRLF is indivisible).

    A final source line without a terminator is complete only if fully present.
    A cutoff exactly after a complete line has no partial line. Empty or
    inconsistent source/location data yields unknown coverage.
    """
    lines = physical_source_lines(content, keepends=True)
    if (
        type(start_line) is not int or type(end_line) is not int
        or start_line < 1 or end_line < start_line
        or len(lines) != end_line - start_line + 1
        or not excerpt or not content.startswith(excerpt)
    ):
        return None
    offset = 0
    complete_end = partial_end = None
    for number, line in enumerate(lines, start_line):
        if len(excerpt) >= offset + len(line):
            complete_end = number
        elif len(excerpt) > offset:
            partial_end = number
            break
        else:
            break
        offset += len(line)
    return _ExcerptCoverage(
        start_line=start_line, end_line=end_line,
        excerpt_sha256=hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
        complete_end_line=complete_end, partial_end_line=partial_end,
        truncated=len(excerpt) < len(content),
    )


class EvidenceBuilder:
    def __init__(self, excerpt_limit: int = DEFAULT_EXCERPT_LIMIT) -> None:
        self.excerpt_limit = max(1, int(excerpt_limit))

    def build(
        self,
        candidates: list[HybridSearchResult],
        project: dict[str, Any],
        *,
        retrieval_strategy_version: str = RETRIEVAL_STRATEGY_VERSION,
    ) -> list[Evidence]:
        repository_id = _repository_id(project)
        repository_url = str(project.get("repo_url", ""))
        evidence: list[Evidence] = []
        for index, candidate in enumerate(candidates, start=1):
            source_names = "+".join(candidate.retrieval_sources)
            hierarchy_only = candidate.retrieval_sources == ["hierarchy"]
            relation_only = candidate.retrieval_sources == ["relation"]
            evidence.append(
                Evidence(
                    evidence_id=f"E{index}",
                    project_id=candidate.project_id,
                    repository_id=repository_id,
                    repository_url=repository_url,
                    repository_revision=candidate.repository_revision,
                    path=candidate.path,
                    language=candidate.language,
                    code_chunk_id=candidate.code_chunk_id,
                    chunk_identity=_chunk_identity(candidate),
                    chunk_type=candidate.chunk_type,
                    symbol_name=candidate.symbol_name,
                    qualified_name=candidate.qualified_name,
                    start_line=candidate.start_line,
                    end_line=candidate.end_line,
                    content_hash=candidate.content_hash,
                    excerpt=candidate.content[: self.excerpt_limit],
                    retrieval_sources=list(candidate.retrieval_sources),
                    lexical_score=candidate.lexical_score,
                    lexical_rank=candidate.lexical_rank,
                    semantic_score=candidate.semantic_score,
                    semantic_rank=candidate.semantic_rank,
                    fusion_score=candidate.fusion_score,
                    fusion_rank=candidate.fusion_rank,
                    selection_reason=(
                        "Selected by versioned chunk hierarchy normalization from "
                        "a real stored chunk; no direct retrieval score was assigned."
                        if hierarchy_only
                        else (
                            "Selected by versioned relation-aware Evidence selection "
                            "from a real stored chunk; no direct retrieval score was assigned."
                            if relation_only
                            else (
                                f"Selected at fusion rank {candidate.fusion_rank} "
                                f"from {source_names} retrieval."
                            )
                        )
                    ),
                    retrieval_strategy_version=retrieval_strategy_version,
                    _excerpt_coverage=_excerpt_coverage(
                        candidate.content, candidate.content[: self.excerpt_limit],
                        candidate.start_line, candidate.end_line,
                    ),
                )
            )
        for item in evidence:
            item._bind_planner_source()
        return evidence

    def build_from_code_chunks(
        self,
        chunks: list[dict[str, Any]],
        project: dict[str, Any],
        *,
        selection_reason: str = "Selected by validated static relation expansion.",
    ) -> list[Evidence]:
        repository_id = _repository_id(project)
        repository_url = str(project.get("repo_url", ""))
        evidence: list[Evidence] = []
        for index, chunk in enumerate(chunks, start=1):
            content = str(chunk["content"])
            evidence.append(
                Evidence(
                    evidence_id=f"E{index}",
                    project_id=str(chunk["project_id"]),
                    repository_id=repository_id,
                    repository_url=repository_url,
                    repository_revision=str(chunk["repository_revision"]),
                    path=str(chunk["path"]),
                    language=str(chunk["language"]),
                    code_chunk_id=int(chunk["id"]),
                    chunk_identity="|".join(
                        [
                            str(chunk["project_id"]),
                            str(chunk["repository_revision"]),
                            str(chunk["path"]),
                            str(chunk["start_line"]),
                            str(chunk["end_line"]),
                            str(chunk["content_hash"]),
                            str(chunk["id"]),
                        ]
                    ),
                    chunk_type=str(chunk["chunk_type"]),
                    symbol_name=str(chunk["symbol_name"]),
                    qualified_name=str(chunk["qualified_name"]),
                    start_line=int(chunk["start_line"]),
                    end_line=int(chunk["end_line"]),
                    content_hash=str(chunk["content_hash"]),
                    excerpt=content[: self.excerpt_limit],
                    retrieval_sources=["relation"],
                    lexical_score=None,
                    lexical_rank=None,
                    semantic_score=None,
                    semantic_rank=None,
                    fusion_score=0.0,
                    fusion_rank=index,
                    selection_reason=selection_reason,
                    retrieval_strategy_version="relation-expansion-v1",
                    _excerpt_coverage=_excerpt_coverage(
                        content, content[: self.excerpt_limit],
                        int(chunk["start_line"]), int(chunk["end_line"]),
                    ),
                )
            )
        for item in evidence:
            item._bind_planner_source()
        return evidence


class CitationValidator:
    def __init__(self, database: Database) -> None:
        self.database = database

    def validate(self, evidence: Evidence) -> Evidence:
        invalid_reason = self._invalid_reason(evidence)
        evidence.validation_status = "valid" if invalid_reason is None else "invalid"
        evidence.invalid_reason = invalid_reason
        if invalid_reason is not None:
            evidence.excerpt = ""
            evidence._excerpt_coverage = None
        return evidence

    def validate_all(
        self,
        evidence: list[Evidence],
    ) -> tuple[list[Evidence], list[str]]:
        valid: list[Evidence] = []
        warnings: list[str] = []
        for item in evidence:
            validated = self.validate(item)
            if validated.validation_status == "valid":
                valid.append(validated)
            else:
                warnings.append(
                    f"Evidence {validated.evidence_id} was rejected: "
                    f"{validated.invalid_reason}."
                )
        return valid, warnings

    def _invalid_reason(self, evidence: Evidence) -> str | None:
        if evidence.project_id == "" or evidence.code_chunk_id < 1:
            return "invalid chunk identity"
        if not _is_safe_relative_path(evidence.path):
            return "unsafe repository path"
        source = self.database.get_evidence_source(
            evidence.project_id,
            evidence.code_chunk_id,
            evidence.path,
        )
        if source is None:
            return "source file or code chunk no longer exists"
        if evidence.repository_id != _repository_id(source):
            return "repository identity mismatch"
        if evidence.repository_url != str(source["repo_url"]):
            return "repository URL mismatch"
        if evidence.repository_revision != str(source["repository_revision"]):
            return "repository revision mismatch"
        if evidence.path != str(source["chunk_path"]):
            return "repository path mismatch"
        if evidence.language.casefold() != str(source["chunk_language"]).casefold():
            return "language mismatch"
        if evidence.start_line < 1 or evidence.end_line < evidence.start_line:
            return "invalid line range"

        file_lines = physical_source_lines(str(source["file_content"]), keepends=True)
        if evidence.end_line > len(file_lines):
            return "line range exceeds stored source"
        current_content = "".join(
            file_lines[evidence.start_line - 1 : evidence.end_line]
        )
        if current_content != str(source["chunk_content"]):
            return "stored source no longer matches code chunk"
        current_hash = hashlib.sha256(current_content.encode("utf-8")).hexdigest()
        if evidence.content_hash != current_hash:
            return "content hash mismatch"
        if evidence.content_hash != str(source["content_hash"]):
            return "code chunk hash mismatch"
        if evidence.symbol_name != str(source["symbol_name"]):
            return "symbol identity mismatch"
        if evidence.qualified_name != str(source["qualified_name"]):
            return "qualified symbol identity mismatch"
        if evidence.start_line != int(source["start_line"]) or evidence.end_line != int(
            source["end_line"]
        ):
            return "code chunk line identity mismatch"
        expected_identity = _chunk_identity_from_source(source)
        if evidence.chunk_identity != expected_identity:
            return "chunk identity mismatch"
        if not current_content.startswith(evidence.excerpt):
            return "excerpt mismatch"
        evidence._excerpt_coverage = _excerpt_coverage(
            current_content, evidence.excerpt, evidence.start_line, evidence.end_line,
        )
        evidence._bind_planner_source()
        return None


def _repository_id(project: dict[str, Any]) -> str:
    owner = str(project.get("owner", ""))
    repo = str(project.get("repo", ""))
    return f"{owner}/{repo}" if owner or repo else ""


def _chunk_identity(candidate: HybridSearchResult) -> str:
    return "|".join(
        [
            candidate.project_id,
            candidate.repository_revision,
            candidate.path,
            str(candidate.start_line),
            str(candidate.end_line),
            candidate.content_hash,
            str(candidate.code_chunk_id),
        ]
    )


def _chunk_identity_from_source(source: dict[str, Any]) -> str:
    return "|".join(
        [
            str(source["project_id"]),
            str(source["repository_revision"]),
            str(source["chunk_path"]),
            str(source["start_line"]),
            str(source["end_line"]),
            str(source["content_hash"]),
            str(source["code_chunk_id"]),
        ]
    )


def _is_safe_relative_path(path: str) -> bool:
    if not path or "\\" in path:
        return False
    posix = PurePosixPath(path)
    windows = PureWindowsPath(path)
    if posix.is_absolute() or windows.is_absolute():
        return False
    if any(part in {"", ".", ".."} for part in posix.parts):
        return False
    return str(posix) == path
