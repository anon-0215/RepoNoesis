from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import PurePosixPath, PureWindowsPath
import time
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from app.database import Database
from app.services.agent_contracts import (
    AgentLimits,
    CancellationToken,
    LookupSymbolInput,
    ReadSourceInput,
    SearchCodeInput,
    ToolCall,
    ToolObservation,
    ValidateEvidenceInput,
    utc_now,
)
from app.services.embedding_service import EmbeddingService
from app.services.evidence import CitationValidator, Evidence, EvidenceBuilder
from app.services.hybrid_retriever import HybridRetriever


ToolHandler = Callable[["ToolContext", BaseModel], tuple[Any, list[str], bool]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    version: str
    description: str
    input_model: type[BaseModel]
    handler: ToolHandler
    timeout_ms: int
    max_results: int
    max_bytes: int


@dataclass
class EvidenceStore:
    _items: dict[str, Evidence] = field(default_factory=dict)
    _owners: dict[str, str] = field(default_factory=dict)
    _next_id: int = 1

    def add(self, owner_id: str, evidence: list[Evidence]) -> list[Evidence]:
        added: list[Evidence] = []
        known_identities = {item.chunk_identity for item in self._items.values()}
        for item in evidence:
            if item.chunk_identity in known_identities:
                continue
            item.evidence_id = f"E{self._next_id}"
            self._next_id += 1
            self._items[item.evidence_id] = item
            self._owners[item.evidence_id] = owner_id
            known_identities.add(item.chunk_identity)
            added.append(item)
        return added

    def get_many(self, owner_id: str, evidence_ids: list[str]) -> list[Evidence]:
        return [
            self._items[evidence_id]
            for evidence_id in evidence_ids
            if self._owners.get(evidence_id) == owner_id and evidence_id in self._items
        ]

    def all(self, owner_id: str) -> list[Evidence]:
        return [
            item
            for evidence_id, item in self._items.items()
            if self._owners.get(evidence_id) == owner_id
        ]


@dataclass
class ToolContext:
    request_id: str
    project_id: str
    repository_id: str
    repository_url: str
    repository_revision: str
    bundle: dict[str, Any]
    database: Database
    embedding_service: EmbeddingService
    evidence_store: EvidenceStore
    limits: AgentLimits
    cancellation: CancellationToken
    deadline_monotonic: float

    def check_active(self) -> None:
        if self.cancellation.cancelled:
            raise ToolCancelled("request was cancelled")
        if time.monotonic() >= self.deadline_monotonic:
            raise ToolDeadlineExceeded("request deadline was reached")


class ToolError(Exception):
    code = "tool_error"


class ToolCancelled(ToolError):
    code = "cancelled"


class ToolDeadlineExceeded(ToolError):
    code = "deadline_exceeded"


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name]

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "version": spec.version,
                "description": spec.description,
                "input_schema": spec.input_model.model_json_schema(),
            }
            for spec in sorted(self._tools.values(), key=lambda item: item.name)
        ]

    def execute(
        self,
        context: ToolContext,
        call: ToolCall,
    ) -> ToolObservation:
        started = time.monotonic()
        call.started_at = utc_now()
        call.status = "running"
        try:
            context.check_active()
            spec = self.get(call.tool_name)
            if call.tool_version != spec.version:
                raise ToolError("unsupported tool version")
            try:
                parameters = spec.input_model.model_validate(call.parameters)
            except ValidationError as exc:
                return self._finish_error(
                    call,
                    started,
                    "rejected",
                    "invalid_parameters",
                    _safe_validation_message(exc),
                )
            results, warnings, handler_truncated = spec.handler(context, parameters)
            context.check_active()
            serialized, size, serialization_truncated = _bounded_serialization(
                results,
                min(spec.max_bytes, context.limits.max_observation_bytes),
                spec.max_results,
            )
            duration_ms = int((time.monotonic() - started) * 1000)
            if duration_ms > min(call.timeout_ms, spec.timeout_ms):
                return self._finish_error(
                    call,
                    started,
                    "timed_out",
                    "tool_timeout",
                    "tool exceeded its cooperative timeout",
                )
            call.status = "succeeded"
            call.ended_at = utc_now()
            return ToolObservation(
                call_id=call.call_id,
                status="succeeded",
                structured_results=serialized,
                warnings=warnings,
                truncated=handler_truncated or serialization_truncated,
                metrics={
                    "duration_ms": duration_ms,
                    "result_count": _result_count(serialized),
                    "output_bytes": size,
                    "timeout_enforcement": "cooperative",
                },
            )
        except KeyError:
            return self._finish_error(
                call, started, "rejected", "unknown_tool", "tool is not registered"
            )
        except ToolCancelled:
            return self._finish_error(
                call, started, "cancelled", "cancelled", "request was cancelled"
            )
        except ToolDeadlineExceeded:
            return self._finish_error(
                call, started, "timed_out", "deadline_exceeded", "request deadline was reached"
            )
        except ToolError as exc:
            return self._finish_error(
                call, started, "failed", exc.code, _safe_error_message(str(exc))
            )
        except Exception as exc:
            return self._finish_error(
                call,
                started,
                "failed",
                "tool_failed",
                f"tool failed with {type(exc).__name__}",
            )

    @staticmethod
    def _finish_error(
        call: ToolCall,
        started: float,
        status: str,
        code: str,
        message: str,
    ) -> ToolObservation:
        call.status = status
        call.ended_at = utc_now()
        return ToolObservation(
            call_id=call.call_id,
            status=status,
            error={"code": code, "message": message[:300]},
            metrics={
                "duration_ms": int((time.monotonic() - started) * 1000),
                "result_count": 0,
                "output_bytes": 0,
                "timeout_enforcement": "cooperative",
            },
        )


def build_m2_tool_registry(limits: AgentLimits) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="search_code",
            version="1",
            description="Search bound repository code chunks and create Evidence.",
            input_model=SearchCodeInput,
            handler=_search_code,
            timeout_ms=limits.default_tool_timeout_ms,
            max_results=limits.max_search_results,
            max_bytes=limits.max_observation_bytes,
        )
    )
    registry.register(
        ToolSpec(
            name="lookup_symbol",
            version="1",
            description="Look up symbol definitions in the bound AST chunk index.",
            input_model=LookupSymbolInput,
            handler=_lookup_symbol,
            timeout_ms=limits.default_tool_timeout_ms,
            max_results=limits.max_search_results,
            max_bytes=limits.max_observation_bytes,
        )
    )
    registry.register(
        ToolSpec(
            name="read_source",
            version="1",
            description="Read a bounded line range from the stored repository snapshot.",
            input_model=ReadSourceInput,
            handler=_read_source,
            timeout_ms=limits.default_tool_timeout_ms,
            max_results=1,
            max_bytes=limits.max_observation_bytes,
        )
    )
    registry.register(
        ToolSpec(
            name="validate_evidence",
            version="1",
            description="Validate request-owned Evidence IDs against the current snapshot.",
            input_model=ValidateEvidenceInput,
            handler=_validate_evidence,
            timeout_ms=limits.default_tool_timeout_ms,
            max_results=limits.max_search_results,
            max_bytes=limits.max_observation_bytes,
        )
    )
    return registry


def build_tool_context(
    *,
    request_id: str,
    bundle: dict[str, Any],
    database: Database,
    embedding_service: EmbeddingService,
    evidence_store: EvidenceStore,
    limits: AgentLimits,
    cancellation: CancellationToken,
    deadline_monotonic: float,
) -> ToolContext:
    project = bundle.get("project") or {}
    project_id = str(project.get("id", ""))
    chunks = database.get_code_chunks(project_id)
    revisions = {str(chunk.get("repository_revision", "")) for chunk in chunks}
    if len(revisions) != 1 or "" in revisions:
        raise ValueError("project must have exactly one bound repository revision")
    repository_id = f"{project.get('owner', '')}/{project.get('repo', '')}".strip("/")
    return ToolContext(
        request_id=request_id,
        project_id=project_id,
        repository_id=repository_id,
        repository_url=str(project.get("repo_url", "")),
        repository_revision=next(iter(revisions)),
        bundle=bundle,
        database=database,
        embedding_service=embedding_service,
        evidence_store=evidence_store,
        limits=limits,
        cancellation=cancellation,
        deadline_monotonic=deadline_monotonic,
    )


def _search_code(
    context: ToolContext,
    parameters: BaseModel,
) -> tuple[Any, list[str], bool]:
    values = SearchCodeInput.model_validate(parameters)
    context.check_active()
    outcome = HybridRetriever(context.database, context.embedding_service).search(
        context.project_id,
        values.query,
        evidence_count=min(values.top_k, context.limits.max_search_results),
        path=values.path,
        language=values.language,
        symbol=values.symbol,
    )
    project = context.bundle.get("project") or {}
    built = EvidenceBuilder().build(outcome.results, project)
    built = [
        item
        for item in built
        if item.project_id == context.project_id
        and item.repository_revision == context.repository_revision
    ]
    added = context.evidence_store.add(context.request_id, built)
    results = [_evidence_summary(item) for item in added]
    return (
        {
            "evidence": results,
            "retrieval_mode": outcome.retrieval_mode,
            "grounding": "unvalidated",
            "retrieval_metrics": {
                "candidate_count": len(outcome.results),
                "new_evidence_count": len(added),
            },
        },
        outcome.warnings,
        len(outcome.results) > context.limits.max_search_results,
    )


def _lookup_symbol(
    context: ToolContext,
    parameters: BaseModel,
) -> tuple[Any, list[str], bool]:
    values = LookupSymbolInput.model_validate(parameters)
    context.check_active()
    chunks = context.database.get_code_chunks(
        context.project_id,
        path=values.path,
        language=values.language,
    )
    needle = values.symbol.casefold()
    ranked: list[tuple[int, dict[str, Any]]] = []
    for chunk in chunks:
        if str(chunk["repository_revision"]) != context.repository_revision:
            continue
        symbol = str(chunk["symbol_name"])
        qualified = str(chunk["qualified_name"])
        candidates = (symbol.casefold(), qualified.casefold())
        score: int | None = None
        if needle == candidates[1]:
            score = 0
        elif needle == candidates[0]:
            score = 1
        elif values.match_mode in {"prefix", "fuzzy"} and any(
            candidate.startswith(needle) for candidate in candidates
        ):
            score = 2
        elif values.match_mode == "fuzzy" and any(
            needle in candidate for candidate in candidates
        ):
            score = 3
        if score is not None:
            ranked.append((score, chunk))
    ranked.sort(
        key=lambda item: (
            item[0],
            item[1]["path"],
            int(item[1]["start_line"]),
            item[1]["qualified_name"],
            int(item[1]["id"]),
        )
    )
    limit = min(values.top_k, context.limits.max_search_results)
    results = [
        {
            "code_chunk_id": int(chunk["id"]),
            "chunk_identity": _chunk_identity_from_chunk(context.project_id, chunk),
            "symbol_name": chunk["symbol_name"],
            "qualified_name": chunk["qualified_name"],
            "symbol_kind": chunk["chunk_type"],
            "path": chunk["path"],
            "language": chunk["language"],
            "start_line": int(chunk["start_line"]),
            "end_line": int(chunk["end_line"]),
            "repository_revision": chunk["repository_revision"],
            "content_hash": chunk["content_hash"],
        }
        for _score, chunk in ranked[:limit]
    ]
    return results, [], len(ranked) > limit


def _read_source(
    context: ToolContext,
    parameters: BaseModel,
) -> tuple[Any, list[str], bool]:
    values = ReadSourceInput.model_validate(parameters)
    if not _is_safe_relative_path(values.path):
        raise ToolError("unsafe repository path")
    if values.end_line < values.start_line:
        raise ToolError("end_line must not be less than start_line")
    requested_lines = values.end_line - values.start_line + 1
    if requested_lines > context.limits.max_source_read_lines:
        raise ToolError("requested line range exceeds server limit")
    context.check_active()
    before = _snapshot_file(context, values.path)
    lines = str(before["content"]).splitlines(keepends=True)
    if values.start_line > len(lines) or values.end_line > len(lines):
        raise ToolError("requested line range exceeds stored source")
    content = "".join(lines[values.start_line - 1 : values.end_line])
    encoded = content.encode("utf-8")
    truncated = False
    if len(encoded) > context.limits.max_source_read_bytes:
        encoded = encoded[: context.limits.max_source_read_bytes]
        content = encoded.decode("utf-8", errors="ignore")
        truncated = True
    after = _snapshot_file(context, values.path)
    if _file_identity(before) != _file_identity(after):
        raise ToolError("stored source changed during read")
    context.check_active()
    return (
        {
            "path": values.path,
            "start_line": values.start_line,
            "end_line": values.end_line,
            "repository_revision": context.repository_revision,
            "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "source_identity_hash": _file_identity(before),
            "content": content,
        },
        [],
        truncated,
    )


def _validate_evidence(
    context: ToolContext,
    parameters: BaseModel,
) -> tuple[Any, list[str], bool]:
    values = ValidateEvidenceInput.model_validate(parameters)
    owned = context.evidence_store.get_many(context.request_id, values.evidence_ids)
    owned_ids = {item.evidence_id for item in owned}
    forged = [value for value in values.evidence_ids if value not in owned_ids]
    valid, warnings = CitationValidator(context.database).validate_all(owned)
    valid_ids = {item.evidence_id for item in valid}
    results = [
        {
            "evidence_id": evidence_id,
            "validated": evidence_id in valid_ids,
            "invalid": evidence_id not in valid_ids,
            "invalid_reason": (
                "evidence ID is not owned by this request"
                if evidence_id in forged
                else next(
                    (
                        item.invalid_reason
                        for item in owned
                        if item.evidence_id == evidence_id
                    ),
                    None,
                )
            ),
            "revision_valid": evidence_id in valid_ids,
            "path_valid": evidence_id in valid_ids,
            "hash_valid": evidence_id in valid_ids,
            "line_range_valid": evidence_id in valid_ids,
        }
        for evidence_id in values.evidence_ids
    ]
    if forged:
        warnings.append("One or more Evidence IDs were rejected as unknown to this request.")
    return results, warnings, False


def _snapshot_file(context: ToolContext, path: str) -> dict[str, Any]:
    bundle = context.database.get_bundle(context.project_id)
    if not bundle:
        raise ToolError("bound project no longer exists")
    chunks = bundle.get("code_chunks", [])
    revisions = {str(chunk.get("repository_revision", "")) for chunk in chunks}
    if revisions != {context.repository_revision}:
        raise ToolError("bound repository revision changed")
    matches = [file for file in bundle.get("files", []) if file.get("path") == path]
    if len(matches) != 1:
        raise ToolError("source file does not exist in the stored snapshot")
    return matches[0]


def _file_identity(file: dict[str, Any]) -> str:
    return hashlib.sha256(
        (
            str(file.get("path", ""))
            + "\0"
            + str(file.get("language", ""))
            + "\0"
            + str(file.get("content", ""))
        ).encode("utf-8")
    ).hexdigest()


def _evidence_summary(item: Evidence) -> dict[str, Any]:
    return {
        "evidence_id": item.evidence_id,
        "chunk_identity": item.chunk_identity,
        "path": item.path,
        "language": item.language,
        "symbol_name": item.symbol_name,
        "qualified_name": item.qualified_name,
        "start_line": item.start_line,
        "end_line": item.end_line,
        "repository_revision": item.repository_revision,
        "content_hash": item.content_hash,
        "retrieval_sources": list(item.retrieval_sources),
        "fusion_rank": item.fusion_rank,
    }


def _chunk_identity_from_chunk(project_id: str, chunk: dict[str, Any]) -> str:
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


def _is_safe_relative_path(path: str) -> bool:
    if not path or "\\" in path:
        return False
    posix = PurePosixPath(path)
    windows = PureWindowsPath(path)
    return (
        not posix.is_absolute()
        and not windows.is_absolute()
        and all(part not in {"", ".", ".."} for part in posix.parts)
        and str(posix) == path
    )


def _bounded_serialization(
    value: Any,
    max_bytes: int,
    max_results: int,
) -> tuple[Any, int, bool]:
    truncated = False
    if isinstance(value, list) and len(value) > max_results:
        value = value[:max_results]
        truncated = True
    elif isinstance(value, dict):
        value = dict(value)
        for key, child in list(value.items()):
            if isinstance(child, list) and len(child) > max_results:
                value[key] = child[:max_results]
                truncated = True
    try:
        encoded = json.dumps(value, ensure_ascii=False, default=_reject_json).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ToolError("tool result is not serializable") from exc
    if len(encoded) <= max_bytes:
        return value, len(encoded), truncated
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        content = value["content"]
        while content and len(encoded) > max_bytes:
            content = content[: max(0, len(content) // 2)]
            value["content"] = content
            encoded = json.dumps(value, ensure_ascii=False).encode("utf-8")
        truncated = True
    elif isinstance(value, list):
        while value and len(encoded) > max_bytes:
            value = value[:-1]
            encoded = json.dumps(value, ensure_ascii=False).encode("utf-8")
        truncated = True
    elif isinstance(value, dict):
        for key in sorted(value, reverse=True):
            if len(encoded) <= max_bytes:
                break
            if isinstance(value[key], list) and value[key]:
                value[key] = value[key][:-1]
                truncated = True
                encoded = json.dumps(value, ensure_ascii=False).encode("utf-8")
    if len(encoded) > max_bytes:
        raise ToolError("tool result exceeds byte limit and cannot be safely truncated")
    return value, len(encoded), truncated


def _result_count(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        for key in ("evidence", "results"):
            if isinstance(value.get(key), list):
                return len(value[key])
        return 1 if value else 0
    return 1 if value is not None else 0


def _reject_json(value: Any) -> Any:
    raise TypeError(f"unsupported type: {type(value).__name__}")


def _safe_validation_message(exc: ValidationError) -> str:
    fields = sorted(
        {
            ".".join(str(part) for part in error.get("loc", ()))
            for error in exc.errors()
        }
    )
    return "invalid fields: " + ", ".join(fields)


def _safe_error_message(message: str) -> str:
    lowered = message.casefold()
    if any(value in lowered for value in ("api_key", "authorization", "token=")):
        return "tool request failed"
    return message[:300]
