from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import time
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from app.database import Database
from app.services.agent_contracts import (
    AgentLimits,
    CancellationToken,
    ExpandRelationsInput,
    GetLearningContextInput,
    LookupSymbolInput,
    ReadSourceInput,
    SearchCodeInput,
    ToolCall,
    ToolObservation,
    RequestBudget,
    ValidateEvidenceInput,
    normalize_repository_relative_path,
    utc_now,
)
from app.services.embedding_service import EmbeddingService
from app.services.evidence import CitationValidator, Evidence, EvidenceBuilder
from app.services.hierarchy_normalization import (
    HIERARCHY_MODE_OFF,
    validate_hierarchy_mode,
)
from app.services.relation_graph import (
    EvidenceChainStore,
    RelationGraphService,
    RelationPath,
    RelationTraversalLimits,
)
from app.services.relation_retrieval import (
    RELATION_MODE_EXPAND_V1,
    RELATION_MODE_OFF,
    validate_relation_mode,
)
from app.services.retrieval_v2 import (
    RETRIEVAL_VERSION_V1,
    retrieve_code,
    validate_retrieval_version,
)
from app.services.symbol_retriever import SymbolRetriever
from app.services.smoke_diagnostics import SmokeDiagnosticsRecorder


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


class EvidenceStore:
    """Request-owned Evidence with one immutable, deduplicated capacity."""

    def __init__(self, capacity: int = 20) -> None:
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 0:
            raise ValueError("EvidenceStore capacity must be a non-negative integer")
        object.__setattr__(self, "_capacity", capacity)
        self._items: dict[str, Evidence] = {}
        self._owners: dict[str, str] = {}
        self._next_id = 1

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_capacity" and hasattr(self, "_capacity"):
            raise AttributeError("EvidenceStore capacity is immutable")
        object.__setattr__(self, name, value)

    @property
    def capacity(self) -> int:
        return self._capacity

    def add(self, owner_id: str, evidence: list[Evidence]) -> list[Evidence]:
        added: list[Evidence] = []
        known_identities = {item.chunk_identity for item in self._items.values()}
        remaining = max(0, self.capacity - len(self.all(owner_id)))
        for item in evidence:
            if item.chunk_identity in known_identities:
                continue
            if remaining <= 0:
                break
            item.evidence_id = f"E{self._next_id}"
            self._next_id += 1
            self._items[item.evidence_id] = item
            self._owners[item.evidence_id] = owner_id
            known_identities.add(item.chunk_identity)
            added.append(item)
            remaining -= 1
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
    chain_store: EvidenceChainStore
    limits: AgentLimits
    cancellation: CancellationToken
    deadline_monotonic: float
    work_deadline_monotonic: float
    diagnostics_recorder: SmokeDiagnosticsRecorder | None = None
    active_tool_deadline_monotonic: float | None = None
    active_tool_deadline_reason: str | None = None
    learning_context: dict[str, Any] = field(default_factory=dict)
    retrieval_version: str = RETRIEVAL_VERSION_V1
    hierarchy_mode: str = HIERARCHY_MODE_OFF
    relation_mode: str = RELATION_MODE_OFF

    def check_active(self) -> None:
        if self.cancellation.cancelled:
            raise ToolCancelled("request was cancelled")
        now = time.monotonic()
        if now >= self.deadline_monotonic:
            raise ToolDeadlineExceeded(
                "deadline_exceeded", "request deadline was reached"
            )
        if now >= self.effective_deadline_monotonic:
            reason = self.active_tool_deadline_reason or "final_answer_not_attempted"
            message = {
                "tool_timeout": "tool cooperative timeout was reached",
                "final_answer_not_attempted": "work cutoff was reached",
            }.get(reason, "request deadline was reached")
            raise ToolDeadlineExceeded(reason, message)

    @property
    def effective_deadline_monotonic(self) -> float:
        if self.active_tool_deadline_monotonic is None:
            return min(self.deadline_monotonic, self.work_deadline_monotonic)
        return min(
            self.deadline_monotonic,
            self.work_deadline_monotonic,
            self.active_tool_deadline_monotonic,
        )

    def remaining_ms(self) -> int:
        return max(0, int((self.effective_deadline_monotonic - time.monotonic()) * 1000))


class ToolError(Exception):
    code = "tool_error"


class ToolCancelled(ToolError):
    code = "cancelled"


class ToolDeadlineExceeded(ToolError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


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
        try:
            spec = self.get(call.tool_name)
        except KeyError:
            spec = None
        return self._execute_resolved(context, call, spec)

    def execute_resolved(
        self,
        context: ToolContext,
        call: ToolCall,
        spec: ToolSpec,
    ) -> ToolObservation:
        """Execute a call with the request-local Registry resolution."""

        return self._execute_resolved(context, call, spec)

    def _execute_resolved(
        self,
        context: ToolContext,
        call: ToolCall,
        spec: ToolSpec | None,
    ) -> ToolObservation:
        started = time.monotonic()
        remaining_at_start = max(
            0, int((context.deadline_monotonic - started) * 1000)
        )
        timeout_ms = max(0, call.timeout_ms)
        if spec is not None:
            timeout_ms = min(timeout_ms, max(0, spec.timeout_ms))
        stage_deadline = RequestBudget.derive_tool_deadline(
            request_deadline_at=context.deadline_monotonic,
            work_cutoff_at=context.work_deadline_monotonic,
            started_at=started,
            timeout_ms=timeout_ms,
        )
        context.active_tool_deadline_monotonic = stage_deadline.deadline_monotonic
        context.active_tool_deadline_reason = stage_deadline.reason
        try:
            observation = self._execute_once(context, call, spec)
        finally:
            context.active_tool_deadline_monotonic = None
            context.active_tool_deadline_reason = None
        ended = time.monotonic()
        duration_ms = max(0, int((ended - started) * 1000))
        remaining_at_end = max(
            0, int((context.deadline_monotonic - ended) * 1000)
        )
        overrun_ms = max(0, int((ended - context.deadline_monotonic) * 1000))
        tool_overrun_ms = max(
            0, int((ended - stage_deadline.deadline_monotonic) * 1000)
        )
        observation.metrics.update(
            {
                "duration_ms": duration_ms,
                "deadline_remaining_at_start_ms": remaining_at_start,
                "deadline_remaining_at_end_ms": remaining_at_end,
                "deadline_overrun_ms": overrun_ms,
                "deadline_overrun": int(overrun_ms > 0),
                "tool_deadline_reason": stage_deadline.reason,
                "tool_deadline_overrun_ms": tool_overrun_ms,
                "tool_deadline_overrun": int(tool_overrun_ms > 0),
            }
        )
        if context.diagnostics_recorder is not None:
            context.diagnostics_recorder.record_stage_duration("tool", duration_ms)
            context.diagnostics_recorder.record_tool_deadline_overrun(
                tool_overrun_ms > 0,
                overrun_ms=tool_overrun_ms,
            )
            context.diagnostics_recorder.record_request_deadline_reached(
                ended >= context.deadline_monotonic
            )
            context.diagnostics_recorder.record_deadline_state(
                remaining_ms=remaining_at_end,
                overrun_ms=overrun_ms,
            )
        return observation

    def _execute_once(
        self,
        context: ToolContext,
        call: ToolCall,
        spec: ToolSpec | None,
    ) -> ToolObservation:
        started = time.monotonic()
        call.started_at = utc_now()
        call.status = "running"
        try:
            context.check_active()
            if spec is None:
                return self._finish_error(
                    call, started, "rejected", "unknown_tool", "tool is not registered"
                )
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
            handler_metrics: dict[str, int | float | str] = {}
            if isinstance(results, dict):
                results = dict(results)
                raw_metrics = results.pop("_observation_metrics", {})
                if isinstance(raw_metrics, dict):
                    handler_metrics = {
                        str(key): value
                        for key, value in raw_metrics.items()
                        if isinstance(value, (int, float, str))
                    }
            serialized, size, serialization_truncated = _bounded_serialization(
                results,
                min(spec.max_bytes, context.limits.max_observation_bytes),
                spec.max_results,
            )
            elapsed_ms = (time.monotonic() - started) * 1000
            duration_ms = int(elapsed_ms)
            if elapsed_ms >= min(call.timeout_ms, spec.timeout_ms):
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
                    **handler_metrics,
                },
            )
        except ToolCancelled:
            return self._finish_error(
                call, started, "cancelled", "cancelled", "request was cancelled"
            )
        except ToolDeadlineExceeded as exc:
            return self._finish_error(
                call, started, "timed_out", exc.code, str(exc)
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
            name="get_learning_context",
            version="1",
            description=(
                "Read the bounded, server-validated learner context already bound "
                "to this request. It is guidance, never repository Evidence."
            ),
            input_model=GetLearningContextInput,
            handler=_get_learning_context,
            timeout_ms=limits.default_tool_timeout_ms,
            max_results=limits.max_learning_state_items,
            max_bytes=limits.max_learning_context_bytes,
        )
    )
    registry.register(
        ToolSpec(
            name="expand_relations",
            version="1",
            description=(
                "Expand bounded static Python relations from request-owned "
                "Evidence or bound symbol nodes."
            ),
            input_model=ExpandRelationsInput,
            handler=_expand_relations,
            timeout_ms=limits.default_tool_timeout_ms,
            max_results=limits.max_relation_edges,
            max_bytes=limits.max_relation_observation_bytes,
        )
    )
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
    chain_store: EvidenceChainStore | None = None,
    limits: AgentLimits,
    cancellation: CancellationToken,
    deadline_monotonic: float,
    work_deadline_monotonic: float | None = None,
    learning_context: dict[str, Any] | None = None,
    retrieval_version: str = RETRIEVAL_VERSION_V1,
    hierarchy_mode: str = HIERARCHY_MODE_OFF,
    relation_mode: str = RELATION_MODE_OFF,
    diagnostics_recorder: SmokeDiagnosticsRecorder | None = None,
) -> ToolContext:
    project = bundle.get("project") or {}
    project_id = str(project.get("id", ""))
    chunks = database.get_code_chunks(project_id)
    revisions = {str(chunk.get("repository_revision", "")) for chunk in chunks}
    project_revision = str(project.get("repository_revision", ""))
    if project_revision:
        revisions.add(project_revision)
    revisions.discard("")
    if len(revisions) != 1:
        raise ValueError("project must have exactly one bound repository revision")
    repository_id = f"{project.get('owner', '')}/{project.get('repo', '')}".strip("/")
    retrieval_version = validate_retrieval_version(retrieval_version)
    hierarchy_mode = validate_hierarchy_mode(
        hierarchy_mode,
        retrieval_version=retrieval_version,
    )
    relation_mode = validate_relation_mode(
        relation_mode,
        retrieval_version=retrieval_version,
    )
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
        chain_store=chain_store or EvidenceChainStore(),
        limits=limits,
        cancellation=cancellation,
        deadline_monotonic=deadline_monotonic,
        work_deadline_monotonic=(
            deadline_monotonic
            if work_deadline_monotonic is None
            else work_deadline_monotonic
        ),
        diagnostics_recorder=diagnostics_recorder,
        learning_context=dict(learning_context or {}),
        retrieval_version=retrieval_version,
        hierarchy_mode=hierarchy_mode,
        relation_mode=relation_mode,
    )


def _get_learning_context(
    context: ToolContext,
    parameters: BaseModel,
) -> tuple[Any, list[str], bool]:
    GetLearningContextInput.model_validate(parameters)
    context.check_active()
    value = dict(context.learning_context)
    value.pop("project_binding", None)
    value["target_states"] = list(value.get("target_states") or [
    ])[: context.limits.max_learning_state_items]
    value["recent_verified_outcomes"] = list(
        value.get("recent_verified_outcomes") or []
    )[: context.limits.max_recent_learning_events]
    plan = value.get("current_plan")
    if isinstance(plan, dict):
        plan = dict(plan)
        plan["steps"] = list(plan.get("steps") or [
        ])[: context.limits.max_plan_steps_in_learning_context]
        value["current_plan"] = plan
    return value, list(value.get("warnings") or []), False


def _search_code(
    context: ToolContext,
    parameters: BaseModel,
) -> tuple[Any, list[str], bool]:
    values = SearchCodeInput.model_validate(parameters)
    context.check_active()
    outcome = retrieve_code(
        context.database,
        context.embedding_service,
        context.project_id,
        values.query,
        retrieval_version=context.retrieval_version,
        evidence_count=min(values.top_k, context.limits.max_search_results),
        path=values.path,
        language=values.language,
        symbol=values.symbol,
        hierarchy_mode=context.hierarchy_mode,
        relation_mode=context.relation_mode,
        check_active=context.check_active,
        diagnostics_recorder=context.diagnostics_recorder,
    )
    context.check_active()
    project = context.bundle.get("project") or {}
    built = EvidenceBuilder().build(
        outcome.results,
        project,
        retrieval_strategy_version=outcome.retrieval_strategy_version,
    )
    context.check_active()
    built = [
        item
        for item in built
        if item.project_id == context.project_id
        and item.repository_revision == context.repository_revision
    ]
    added = context.evidence_store.add(context.request_id, built)
    if context.relation_mode == RELATION_MODE_EXPAND_V1:
        relation_audit = outcome.audit.get("relation", {})
        selected_paths = relation_audit.get("selected_relation_paths", [])
        evidence_by_identity = {
            item.chunk_identity: item
            for item in context.evidence_store.all(context.request_id)
        }
        for item in selected_paths:
            if not isinstance(item, dict):
                continue
            seed = evidence_by_identity.get(str(item.get("seed_chunk_identity", "")))
            target = evidence_by_identity.get(str(item.get("target_chunk_identity", "")))
            if seed is None or target is None:
                continue
            edge_id = str(item.get("edge_id", ""))
            relation_type = str(item.get("relation_type", ""))
            direction = str(item.get("direction", ""))
            if not edge_id or not relation_type or direction not in {"incoming", "outgoing"}:
                continue
            context.chain_store.add(
                owner_id=context.request_id,
                project_id=context.project_id,
                repository_revision=context.repository_revision,
                seed_evidence_ids=[seed.evidence_id],
                supporting_evidence_ids=[target.evidence_id],
                path=RelationPath(
                    [str(item["seed_node_id"]), str(item["target_node_id"])],
                    [edge_id],
                    "resolved",
                ),
                edges_by_id={
                    edge_id: {"edge_id": edge_id, "relation_type": relation_type}
                },
                truncated=bool(relation_audit.get("truncated", False)),
                warnings=list(relation_audit.get("warnings", [])),
                contract_version="relation_expansion_v1@1",
                ordered_directions=[direction],
            )
    results = [_evidence_summary(item) for item in added]
    payload = {
        "evidence": results,
        "retrieval_mode": outcome.retrieval_mode,
        "grounding": "unvalidated",
        "retrieval_metrics": {
            "candidate_count": len(outcome.results),
            "new_evidence_count": len(added),
        },
        "_observation_metrics": {
            "result_count": len(outcome.results),
        },
    }
    if context.retrieval_version != RETRIEVAL_VERSION_V1:
        payload.update(
            {
                "retrieval_version": outcome.retrieval_version,
                "retrieval_strategy_version": outcome.retrieval_strategy_version,
                "retrieval_audit": outcome.audit,
            }
        )
    return (
        payload,
        outcome.warnings,
        len(outcome.results) > context.limits.max_search_results,
    )


def _lookup_symbol(
    context: ToolContext,
    parameters: BaseModel,
) -> tuple[Any, list[str], bool]:
    values = LookupSymbolInput.model_validate(parameters)
    context.check_active()
    limit = min(values.top_k, context.limits.max_search_results)
    ranked = SymbolRetriever(context.database).search(
        context.project_id,
        values.symbol,
        top_k=limit + 1,
        path=values.path,
        language=values.language,
        match_mode=values.match_mode,
        explicit_symbol=True,
        repository_revision=context.repository_revision,
    )
    truncated = len(ranked) > limit
    results = [
        {
            "code_chunk_id": item.code_chunk_id,
            "chunk_identity": item.chunk_identity,
            "symbol_name": item.symbol_name,
            "qualified_name": item.qualified_name,
            "symbol_kind": item.chunk_type,
            "path": item.path,
            "language": item.language,
            "start_line": item.start_line,
            "end_line": item.end_line,
            "repository_revision": item.repository_revision,
            "content_hash": item.content_hash,
            "candidate_source": item.candidate_source,
            "symbol_match_type": item.symbol_match_type,
            "symbol_score": item.symbol_score,
            "symbol_rank": item.symbol_rank,
            "match_reasons": list(item.match_reasons),
            "relation_node_id": next(
                (
                    str(node["node_id"])
                    for node in context.database.get_relation_nodes(
                        context.project_id,
                        context.repository_revision,
                        code_chunk_ids=[item.code_chunk_id],
                    )
                ),
                None,
            ),
        }
        for item in ranked[:limit]
    ]
    return results, [], truncated


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


def _expand_relations(
    context: ToolContext,
    parameters: BaseModel,
) -> tuple[Any, list[str], bool]:
    values = ExpandRelationsInput.model_validate(parameters)
    context.check_active()
    if context.relation_mode == RELATION_MODE_EXPAND_V1:
        raise ToolError(
            "retrieval-time relation expansion already owns the request relation budget"
        )
    index_status = context.database.get_relation_index_status(
        context.project_id, context.repository_revision
    )
    if index_status is None:
        return (
            {
                "analysis_mode": "retrieval_only",
                "relation_support": "unavailable",
                "paths": [],
                "nodes": [],
                "edges": [],
                "supporting_evidence_ids": [],
                "evidence_chains": [],
                "unresolved_count": 0,
                "ambiguous_count": 0,
                "external_count": 0,
                "_observation_metrics": {
                    "seed_count": 0,
                    "node_count": 0,
                    "edge_count": 0,
                    "path_count": 0,
                    "evidence_count": 0,
                },
            },
            ["No relation index exists for the bound revision; retrieval-only mode used."],
            False,
        )

    owned = context.evidence_store.get_many(
        context.request_id, values.seed_evidence_ids
    )
    if len(owned) != len(set(values.seed_evidence_ids)):
        raise ToolError("one or more Evidence seeds are not owned by this request")
    evidence_nodes = context.database.get_relation_nodes(
        context.project_id,
        context.repository_revision,
        code_chunk_ids=[item.code_chunk_id for item in owned],
    )
    symbol_nodes = context.database.get_relation_nodes(
        context.project_id,
        context.repository_revision,
        node_ids=sorted(set(values.seed_symbol_ids)),
    )
    if len(symbol_nodes) != len(set(values.seed_symbol_ids)):
        raise ToolError("one or more symbol seeds are outside the bound revision")
    file_seed_nodes: list[dict[str, Any]] = []
    if "imports" in values.relation_types:
        seed_paths = {
            item.path for item in owned
        } | {str(item["path"]) for item in symbol_nodes}
        for path in sorted(seed_paths):
            file_seed_nodes.extend(
                item
                for item in context.database.get_relation_nodes(
                    context.project_id,
                    context.repository_revision,
                    path=path,
                )
                if item["node_type"] == "file"
            )
    seed_node_ids = sorted(
        {
            str(item["node_id"])
            for item in [*evidence_nodes, *symbol_nodes, *file_seed_nodes]
        }
    )
    if not seed_node_ids:
        raise ToolError("relation seeds did not resolve to indexed nodes")
    if len(seed_node_ids) > context.limits.max_relation_seed_nodes:
        raise ToolError("relation seed count exceeds server limit")

    effective_depth = min(values.max_depth, context.limits.max_relation_depth)
    effective_per_node = min(
        values.per_node_limit, context.limits.max_relation_neighbors_per_node
    )
    traversal_limits = RelationTraversalLimits(
        max_depth=context.limits.max_relation_depth,
        per_node_limit=effective_per_node,
        max_nodes=context.limits.max_relation_nodes,
        max_edges=context.limits.max_relation_edges,
        max_paths=context.limits.max_relation_paths,
        max_output_bytes=context.limits.max_relation_observation_bytes,
    )
    traversal = RelationGraphService(context.database).expand(
        project_id=context.project_id,
        repository_revision=context.repository_revision,
        seed_node_ids=seed_node_ids,
        relation_types=list(values.relation_types),
        direction=values.direction,
        max_depth=effective_depth,
        limits=traversal_limits,
        check_active=context.check_active,
    )

    chunk_ids = sorted(
        {
            int(node["code_chunk_id"])
            for node in traversal.nodes
            if node.get("code_chunk_id") is not None
        }
    )[: context.limits.max_relation_evidence_items]
    all_chunks = {
        int(chunk["id"]): chunk
        for chunk in context.database.get_code_chunks(context.project_id)
        if str(chunk.get("repository_revision", ""))
        == context.repository_revision
    }
    relation_chunks = [all_chunks[value] for value in chunk_ids if value in all_chunks]
    built = EvidenceBuilder().build_from_code_chunks(
        relation_chunks,
        context.bundle.get("project") or {},
    )
    context.evidence_store.add(context.request_id, built)
    evidence_by_chunk = {
        item.code_chunk_id: item
        for item in context.evidence_store.all(context.request_id)
    }
    supporting_ids = sorted(
        {
            evidence_by_chunk[chunk_id].evidence_id
            for chunk_id in chunk_ids
            if chunk_id in evidence_by_chunk
        }
    )
    seed_ids = sorted(set(values.seed_evidence_ids))
    edges_by_id = {str(item["edge_id"]): item for item in traversal.edges}
    nodes_by_id = {str(item["node_id"]): item for item in traversal.nodes}
    chains = []
    for path in traversal.paths:
        if not path.edge_ids or path.resolution_status not in {"resolved", "ambiguous"}:
            continue
        path_evidence_ids = sorted(
            {
                evidence_by_chunk[int(nodes_by_id[node_id]["code_chunk_id"])].evidence_id
                for node_id in path.node_ids
                if node_id in nodes_by_id
                and nodes_by_id[node_id].get("code_chunk_id") is not None
                and int(nodes_by_id[node_id]["code_chunk_id"]) in evidence_by_chunk
            }
        )
        chain = context.chain_store.add(
            owner_id=context.request_id,
            project_id=context.project_id,
            repository_revision=context.repository_revision,
            seed_evidence_ids=seed_ids,
            supporting_evidence_ids=path_evidence_ids,
            path=path,
            edges_by_id=edges_by_id,
            truncated=traversal.truncated,
            warnings=traversal.warnings,
        )
        chains.append(chain.public_summary())

    node_summaries = [
        {
            "node_id": node["node_id"],
            "node_type": node["node_type"],
            "path": node["path"],
            "qualified_name": node["qualified_name"],
            "start_line": node["start_line"],
            "end_line": node["end_line"],
        }
        for node in traversal.nodes
    ]
    edge_summaries = [
        {
            "edge_id": edge["edge_id"],
            "relation_type": edge["relation_type"],
            "source_node_id": edge["source_node_id"],
            "target_node_id": edge["target_node_id"],
            "source_line": edge["source_start_line"],
            "resolution_status": edge["resolution_status"],
            "resolution_rule": edge["resolution_rule"],
        }
        for edge in traversal.edges
    ]
    path_summaries = [
        {
            "node_ids": list(path.node_ids),
            "edge_ids": list(path.edge_ids),
            "path_depth": path.depth,
            "resolution_status": path.resolution_status,
        }
        for path in traversal.paths
    ]
    warnings = [*index_status.get("warnings", []), *traversal.warnings]
    return (
        {
            "analysis_mode": "relation_expanded",
            "relation_support": (
                "partial" if index_status["status"] == "partial" else "python_static"
            ),
            "paths": path_summaries,
            "nodes": node_summaries,
            "edges": edge_summaries,
            "supporting_evidence_ids": supporting_ids,
            "evidence_chains": chains,
            "unresolved_count": traversal.unresolved_count,
            "ambiguous_count": traversal.ambiguous_count,
            "external_count": traversal.external_count,
            "_observation_metrics": {
                "duration_ms": traversal.duration_ms,
                "seed_count": len(seed_node_ids),
                "node_count": len(traversal.nodes),
                "edge_count": len(traversal.edges),
                "path_count": len(traversal.paths),
                "evidence_count": len(supporting_ids),
            },
        },
        list(dict.fromkeys(warnings)),
        traversal.truncated,
    )


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


def _is_safe_relative_path(path: str) -> bool:
    try:
        normalized = normalize_repository_relative_path(path)
    except ValueError:
        return False
    return normalized == path


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
