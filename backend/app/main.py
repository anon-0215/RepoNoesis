from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import Iterator, Mapping, Sequence, Set
import json
import logging
import math
from pathlib import Path
from threading import RLock
import time
from typing import Any, Callable, Literal
import uuid

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    field_validator,
    model_validator,
)

from app.config import (
    get_agent_limits,
    get_embedding_settings,
    get_env_value,
    get_product_config_status,
    get_repository_settings,
)
from app.database import Database, SCHEMA_VERSION
from app.services.analyzer import analyze_snapshot
from app.services.agent_core import run_bounded_agent
from app.services.agent_contracts import (
    RequestBudget,
    normalize_repository_relative_path,
)
from app.services.ask_diagnostics import (
    ask_result_is_failure,
    ask_failure_http_status,
    build_ask_failure_detail,
    build_ask_success_diagnostics,
    format_ask_failure_log,
    format_ask_success_log,
    normalize_provider_failure_code,
)
from app.services.code_chunker import extract_python_code_chunks_from_files
from app.services.embedding_indexer import EmbeddingIndexer
from app.services.embedding_service import (
    CODE_CHUNK_TEXT_FORMAT_VERSION,
    EmbeddingService,
    build_code_chunk_embedding_input_hash,
)
from app.services.github_client import fetch_repository
from app.services.hierarchy_normalization import validate_hierarchy_mode
from app.services.learning_agent import build_learning_path
from app.services.learning_continuity import (
    LearningContinuityError,
    LearningContinuityService,
)
from app.services.learning_contracts import (
    CreateGoalRequest,
    CreatePlanRequest,
    CreateTaskRequest,
    EvaluationCorrectionRequest,
    GoalStatusRequest,
    SelfReportRequest,
    SubmitAttemptRequest,
)
from app.services.learning_service import LearningError, LearningService
from app.services.llm_client import LLMClient, ProviderError
from app.services.smoke_diagnostics import SmokeDiagnosticsRecorder
from app.services.repository_import import (
    RepositoryImportError,
    import_repository,
    remove_persisted_git_checkout,
)
from app.services.report import generate_report
from app.services.relation_analysis import index_project_relations
from app.services.relation_retrieval import validate_relation_mode
from app.services.workspace_service import (
    DEFAULT_WORKSPACE_LIMIT,
    MAX_WORKSPACE_LIMIT,
    WorkspaceCorrupt,
    WorkspaceNotFound,
    WorkspaceService,
    WorkspaceUnavailable,
)
from app.services.workspace_update import WorkspaceUpdateError, WorkspaceUpdateService


class _DeferredValue:
    """Construct a runtime service only when application startup or a route uses it."""

    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory
        self._value: Any | None = None
        self._lock = RLock()

    def initialize(self) -> Any:
        if self._value is None:
            with self._lock:
                if self._value is None:
                    self._value = self._factory()
        return self._value

    def __getattr__(self, name: str) -> Any:
        return getattr(self.initialize(), name)


def _initialize_runtime_value(value: Any) -> Any:
    return value.initialize() if isinstance(value, _DeferredValue) else value


db = _DeferredValue(Database)
llm = LLMClient()
learning_service = _DeferredValue(
    lambda: LearningService(_initialize_runtime_value(db), llm)
)
embedding_service = EmbeddingService(get_embedding_settings())
agent_limits = get_agent_limits()
repository_settings = get_repository_settings()
logger = logging.getLogger(__name__)
success_logger = logging.getLogger("uvicorn.error")
_active_import_lock = RLock()
_active_import_project_ids: set[str] = set()


def _project_integrity(project_id: str) -> dict[str, Any]:
    identity = embedding_service.ensure_effective_embedding_identity()
    integrity = db.get_project_index_integrity(
        project_id,
        model_name=identity.model_name,
        backend_model_identity=identity.backend_model_identity,
        model_identity=identity.model_identity,
        identity_schema_version=identity.identity_schema_version,
        resolved_revision=identity.resolved_revision or "",
        text_format_version=identity.text_format_version,
        embedding_config_hash=identity.embedding_config_hash,
        embedding_dimension=identity.dimension,
        normalized=identity.normalized,
    )
    if integrity is None:
        raise LookupError("Project does not exist.")
    chunks = db.get_code_chunks(project_id)
    input_hashes = {
        int(chunk["id"]): build_code_chunk_embedding_input_hash(
            chunk, embedding_service.settings
        )
        for chunk in chunks
    }
    missing = db.get_code_chunks_missing_embeddings(
        project_id,
        identity.model_name,
        identity.backend_model_identity,
        CODE_CHUNK_TEXT_FORMAT_VERSION,
        identity.embedding_config_hash,
        identity.normalized,
        input_hashes,
        effective_identity=identity,
    )
    dimensions = db.get_fresh_embedding_dimensions_for_project(
        project_id,
        identity.model_name,
        identity.backend_model_identity,
        CODE_CHUNK_TEXT_FORMAT_VERSION,
        identity.embedding_config_hash,
        identity.normalized,
        effective_identity=identity,
    )
    integrity["missing_embeddings"] = len(missing)
    integrity["fresh_embeddings"] = max(0, len(chunks) - len(missing))
    integrity["coverage"] = (
        integrity["fresh_embeddings"] / len(chunks) if chunks else 0.0
    )
    integrity["ready"] = bool(
        chunks
        and not missing
        and dimensions == [identity.dimension]
        and integrity["relation_status"] == "complete"
        and integrity["revision_consistent"]
    )
    return integrity


def _log_ask_failure(detail: dict[str, Any]) -> None:
    logger.warning(format_ask_failure_log(detail), extra={"ask_failure": detail})


def _log_ask_success(diagnostics: dict[str, Any]) -> None:
    success_logger.info(
        format_ask_success_log(diagnostics),
        extra={"ask_success": diagnostics},
    )


def _prepare_ask_success_diagnostics(
    *,
    request_id: str,
    result: dict[str, Any],
    recorder_snapshot: dict[str, Any],
    retrieval_version: str,
    hierarchy_mode: str,
    relation_mode: str,
) -> dict[str, Any]:
    """Prepare bounded observability without making it a success gate."""

    try:
        return build_ask_success_diagnostics(
            result=result,
            recorder_snapshot=recorder_snapshot,
            retrieval_version=retrieval_version,
            hierarchy_mode=hierarchy_mode,
            relation_mode=relation_mode,
        )
    except Exception:
        # request_id is generated by this route. Every other value is a fixed
        # primitive so this fallback cannot recurse into projection code or
        # carry an exception, model output, prompt, or source text.
        return {
            "request_id": request_id,
            "success_stage": "response_validated",
            "core_validation_passed": True,
            "observability_degraded": True,
            "evidence_count": 0,
            "citation_count": 0,
        }


@asynccontextmanager
async def app_lifespan(_app: FastAPI):
    database = _initialize_runtime_value(db)
    _initialize_runtime_value(learning_service)
    WorkspaceUpdateService(database, repository_settings, embedding_service).recover_interrupted_runs()
    LearningContinuityService(database).recover_interrupted()
    yield

app = FastAPI(title="源鉴 RepoNoesis API", version="0.1.0", lifespan=app_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    repo_url: str | None = Field(default=None, examples=["https://github.com/tiangolo/fastapi"])
    source_type: Literal["local", "git_url"] | None = None
    source: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> "AnalyzeRequest":
        product_source = self.source_type is not None or self.source is not None
        if product_source:
            if self.repo_url is not None or self.source_type is None or not (self.source or "").strip():
                raise ValueError("Provide source_type and source for product imports, without repo_url.")
        elif not (self.repo_url or "").strip():
            raise ValueError("A repository source is required.")
        return self


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    path: str | None = None
    language: str | None = None
    symbol: str | None = None
    evidence_count: int = Field(default=5, ge=1, le=8)
    retrieval_version: Literal["v1", "v2"] = "v1"
    hierarchy_mode: Literal["off", "normalize_v1"] = "off"
    relation_mode: Literal["off", "expand_v1"] = "off"

    @field_validator("path")
    @classmethod
    def validate_repository_path(cls, value: str | None) -> str | None:
        if value is not None:
            normalize_repository_relative_path(value)
        return value

    @model_validator(mode="after")
    def validate_retrieval_hierarchy_pair(self) -> "AskRequest":
        validate_hierarchy_mode(
            self.hierarchy_mode,
            retrieval_version=self.retrieval_version,
        )
        validate_relation_mode(
            self.relation_mode,
            retrieval_version=self.retrieval_version,
        )
        return self


class CitationResponse(BaseModel):
    path: str
    summary: str
    snippet: str
    qualified_name: str = ""
    start_line: int = 0
    end_line: int = 0


class EvidenceResponse(BaseModel):
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
    lexical_score: FiniteFloat | None
    lexical_rank: int | None
    semantic_score: FiniteFloat | None
    semantic_rank: int | None
    fusion_score: FiniteFloat
    fusion_rank: int
    selection_reason: str
    validation_status: Literal["valid", "invalid", "unvalidated"]
    invalid_reason: str | None
    retrieval_strategy_version: str


class EvidenceChainResponse(BaseModel):
    chain_id: str
    relation_types: list[str]
    path_length: int
    seed_evidence_ids: list[str]
    supporting_evidence_ids: list[str]
    resolution_status: str
    truncated: bool


class AskResponse(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    request_id: str = Field(..., min_length=1, max_length=64)
    answer: str
    citations: list[CitationResponse]
    evidence_schema_version: Literal[1]
    evidence: list[EvidenceResponse]
    grounding_status: Literal["grounded", "insufficient_evidence", "degraded"]
    retrieval_mode: Literal["hybrid", "lexical", "legacy"]
    warnings: list[str]
    agent_schema_version: Literal[1]
    agent_mode: Literal["bounded", "deterministic_fallback"]
    agent_status: Literal[
        "completed",
        "insufficient_evidence",
        "degraded",
        "budget_exhausted",
        "tool_budget_exhausted",
        "final_answer_failed",
        "cancelled",
        "failed",
    ]
    agent_trace: list[dict[str, Any]]
    budget_usage: dict[str, Any]
    relation_schema_version: Literal[1]
    analysis_mode: Literal["retrieval_only", "relation_expanded"]
    evidence_chains: list[EvidenceChainResponse]
    relation_summary: dict[str, Any]
    learning_schema_version: Literal[1]
    learning_mode: Literal["disabled", "profiled", "adaptive", "degraded"]
    learning_context_summary: dict[str, Any]
    learning_plan_summary: dict[str, Any]
    recommended_next_action: dict[str, Any] | None
    learning_warnings: list[str]
    answer_mode: Literal["llm_grounded", "deterministic"]

    @model_validator(mode="before")
    @classmethod
    def reject_unsafe_response_metadata(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value

        pending = [
            value.get(field_name)
            for field_name in (
                "agent_trace",
                "budget_usage",
                "relation_summary",
                "learning_context_summary",
                "learning_plan_summary",
                "recommended_next_action",
            )
        ]
        visited: set[int] = set()
        while pending:
            current = pending.pop()
            if isinstance(current, Iterator):
                raise ValueError("response metadata contains a one-shot iterator")
            if isinstance(current, float) and not math.isfinite(current):
                raise ValueError("response metadata contains a non-finite float")
            if isinstance(current, Mapping):
                identity = id(current)
                if identity in visited:
                    continue
                visited.add(identity)
                pending.extend(current.keys())
                pending.extend(current.values())
            elif isinstance(current, Set):
                identity = id(current)
                if identity in visited:
                    continue
                visited.add(identity)
                pending.extend(current)
            elif isinstance(current, Sequence) and not isinstance(
                current, (str, bytes, bytearray)
            ):
                identity = id(current)
                if identity in visited:
                    continue
                visited.add(identity)
                pending.extend(current)
        return value


class LearningResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    learning_schema_version: Literal[1]


class LearningListResponse(BaseModel):
    learning_schema_version: Literal[1]
    items: list[dict[str, Any]]


class WorkspaceSummaryResponse(BaseModel):
    workspace_id: str
    display_name: str
    source_type: str
    project_status: str
    repository_revision: str
    openable: bool
    project_id: str | None = None
    total_chunks: int = 0
    embedding_count: int = 0
    created_at: str
    updated_at: str


class WorkspaceListResponse(BaseModel):
    items: list[WorkspaceSummaryResponse]
    total: int
    limit: int
    offset: int


class WorkspaceSnapshotResponse(BaseModel):
    project_id: str
    repository_revision: str
    status: str
    primary_language: str
    frameworks: list[str]
    updated_at: str


class WorkspaceUpdateRunResponse(BaseModel):
    run_id: str
    workspace_id: str
    target_revision: str
    status: Literal["pending", "running", "succeeded", "failed"]
    phase: str
    result: Literal["", "unchanged", "activated"]
    stats: dict[str, Any]
    error_code: str
    error_message: str
    retryable: bool
    retry_count: int
    active_project_id: str | None = None
    created_at: str
    started_at: str | None
    finished_at: str | None
    updated_at: str


class WorkspaceRevisionCheckResponse(BaseModel):
    workspace_id: str
    current_revision: str
    available_revision: str
    state: Literal["unchanged", "update_available"]


class WorkspaceDetailResponse(WorkspaceSummaryResponse):
    active_snapshot: WorkspaceSnapshotResponse
    latest_update_run: WorkspaceUpdateRunResponse | None = None
    learning_continuity: "LearningContinuityResponse | None" = None


class LearningContinuityResponse(BaseModel):
    transition_id: str | None
    workspace_id: str
    status: Literal["not_required", "pending", "running", "succeeded", "failed"]
    activation_version: int
    mapping_config_identity: str | None = None
    source_revision: str | None = None
    target_revision: str | None = None
    stats: dict[str, int]
    error_code: str = ""
    error_message: str = ""
    retryable: bool
    retry_count: int = 0
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    updated_at: str | None = None


class LearningContinuityImpactResponse(BaseModel):
    source_target_id: str
    target_target_id: str | None
    mapping_status: Literal[
        "unchanged_exact", "renamed_exact", "modified", "deleted",
        "ambiguous", "unmapped", "incompatible",
    ]
    mapping_rule: str
    source_mastery_status: str
    derived_mastery_status: str
    source_path: str
    target_path: str
    source_qualified_name: str
    target_qualified_name: str
    review_reason: str


class LearningContinuityImpactsResponse(BaseModel):
    transition_id: str
    workspace_id: str
    status: Literal["pending", "running", "succeeded", "failed"]
    items: list[LearningContinuityImpactResponse]


@app.get("/api/health")
def health() -> dict[str, Any]:
    embedding_identity = embedding_service.get_model_identity()
    return {
        "ok": True,
        "llm_available": llm.available,
        "github_token_configured": bool(get_env_value("GITHUB_TOKEN")),
        "embedding_enabled": embedding_service.settings.enabled,
        "embedding_available": embedding_service.is_available(),
        "embedding_model": embedding_identity.model_name,
        "embedding_model_revision": embedding_identity.model_revision,
        "embedding_device": embedding_identity.device,
        "database": str(Path(db.path)),
        "database_schema_version": SCHEMA_VERSION,
        "configuration": get_product_config_status(),
    }


@app.get("/api/config/status")
def configuration_status() -> dict[str, Any]:
    return get_product_config_status()


@app.get("/api/workspaces", response_model=WorkspaceListResponse)
def list_workspaces(
    limit: int = Query(default=DEFAULT_WORKSPACE_LIMIT, ge=1, le=MAX_WORKSPACE_LIMIT),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    if not 1 <= limit <= MAX_WORKSPACE_LIMIT or offset < 0:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_pagination",
                "message": f"limit must be between 1 and {MAX_WORKSPACE_LIMIT}, and offset must be non-negative.",
                "retryable": False,
            },
        )
    return WorkspaceService(db).list_workspaces(limit=limit, offset=offset)


@app.get("/api/workspaces/{workspace_id}", response_model=WorkspaceDetailResponse)
def get_workspace(workspace_id: str) -> dict[str, Any]:
    try:
        result = WorkspaceService(db).get_workspace(workspace_id)
        latest = db.get_latest_update_run(workspace_id)
        result["latest_update_run"] = _public_update_run(latest) if latest else None
        result["learning_continuity"] = _public_continuity(
            LearningContinuityService(db).get_current(workspace_id)
        )
        return result
    except WorkspaceNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "workspace_not_found",
                "message": "The requested workspace does not exist.",
                "retryable": False,
            },
        ) from exc
    except WorkspaceCorrupt as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "workspace_corrupt",
                "message": "The workspace snapshot association is incomplete or inconsistent.",
                "retryable": False,
            },
        ) from exc
    except WorkspaceUnavailable as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "workspace_not_openable",
                "message": "The workspace has no completed snapshot that can be reopened.",
                "retryable": False,
            },
        ) from exc


def _update_service() -> WorkspaceUpdateService:
    return WorkspaceUpdateService(db, repository_settings, embedding_service)


@app.post(
    "/api/workspaces/{workspace_id}/revision/check",
    response_model=WorkspaceRevisionCheckResponse,
)
def check_workspace_revision(workspace_id: str) -> dict[str, Any]:
    try:
        return _update_service().check_revision(workspace_id)
    except RepositoryImportError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.to_safe_dict()) from exc
    except WorkspaceUpdateError as exc:
        raise _workspace_update_http_error(exc) from exc


@app.post(
    "/api/workspaces/{workspace_id}/refresh",
    response_model=WorkspaceUpdateRunResponse,
)
def start_workspace_refresh(
    workspace_id: str, background_tasks: BackgroundTasks
) -> dict[str, Any]:
    embedding_status = get_product_config_status()["embedding"]
    if not embedding_status["ready"]:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "embedding_not_configured",
                "message": "Configure the local BGE-M3 provider before refreshing a workspace.",
                "retryable": False,
            },
        )
    try:
        service = _update_service()
        run = service.start_refresh(workspace_id)
        if run["status"] == "pending":
            background_tasks.add_task(service.execute_run, workspace_id, run["run_id"])
        return _public_update_run(run)
    except RepositoryImportError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.to_safe_dict()) from exc
    except WorkspaceUpdateError as exc:
        raise _workspace_update_http_error(exc) from exc


@app.get(
    "/api/workspaces/{workspace_id}/runs/{run_id}",
    response_model=WorkspaceUpdateRunResponse,
)
def get_workspace_update_run(workspace_id: str, run_id: str) -> dict[str, Any]:
    try:
        return _public_update_run(_update_service().get_run(workspace_id, run_id))
    except WorkspaceUpdateError as exc:
        raise _workspace_update_http_error(exc) from exc


@app.post(
    "/api/workspaces/{workspace_id}/runs/{run_id}/retry",
    response_model=WorkspaceUpdateRunResponse,
)
def retry_workspace_update_run(
    workspace_id: str, run_id: str, background_tasks: BackgroundTasks
) -> dict[str, Any]:
    try:
        service = _update_service()
        run = service.retry_run(workspace_id, run_id)
        background_tasks.add_task(service.execute_run, workspace_id, run_id)
        return _public_update_run(run)
    except WorkspaceUpdateError as exc:
        raise _workspace_update_http_error(exc) from exc


def _workspace_update_http_error(exc: WorkspaceUpdateError) -> HTTPException:
    return HTTPException(
        status_code=exc.status_code,
        detail={
            "code": exc.code,
            "message": exc.message,
            "retryable": exc.retryable,
        },
    )


def _public_update_run(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run["run_id"],
        "workspace_id": run["workspace_id"],
        "target_revision": run["target_revision"],
        "status": run["status"],
        "phase": run["phase"],
        "result": run["result"],
        "stats": run["stats"],
        "error_code": run["error_code"],
        "error_message": run["error_message"],
        "retryable": run["retryable"],
        "retry_count": run["retry_count"],
        "active_project_id": run["project_id"] if run["result"] == "activated" else None,
        "created_at": run["created_at"],
        "started_at": run["started_at"],
        "finished_at": run["finished_at"],
        "updated_at": run["updated_at"],
    }


@app.get(
    "/api/workspaces/{workspace_id}/learning-continuity",
    response_model=LearningContinuityResponse,
)
def get_workspace_learning_continuity(workspace_id: str) -> dict[str, Any]:
    try:
        return _public_continuity(LearningContinuityService(db).get_current(workspace_id))
    except LearningContinuityError as exc:
        raise _continuity_http_error(exc) from exc


@app.get(
    "/api/workspaces/{workspace_id}/learning-continuity/{transition_id}",
    response_model=LearningContinuityResponse,
)
def get_learning_continuity_transition(
    workspace_id: str, transition_id: str
) -> dict[str, Any]:
    try:
        return _public_continuity(
            LearningContinuityService(db).get_transition(workspace_id, transition_id)
        )
    except LearningContinuityError as exc:
        raise _continuity_http_error(exc) from exc


@app.get(
    "/api/workspaces/{workspace_id}/learning-continuity/{transition_id}/targets",
    response_model=LearningContinuityImpactsResponse,
)
def get_learning_continuity_impacts(
    workspace_id: str, transition_id: str
) -> dict[str, Any]:
    try:
        return LearningContinuityService(db).get_impacts(workspace_id, transition_id)
    except LearningContinuityError as exc:
        raise _continuity_http_error(exc) from exc


@app.post(
    "/api/workspaces/{workspace_id}/learning-continuity/{transition_id}/retry",
    response_model=LearningContinuityResponse,
)
def retry_learning_continuity_transition(
    workspace_id: str,
    transition_id: str,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    try:
        service = LearningContinuityService(db)
        transition = service.retry(workspace_id, transition_id)
        background_tasks.add_task(service.execute, workspace_id, transition_id)
        return _public_continuity(transition)
    except LearningContinuityError as exc:
        raise _continuity_http_error(exc) from exc


def _public_continuity(value: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "transition_id", "workspace_id", "status", "activation_version",
        "mapping_config_identity", "source_revision", "target_revision", "stats",
        "error_code", "error_message", "retryable", "retry_count", "created_at",
        "started_at", "finished_at", "updated_at",
    }
    return {key: value[key] for key in allowed if key in value}


def _continuity_http_error(exc: LearningContinuityError) -> HTTPException:
    return HTTPException(
        status_code=exc.status_code,
        detail={"code": exc.code, "message": exc.message, "retryable": exc.retryable},
    )


def _workspace_id_or_409(project_id: str) -> str:
    workspace = db.get_workspace_for_project(project_id)
    if workspace is None:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "workspace_corrupt",
                "message": "The project is not linked to a valid workspace.",
                "retryable": False,
            },
        )
    return str(workspace["id"])


@app.post("/api/projects/analyze")
def analyze_project(request: AnalyzeRequest) -> dict[str, Any]:
    product_import = request.source_type is not None
    if product_import:
        request_id = str(uuid.uuid4())
        embedding_status = get_product_config_status()["embedding"]
        if not embedding_status["ready"]:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "embedding_not_configured",
                    "message": "Configure the local BGE-M3 provider in the backend .env and restart the backend.",
                    "retryable": False,
                },
            )
        try:
            imported = import_repository(
                request.source_type or "",
                request.source or "",
                repository_settings,
                request_id=request_id,
            )
            snapshot = imported.snapshot
        except RepositoryImportError as exc:
            detail = exc.to_safe_dict(request_id=request_id)
            if exc.safe_stage is None:
                logger.warning(
                    "Repository import rejected.",
                    extra={"repository_import": detail},
                )
            raise HTTPException(status_code=exc.status_code, detail=detail) from exc
        existing = db.get_project_by_source_identity(imported.source_identity)
        if existing and existing["status"] == "done":
            integrity = _project_integrity(existing["id"])
            if integrity["ready"]:
                return {
                    "project_id": existing["id"],
                    "workspace_id": _workspace_id_or_409(existing["id"]),
                    "status": "done",
                    "import_action": "reused",
                    "index_integrity": integrity,
                }
            db.set_project_status(existing["id"], "incomplete")
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "existing_import_incomplete",
                    "message": "The existing import has an incomplete local index.",
                    "retryable": False,
                },
            )
        if existing and existing["status"] == "analyzing":
            with _active_import_lock:
                active = existing["id"] in _active_import_project_ids
            if active:
                return {
                    "project_id": existing["id"],
                    "workspace_id": _workspace_id_or_409(existing["id"]),
                    "status": "analyzing",
                    "import_action": "reused",
                }
            db.set_project_status(existing["id"], "interrupted")
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "existing_import_interrupted",
                    "message": "The existing import was interrupted and must be rebuilt or deleted.",
                    "retryable": False,
                },
            )
        if existing:
            project_id = existing["id"]
            db.begin_reanalysis(project_id)
        else:
            project_id = db.create_project(snapshot.to_dict())
    else:
        try:
            snapshot = fetch_repository(request.repo_url or "")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        project_id = db.create_project(snapshot.to_dict())
    embedding_index: dict[str, Any] | None = None
    relation_index: dict[str, Any] | None = None
    with _active_import_lock:
        _active_import_project_ids.add(project_id)
    try:
        analysis = analyze_snapshot(snapshot)
        chunk_result = extract_python_code_chunks_from_files(
            snapshot.files,
            snapshot.repository_revision,
        )
        if chunk_result.warnings:
            analysis["code_chunk_warnings"] = [
                warning.to_dict() for warning in chunk_result.warnings
            ]
        project = {
            "id": project_id,
            "repo": snapshot.repo,
            "repo_url": snapshot.repo_url,
        }
        # Product imports are deterministic until the user explicitly asks a
        # question; importing/indexing a repository must never incur LLM cost.
        learning_steps = build_learning_path(
            project, analysis, None if product_import else llm
        )
        enriched_files = [file.to_dict() for file in snapshot.files]
        enriched_by_path = {file["path"]: file for file in enriched_files}
        for public_file in analysis["files"]:
            enriched_by_path[public_file["path"]].update(public_file)
        db.save_analysis(
            project_id,
            analysis,
            list(enriched_by_path.values()),
            learning_steps,
            [chunk.to_dict() for chunk in chunk_result.chunks],
            finalize=not product_import,
        )
        db.set_project_status(project_id, "relation_indexing")
        try:
            relation_result = index_project_relations(db, project_id)
            relation_index = {
                "status": relation_result.status,
                "relation_schema_version": 1,
                "parsed_files": relation_result.parsed_files,
                "failed_files": relation_result.failed_files,
                "unsupported_files": relation_result.unsupported_files,
                "node_count": len(relation_result.nodes),
                "edge_count": len(relation_result.edges),
                "warnings": relation_result.warnings,
            }
        except Exception as exc:
            relation_index = {
                "status": "warning",
                "relation_schema_version": 1,
                "warnings": [
                    "Static relation indexing failed after M1/M2 analysis was saved: "
                    f"{type(exc).__name__}."
                ],
            }
        if embedding_service.settings.enabled:
            db.set_project_status(project_id, "embedding_indexing")
            try:
                embedding_index = EmbeddingIndexer(db, embedding_service).index_project(
                    project_id
                ).to_dict()
            except Exception as exc:
                if product_import:
                    raise HTTPException(
                        status_code=503,
                        detail={
                            "code": "embedding_unavailable",
                            "message": "Local BGE-M3 indexing failed in offline mode. Verify the configured model path, device, and installed backend dependencies.",
                            "retryable": False,
                        },
                    ) from exc
                embedding_index = {
                    "status": "warning",
                    "warnings": [f"Embedding indexing failed after analysis was saved: {exc}"],
                }
        if product_import:
            db.set_project_status(project_id, "validating")
            integrity = _project_integrity(project_id)
            failed_chunks = int((embedding_index or {}).get("failed_chunks", 0))
            if failed_chunks > 0 or not integrity["ready"]:
                db.set_project_status(project_id, "incomplete")
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "embedding_index_incomplete",
                        "message": "The local embedding index is incomplete.",
                        "retryable": True,
                        "total_chunks": integrity["total_chunks"],
                        "fresh_embeddings": integrity["fresh_embeddings"],
                        "missing_embeddings": integrity["missing_embeddings"],
                    },
                )
            db.set_project_status(project_id, "done")
        else:
            db.set_project_status(project_id, "done")
    except HTTPException as exc:
        retained_incomplete = isinstance(exc.detail, dict) and exc.detail.get("code") == "embedding_index_incomplete"
        if product_import and not retained_incomplete:
            _rollback_product_import(project_id)
        else:
            db.mark_failed(project_id, "Product analysis failed; see the API diagnostic.")
        raise
    except Exception as exc:
        if product_import:
            rollback_complete = _rollback_product_import(project_id)
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "repository_analysis_failed",
                    "message": "Repository analysis failed before the import could be committed.",
                    "retryable": False,
                    "request_id": request_id,
                    "safe_stage": "analysis",
                    "cleanup_pending": False,
                    "rollback_pending": not rollback_complete,
                },
            ) from exc
        db.mark_failed(project_id, str(exc))
        raise HTTPException(status_code=500, detail=f"分析失败：{exc}") from exc
    finally:
        with _active_import_lock:
            _active_import_project_ids.discard(project_id)

    response: dict[str, Any] = {
        "project_id": project_id,
        "workspace_id": _workspace_id_or_409(project_id),
        "status": "done",
        "import_action": "analyzed" if product_import else "legacy_analyzed",
    }
    if embedding_index is not None:
        response["embedding_index"] = embedding_index
    if relation_index is not None:
        response["relation_index"] = relation_index
    return response


@app.delete("/api/projects/{project_id}")
def delete_project(project_id: str) -> dict[str, Any]:
    project = db.get_project(project_id)
    if project is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "project_not_found", "message": "The project does not exist.", "retryable": False},
        )
    with _active_import_lock:
        if project_id in _active_import_project_ids:
            raise HTTPException(
                status_code=409,
                detail={"code": "project_import_active", "message": "The project import is active.", "retryable": True},
            )
    if project.get("source_type") == "git_url":
        cleanup = remove_persisted_git_checkout(
            str(project.get("source_location") or ""),
            str(project.get("repository_revision") or ""),
            repository_settings,
        )
        if cleanup.cleanup_pending:
            db.set_project_status(project_id, "cleanup_pending")
            return {"deleted": False, "cleanup_pending": True, "retryable": True}
    try:
        db.delete_project(project_id)
    except Exception as exc:
        logger.error("Project database deletion failed.")
        raise HTTPException(
            status_code=500,
            detail={"code": "project_delete_failed", "message": "The project could not be deleted.", "retryable": True},
        ) from exc
    return {"deleted": True, "cleanup_pending": False, "retryable": False}


def _rollback_product_import(project_id: str) -> bool:
    try:
        db.delete_product_import(project_id)
        return True
    except Exception:
        logger.error(
            "Product import database rollback failed.",
            extra={"repository_import": {"status": "database_rollback_failed"}},
        )
        return False


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict[str, Any]:
    project = db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    analysis = project.get("analysis", {})
    return {
        "project": {key: value for key, value in project.items() if key != "analysis"},
        "overview": analysis.get("overview", ""),
        "stats": analysis.get("stats", {}),
        "start_commands": analysis.get("start_commands", []),
        "core_files": [
            file for file in analysis.get("files", []) if file.get("is_core")
        ][:12],
        "modules": analysis.get("modules", []),
    }


@app.get("/api/projects/{project_id}/map")
def get_project_map(project_id: str) -> dict[str, Any]:
    bundle = _bundle_or_404(project_id)
    analysis = bundle.get("analysis", {})
    return {
        "tree": analysis.get("tree", {}),
        "modules": bundle.get("modules", []),
        "dependency_edges": analysis.get("dependency_edges", []),
        "core_files": [file for file in bundle.get("files", []) if file.get("is_core")],
    }


@app.get("/api/projects/{project_id}/learning-path")
def get_learning_path(project_id: str) -> dict[str, Any]:
    bundle = _bundle_or_404(project_id)
    return {"steps": bundle.get("learning_steps", [])}


@app.post("/api/projects/{project_id}/ask", response_model=AskResponse)
def ask_project(project_id: str, request: AskRequest) -> dict[str, Any]:
    started = time.monotonic()
    request_id = str(uuid.uuid4())
    recorder = SmokeDiagnosticsRecorder()
    request_budget = RequestBudget.create(started_at=started, limits=agent_limits)
    recorder.begin_request(
        deadline_budget_ms=request_budget.total_budget_ms,
        remaining_ms=request_budget.request_remaining_ms(started),
    )
    bundle = _bundle_or_404(project_id)
    product_project = bundle["project"].get("source_type") in {"local", "git_url"}
    if product_project:
        try:
            llm.require_available()
        except ProviderError as exc:
            now = time.monotonic()
            recorder.record_route_elapsed(int((now - started) * 1000))
            if request_budget.request_expired(now):
                recorder.record_agent_failure("deadline_exceeded")
                recorder.record_deadline_state(
                    remaining_ms=0,
                    overrun_ms=max(
                        0,
                        int((now - request_budget.request_deadline_at) * 1000),
                    ),
                )
                recorder.record_request_deadline_reached(True)
            detail = build_ask_failure_detail(
                result={
                    "request_id": request_id,
                    "budget_usage": {"elapsed_ms": int((now - started) * 1000)},
                },
                recorder_snapshot=recorder.snapshot(),
                retrieval_version=request.retrieval_version,
                hierarchy_mode=request.hierarchy_mode,
                relation_mode=request.relation_mode,
                retryable=exc.retryable,
                terminal_reason=normalize_provider_failure_code(exc.code),
            )
            _log_ask_failure(detail)
            raise HTTPException(
                status_code=(
                    ask_failure_http_status(detail)
                    if detail["code"] == "deadline_exceeded"
                    else exc.status_code
                ),
                detail=detail,
            ) from exc
    learning_context = learning_service.get_learning_context(project_id)
    try:
        result = run_bounded_agent(
            request.question,
            bundle,
            llm,
            db,
            embedding_service,
            path=request.path,
            language=request.language,
            symbol=request.symbol,
            evidence_count=request.evidence_count,
            limits=agent_limits,
            learning_context=learning_context,
            retrieval_version=request.retrieval_version,
            hierarchy_mode=request.hierarchy_mode,
            relation_mode=request.relation_mode,
            diagnostics_recorder=recorder,
            request_id=request_id,
            request_budget=request_budget,
            allow_planner_failure_fallback=not product_project,
        )
    except ProviderError as exc:
        now = time.monotonic()
        route_elapsed_ms = int((now - started) * 1000)
        recorder.record_route_elapsed(route_elapsed_ms)
        canonical_budget_reason = exc.code in {
            "deadline_exceeded",
            "planner_budget_exhausted",
            "final_answer_not_attempted",
            "tool_timeout",
        }
        if exc.code == "deadline_exceeded" or request_budget.request_expired(now):
            recorder.record_agent_failure("deadline_exceeded")
            recorder.record_deadline_state(
                remaining_ms=request_budget.request_remaining_ms(now),
                overrun_ms=max(
                    0, int((now - request_budget.request_deadline_at) * 1000)
                ),
            )
            recorder.record_request_deadline_reached(True)
        elif canonical_budget_reason:
            recorder.record_agent_failure(exc.code)
        terminal_reason = (
            exc.code
            if canonical_budget_reason
            else normalize_provider_failure_code(exc.code)
        )
        detail = build_ask_failure_detail(
            result={
                "request_id": request_id,
                "budget_usage": {
                    "elapsed_ms": route_elapsed_ms,
                    "limits": {"total_deadline_ms": agent_limits.total_deadline_ms},
                },
            },
            recorder_snapshot=recorder.snapshot(),
            retrieval_version=request.retrieval_version,
            hierarchy_mode=request.hierarchy_mode,
            relation_mode=request.relation_mode,
            retryable=exc.retryable,
            terminal_reason=terminal_reason,
        )
        _log_ask_failure(detail)
        raise HTTPException(
            status_code=(
                ask_failure_http_status(detail)
                if detail["code"] == "deadline_exceeded"
                else exc.status_code
            ),
            detail=detail,
        ) from exc
    recorder.record_route_elapsed(int((time.monotonic() - started) * 1000))
    recorder_snapshot = recorder.snapshot()
    if ask_result_is_failure(
        result,
        recorder_snapshot,
        product_project=product_project,
    ):
        detail = build_ask_failure_detail(
            result=result,
            recorder_snapshot=recorder_snapshot,
            retrieval_version=request.retrieval_version,
            hierarchy_mode=request.hierarchy_mode,
            relation_mode=request.relation_mode,
        )
        _log_ask_failure(detail)
        status_code = ask_failure_http_status(detail)
        raise HTTPException(status_code=status_code, detail=detail)
    response_candidate = dict(result)
    response_candidate["request_id"] = request_id
    try:
        initially_validated_response = AskResponse.model_validate(response_candidate)
        serialized_response = initially_validated_response.model_dump_json()
        validated_response = AskResponse.model_validate_json(serialized_response)
        validated_payload = json.loads(validated_response.model_dump_json())
    except Exception as exc:
        now = time.monotonic()
        recorder.record_route_elapsed(int((now - started) * 1000))
        terminal_reason = "response_contract_invalid"
        if request_budget.request_expired(now):
            terminal_reason = "deadline_exceeded"
            recorder.record_agent_failure("deadline_exceeded")
            recorder.record_deadline_state(
                remaining_ms=0,
                overrun_ms=max(
                    0,
                    int((now - request_budget.request_deadline_at) * 1000),
                ),
            )
            recorder.record_request_deadline_reached(True)
        detail = build_ask_failure_detail(
            result={
                "request_id": request_id,
                "budget_usage": response_candidate.get("budget_usage", {}),
            },
            recorder_snapshot=recorder.snapshot(),
            retrieval_version=request.retrieval_version,
            hierarchy_mode=request.hierarchy_mode,
            relation_mode=request.relation_mode,
            terminal_reason=terminal_reason,
        )
        _log_ask_failure(detail)
        raise HTTPException(
            status_code=ask_failure_http_status(detail), detail=detail
        ) from exc
    success_diagnostics = _prepare_ask_success_diagnostics(
        request_id=request_id,
        result=validated_payload,
        recorder_snapshot=recorder.snapshot(),
        retrieval_version=request.retrieval_version,
        hierarchy_mode=request.hierarchy_mode,
        relation_mode=request.relation_mode,
    )
    before_save = time.monotonic()
    if request_budget.request_expired(before_save):
        recorder.record_agent_failure("deadline_exceeded")
        recorder.record_deadline_state(
            remaining_ms=0,
            overrun_ms=max(
                0,
                int((before_save - request_budget.request_deadline_at) * 1000),
            ),
        )
        recorder.record_request_deadline_reached(True)
        recorder.record_route_elapsed(int((before_save - started) * 1000))
        detail = build_ask_failure_detail(
            result=validated_payload,
            recorder_snapshot=recorder.snapshot(),
            retrieval_version=request.retrieval_version,
            hierarchy_mode=request.hierarchy_mode,
            relation_mode=request.relation_mode,
            terminal_reason="deadline_exceeded",
        )
        _log_ask_failure(detail)
        raise HTTPException(status_code=504, detail=detail)
    try:
        db.save_chat_answer(
            project_id,
            request.question,
            validated_payload["answer"],
            validated_payload["citations"],
        )
    except Exception as exc:
        recorder.record_route_elapsed(int((time.monotonic() - started) * 1000))
        detail = build_ask_failure_detail(
            result=validated_payload,
            recorder_snapshot=recorder.snapshot(),
            retrieval_version=request.retrieval_version,
            hierarchy_mode=request.hierarchy_mode,
            relation_mode=request.relation_mode,
            terminal_reason="persistence_failed",
        )
        _log_ask_failure(detail)
        raise HTTPException(status_code=500, detail=detail) from exc
    try:
        _log_ask_success(success_diagnostics)
    except Exception:
        # Persistence is the business terminal boundary. Logging handlers,
        # filters, formatters, and serializers are best-effort after it.
        pass
    return validated_payload


@app.post(
    "/api/projects/{project_id}/learning/goals",
    response_model=LearningResponse,
)
def create_learning_goal(
    project_id: str, request: CreateGoalRequest
) -> dict[str, Any]:
    return _learning_call(learning_service.create_goal, project_id, request)


@app.get(
    "/api/projects/{project_id}/learning/goals",
    response_model=LearningListResponse,
)
def get_learning_goals(project_id: str) -> dict[str, Any]:
    items = _learning_call(learning_service.get_goals, project_id)
    return {"learning_schema_version": 1, "items": items}


@app.patch(
    "/api/projects/{project_id}/learning/goals/{goal_id}",
    response_model=LearningResponse,
)
def update_learning_goal(
    project_id: str, goal_id: str, request: GoalStatusRequest
) -> dict[str, Any]:
    return _learning_call(
        learning_service.set_goal_status, project_id, goal_id, request.status
    )


@app.post(
    "/api/projects/{project_id}/learning/plans",
    response_model=LearningResponse,
)
def create_learning_plan(
    project_id: str, request: CreatePlanRequest
) -> dict[str, Any]:
    return _learning_call(learning_service.create_plan, project_id, request)


@app.get(
    "/api/projects/{project_id}/learning/plans/current",
    response_model=LearningResponse | None,
)
def get_current_learning_plan(
    project_id: str, goal_id: str | None = None
) -> dict[str, Any] | None:
    return _learning_call(
        learning_service.get_current_plan, project_id, goal_id
    )


@app.get(
    "/api/projects/{project_id}/learning/state",
    response_model=LearningListResponse,
)
def get_learner_state(project_id: str) -> dict[str, Any]:
    items = _learning_call(learning_service.get_states, project_id)
    return {"learning_schema_version": 1, "items": items}


@app.post(
    "/api/projects/{project_id}/learning/tasks",
    response_model=LearningResponse,
)
def create_learning_task(
    project_id: str, request: CreateTaskRequest
) -> dict[str, Any]:
    return _learning_call(learning_service.create_task, project_id, request)


@app.get(
    "/api/projects/{project_id}/learning/tasks/{task_id}",
    response_model=LearningResponse,
)
def get_learning_task(project_id: str, task_id: str) -> dict[str, Any]:
    return _learning_call(learning_service.get_task, project_id, task_id)


@app.post(
    "/api/projects/{project_id}/learning/tasks/{task_id}/attempts",
    response_model=LearningResponse,
)
def submit_learning_attempt(
    project_id: str, task_id: str, request: SubmitAttemptRequest
) -> dict[str, Any]:
    return _learning_call(
        learning_service.submit_attempt, project_id, task_id, request
    )


@app.post(
    "/api/projects/{project_id}/learning/events/{event_id}/corrections",
    response_model=LearningResponse,
)
def correct_learning_evaluation(
    project_id: str, event_id: str, request: EvaluationCorrectionRequest
) -> dict[str, Any]:
    return _learning_call(
        learning_service.correct_evaluation, project_id, event_id, request
    )


@app.post(
    "/api/projects/{project_id}/learning/self-reports",
    response_model=LearningResponse,
)
def submit_learning_self_report(
    project_id: str, request: SelfReportRequest
) -> dict[str, Any]:
    return _learning_call(
        learning_service.submit_self_report, project_id, request
    )


@app.get(
    "/api/projects/{project_id}/learning/next-action",
    response_model=LearningResponse,
)
def get_recommended_learning_action(project_id: str) -> dict[str, Any]:
    context = learning_service.get_learning_context(project_id)
    return {
        "learning_schema_version": 1,
        "learning_mode": context["learning_mode"],
        "recommended_next_action": context.get("recommended_next_action"),
        "warnings": context.get("warnings", []),
    }


@app.post(
    "/api/projects/{project_id}/learning/revalidate",
    response_model=LearningResponse,
)
def revalidate_learning_state(project_id: str) -> dict[str, Any]:
    return _learning_call(learning_service.revalidate_project, project_id)


@app.get("/api/projects/{project_id}/report")
def get_report(project_id: str) -> dict[str, str]:
    bundle = _bundle_or_404(project_id)
    return {"markdown": generate_report(bundle)}


def _bundle_or_404(project_id: str) -> dict[str, Any]:
    bundle = db.get_bundle(project_id)
    if not bundle:
        raise HTTPException(status_code=404, detail="项目不存在")
    return bundle


def _learning_call(function: Any, *args: Any) -> Any:
    try:
        return function(*args)
    except LearningError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
