from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import time
from contextlib import contextmanager
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import BoundedSemaphore, RLock, local
from typing import Any, Callable, Protocol, Sequence

from app.config import EmbeddingSettings, clamp_embedding_max_length, get_embedding_settings
from app.database import _as_float32


CODE_CHUNK_TEXT_FORMAT_VERSION = "code-chunk-v1"
EMBEDDING_CONFIG_HASH_VERSION = "embedding-config-v1"
EFFECTIVE_EMBEDDING_IDENTITY_VERSION = "embedding-effective-v1"


def _check_active(callback: Callable[[], None] | None) -> None:
    if callback is not None:
        callback()


class EmbeddingError(RuntimeError):
    pass


class EmbeddingConfigurationError(EmbeddingError):
    pass


class EmbeddingModelLoadError(EmbeddingError):
    pass


class EmbeddingEncodeError(EmbeddingError):
    pass


class SemanticCapacityUnavailable(EmbeddingError):
    pass


@dataclass(frozen=True)
class EmbeddingModelIdentity:
    model_name: str
    model_identity: str
    device: str
    configured_revision: str | None = None
    resolved_revision: str | None = None
    local_snapshot_identity: str | None = None

    @property
    def model_revision(self) -> str:
        """Backward-compatible alias for the composite cache identity."""
        return self.model_identity

    def to_dict(self) -> dict[str, str | None]:
        return {
            "model_name": self.model_name,
            "model_identity": self.model_identity,
            "device": self.device,
            "configured_revision": self.configured_revision,
            "resolved_revision": self.resolved_revision,
            "local_snapshot_identity": self.local_snapshot_identity,
        }


@dataclass(frozen=True)
class EffectiveEmbeddingIdentity:
    identity_schema_version: str
    provider: str
    backend_type: str
    model_name: str
    configured_revision: str | None
    resolved_revision: str | None
    local_snapshot_identity: str | None
    backend_model_identity: str
    model_identity: str
    dimension: int
    normalized: bool
    text_format_version: str
    document_prefix_identity: str
    query_prefix_identity: str
    max_length: int
    batch_size: int
    pooling_identity: str
    embedding_config_hash: str
    device: str
    dtype: str
    cache_identity: str
    is_real: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity_schema_version": self.identity_schema_version,
            "provider": self.provider,
            "backend_type": self.backend_type,
            "model_name": self.model_name,
            "configured_revision": self.configured_revision,
            "resolved_revision": self.resolved_revision,
            "local_snapshot_identity": self.local_snapshot_identity,
            "backend_model_identity": self.backend_model_identity,
            "model_identity": self.model_identity,
            "dimension": self.dimension,
            "normalized": self.normalized,
            "text_format_version": self.text_format_version,
            "document_prefix_identity": self.document_prefix_identity,
            "query_prefix_identity": self.query_prefix_identity,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "pooling_identity": self.pooling_identity,
            "embedding_config_hash": self.embedding_config_hash,
            "device": self.device,
            "dtype": self.dtype,
            "cache_identity": self.cache_identity,
            "is_real": self.is_real,
        }


class EmbeddingBackend(Protocol):
    def load_model(
        self,
        model_name_or_path: str,
        device: str,
        cache_dir: Path,
        max_length: int,
        model_revision: str,
    ) -> None:
        ...

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int,
        normalize: bool,
    ) -> Any:
        ...

    def get_embedding_dimension(self) -> int | None:
        ...

    def get_model_revision(self) -> str | None:
        ...

    def unload_model(self) -> None:
        ...


class SentenceTransformerEmbeddingBackend:
    def __init__(self) -> None:
        self._model: Any | None = None
        self._resolved_revision: str | None = None
        self.local_files_only = False

    def load_model(
        self,
        model_name_or_path: str,
        device: str,
        cache_dir: Path,
        max_length: int,
        model_revision: str = "",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingModelLoadError(
                "sentence-transformers is not installed; install backend requirements "
                "before enabling embeddings."
            ) from exc

        cache_dir.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {
            "device": device,
            "cache_folder": str(cache_dir),
        }
        if self.local_files_only:
            kwargs["local_files_only"] = True
        if model_revision:
            kwargs["revision"] = model_revision
        model = SentenceTransformer(
            model_name_or_path,
            **kwargs,
        )
        if max_length > 0 and hasattr(model, "max_seq_length"):
            model.max_seq_length = max_length
        self._model = model
        self._resolved_revision = _extract_sentence_transformer_revision(model)

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int,
        normalize: bool,
    ) -> Any:
        if self._model is None:
            raise EmbeddingModelLoadError("embedding model is not loaded")
        return self._model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def get_embedding_dimension(self) -> int | None:
        if self._model is None:
            return None
        dimension = self._model.get_sentence_embedding_dimension()
        return int(dimension) if dimension else None

    def get_model_revision(self) -> str | None:
        if self._model is None:
            return None
        return self._resolved_revision or None

    def unload_model(self) -> None:
        self._model = None
        self._resolved_revision = None


class EmbeddingService:
    def __init__(
        self,
        settings: EmbeddingSettings | None = None,
        backend_factory: Callable[[], EmbeddingBackend] | None = None,
        cuda_available: Callable[[], bool] | None = None,
    ) -> None:
        self.settings = settings or get_embedding_settings()
        self._backend_is_injected = backend_factory is not None
        self._backend_factory = backend_factory or SentenceTransformerEmbeddingBackend
        self._cuda_available = cuda_available or _torch_cuda_available
        self._backend: EmbeddingBackend | None = None
        self._identity: EmbeddingModelIdentity | None = None
        self._dimension: int | None = None
        self._lock = RLock()
        # A timed-out native load/encode may keep running. Slots remain held
        # until that work really exits; no request-owned results are shared.
        self._semantic_slots = BoundedSemaphore(2)
        self._semantic_executor: ThreadPoolExecutor | None = None
        self._semantic_executor_lock = RLock()
        self._load_state = "unready"
        self._task_local = local()
        self._tasks: dict[int, dict[str, Any]] = {}
        self._last_task: dict[str, Any] | None = None
        self._next_task_id = 0
        self._closed = False
        self._preparation_future: Future[Any] | None = None
        self._preparation_state = "unready"
        self._preparation_summary: dict[str, Any] | None = None

    @property
    def model_load_state(self) -> str:
        if self._preparation_state == "loading":
            return "loading"
        if self._preparation_state == "failed":
            return "failed"
        return self._load_state

    def runtime_status(self) -> dict[str, Any]:
        """Bounded, content-free snapshots; never wait for the native model lock."""
        with self._semantic_executor_lock:
            now = time.monotonic()
            tasks = [{**{k: v for k, v in task.items() if k != "started_at"},
                      "elapsed_ms": int((now - task["started_at"]) * 1000)}
                     for task in self._tasks.values()]
            return {
                "model_state": self.model_load_state,
                "preparation_state": self._preparation_state,
                "ready": self.model_load_state == "ready" and not self._closed,
                "closed": self._closed,
                "active_tasks": len(tasks),
                "capacity": 2,
                "tasks": tasks,
                "last_task": dict(self._last_task) if self._last_task else None,
                "preparation": dict(self._preparation_summary) if self._preparation_summary else None,
            }

    @contextmanager
    def observe_phase(self, phase: str):
        task_id = getattr(self._task_local, "task_id", None)
        started = time.monotonic()
        with self._semantic_executor_lock:
            task = self._tasks.get(task_id)
            previous = task.get("phase", "semantic") if task else "semantic"
            if task is not None:
                task["phase"] = phase
        try:
            yield
        except BaseException:
            with self._semantic_executor_lock:
                if task is not None:
                    task.setdefault("failed_phase", phase)
            raise
        finally:
            with self._semantic_executor_lock:
                if task is not None:
                    key = phase + "_ms"
                    task[key] = task.get(key, 0) + int((time.monotonic()-started)*1000)
                    task["phase"] = previous

    @contextmanager
    def _model_lock(self):
        with self.observe_phase("lock_wait"):
            self._lock.acquire()
        try:
            yield
        finally:
            self._lock.release()

    def start_preparation(self) -> Future[Any] | None:
        """Normal startup owns one local load/first encode, never a reindex."""
        with self._semantic_executor_lock:
            if not self.settings.enabled or self._closed:
                return None
            if self._preparation_future is not None:
                return self._preparation_future
            self._preparation_state = "loading"
            def prepare() -> None:
                try:
                    with self.observe_phase("model_identity"):
                        self.load_model(local_files_only=True)
                    if not self._closed:
                        self.encode_query("source code", local_files_only=True)
                        self._preparation_state = "ready"
                except BaseException:
                    self._preparation_state = "failed"
                    raise
            try:
                self._preparation_future = self.submit_semantic_work(prepare, purpose="startup")
            except (EmbeddingError, OSError, RuntimeError) as exc:
                self._preparation_state = "failed"
                self._preparation_summary = {
                    "purpose": "startup", "status": "failed", "failed_phase": "submission",
                    "exception_type": type(exc).__name__,
                    "cause_type": type(exc.__cause__).__name__ if exc.__cause__ else None,
                }
                raise
            return self._preparation_future

    def close(self) -> None:
        """Stop submissions; running native work retains its slot until return."""
        with self._semantic_executor_lock:
            self._closed = True
            executor = self._semantic_executor
            idle = not self._tasks
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        if idle:
            self.unload_model()

    def submit_semantic_work(self, work: Callable[[], Any], *, purpose: str = "retrieval") -> Future[Any]:
        if not self._semantic_slots.acquire(blocking=False):
            raise SemanticCapacityUnavailable("semantic workers are occupied")
        try:
            with self._semantic_executor_lock:
                if self._closed:
                    raise SemanticCapacityUnavailable("semantic service is closed")
                if self._semantic_executor is None:
                    self._semantic_executor = ThreadPoolExecutor(
                        max_workers=2, thread_name_prefix="rn-semantic"
                    )
                executor = self._semantic_executor
                self._next_task_id += 1
                task_id = self._next_task_id
                task = {"task_id": task_id, "purpose": purpose, "started_at": time.monotonic(),
                        "phase": "queued", "status": "running"}
                self._tasks[task_id] = task

            def run() -> Any:
                self._task_local.task_id = task_id
                task["phase"] = "semantic"
                try:
                    return work()
                except BaseException as exc:
                    task["status"] = "failed"
                    task["exception_type"] = type(exc).__name__
                    task["cause_type"] = type(exc.__cause__).__name__ if exc.__cause__ else None
                    raise
                finally:
                    self._task_local.task_id = None

            future = executor.submit(run)
            def finished(done: Future[Any]) -> None:
                with self._semantic_executor_lock:
                    task["status"] = ("cancelled" if done.cancelled() else
                                      "failed" if done.exception() is not None else "completed")
                    task["elapsed_ms"] = int((time.monotonic()-task["started_at"])*1000)
                    self._last_task = {k:v for k,v in task.items() if k != "started_at"}
                    if purpose == "startup":
                        self._preparation_summary = dict(self._last_task)
                    self._tasks.pop(task_id, None)
                    self._semantic_slots.release()
                    unload = self._closed and not self._tasks
                if unload:
                    self.unload_model()
            future.add_done_callback(finished)
            return future
        except BaseException:
            with self._semantic_executor_lock:
                if "task_id" in locals():
                    self._tasks.pop(task_id, None)
            self._semantic_slots.release()
            raise

    def load_model(
        self,
        local_files_only: bool = False,
        *,
        check_active: Callable[[], None] | None = None,
    ) -> None:
        _check_active(check_active)
        with self._model_lock():
            _check_active(check_active)
            if self._backend is not None:
                return
            self._load_state = "loading"
            try:
                if not self.settings.enabled:
                    raise EmbeddingConfigurationError("embeddings are disabled by EMBEDDING_ENABLED")
                if self.settings.provider != "local_bge_m3":
                    raise EmbeddingConfigurationError(
                        "EMBEDDING_PROVIDER must be local_bge_m3 for the local product path"
                    )
                local_files_only = bool(local_files_only or self.settings.offline)
                device = resolve_embedding_device(self.settings.device, self._cuda_available)
                identity = build_model_identity(
                    self.settings.model_name_or_path,
                    device,
                    self.settings.model_revision,
                )
            except EmbeddingError:
                self._load_state = "failed"
                raise
            except (OSError, RuntimeError) as exc:
                self._load_state = "failed"
                raise EmbeddingModelLoadError("embedding initialization failed") from exc
            try:
                _check_active(check_active)
                self._load_state = "loading"
                backend = self._backend_factory()
                if isinstance(backend, SentenceTransformerEmbeddingBackend):
                    backend.local_files_only = local_files_only
                with self.observe_phase("model_load"):
                    backend.load_model(
                        self.settings.model_name_or_path,
                        device,
                        self.settings.cache_dir,
                        clamp_embedding_max_length(self.settings.max_length),
                        self.settings.model_revision,
                    )
            except EmbeddingError:
                self._load_state = "failed"
                raise
            except Exception as exc:
                self._load_state = "failed"
                suffix = (
                    "; offline mode is enabled, so place BGE-M3 in the configured local path or cache"
                    if local_files_only
                    else ""
                )
                _check_active(check_active)
                raise EmbeddingModelLoadError(
                    f"failed to load embedding model {identity.model_name} on {device}{suffix}"
                ) from exc
            backend_revision = _normalize_commit_sha(_backend_model_revision(backend))
            resolved_revision = identity.resolved_revision
            configured_commit = _normalize_commit_sha(identity.configured_revision)
            if backend_revision is not None:
                if configured_commit is not None and backend_revision != configured_commit:
                    self._load_state = "failed"
                    raise EmbeddingModelLoadError(
                        "loaded embedding revision does not match configured revision"
                    )
                if resolved_revision is not None and backend_revision != resolved_revision:
                    self._load_state = "failed"
                    raise EmbeddingModelLoadError(
                        "loaded embedding revision does not match verified local snapshot"
                    )
                resolved_revision = backend_revision
            loaded_identity = EmbeddingModelIdentity(
                identity.model_name,
                _compose_model_identity(
                    identity.model_name,
                    identity.configured_revision,
                    resolved_revision,
                    identity.local_snapshot_identity,
                ),
                identity.device,
                identity.configured_revision,
                resolved_revision,
                identity.local_snapshot_identity,
            )
            try:
                dimension = backend.get_embedding_dimension()
            except Exception as exc:
                self._load_state = "failed"
                raise EmbeddingModelLoadError("embedding dimension is unavailable") from exc
            self._identity = loaded_identity
            self._dimension = dimension
            self._backend = backend
            self._load_state = "ready"

    def encode_documents(
        self,
        texts: Sequence[str],
        local_files_only: bool = False,
        *,
        check_active: Callable[[], None] | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        prefixed = [self.settings.document_prefix + text for text in texts]
        return self._encode(
            prefixed,
            "documents",
            local_files_only=local_files_only,
            check_active=check_active,
        )

    def encode_query(
        self,
        text: str,
        local_files_only: bool = False,
        *,
        check_active: Callable[[], None] | None = None,
    ) -> list[float]:
        query = text.strip()
        if not query:
            raise EmbeddingEncodeError("embedding query must not be empty")
        return self._encode(
            [self.settings.query_prefix + query],
            "query",
            local_files_only=local_files_only,
            check_active=check_active,
        )[0]

    def get_model_identity(self) -> EmbeddingModelIdentity:
        if self._identity is not None:
            return self._identity
        device = (
            self.settings.device
            # A failed initialization has no verified runtime device. Health
            # must not repeat the failed probe or infer a CPU fallback.
            if self.settings.device != "auto" or self.model_load_state == "failed"
            else ("cuda" if self._cuda_available() else "cpu")
        )
        return build_model_identity(
            self.settings.model_name_or_path,
            device,
            self.settings.model_revision,
        )

    def get_backend_identity(self) -> EmbeddingModelIdentity:
        return self.get_model_identity()

    def ensure_model_identity(
        self,
        local_files_only: bool = False,
    ) -> EmbeddingModelIdentity:
        if _needs_loaded_revision(self.settings) and self._identity is None:
            self.load_model(local_files_only=local_files_only)
        return self.get_model_identity()

    def ensure_backend_identity(
        self,
        local_files_only: bool = False,
    ) -> EmbeddingModelIdentity:
        return self.ensure_model_identity(local_files_only=local_files_only)

    def get_effective_embedding_identity(
        self,
        local_files_only: bool = False,
        *,
        check_active: Callable[[], None] | None = None,
    ) -> EffectiveEmbeddingIdentity:
        _check_active(check_active)
        dimension = self.get_embedding_dimension(
            local_files_only=local_files_only,
            check_active=check_active,
        )
        if dimension is None:
            raise EmbeddingConfigurationError("embedding dimension is unavailable")
        return build_effective_embedding_identity(
            self.get_model_identity(),
            self.settings,
            dimension,
            is_real=not self._backend_is_injected,
        )

    def ensure_effective_embedding_identity(
        self,
        local_files_only: bool = False,
        *,
        check_active: Callable[[], None] | None = None,
    ) -> EffectiveEmbeddingIdentity:
        _check_active(check_active)
        value = self.get_effective_embedding_identity(
            local_files_only=local_files_only,
            check_active=check_active,
        )
        _check_active(check_active)
        return value

    def get_embedding_dimension(
        self,
        local_files_only: bool = False,
        *,
        check_active: Callable[[], None] | None = None,
    ) -> int | None:
        _check_active(check_active)
        if self._dimension is None:
            self.load_model(
                local_files_only=local_files_only,
                check_active=check_active,
            )
        _check_active(check_active)
        return self._dimension

    def unload_model(self) -> None:
        with self._model_lock():
            if self._backend is not None:
                self._backend.unload_model()
            self._backend = None
            self._identity = None
            self._dimension = None
            self._load_state = "unready"

    def is_available(self) -> bool:
        if not self.settings.enabled:
            return False
        if self._backend is not None:
            return True
        if self._backend_is_injected:
            return True
        return importlib.util.find_spec("sentence_transformers") is not None

    def _encode(
        self,
        texts: Sequence[str],
        operation: str,
        local_files_only: bool = False,
        check_active: Callable[[], None] | None = None,
    ) -> list[list[float]]:
        _check_active(check_active)
        with self._model_lock():
            self.load_model(
                local_files_only=local_files_only,
                check_active=check_active,
            )
            _check_active(check_active)
            assert self._backend is not None
            identity = self.get_model_identity()
            try:
                _check_active(check_active)
                with self.observe_phase("query_encode" if operation == "query" else "document_encode"):
                    raw_vectors = self._backend.encode(
                        texts,
                        batch_size=self.settings.batch_size,
                        normalize=self.settings.normalize,
                    )
                _check_active(check_active)
            except EmbeddingError:
                raise
            except Exception as exc:
                raise EmbeddingEncodeError(
                    f"failed to encode {len(texts)} {operation} with {identity.model_name}"
                ) from exc
            vectors = coerce_embedding_batch(raw_vectors, self.settings.normalize)
            if len(vectors) != len(texts):
                raise EmbeddingEncodeError(
                    f"embedding backend returned {len(vectors)} vectors for "
                    f"{len(texts)} {operation}"
                )
            if vectors:
                dimension = len(vectors[0])
                if any(len(vector) != dimension for vector in vectors):
                    raise EmbeddingEncodeError("embedding backend returned inconsistent dimensions")
                self._dimension = dimension
                if operation == "query" and self._preparation_state == "failed":
                    self._preparation_state = "ready"
            return vectors


def build_code_chunk_document_text(chunk: dict[str, Any]) -> str:
    path = str(chunk.get("path", "")).replace("\\", "/").lstrip("/")
    chunk_type = str(chunk.get("chunk_type", ""))
    symbol = str(chunk.get("qualified_name") or chunk.get("symbol_name") or "")
    content = str(chunk.get("content", ""))
    return "\n".join(
        [
            f"format: {CODE_CHUNK_TEXT_FORMAT_VERSION}",
            f"path: {path}",
            f"type: {chunk_type}",
            f"symbol: {symbol}",
            "code:",
            content,
        ]
    )


def resolve_embedding_device(
    requested_device: str,
    cuda_available: Callable[[], bool] | None = None,
) -> str:
    requested = (requested_device or "auto").lower()
    has_cuda = (cuda_available or _torch_cuda_available)()
    if requested == "auto":
        return "cuda" if has_cuda else "cpu"
    if requested == "cpu":
        return "cpu"
    if requested == "cuda" or requested.startswith("cuda:"):
        if not has_cuda:
            raise EmbeddingConfigurationError(
                f"EMBEDDING_DEVICE={requested_device} was requested, but CUDA is not available"
            )
        return requested
    raise EmbeddingConfigurationError(f"unsupported EMBEDDING_DEVICE value: {requested_device}")


def build_model_identity(
    model_name_or_path: str,
    device: str,
    model_revision: str = "",
) -> EmbeddingModelIdentity:
    raw = model_name_or_path.strip() or "BAAI/bge-m3"
    configured_revision = model_revision.strip()
    path = Path(raw)
    looks_local = path.exists() or path.is_absolute() or "\\" in raw or raw.startswith(".")
    if looks_local:
        resolved = path.expanduser().resolve(strict=False)
        digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16]
        safe_name = f"local:{resolved.name or 'embedding-model'}"
        local_snapshot_identity = f"path-sha256:{digest}"
        resolved_revision = _resolve_local_snapshot_revision(
            resolved,
            configured_revision or None,
        )
        model_identity = _compose_model_identity(
            safe_name,
            configured_revision or None,
            resolved_revision,
            local_snapshot_identity,
        )
        return EmbeddingModelIdentity(
            safe_name,
            model_identity,
            device,
            configured_revision or None,
            resolved_revision,
            local_snapshot_identity,
        )
    return EmbeddingModelIdentity(
        raw,
        configured_revision or raw,
        device,
        configured_revision or None,
        None,
        None,
    )


def _resolve_local_snapshot_revision(
    snapshot_path: Path,
    configured_revision: str | None,
) -> str | None:
    """Resolve a commit only from a verified Hugging Face snapshot layout."""
    if not _is_hugging_face_snapshot_layout(snapshot_path):
        return None

    directory_revision = _normalize_commit_sha(snapshot_path.name)
    configured_commit = _normalize_commit_sha(configured_revision)
    if directory_revision is None or configured_commit is None:
        return None
    if configured_commit != directory_revision:
        raise EmbeddingConfigurationError(
            "configured embedding revision does not match local snapshot directory"
        )

    refs_main = snapshot_path.parent.parent / "refs" / "main"
    if refs_main.exists():
        try:
            refs_revision = _normalize_commit_sha(
                refs_main.read_text(encoding="utf-8").strip()
            )
        except OSError as exc:
            raise EmbeddingConfigurationError(
                "unable to read local embedding snapshot refs/main"
            ) from exc
        if refs_revision != directory_revision:
            raise EmbeddingConfigurationError(
                "local embedding snapshot refs/main does not match snapshot directory"
            )
    return directory_revision


def _is_hugging_face_snapshot_layout(snapshot_path: Path) -> bool:
    model_root = snapshot_path.parent.parent
    return (
        snapshot_path.is_dir()
        and snapshot_path.parent.name == "snapshots"
        and model_root.name.startswith("models--")
        and (snapshot_path / "config.json").is_file()
        and (snapshot_path / "modules.json").is_file()
        and (snapshot_path / "1_Pooling" / "config.json").is_file()
    )


def _compose_model_identity(
    model_name: str,
    configured_revision: str | None,
    resolved_revision: str | None,
    local_snapshot_identity: str | None,
) -> str:
    if local_snapshot_identity is None and resolved_revision is None:
        return configured_revision or model_name
    payload = json.dumps(
        {
            "provider": "sentence-transformers",
            "model_name": model_name,
            "configured_revision": configured_revision,
            "resolved_revision": resolved_revision,
            "local_snapshot_identity": local_snapshot_identity,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"model-sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _normalize_commit_sha(value: str | None) -> str | None:
    candidate = (value or "").strip().lower()
    return candidate if re.fullmatch(r"[0-9a-f]{40}", candidate) else None


def build_embedding_config_hash(settings: EmbeddingSettings) -> str:
    payload = {
        "version": EMBEDDING_CONFIG_HASH_VERSION,
        "text_format_version": CODE_CHUNK_TEXT_FORMAT_VERSION,
        "query_prefix": settings.query_prefix,
        "document_prefix": settings.document_prefix,
        "max_length": clamp_embedding_max_length(settings.max_length),
        "normalize": bool(settings.normalize),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_effective_embedding_identity(
    backend_identity: EmbeddingModelIdentity,
    settings: EmbeddingSettings,
    dimension: int,
    *,
    is_real: bool,
) -> EffectiveEmbeddingIdentity:
    if int(dimension) < 1:
        raise EmbeddingConfigurationError("embedding dimension must be positive")
    embedding_config_hash = build_embedding_config_hash(settings)
    payload = {
        "identity_schema_version": EFFECTIVE_EMBEDDING_IDENTITY_VERSION,
        "provider": "sentence-transformers",
        "backend_type": "sentence-transformers",
        "model_name": backend_identity.model_name,
        "configured_revision": backend_identity.configured_revision,
        "resolved_revision": backend_identity.resolved_revision,
        "local_snapshot_identity": backend_identity.local_snapshot_identity,
        "backend_model_identity": backend_identity.model_identity,
        "dimension": int(dimension),
        "normalized": bool(settings.normalize),
        "text_format_version": CODE_CHUNK_TEXT_FORMAT_VERSION,
        "document_prefix_identity": _text_identity(settings.document_prefix),
        "query_prefix_identity": _text_identity(settings.query_prefix),
        "max_length": clamp_embedding_max_length(settings.max_length),
        "batch_size": max(1, int(settings.batch_size)),
        "pooling_identity": _text_identity(
            f"model-defined:{backend_identity.model_identity}"
        ),
        "embedding_config_hash": embedding_config_hash,
        "device": backend_identity.device,
        "dtype": "float32",
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return EffectiveEmbeddingIdentity(
        **payload,
        model_identity=f"embedding-sha256:{digest}",
        cache_identity=digest,
        is_real=is_real,
    )


def _text_identity(value: str) -> str:
    return f"text-sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def build_embedding_input_hash(final_embedding_text: str) -> str:
    return hashlib.sha256(final_embedding_text.encode("utf-8")).hexdigest()


def build_code_chunk_embedding_input_hash(
    chunk: dict[str, Any],
    settings: EmbeddingSettings,
) -> str:
    final_text = settings.document_prefix + build_code_chunk_document_text(chunk)
    return build_embedding_input_hash(final_text)


def coerce_embedding_batch(raw_vectors: Any, normalize: bool) -> list[list[float]]:
    if hasattr(raw_vectors, "tolist"):
        raw_vectors = raw_vectors.tolist()
    if raw_vectors is None:
        return []
    vectors = list(raw_vectors)
    if not vectors:
        return []
    if vectors and not isinstance(vectors[0], (list, tuple)):
        vectors = [vectors]
    coerced = [_coerce_vector(vector) for vector in vectors]
    return [_normalize_vector(vector) for vector in coerced] if normalize else coerced


def _coerce_vector(vector: Sequence[float]) -> list[float]:
    values = [_as_float32(value) for value in vector]
    if not values:
        raise EmbeddingEncodeError("embedding backend returned an empty vector")
    return values


def _normalize_vector(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return [_as_float32(value) for value in vector]
    return [_as_float32(value / norm) for value in vector]


def _torch_cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _needs_loaded_revision(settings: EmbeddingSettings) -> bool:
    if settings.model_revision.strip():
        return False
    raw = settings.model_name_or_path.strip()
    path = Path(raw)
    looks_local = path.exists() or path.is_absolute() or "\\" in raw or raw.startswith(".")
    return bool(raw) and not looks_local


def _backend_model_revision(backend: EmbeddingBackend) -> str | None:
    getter = getattr(backend, "get_model_revision", None)
    if getter is None:
        return None
    try:
        revision = getter()
    except Exception:
        return None
    return str(revision).strip() or None


def _extract_sentence_transformer_revision(model: Any) -> str | None:
    objects: list[Any] = [model]
    modules = getattr(model, "_modules", None)
    if isinstance(modules, dict):
        objects.extend(modules.values())
    for item in list(objects):
        objects.extend(
            candidate
            for candidate in (
                getattr(item, "auto_model", None),
                getattr(item, "tokenizer", None),
            )
            if candidate is not None
        )

    for item in objects:
        config = getattr(item, "config", None)
        commit_hash = getattr(config, "_commit_hash", None)
        if commit_hash:
            return str(commit_hash)
        init_kwargs = getattr(item, "init_kwargs", None)
        if isinstance(init_kwargs, dict) and init_kwargs.get("_commit_hash"):
            return str(init_kwargs["_commit_hash"])
        for attr in ("name_or_path", "_name_or_path"):
            value = getattr(item, attr, None)
            commit_hash = _commit_from_cache_path(str(value)) if value else None
            if commit_hash:
                return commit_hash
    return None


def _commit_from_cache_path(value: str) -> str | None:
    match = re.search(r"[\\/]+snapshots[\\/]+([0-9a-f]{40})(?:[\\/]|$)", value)
    return match.group(1) if match else None
