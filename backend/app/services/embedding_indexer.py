from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any, Literal, Protocol, Sequence

from app.database import Database
from app.services.embedding_service import (
    CODE_CHUNK_TEXT_FORMAT_VERSION,
    EmbeddingError,
    EffectiveEmbeddingIdentity,
    build_code_chunk_embedding_input_hash,
    build_code_chunk_document_text,
)


class EffectiveEmbeddingProvider(Protocol):
    settings: Any

    def get_backend_identity(self) -> Any: ...

    def ensure_effective_embedding_identity(
        self,
        local_files_only: bool = False,
    ) -> EffectiveEmbeddingIdentity: ...

    def encode_documents(
        self,
        texts: Sequence[str],
        local_files_only: bool = False,
    ) -> list[list[float]]: ...


@dataclass
class EmbeddingIndexStats:
    total_chunks: int
    cached_chunks: int
    generated_chunks: int
    failed_chunks: int
    model_name: str
    configured_revision: str | None
    resolved_revision: str | None
    local_snapshot_identity: str | None
    backend_model_identity: str
    model_identity: str
    dimension: int | None
    device: str
    duration_ms: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EmbeddingIndexer:
    def __init__(
        self,
        database: Database,
        embedding_service: EffectiveEmbeddingProvider,
    ) -> None:
        self.database = database
        self.embedding_service = embedding_service

    def index_project(
        self,
        project_id: str,
        *,
        artifact: Any | None = None,
        artifact_mode: Literal["create", "extend", "resume"] = "resume",
    ) -> EmbeddingIndexStats:
        started = perf_counter()
        chunks = self.database.get_code_chunks(project_id)
        backend_identity = self.embedding_service.get_backend_identity()
        warnings: list[str] = []
        if not self.embedding_service.settings.enabled:
            warnings.append("Embeddings are disabled by EMBEDDING_ENABLED=false.")
            return EmbeddingIndexStats(
                total_chunks=len(chunks),
                cached_chunks=0,
                generated_chunks=0,
                failed_chunks=0,
                model_name=backend_identity.model_name,
                configured_revision=backend_identity.configured_revision,
                resolved_revision=backend_identity.resolved_revision,
                local_snapshot_identity=backend_identity.local_snapshot_identity,
                backend_model_identity=backend_identity.model_identity,
                model_identity=backend_identity.model_identity,
                dimension=None,
                device=backend_identity.device,
                duration_ms=_elapsed_ms(started),
                warnings=warnings,
            )
        if artifact is not None:
            artifact.preflight(chunks, mode=artifact_mode)
        identity = self.embedding_service.ensure_effective_embedding_identity()
        if artifact is not None:
            artifact.prepare(chunks, identity, mode=artifact_mode)
        embedding_config_hash = identity.embedding_config_hash
        embedding_input_hashes = {
            int(chunk["id"]): build_code_chunk_embedding_input_hash(
                chunk,
                self.embedding_service.settings,
            )
            for chunk in chunks
        }

        missing = self.database.get_code_chunks_missing_embeddings(
            project_id,
            identity.model_name,
            identity.backend_model_identity,
            CODE_CHUNK_TEXT_FORMAT_VERSION,
            embedding_config_hash,
            self.embedding_service.settings.normalize,
            embedding_input_hashes,
            effective_identity=identity,
        )
        generated = 0
        failed = 0
        dimension: int | None = None
        batch_size = max(1, self.embedding_service.settings.batch_size)
        for batch in _batches(missing, batch_size):
            successes, batch_failed, batch_warnings = self._encode_batch(project_id, batch)
            failed += batch_failed
            warnings.extend(batch_warnings)
            if not successes:
                continue
            try:
                self.database.upsert_code_chunk_embeddings(
                    [
                        {
                            "code_chunk_id": chunk["id"],
                            "content_hash": chunk["content_hash"],
                            "embedding_input_hash": build_code_chunk_embedding_input_hash(
                                chunk,
                                self.embedding_service.settings,
                            ),
                            "model_name": identity.model_name,
                            # Historical column retained for legacy readers; the wrapper identity
                            # below is authoritative for the effective cache protocol.
                            "model_revision": identity.backend_model_identity,
                            "identity_schema_version": identity.identity_schema_version,
                            "wrapper_model_identity": identity.model_identity,
                            "resolved_revision": identity.resolved_revision or "",
                            "identity_eligible": True,
                            "text_format_version": CODE_CHUNK_TEXT_FORMAT_VERSION,
                            "embedding_config_hash": embedding_config_hash,
                            "embedding_dimension": len(vector),
                            "embedding_dtype": "float32",
                            "normalized": self.embedding_service.settings.normalize,
                            "vector": vector,
                        }
                        for chunk, vector in successes
                    ]
                )
            except Exception as exc:
                failed += len(successes)
                warnings.append(
                    f"Failed to store {len(successes)} embeddings for project {project_id}: {exc}"
                )
                continue
            generated += len(successes)
            dimension = len(successes[0][1])
            if artifact is not None:
                artifact.update_progress(
                    len(chunks) - len(missing) + generated,
                    status="indexing",
                )

        cached_dimensions = self.database.get_fresh_embedding_dimensions_for_project(
            project_id,
            identity.model_name,
            identity.backend_model_identity,
            CODE_CHUNK_TEXT_FORMAT_VERSION,
            embedding_config_hash,
            self.embedding_service.settings.normalize,
            effective_identity=identity,
        )
        if dimension is None and len(cached_dimensions) == 1:
            dimension = cached_dimensions[0]
        elif len(cached_dimensions) > 1:
            warnings.append(
                f"Project {project_id} has inconsistent cached embedding dimensions: "
                f"{cached_dimensions}"
            )

        stats = EmbeddingIndexStats(
            total_chunks=len(chunks),
            cached_chunks=len(chunks) - len(missing),
            generated_chunks=generated,
            failed_chunks=failed,
            model_name=identity.model_name,
            configured_revision=identity.configured_revision,
            resolved_revision=identity.resolved_revision,
            local_snapshot_identity=identity.local_snapshot_identity,
            backend_model_identity=identity.backend_model_identity,
            model_identity=identity.model_identity,
            dimension=dimension,
            device=identity.device,
            duration_ms=_elapsed_ms(started),
            warnings=warnings,
        )
        if artifact is not None:
            indexed_count = stats.cached_chunks + stats.generated_chunks
            artifact.complete(
                indexed_count,
                partial=stats.failed_chunks > 0 or indexed_count != stats.total_chunks,
            )
        return stats

    def _encode_batch(
        self,
        project_id: str,
        chunks: Sequence[dict[str, Any]],
    ) -> tuple[list[tuple[dict[str, Any], list[float]]], int, list[str]]:
        texts = [build_code_chunk_document_text(chunk) for chunk in chunks]
        try:
            vectors = self.embedding_service.encode_documents(texts)
            return self._validate_vectors(chunks, vectors), 0, []
        except EmbeddingError as exc:
            if len(chunks) == 1:
                chunk = chunks[0]
                return [], 1, [
                    (
                        f"Failed to encode code chunk {chunk['id']} "
                        f"{chunk['path']}:{chunk['qualified_name']} in project {project_id}: {exc}"
                    )
                ]
        except Exception as exc:
            if len(chunks) == 1:
                chunk = chunks[0]
                return [], 1, [
                    (
                        f"Failed to encode code chunk {chunk['id']} "
                        f"{chunk['path']}:{chunk['qualified_name']} in project {project_id}: {exc}"
                    )
                ]

        successes: list[tuple[dict[str, Any], list[float]]] = []
        failed = 0
        warnings = [
            f"Batch embedding failed for project {project_id}; retrying chunks individually."
        ]
        for chunk, text in zip(chunks, texts):
            chunk_successes, chunk_failed, chunk_warnings = self._encode_batch(project_id, [chunk])
            successes.extend(chunk_successes)
            failed += chunk_failed
            warnings.extend(chunk_warnings)
        return successes, failed, warnings

    @staticmethod
    def _validate_vectors(
        chunks: Sequence[dict[str, Any]],
        vectors: Sequence[Sequence[float]],
    ) -> list[tuple[dict[str, Any], list[float]]]:
        if len(vectors) != len(chunks):
            raise ValueError(
                f"embedding count mismatch: got {len(vectors)} for {len(chunks)} chunks"
            )
        validated: list[tuple[dict[str, Any], list[float]]] = []
        dimension: int | None = None
        for chunk, vector in zip(chunks, vectors):
            values = list(vector)
            if not values:
                raise ValueError(f"empty embedding vector for code chunk {chunk['id']}")
            if dimension is None:
                dimension = len(values)
            elif len(values) != dimension:
                raise ValueError("embedding backend returned inconsistent dimensions")
            validated.append((chunk, values))
        return validated


def _batches(items: Sequence[dict[str, Any]], size: int) -> list[Sequence[dict[str, Any]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _elapsed_ms(started: float) -> int:
    return int((perf_counter() - started) * 1000)
