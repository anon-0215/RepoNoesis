from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

from app.m5.identity import identity_digest, require_identity_digest
from app.services.embedding_service import EffectiveEmbeddingIdentity


DENSE_ARTIFACT_SCHEMA_VERSION = 1
DENSE_CHECKPOINT_SCHEMA_VERSION = 1
DENSE_INDEX_SCHEMA_VERSION = "sqlite-dense-v1"


class DenseArtifactError(ValueError):
    pass


class DenseArtifactLegacyError(DenseArtifactError):
    pass


class StandaloneDenseArtifact:
    def __init__(
        self,
        root: Path,
        *,
        repository_id: str,
        repository_revision: str,
        repository_content_identity: str,
    ) -> None:
        self.root = root.resolve()
        self.manifest_path = self.root / "manifest.json"
        self.checkpoint_path = self.root / "checkpoint.json"
        self.repository = {
            "repository_id": repository_id,
            "repository_revision": repository_revision,
            "repository_content_identity": repository_content_identity,
        }
        self._manifest: dict[str, Any] | None = None

    def preflight(
        self,
        chunks: Sequence[dict[str, Any]],
        *,
        mode: Literal["create", "extend", "resume"],
    ) -> None:
        """Reject malformed or repository-incompatible metadata before model loading."""
        existing = self._load_valid_state()
        if existing is None:
            if mode != "create":
                raise DenseArtifactLegacyError(
                    "standalone dense manifest is missing; legacy artifact cannot resume"
                )
            return
        if mode == "create":
            raise DenseArtifactError("standalone dense artifact already exists; resume is required")

        inventory = build_chunk_inventory(chunks)
        identity = existing["artifact_identity"]
        if identity["repository"] != self.repository:
            raise DenseArtifactError("standalone dense repository identity mismatch")
        if mode == "resume":
            expected_inventory = f"inventory-sha256:{identity_digest(inventory)}"
            if identity["chunk_inventory_identity"] != expected_inventory:
                raise DenseArtifactError("standalone dense chunk inventory identity mismatch")
            if identity["chunk_count"] != len(inventory):
                raise DenseArtifactError("standalone dense chunk count mismatch")
            return
        if not set(existing["chunk_identities"]).issubset(set(inventory)):
            raise DenseArtifactError("standalone dense inventory extension removed existing chunks")

    def prepare(
        self,
        chunks: Sequence[dict[str, Any]],
        effective_identity: EffectiveEmbeddingIdentity,
        *,
        mode: Literal["create", "extend", "resume"],
    ) -> None:
        inventory = build_chunk_inventory(chunks)
        expected = _artifact_identity(self.repository, inventory, effective_identity)
        expected_digest = identity_digest(expected)
        existing = self._load_valid_state()

        if existing is None:
            if mode != "create":
                raise DenseArtifactLegacyError(
                    "standalone dense manifest is missing; legacy artifact cannot resume"
                )
            now = _utc_now()
            manifest = {
                "artifact_schema_version": DENSE_ARTIFACT_SCHEMA_VERSION,
                "index_schema_version": DENSE_INDEX_SCHEMA_VERSION,
                "artifact_identity": expected,
                "artifact_identity_digest": expected_digest,
                "repository": dict(self.repository),
                "chunk_inventory_identity": expected["chunk_inventory_identity"],
                "chunk_identities": inventory,
                "chunk_count": len(inventory),
                "effective_embedding_identity": effective_identity.to_dict(),
                "resolved_revision": effective_identity.resolved_revision,
                "dimension": effective_identity.dimension,
                "normalized": effective_identity.normalized,
                "text_format_version": effective_identity.text_format_version,
                "embedding_config_hash": effective_identity.embedding_config_hash,
                "created_at": now,
                "updated_at": now,
                "indexed_chunk_count": 0,
                "checkpoint_status": "indexing",
            }
            self.root.mkdir(parents=True, exist_ok=True)
            self._write_state(manifest, indexed_count=0, status="indexing")
            self._manifest = manifest
            return

        if mode == "create":
            raise DenseArtifactError("standalone dense artifact already exists; resume is required")
        if mode == "resume":
            require_identity_digest(
                existing,
                expected_digest,
                field="artifact_identity_digest",
                label="standalone dense manifest",
                error_type=DenseArtifactError,
            )
            self._manifest = existing
            return

        previous_identity = dict(existing["artifact_identity"])
        previous_inventory = set(existing["chunk_identities"])
        if not previous_inventory.issubset(set(inventory)):
            raise DenseArtifactError("standalone dense inventory extension removed existing chunks")
        expected_without_inventory = dict(expected)
        previous_without_inventory = dict(previous_identity)
        expected_without_inventory.pop("chunk_inventory_identity", None)
        expected_without_inventory.pop("chunk_count", None)
        previous_without_inventory.pop("chunk_inventory_identity", None)
        previous_without_inventory.pop("chunk_count", None)
        if identity_digest(expected_without_inventory) != identity_digest(previous_without_inventory):
            raise DenseArtifactError("standalone dense artifact identity mismatch")

        updated = dict(existing)
        updated.update(
            {
                "artifact_identity": expected,
                "artifact_identity_digest": expected_digest,
                "chunk_inventory_identity": expected["chunk_inventory_identity"],
                "chunk_identities": inventory,
                "chunk_count": len(inventory),
                "updated_at": _utc_now(),
                "checkpoint_status": "indexing",
            }
        )
        self._write_state(
            updated,
            indexed_count=int(existing.get("indexed_chunk_count", 0)),
            status="indexing",
        )
        self._manifest = updated

    def _load_valid_state(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        existing = _load_json(self.manifest_path)
        _validate_manifest_shape(existing)
        if not self.checkpoint_path.exists():
            raise DenseArtifactLegacyError(
                "standalone dense checkpoint is missing; artifact cannot resume"
            )
        checkpoint = _load_json(self.checkpoint_path)
        _validate_checkpoint_shape(checkpoint)
        _validate_manifest_consistency(existing)
        _validate_checkpoint_consistency(checkpoint, existing)
        require_identity_digest(
            checkpoint,
            str(existing["artifact_identity_digest"]),
            field="artifact_identity_digest",
            label="standalone dense checkpoint",
            error_type=DenseArtifactError,
        )
        return existing

    def update_progress(self, indexed_count: int, *, status: str = "indexing") -> bool:
        if self._manifest is None:
            raise DenseArtifactError("standalone dense artifact was not prepared")
        target = dict(self._manifest)
        target["indexed_chunk_count"] = int(indexed_count)
        target["checkpoint_status"] = status
        checkpoint = _checkpoint_state(
            target,
            indexed_count=int(indexed_count),
            status=status,
            updated_at=None,
        )
        current_checkpoint = _load_json(self.checkpoint_path)
        if (
            _semantic_state(target) == _semantic_state(self._manifest)
            and _semantic_state(checkpoint) == _semantic_state(current_checkpoint)
        ):
            return False

        now = _utc_now()
        target["updated_at"] = now
        self._write_state(
            target,
            indexed_count=int(indexed_count),
            status=status,
            updated_at=now,
        )
        self._manifest = target
        return True

    def complete(self, indexed_count: int, *, partial: bool = False) -> bool:
        return self.update_progress(
            indexed_count,
            status="partial" if partial else "completed",
        )

    def _write_state(
        self,
        manifest: dict[str, Any],
        *,
        indexed_count: int,
        status: str,
        updated_at: str | None = None,
    ) -> None:
        timestamp = updated_at or str(manifest["updated_at"])
        checkpoint = _checkpoint_state(
            manifest,
            indexed_count=indexed_count,
            status=status,
            updated_at=timestamp,
        )
        originals = {
            self.checkpoint_path: _read_bytes_if_exists(self.checkpoint_path),
            self.manifest_path: _read_bytes_if_exists(self.manifest_path),
        }
        try:
            _atomic_json(self.checkpoint_path, checkpoint)
            _atomic_json(self.manifest_path, manifest)
        except Exception as exc:
            try:
                for path, content in originals.items():
                    _restore_bytes(path, content)
            except Exception as rollback_exc:
                raise DenseArtifactError(
                    "standalone dense metadata write and rollback both failed"
                ) from rollback_exc
            raise DenseArtifactError("standalone dense metadata write failed") from exc
        finally:
            for path in (self.checkpoint_path, self.manifest_path):
                path.with_name(path.name + ".tmp").unlink(missing_ok=True)
                path.with_name(path.name + ".rollback").unlink(missing_ok=True)


def build_chunk_inventory(chunks: Sequence[dict[str, Any]]) -> list[str]:
    identities = []
    for chunk in chunks:
        payload = {
            "repository_revision": str(chunk.get("repository_revision") or ""),
            "path": str(chunk["path"]).replace("\\", "/").lstrip("/"),
            "chunk_type": str(chunk["chunk_type"]),
            "qualified_name": str(chunk["qualified_name"]),
            "start_line": int(chunk["start_line"]),
            "end_line": int(chunk["end_line"]),
            "content_hash": str(chunk["content_hash"]),
        }
        identities.append(f"chunk-sha256:{identity_digest(payload)}")
    if len(identities) != len(set(identities)):
        raise DenseArtifactError("standalone dense chunk inventory contains duplicates")
    return sorted(identities)


def _artifact_identity(
    repository: dict[str, str],
    inventory: list[str],
    effective_identity: EffectiveEmbeddingIdentity,
) -> dict[str, Any]:
    return {
        "artifact_schema_version": DENSE_ARTIFACT_SCHEMA_VERSION,
        "index_schema_version": DENSE_INDEX_SCHEMA_VERSION,
        "repository": dict(repository),
        "chunk_inventory_identity": f"inventory-sha256:{identity_digest(inventory)}",
        "chunk_count": len(inventory),
        "effective_embedding_identity": effective_identity.to_dict(),
        "wrapper_model_identity": effective_identity.model_identity,
        "resolved_revision": effective_identity.resolved_revision,
        "dimension": effective_identity.dimension,
        "normalized": effective_identity.normalized,
        "text_format_version": effective_identity.text_format_version,
        "embedding_config_hash": effective_identity.embedding_config_hash,
    }


def _validate_manifest_shape(value: dict[str, Any]) -> None:
    required = {
        "artifact_schema_version",
        "index_schema_version",
        "artifact_identity",
        "artifact_identity_digest",
        "repository",
        "chunk_inventory_identity",
        "chunk_identities",
        "chunk_count",
        "effective_embedding_identity",
        "resolved_revision",
        "dimension",
        "normalized",
        "text_format_version",
        "embedding_config_hash",
        "created_at",
        "updated_at",
        "indexed_chunk_count",
        "checkpoint_status",
    }
    if value.get("artifact_schema_version") != DENSE_ARTIFACT_SCHEMA_VERSION:
        raise DenseArtifactLegacyError("standalone dense manifest schema is incompatible")
    if not required.issubset(value):
        raise DenseArtifactLegacyError("standalone dense manifest lacks required identity fields")
    if identity_digest(value["artifact_identity"]) != value["artifact_identity_digest"]:
        raise DenseArtifactError("standalone dense manifest identity checksum mismatch")


def _validate_checkpoint_shape(value: dict[str, Any]) -> None:
    if value.get("checkpoint_schema_version") != DENSE_CHECKPOINT_SCHEMA_VERSION:
        raise DenseArtifactLegacyError("standalone dense checkpoint schema is incompatible")
    required = {
        "artifact_identity_digest",
        "effective_embedding_identity",
        "chunk_inventory_identity",
        "indexed_chunk_count",
        "status",
        "updated_at",
    }
    if not required.issubset(value):
        raise DenseArtifactLegacyError("standalone dense checkpoint lacks required identity fields")


def _validate_manifest_consistency(value: dict[str, Any]) -> None:
    identity = value["artifact_identity"]
    pairs = {
        "artifact_schema_version": value["artifact_schema_version"],
        "index_schema_version": value["index_schema_version"],
        "repository": value["repository"],
        "chunk_inventory_identity": value["chunk_inventory_identity"],
        "effective_embedding_identity": value["effective_embedding_identity"],
        "resolved_revision": value["resolved_revision"],
        "dimension": value["dimension"],
        "normalized": value["normalized"],
        "text_format_version": value["text_format_version"],
        "embedding_config_hash": value["embedding_config_hash"],
    }
    if any(identity.get(key) != expected for key, expected in pairs.items()):
        raise DenseArtifactError("standalone dense manifest identity fields are inconsistent")
    if identity.get("chunk_count") != value.get("chunk_count"):
        raise DenseArtifactError("standalone dense manifest chunk count is inconsistent")
    if value["chunk_count"] != len(value["chunk_identities"]):
        raise DenseArtifactError("standalone dense chunk inventory length is inconsistent")
    effective = value["effective_embedding_identity"]
    effective_pairs = {
        "model_identity": identity.get("wrapper_model_identity"),
        "resolved_revision": value["resolved_revision"],
        "dimension": value["dimension"],
        "normalized": value["normalized"],
        "text_format_version": value["text_format_version"],
        "embedding_config_hash": value["embedding_config_hash"],
    }
    if any(effective.get(key) != expected for key, expected in effective_pairs.items()):
        raise DenseArtifactError("standalone dense effective identity fields are inconsistent")
    indexed_count = value["indexed_chunk_count"]
    if not isinstance(indexed_count, int) or not 0 <= indexed_count <= value["chunk_count"]:
        raise DenseArtifactError("standalone dense indexed chunk count is invalid")
    if identity_digest(value["chunk_identities"]) != str(
        value["chunk_inventory_identity"]
    ).removeprefix("inventory-sha256:"):
        raise DenseArtifactError("standalone dense chunk inventory checksum mismatch")


def _validate_checkpoint_consistency(
    checkpoint: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    if checkpoint["effective_embedding_identity"] != manifest["effective_embedding_identity"]:
        raise DenseArtifactError("standalone dense checkpoint embedding identity mismatch")
    if checkpoint["chunk_inventory_identity"] != manifest["chunk_inventory_identity"]:
        raise DenseArtifactError("standalone dense checkpoint inventory identity mismatch")
    if checkpoint["indexed_chunk_count"] != manifest["indexed_chunk_count"]:
        raise DenseArtifactError("standalone dense checkpoint progress mismatch")
    if checkpoint["status"] != manifest["checkpoint_status"]:
        raise DenseArtifactError("standalone dense checkpoint status mismatch")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DenseArtifactError(f"standalone dense metadata is unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        raise DenseArtifactError(f"standalone dense metadata is invalid: {path.name}")
    return value


def _checkpoint_state(
    manifest: dict[str, Any],
    *,
    indexed_count: int,
    status: str,
    updated_at: str | None,
) -> dict[str, Any]:
    value = {
        "checkpoint_schema_version": DENSE_CHECKPOINT_SCHEMA_VERSION,
        "artifact_identity_digest": str(manifest["artifact_identity_digest"]),
        "effective_embedding_identity": manifest["effective_embedding_identity"],
        "chunk_inventory_identity": manifest["chunk_inventory_identity"],
        "indexed_chunk_count": int(indexed_count),
        "status": status,
    }
    if updated_at is not None:
        value["updated_at"] = updated_at
    return value


def _semantic_state(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "updated_at"}


def _read_bytes_if_exists(path: Path) -> bytes | None:
    return path.read_bytes() if path.exists() else None


def _restore_bytes(path: Path, content: bytes | None) -> None:
    if content is None:
        path.unlink(missing_ok=True)
        return
    temporary = path.with_name(path.name + ".rollback")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
