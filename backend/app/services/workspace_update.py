from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import PurePosixPath
from typing import Any, Callable

from app.config import RepositorySettings
from app.database import CODE_CHUNKER_VERSION, Database
from app.models import RepositorySnapshot
from app.services.analyzer import analyze_snapshot
from app.services.code_chunker import extract_python_code_chunks_from_files
from app.services.embedding_indexer import EmbeddingIndexer
from app.services.embedding_service import (
    CODE_CHUNK_TEXT_FORMAT_VERSION,
    build_code_chunk_embedding_input_hash,
    build_embedding_config_hash,
)
from app.services.learning_agent import build_learning_path
from app.services.learning_continuity import LearningContinuityService
from app.services.relation_analysis import index_project_relations
from app.services.repository_import import (
    ImportedRepository,
    RepositoryImportError,
    import_repository,
)


Importer = Callable[[str, str, RepositorySettings], ImportedRepository]


class WorkspaceUpdateError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 409,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable


class WorkspaceUpdateService:
    def __init__(
        self,
        database: Database,
        repository_settings: RepositorySettings,
        embedding_service: Any,
        *,
        importer: Importer = import_repository,
    ) -> None:
        self.database = database
        self.repository_settings = repository_settings
        self.embedding_service = embedding_service
        self.importer = importer
        self.fail_at_phase: str | None = None
        self.continuity_fail_before_publish = False

    def check_revision(self, workspace_id: str) -> dict[str, Any]:
        workspace = self._workspace(workspace_id)
        imported = self._import_workspace(workspace)
        current = str(workspace["repository_revision"])
        available = imported.snapshot.repository_revision
        return {
            "workspace_id": workspace_id,
            "current_revision": current,
            "available_revision": available,
            "state": "unchanged" if current == available else "update_available",
        }

    def start_refresh(self, workspace_id: str) -> dict[str, Any]:
        workspace = self._workspace(workspace_id)
        imported = self._import_workspace(workspace)
        target_revision = imported.snapshot.repository_revision
        config_identity = self._config_identity()
        try:
            return self.database.create_or_get_update_run(
                workspace_id, target_revision, config_identity
            )
        except LookupError as exc:
            raise WorkspaceUpdateError(
                "workspace_not_found", "The requested workspace does not exist.", status_code=404
            ) from exc

    def retry_run(self, workspace_id: str, run_id: str) -> dict[str, Any]:
        run = self.database.retry_update_run(workspace_id, run_id)
        if run is None:
            raise WorkspaceUpdateError(
                "update_run_not_found", "The requested update run does not exist.", status_code=404
            )
        if run["status"] != "pending":
            raise WorkspaceUpdateError(
                "update_not_retryable", "The update run is not in a retryable failed state."
            )
        return run

    def get_run(self, workspace_id: str, run_id: str) -> dict[str, Any]:
        run = self.database.get_update_run(workspace_id, run_id)
        if run is None:
            raise WorkspaceUpdateError(
                "update_run_not_found", "The requested update run does not exist.", status_code=404
            )
        return run

    def recover_interrupted_runs(self) -> int:
        return self.database.recover_interrupted_update_runs()

    def execute_run(self, workspace_id: str, run_id: str) -> dict[str, Any]:
        if not self.database.claim_update_run(workspace_id, run_id):
            return self.get_run(workspace_id, run_id)
        run = self.get_run(workspace_id, run_id)
        phase = "revision_resolution"
        try:
            workspace = self._workspace(workspace_id)
            if workspace["active_project_id"] != run["base_project_id"]:
                raise WorkspaceUpdateError(
                    "active_snapshot_changed",
                    "The active snapshot changed before this update started.",
                    retryable=True,
                )
            imported = self._import_workspace(workspace)
            snapshot = imported.snapshot
            if snapshot.repository_revision != run["target_revision"]:
                raise WorkspaceUpdateError(
                    "revision_changed_during_update",
                    "The repository revision changed while the update was starting.",
                    retryable=True,
                )
            self._checkpoint(phase)

            conflict = self.database.get_project_by_source_identity(imported.source_identity)
            if conflict is not None and conflict["id"] != run.get("project_id"):
                raise WorkspaceUpdateError(
                    "revision_snapshot_conflict",
                    "This revision already belongs to another historical snapshot and was not merged automatically.",
                )

            base_bundle = self.database.get_bundle(run["base_project_id"])
            if base_bundle is None:
                raise WorkspaceUpdateError(
                    "active_snapshot_missing", "The active snapshot is unavailable."
                )

            phase = "manifest_diff"
            self.database.update_run_phase(workspace_id, run_id, phase)
            old_manifest = build_manifest(base_bundle.get("files", []))
            new_files = [file.to_dict() for file in snapshot.files]
            new_manifest = build_manifest(new_files)
            diff = diff_manifests(old_manifest, new_manifest)
            stats: dict[str, Any] = {
                "files": diff["counts"],
                "chunks": {"reused": 0, "recomputed": 0, "removed": 0},
                "embeddings": {"reused": 0, "generated": 0, "failed": 0},
                "relations": {"mode": "full", "nodes": 0, "edges": 0},
            }
            self.database.update_run_phase(workspace_id, run_id, phase, stats)
            self._checkpoint(phase)

            base_revision_row = self.database.get_workspace_revision(
                workspace_id, str(workspace["repository_revision"])
            )
            project_id = self.database.create_staging_project(
                workspace_id,
                snapshot.to_dict(),
                run["base_project_id"],
                new_manifest["identity"],
                CODE_CHUNKER_VERSION,
                run_id,
            )

            phase = "source_analysis"
            self.database.update_run_phase(workspace_id, run_id, phase, stats)
            analysis = analyze_snapshot(snapshot)
            project = {"id": project_id, "repo": snapshot.repo, "repo_url": snapshot.repo_url}
            learning_steps = build_learning_path(project, analysis, None)
            enriched_files = self._enriched_files(snapshot, analysis)
            self._checkpoint(phase)

            phase = "chunk_update"
            self.database.update_run_phase(workspace_id, run_id, phase, stats)
            chunks, warnings, chunk_stats = self._incremental_chunks(
                base_bundle,
                snapshot,
                diff,
                base_chunker_version=str((base_revision_row or {}).get("chunker_version", "")),
            )
            stats["chunks"] = chunk_stats
            if warnings:
                analysis["code_chunk_warnings"] = warnings
            self.database.save_analysis(
                project_id, analysis, enriched_files, learning_steps, chunks
            )
            self.database.update_run_phase(workspace_id, run_id, phase, stats)
            self._checkpoint(phase)

            phase = "relation_update"
            self.database.update_run_phase(workspace_id, run_id, phase, stats)
            relation = index_project_relations(self.database, project_id)
            stats["relations"] = {
                "mode": "full",
                "status": relation.status,
                "nodes": len(relation.nodes),
                "edges": len(relation.edges),
            }
            self.database.update_run_phase(workspace_id, run_id, phase, stats)
            self._checkpoint(phase)

            phase = "embedding_update"
            self.database.update_run_phase(workspace_id, run_id, phase, stats)
            reused, embedding_identity = self._reuse_embeddings(
                run["base_project_id"], project_id
            )
            embedding_stats = EmbeddingIndexer(
                self.database, self.embedding_service
            ).index_project(project_id)
            stats["embeddings"] = {
                "reused": reused,
                "generated": embedding_stats.generated_chunks,
                "cached": embedding_stats.cached_chunks,
                "failed": embedding_stats.failed_chunks,
                "total": embedding_stats.total_chunks,
            }
            self.database.set_workspace_revision_embedding_identity(
                workspace_id, project_id, embedding_identity
            )
            self.database.update_run_phase(workspace_id, run_id, phase, stats)
            self._checkpoint(phase)

            phase = "snapshot_validation"
            self.database.update_run_phase(workspace_id, run_id, phase, stats)
            self._validate_snapshot(
                project_id,
                snapshot.repository_revision,
                new_manifest["identity"],
                embedding_stats,
            )
            self.database.update_run_phase(workspace_id, run_id, phase, stats)
            self._checkpoint(phase)

            phase = "activation"
            self.database.update_run_phase(workspace_id, run_id, phase, stats)
            self._checkpoint(phase)
            self.database.activate_workspace_snapshot(
                workspace_id,
                run_id,
                project_id=project_id,
                expected_active_project_id=run["base_project_id"],
                source_identity=imported.source_identity,
                stats=stats,
            )
            continuity = LearningContinuityService(self.database)
            continuity.fail_before_publish = self.continuity_fail_before_publish
            transition = continuity.get_current(workspace_id)
            if transition.get("transition_id"):
                continuity.execute(workspace_id, str(transition["transition_id"]))
            return self.get_run(workspace_id, run_id)
        except RepositoryImportError as exc:
            self.database.fail_update_run(
                workspace_id,
                run_id,
                code=exc.code,
                message=exc.message,
                retryable=exc.retryable,
            )
        except WorkspaceUpdateError as exc:
            self.database.fail_update_run(
                workspace_id,
                run_id,
                code=exc.code,
                message=exc.message,
                retryable=exc.retryable,
            )
        except Exception as exc:
            self.database.fail_update_run(
                workspace_id,
                run_id,
                code=f"update_{phase}_failed",
                message=f"Workspace update failed during {phase}: {type(exc).__name__}.",
                retryable=True,
            )
        return self.get_run(workspace_id, run_id)

    def _workspace(self, workspace_id: str) -> dict[str, Any]:
        workspace = self.database.get_workspace_record(workspace_id)
        if workspace is None:
            raise WorkspaceUpdateError(
                "workspace_not_found", "The requested workspace does not exist.", status_code=404
            )
        if (
            workspace.get("revision_workspace_id") != workspace_id
            or workspace.get("revision_project_id") != workspace.get("active_project_id")
            or workspace.get("linked_revision") != workspace.get("repository_revision")
            or workspace.get("linked_activation_status") != "active"
        ):
            raise WorkspaceUpdateError(
                "workspace_corrupt",
                "The workspace snapshot association is incomplete or inconsistent.",
            )
        if workspace.get("project_status") != "done":
            raise WorkspaceUpdateError(
                "workspace_not_openable", "The workspace has no completed active snapshot."
            )
        if workspace.get("source_type") not in {"local", "git_url"}:
            raise WorkspaceUpdateError(
                "workspace_source_unsupported",
                "Revision refresh is supported only for local or public HTTPS Git workspaces.",
                status_code=422,
            )
        return workspace

    def _import_workspace(self, workspace: dict[str, Any]) -> ImportedRepository:
        return self.importer(
            str(workspace["source_type"]),
            str(workspace["source_location"]),
            self.repository_settings,
        )

    def _config_identity(self) -> str:
        backend = self.embedding_service.get_backend_identity()
        payload = {
            "chunker_version": CODE_CHUNKER_VERSION,
            "embedding_backend": backend.model_identity,
            "embedding_config": build_embedding_config_hash(
                self.embedding_service.settings
            ),
            "embedding_enabled": bool(self.embedding_service.settings.enabled),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return f"update-config-sha256:{digest}"

    @staticmethod
    def _enriched_files(
        snapshot: RepositorySnapshot, analysis: dict[str, Any]
    ) -> list[dict[str, Any]]:
        enriched = [file.to_dict() for file in snapshot.files]
        by_path = {file["path"]: file for file in enriched}
        for public_file in analysis.get("files", []):
            if public_file["path"] in by_path:
                by_path[public_file["path"]].update(public_file)
        return list(by_path.values())

    def _incremental_chunks(
        self,
        base_bundle: dict[str, Any],
        snapshot: RepositorySnapshot,
        diff: dict[str, Any],
        *,
        base_chunker_version: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
        old_chunks = list(base_bundle.get("code_chunks", []))
        old_by_path: dict[str, list[dict[str, Any]]] = {}
        for chunk in old_chunks:
            old_by_path.setdefault(str(chunk["path"]), []).append(chunk)
        can_reuse = base_chunker_version == CODE_CHUNKER_VERSION
        chunks: list[dict[str, Any]] = []
        reused_paths: set[str] = set()
        if can_reuse:
            for path in diff["unchanged"]:
                reused_paths.add(path)
                chunks.extend(
                    _clone_chunks(old_by_path.get(path, []), path, snapshot.repository_revision)
                )
            for old_path, new_path in diff["renamed"].items():
                reused_paths.add(new_path)
                chunks.extend(
                    _clone_chunks(
                        old_by_path.get(old_path, []), new_path, snapshot.repository_revision
                    )
                )
        parse_files = [
            file
            for file in snapshot.files
            if file.path not in reused_paths
            and PurePosixPath(file.path).suffix.lower() == ".py"
        ]
        extracted = extract_python_code_chunks_from_files(
            parse_files, snapshot.repository_revision
        )
        chunks.extend(chunk.to_dict() for chunk in extracted.chunks)
        reused_count = len(chunks) - len(extracted.chunks)
        surviving_keys = {
            (
                chunk["path"],
                chunk["chunk_type"],
                chunk["qualified_name"],
                int(chunk["start_line"]),
                int(chunk["end_line"]),
                chunk["content_hash"],
            )
            for chunk in chunks
        }
        removed = sum(
            1
            for chunk in old_chunks
            if (
                chunk["path"],
                chunk["chunk_type"],
                chunk["qualified_name"],
                int(chunk["start_line"]),
                int(chunk["end_line"]),
                chunk["content_hash"],
            )
            not in surviving_keys
        )
        return (
            chunks,
            [warning.to_dict() for warning in extracted.warnings],
            {"reused": reused_count, "recomputed": len(extracted.chunks), "removed": removed},
        )

    def _reuse_embeddings(self, old_project_id: str, new_project_id: str) -> tuple[int, str]:
        if not self.embedding_service.settings.enabled:
            return 0, "disabled"
        identity = self.embedding_service.ensure_effective_embedding_identity()
        config_hash = identity.embedding_config_hash
        try:
            old_embeddings = self.database.get_code_chunk_embeddings_for_project(
                old_project_id,
                identity.model_name,
                identity.backend_model_identity,
                CODE_CHUNK_TEXT_FORMAT_VERSION,
                config_hash,
                self.embedding_service.settings.normalize,
                effective_identity=identity,
            )
        except ValueError:
            old_embeddings = []
        by_input = {
            (item["content_hash"], item["embedding_input_hash"]): item
            for item in old_embeddings
        }
        records: list[dict[str, Any]] = []
        for chunk in self.database.get_code_chunks(new_project_id):
            input_hash = build_code_chunk_embedding_input_hash(
                chunk, self.embedding_service.settings
            )
            source = by_input.get((chunk["content_hash"], input_hash))
            if source is None:
                continue
            records.append(
                {
                    "code_chunk_id": chunk["id"],
                    "content_hash": chunk["content_hash"],
                    "embedding_input_hash": input_hash,
                    "model_name": identity.model_name,
                    "model_revision": identity.backend_model_identity,
                    "identity_schema_version": identity.identity_schema_version,
                    "wrapper_model_identity": identity.model_identity,
                    "resolved_revision": identity.resolved_revision or "",
                    "identity_eligible": True,
                    "text_format_version": CODE_CHUNK_TEXT_FORMAT_VERSION,
                    "embedding_config_hash": config_hash,
                    "embedding_dimension": source["embedding_dimension"],
                    "embedding_dtype": source["embedding_dtype"],
                    "normalized": source["normalized"],
                    "vector": source["vector"],
                }
            )
        self.database.upsert_code_chunk_embeddings(records)
        return len(records), identity.model_identity

    def _validate_snapshot(
        self,
        project_id: str,
        revision: str,
        manifest_identity: str,
        embedding_stats: Any,
    ) -> None:
        bundle = self.database.get_bundle(project_id)
        if bundle is None or bundle["project"]["status"] != "done":
            raise WorkspaceUpdateError("snapshot_validation_failed", "The staging snapshot is incomplete.")
        if bundle["project"]["repository_revision"] != revision:
            raise WorkspaceUpdateError("snapshot_validation_failed", "The staging revision is inconsistent.")
        if any(chunk["repository_revision"] != revision for chunk in bundle["code_chunks"]):
            raise WorkspaceUpdateError("snapshot_validation_failed", "A code chunk belongs to another revision.")
        if build_manifest(bundle["files"])["identity"] != manifest_identity:
            raise WorkspaceUpdateError("snapshot_validation_failed", "The persisted file manifest is inconsistent.")
        relation = self.database.get_relation_index_status(project_id, revision)
        if relation is None:
            raise WorkspaceUpdateError("snapshot_validation_failed", "The relation index is unavailable.")
        if embedding_stats.failed_chunks or (
            embedding_stats.cached_chunks + embedding_stats.generated_chunks
            != embedding_stats.total_chunks
        ):
            raise WorkspaceUpdateError("snapshot_validation_failed", "The embedding index is incomplete.")

    def _checkpoint(self, phase: str) -> None:
        if self.fail_at_phase == phase:
            raise RuntimeError("injected update failure")


def build_manifest(files: list[Any]) -> dict[str, Any]:
    entries: dict[str, dict[str, Any]] = {}
    for file in files:
        path = str(file.get("path", "") if isinstance(file, dict) else file.path).replace("\\", "/").lstrip("/")
        content = str(file.get("content", "") if isinstance(file, dict) else file.content)
        size = int(file.get("size", len(content.encode("utf-8"))) if isinstance(file, dict) else file.size)
        entries[path] = {
            "path": path,
            "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "size": size,
        }
    payload = [entries[path] for path in sorted(entries)]
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"identity": f"manifest-sha256:{digest}", "entries": entries}


def diff_manifests(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    old_entries = old["entries"]
    new_entries = new["entries"]
    shared = set(old_entries) & set(new_entries)
    unchanged = sorted(
        path
        for path in shared
        if old_entries[path]["content_hash"] == new_entries[path]["content_hash"]
    )
    modified = sorted(shared - set(unchanged))
    deleted = set(old_entries) - set(new_entries)
    added = set(new_entries) - set(old_entries)

    deleted_counts = Counter(old_entries[path]["content_hash"] for path in deleted)
    added_counts = Counter(new_entries[path]["content_hash"] for path in added)
    deleted_by_hash = {old_entries[path]["content_hash"]: path for path in deleted}
    added_by_hash = {new_entries[path]["content_hash"]: path for path in added}
    renamed: dict[str, str] = {}
    for content_hash in sorted(set(deleted_counts) & set(added_counts)):
        if deleted_counts[content_hash] == 1 and added_counts[content_hash] == 1:
            old_path = deleted_by_hash[content_hash]
            new_path = added_by_hash[content_hash]
            renamed[old_path] = new_path
            deleted.remove(old_path)
            added.remove(new_path)
    return {
        "unchanged": unchanged,
        "modified": modified,
        "deleted": sorted(deleted),
        "added": sorted(added),
        "renamed": renamed,
        "counts": {
            "added": len(added),
            "modified": len(modified),
            "deleted": len(deleted),
            "renamed": len(renamed),
            "unchanged": len(unchanged),
        },
    }


def _clone_chunks(
    chunks: list[dict[str, Any]], path: str, revision: str
) -> list[dict[str, Any]]:
    keys = (
        "language",
        "chunk_type",
        "symbol_name",
        "qualified_name",
        "parent_symbol",
        "start_line",
        "end_line",
        "content",
        "content_hash",
    )
    return [
        {"repository_revision": revision, "path": path, **{key: chunk[key] for key in keys}}
        for chunk in chunks
    ]
