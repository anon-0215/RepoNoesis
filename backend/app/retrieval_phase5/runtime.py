from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import EmbeddingSettings
from app.database import Database
from app.m5.dataset import BenchmarkDatasetValidator
from app.retrieval_phase5.artifacts import write_result_artifacts
from app.retrieval_phase5.contracts import (
    BGE_M3_SNAPSHOT_REVISION,
    FROZEN_PATHS,
    build_frozen_manifest,
    file_hash,
    immutable_write_json,
    manifest_run_identity,
)
from app.retrieval_phase5.runner import (
    CountingEmbeddingService,
    Phase5Harness,
    Phase5RunError,
    load_click_benchmark,
    relation_graph_identity,
)
from app.services.embedding_indexer import EmbeddingIndexer
from app.services.embedding_service import EmbeddingService


@dataclass(frozen=True)
class RuntimeConfig:
    dataset_directory: Path
    repository_root: Path
    source_database: Path
    model_snapshot: Path
    artifact_root: Path
    resume_database: Path | None = None


class OfflineNetworkGuard(AbstractContextManager["OfflineNetworkGuard"]):
    def __init__(self) -> None:
        self.attempt_count = 0
        self._environment: dict[str, str | None] = {}
        self._create_connection: Any = None
        self._connect: Any = None
        self._connect_ex: Any = None

    def __enter__(self) -> "OfflineNetworkGuard":
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            self._environment[name] = os.environ.get(name)
            os.environ[name] = "1"
        self._create_connection = socket.create_connection
        self._connect = socket.socket.connect
        self._connect_ex = socket.socket.connect_ex

        def blocked_create_connection(*_args: Any, **_kwargs: Any) -> Any:
            self.attempt_count += 1
            raise Phase5RunError("network access is forbidden during Phase 5")

        def blocked_connect(_socket: Any, *_args: Any, **_kwargs: Any) -> Any:
            self.attempt_count += 1
            raise Phase5RunError("network access is forbidden during Phase 5")

        def blocked_connect_ex(_socket: Any, *_args: Any, **_kwargs: Any) -> int:
            self.attempt_count += 1
            raise Phase5RunError("network access is forbidden during Phase 5")

        socket.create_connection = blocked_create_connection
        socket.socket.connect = blocked_connect
        socket.socket.connect_ex = blocked_connect_ex
        return self

    def __exit__(self, *_exc: Any) -> None:
        socket.create_connection = self._create_connection
        socket.socket.connect = self._connect
        socket.socket.connect_ex = self._connect_ex
        for name, value in self._environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_formal_evaluation(config: RuntimeConfig) -> dict[str, Any]:
    started = time.perf_counter()
    _preflight_paths(config)
    validation = BenchmarkDatasetValidator(
        config.dataset_directory,
        config.repository_root,
    ).validate()
    if not validation.valid:
        raise Phase5RunError("benchmark validation failed: " + "; ".join(validation.errors[:5]))
    benchmark = load_click_benchmark(config.dataset_directory)
    source_hash_before = file_hash(config.source_database)
    source = _inspect_source_database(
        config.source_database,
        expected_revision=benchmark.repository_revision,
    )
    source_graph = relation_graph_identity(config.source_database, source["project_id"])
    if source_graph["status"] != "complete":
        raise Phase5RunError("source relation graph is not complete")
    if source_graph["repository_revision"] != benchmark.repository_revision:
        raise Phase5RunError("source relation graph revision differs from Click gold")
    dependency_hash_before = _dependency_identity()
    snapshot_metadata_before = _snapshot_metadata_identity(config.model_snapshot)
    timestamp = datetime.now(timezone.utc).isoformat()
    staging = config.artifact_root.resolve() / (
        ".retrieval-v2-phase5-staging-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    staging.mkdir(parents=True, exist_ok=False)
    copied_database = staging / "phase5.sqlite"
    embedding_resume = config.resume_database is not None
    copy_source = config.resume_database if embedding_resume else config.source_database
    assert copy_source is not None
    shutil.copy2(copy_source, copied_database)
    historical_embedding_rows = 0 if embedding_resume else _clear_copied_embeddings(copied_database)
    database = Database(copied_database)
    copied_graph_before = relation_graph_identity(copied_database, source["project_id"])
    if copied_graph_before != source_graph:
        raise Phase5RunError("copied relation graph identity differs from the frozen source")

    model_cache = staging / "runtime_model_cache"
    settings = EmbeddingSettings(
        enabled=True,
        model_name_or_path=str(config.model_snapshot.resolve()),
        model_revision=config.model_snapshot.name,
        device="cuda",
        batch_size=8,
        max_length=8192,
        normalize=True,
        cache_dir=model_cache,
        query_prefix="",
        document_prefix="",
    )
    service = CountingEmbeddingService(EmbeddingService(settings))
    import torch

    if not torch.cuda.is_available():
        raise Phase5RunError("CUDA is unavailable for the frozen formal run")
    torch.cuda.reset_peak_memory_stats()
    with OfflineNetworkGuard() as network_guard:
        smoke_started = time.perf_counter()
        smoke_vector = service.encode_query("RepoNoesis Retrieval v2 Phase 5 smoke", local_files_only=True)
        smoke_latency_ms = (time.perf_counter() - smoke_started) * 1000
        identity = service.ensure_effective_embedding_identity(local_files_only=True)
        if len(smoke_vector) != identity.dimension:
            raise Phase5RunError("smoke vector dimension differs from the effective identity")
        indexer = EmbeddingIndexer(database, service)
        index_started = time.perf_counter()
        first_index = indexer.index_project(source["project_id"])
        first_index_ms = (time.perf_counter() - index_started) * 1000
        second_started = time.perf_counter()
        second_index = indexer.index_project(source["project_id"])
        second_index_ms = (time.perf_counter() - second_started) * 1000
        if embedding_resume:
            if first_index.generated_chunks != 0 or first_index.cached_chunks != source["chunk_count"]:
                raise Phase5RunError("resumed document embeddings do not match the complete Click corpus")
        elif first_index.generated_chunks != source["chunk_count"] or first_index.cached_chunks != 0:
            raise Phase5RunError("frozen document indexing did not generate the complete Click corpus")
        if second_index.generated_chunks != 0 or second_index.cached_chunks != source["chunk_count"]:
            raise Phase5RunError("frozen document embedding cache did not produce a complete no-encode hit")
        copied_graph_after = relation_graph_identity(copied_database, source["project_id"])
        if copied_graph_after != copied_graph_before:
            raise Phase5RunError("embedding indexing changed the frozen relation graph")
        environment = _environment_snapshot(torch)
        pooling = _pooling_configuration(config.model_snapshot)
        manifest = build_frozen_manifest(
            repository_commit=_git(Path(__file__).resolve().parents[3], "rev-parse", "HEAD"),
            branch=_git(Path(__file__).resolve().parents[3], "branch", "--show-current"),
            corpus_project_id=source["project_id"],
            corpus_repository_revision=benchmark.repository_revision,
            corpus_database_hash=source_hash_before,
            relation_graph_hash=source_graph["graph_hash"],
            dataset_hash=benchmark.dataset_hash,
            query_hash=benchmark.query_hash,
            gold_hash=benchmark.gold_hash,
            matcher_hash=benchmark.matcher_hash,
            query_count=len(benchmark.scenarios),
            answerable_query_count=len(benchmark.answerable),
            embedding_identity=identity,
            model_local_path=str(config.model_snapshot.resolve()),
            cache_namespace="retrieval-v2-phase5-click-v1",
            environment=environment,
            source_files={
                "query_file": "benchmarks/m5/datasets/pilot-v1/scenarios.jsonl#repo_id=click",
                "gold_file": "benchmarks/m5/datasets/pilot-v1/scenarios.jsonl#repo_id=click",
            },
            timestamp=timestamp,
        )
        manifest.update(
            {
                "query_prefix": settings.query_prefix,
                "document_prefix": settings.document_prefix,
                "pooling": pooling,
                "pooling_identity": identity.pooling_identity,
                "benchmark_validation": validation.to_dict(),
                "source_database_path": str(config.source_database.resolve()),
                "source_database_hash_before": source_hash_before,
                "source_relation_graph": source_graph,
                "copied_relation_graph_before_embedding": copied_graph_before,
                "copied_relation_graph_after_embedding": copied_graph_after,
                "historical_fake_embedding_rows_removed_from_copy": historical_embedding_rows,
                "document_embedding_mode": "resume_frozen_cache" if embedding_resume else "encode_fresh",
                "resume_database_hash": file_hash(config.resume_database) if config.resume_database else None,
                "model_snapshot_metadata_identity_before": snapshot_metadata_before,
                "model_smoke": {
                    "status": "succeeded",
                    "latency_ms": smoke_latency_ms,
                    "query_encode_count": 1,
                    "dimension": len(smoke_vector),
                },
                "document_embedding_index": {
                    "document_count": source["chunk_count"],
                    "chunk_count": source["chunk_count"],
                    "first": first_index.to_dict(),
                    "second": second_index.to_dict(),
                    "first_latency_ms": first_index_ms,
                    "second_latency_ms": second_index_ms,
                    "document_encode_calls": service.document_encode_calls,
                    "document_encode_items": service.document_encode_items,
                },
            }
        )
        run_id = manifest_run_identity(manifest)
        final_directory = config.artifact_root.resolve() / "runs" / run_id
        final_directory.parent.mkdir(parents=True, exist_ok=True)
        if final_directory.exists():
            raise FileExistsError(f"formal run already exists: {run_id}")
        immutable_write_json(staging / "manifest.json", manifest)
        staging.rename(final_directory)
        copied_database = final_directory / copied_database.name
        database = Database(copied_database)
        harness = Phase5Harness(
            database=database,
            embedding_service=service,
            project_id=source["project_id"],
            scenarios=benchmark.scenarios,
            formal=True,
        )
        forward = harness.run_matrix(path_order=[item.path_id for item in FROZEN_PATHS])
        reverse = harness.run_matrix(
            path_order=["E", "C", "A", "D", "B"],
            scenario_order=[item.scenario_id for item in reversed(benchmark.scenarios)],
        )
        subset_scenarios = benchmark.answerable[:3]
        subset = Phase5Harness(
            database=database,
            embedding_service=service,
            project_id=source["project_id"],
            scenarios=subset_scenarios,
            formal=True,
        ).run_matrix(path_order=[item.path_id for item in FROZEN_PATHS])
        determinism = _determinism_summary(forward, reverse, subset)
        if not determinism["passed"]:
            raise Phase5RunError("formal deterministic replay changed rank, identity, or gold match")
        manifest["formal_execution"] = {
            "network_attempt_count": network_guard.attempt_count,
            "query_encode_calls_total_including_smoke": service.query_encode_calls,
            "query_encode_items_total_including_smoke": service.query_encode_items,
            "document_encode_calls_total": service.document_encode_calls,
            "document_encode_items_total": service.document_encode_items,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "normal_order": [item.path_id for item in FROZEN_PATHS],
            "interleaved_order": ["E", "C", "A", "D", "B"],
            "interleaved_query_order": "reverse",
            "repeated_subset_query_ids": [item.scenario_id for item in subset_scenarios],
        }
        manifest["model_snapshot_metadata_identity_after"] = _snapshot_metadata_identity(
            config.model_snapshot
        )
        manifest["dependency_inventory_hash_before"] = dependency_hash_before
        manifest["dependency_inventory_hash_after"] = _dependency_identity()
        manifest["source_database_hash_after"] = file_hash(config.source_database)
        manifest["formal_database_hash_after_evaluation"] = file_hash(copied_database)
        if manifest["source_database_hash_after"] != source_hash_before:
            raise Phase5RunError("formal run modified the historical source database")
        if manifest["model_snapshot_metadata_identity_after"] != snapshot_metadata_before:
            raise Phase5RunError("formal run modified the read-only model snapshot")
        if manifest["dependency_inventory_hash_after"] != dependency_hash_before:
            raise Phase5RunError("dependency inventory changed during formal evaluation")
        if network_guard.attempt_count:
            raise Phase5RunError("formal evaluation attempted network access")
        # The initially frozen manifest is retained verbatim. Runtime completion data is
        # written separately so results cannot retroactively change the protocol identity.
        immutable_write_json(final_directory / "runtime_completion.json", {
            key: manifest[key]
            for key in (
                "formal_execution", "model_snapshot_metadata_identity_after",
                "dependency_inventory_hash_before", "dependency_inventory_hash_after",
                "source_database_hash_after", "formal_database_hash_after_evaluation",
            )
        })
        hashes = write_result_artifacts(
            final_directory,
            manifest=json.loads((final_directory / "manifest.json").read_text(encoding="utf-8")),
            records_by_path=forward,
            determinism=determinism,
        )
    return {
        "status": "completed",
        "run_id": run_id,
        "run_directory": str(final_directory),
        "result_hash": hashes["result_hash"],
        "elapsed_seconds": time.perf_counter() - started,
        "benchmark": validation.to_dict(),
        "source": source,
        "embedding": identity.to_dict(),
        "network_attempt_count": network_guard.attempt_count,
        "determinism": determinism,
    }


def _preflight_paths(config: RuntimeConfig) -> None:
    for label, path in (
        ("dataset", config.dataset_directory),
        ("repository root", config.repository_root),
        ("source database", config.source_database),
        ("model snapshot", config.model_snapshot),
    ):
        if not path.exists():
            raise Phase5RunError(f"{label} does not exist: {path}")
    if not config.source_database.is_file():
        raise Phase5RunError("source database is not a file")
    required = ("config.json", "modules.json", "1_Pooling/config.json")
    if any(not (config.model_snapshot / value).is_file() for value in required):
        raise Phase5RunError("local BGE-M3 snapshot is incomplete")
    if config.model_snapshot.name != BGE_M3_SNAPSHOT_REVISION:
        raise Phase5RunError("local BGE-M3 snapshot revision differs from the frozen revision")
    if config.resume_database is not None:
        resume = config.resume_database.resolve()
        artifact_root = config.artifact_root.resolve()
        if not resume.is_file() or artifact_root not in resume.parents:
            raise Phase5RunError("resume database must be a file inside the Phase 5 artifact root")
        if resume.name != "phase5.sqlite":
            raise Phase5RunError("resume database must be the generated phase5.sqlite")


def _inspect_source_database(path: Path, *, expected_revision: str) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT id, repo_url, repo, repository_revision FROM projects WHERE repo = 'click'"
        ).fetchall()
        if len(rows) != 1:
            raise Phase5RunError("source database must contain exactly one Click project")
        project = dict(rows[0])
        if project["repository_revision"] != expected_revision:
            raise Phase5RunError("source database Click revision differs from benchmark")
        chunk_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM code_chunks WHERE project_id = ?",
                (project["id"],),
            ).fetchone()[0]
        )
        if chunk_count < 1:
            raise Phase5RunError("source database Click corpus is empty")
        return {**project, "project_id": project["id"], "chunk_count": chunk_count}
    finally:
        connection.close()


def _clear_copied_embeddings(path: Path) -> int:
    connection = sqlite3.connect(path)
    try:
        count = int(connection.execute("SELECT COUNT(*) FROM code_chunk_embeddings").fetchone()[0])
        connection.execute("DELETE FROM code_chunk_embeddings")
        connection.commit()
        remaining = int(connection.execute("SELECT COUNT(*) FROM code_chunk_embeddings").fetchone()[0])
        if remaining:
            raise Phase5RunError("copied database still contains historical embedding rows")
        return count
    finally:
        connection.close()


def _determinism_summary(
    forward: dict[str, list[dict[str, Any]]],
    reverse: dict[str, list[dict[str, Any]]],
    subset: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    mismatches: list[dict[str, str]] = []
    subset_ids = {
        item["query_id"] for records in subset.values() for item in records
    }
    for path in sorted(forward):
        forward_by_id = {item["query_id"]: item for item in forward[path]}
        reverse_by_id = {item["query_id"]: item for item in reverse[path]}
        subset_by_id = {item["query_id"]: item for item in subset[path]}
        for query_id, first in forward_by_id.items():
            second = reverse_by_id[query_id]
            if _rank_identity(first) != _rank_identity(second):
                mismatches.append({"path_id": path, "query_id": query_id, "comparison": "interleaved"})
            if query_id in subset_ids and _rank_identity(first) != _rank_identity(subset_by_id[query_id]):
                mismatches.append({"path_id": path, "query_id": query_id, "comparison": "repeated_subset"})
    return {
        "passed": not mismatches,
        "mismatches": mismatches,
        "normal_result_identity": _matrix_identity(forward),
        "interleaved_result_identity": _matrix_identity(reverse),
        "repeated_subset_result_identity": _matrix_identity(subset),
        "rank_identity_gold_match_stable": not mismatches,
    }


def _rank_identity(record: dict[str, Any]) -> list[tuple[Any, ...]]:
    return [
        (
            int(item.get("rank", 0)),
            str(item.get("chunk_identity", "")),
            bool(item.get("gold_match")),
        )
        for item in record.get("candidates", [])
    ]


def _matrix_identity(matrix: dict[str, list[dict[str, Any]]]) -> str:
    value = {
        path: {
            item["query_id"]: _rank_identity(item)
            for item in sorted(records, key=lambda record: record["query_id"])
        }
        for path, records in sorted(matrix.items())
    }
    return __import__("hashlib").sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _environment_snapshot(torch: Any) -> dict[str, Any]:
    return {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_initialized": torch.cuda.is_initialized(),
        "sentence_transformers_version": importlib.metadata.version("sentence-transformers"),
        "transformers_version": importlib.metadata.version("transformers"),
        "platform": sys.platform,
    }


def _dependency_identity() -> str:
    values = sorted(
        (distribution.metadata["Name"].casefold(), distribution.version)
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    )
    return __import__("hashlib").sha256(
        json.dumps(values, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _snapshot_metadata_identity(path: Path) -> str:
    values = [
        (
            item.relative_to(path).as_posix(),
            item.stat().st_size,
            item.stat().st_mtime_ns,
        )
        for item in sorted(path.rglob("*"))
        if item.is_file()
    ]
    return __import__("hashlib").sha256(
        json.dumps(values, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _pooling_configuration(path: Path) -> dict[str, Any]:
    value = json.loads((path / "1_Pooling" / "config.json").read_text(encoding="utf-8"))
    return {
        key: item
        for key, item in sorted(value.items())
        if key.startswith("pooling_mode_") or key == "word_embedding_dimension"
    }


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    ).stdout.strip()
