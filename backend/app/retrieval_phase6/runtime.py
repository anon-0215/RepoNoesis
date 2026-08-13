from __future__ import annotations

import shutil
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import EmbeddingSettings
from app.database import Database
from app.retrieval_phase5.contracts import (
    BGE_M3_SNAPSHOT_REVISION,
    FROZEN_PATHS,
    canonical_hash,
    ensure_formal_embedding_identity,
    file_hash,
    immutable_write_json,
)
from app.retrieval_phase5.runner import CountingEmbeddingService, relation_graph_identity
from app.retrieval_phase5.runtime import (
    OfflineNetworkGuard,
    _clear_copied_embeddings,
    _dependency_identity,
    _environment_snapshot,
    _git,
    _pooling_configuration,
    _snapshot_metadata_identity,
)
from app.retrieval_phase6 import EVALUATION_VERSION
from app.retrieval_phase6.artifacts import write_phase6_artifacts
from app.retrieval_phase6.contracts import Phase6BenchmarkSnapshot, load_phase6_benchmark
from app.retrieval_phase6.runner import Phase6Harness, phase6_determinism_summary
from app.services.embedding_indexer import EmbeddingIndexer
from app.services.embedding_service import EmbeddingService


class Phase6RunError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeConfig:
    phase6_benchmark_directory: Path
    click_dataset_directory: Path
    source_database: Path
    model_snapshot: Path
    artifact_root: Path


def run_formal_evaluation(config: RuntimeConfig) -> dict[str, Any]:
    started = time.perf_counter()
    _preflight_paths(config)
    benchmark = load_phase6_benchmark(
        config.phase6_benchmark_directory,
        config.click_dataset_directory,
    )
    source_hash_before = file_hash(config.source_database)
    sources = inspect_phase6_source_database(config.source_database, benchmark)
    dependency_hash_before = _dependency_identity()
    snapshot_identity_before = _snapshot_metadata_identity(config.model_snapshot)
    timestamp = datetime.now(timezone.utc).isoformat()
    staging = config.artifact_root.resolve() / (
        ".retrieval-v2-phase6-staging-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    staging.mkdir(parents=True, exist_ok=False)
    copied_database = staging / "phase6.sqlite"
    shutil.copy2(config.source_database, copied_database)
    historical_rows = _clear_copied_embeddings(copied_database)
    database = Database(copied_database)
    graphs_before = {
        repo: relation_graph_identity(copied_database, source["project_id"])
        for repo, source in sorted(sources.items())
    }
    for repo, graph in graphs_before.items():
        if graph != sources[repo]["relation_graph"]:
            raise Phase6RunError(f"copied {repo} relation graph differs from the frozen source")

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
        raise Phase6RunError("CUDA is unavailable for the frozen formal run")
    torch.cuda.reset_peak_memory_stats()
    with OfflineNetworkGuard() as network_guard:
        smoke_started = time.perf_counter()
        smoke_vector = service.encode_query(
            "RepoNoesis Retrieval v2 Phase 6 cross-repository smoke",
            local_files_only=True,
        )
        smoke_latency_ms = (time.perf_counter() - smoke_started) * 1000
        identity = service.ensure_effective_embedding_identity(local_files_only=True)
        try:
            ensure_formal_embedding_identity(identity, backend=getattr(service.service, "_backend", None))
        except ValueError as exc:
            raise Phase6RunError(str(exc)) from exc
        if len(smoke_vector) != 1024:
            raise Phase6RunError("formal BGE-M3 smoke vector dimension is not 1024")

        indexer = EmbeddingIndexer(database, service)
        index_results: dict[str, Any] = {}
        for repo in benchmark.repository_ids:
            source = sources[repo]
            first_started = time.perf_counter()
            first = indexer.index_project(source["project_id"])
            first_ms = (time.perf_counter() - first_started) * 1000
            second_started = time.perf_counter()
            second = indexer.index_project(source["project_id"])
            second_ms = (time.perf_counter() - second_started) * 1000
            if first.generated_chunks != source["chunk_count"] or first.cached_chunks != 0:
                raise Phase6RunError(f"{repo} fresh embedding index did not cover the complete corpus")
            if second.generated_chunks != 0 or second.cached_chunks != source["chunk_count"]:
                raise Phase6RunError(f"{repo} second embedding pass was not a complete cache hit")
            index_results[repo] = {
                "namespace": _cache_namespace(repo, source["repository_revision"], identity.cache_identity),
                "project_id": source["project_id"],
                "repository_revision": source["repository_revision"],
                "chunk_count": source["chunk_count"],
                "first": first.to_dict(),
                "second": second.to_dict(),
                "first_latency_ms": first_ms,
                "second_latency_ms": second_ms,
            }
        embedding_rows = _embedding_rows_by_repository(copied_database, sources)
        for repo, source in sources.items():
            if embedding_rows.get(repo) != source["chunk_count"]:
                raise Phase6RunError(f"{repo} embedding rows are not isolated to its complete chunk set")
        graphs_after = {
            repo: relation_graph_identity(copied_database, source["project_id"])
            for repo, source in sorted(sources.items())
        }
        if graphs_after != graphs_before:
            raise Phase6RunError("embedding indexing changed a frozen relation graph")

        manifest = _build_manifest(
            benchmark=benchmark,
            sources=sources,
            source_database=config.source_database,
            source_database_hash=source_hash_before,
            model_snapshot=config.model_snapshot,
            embedding_identity=identity.to_dict(),
            pooling=_pooling_configuration(config.model_snapshot),
            environment=_environment_snapshot(torch),
            index_results=index_results,
            embedding_rows=embedding_rows,
            historical_rows=historical_rows,
            smoke_latency_ms=smoke_latency_ms,
            timestamp=timestamp,
        )
        run_id = f"retrieval-v2-phase6-{canonical_hash(manifest)[:24]}"
        final_directory = config.artifact_root.resolve() / "runs" / run_id
        final_directory.parent.mkdir(parents=True, exist_ok=True)
        if final_directory.exists():
            raise FileExistsError(f"formal run already exists: {run_id}")
        immutable_write_json(staging / "manifest.json", manifest)
        staging.rename(final_directory)
        copied_database = final_directory / "phase6.sqlite"
        database = Database(copied_database)
        projects = {repo: value["project_id"] for repo, value in sources.items()}
        harness = Phase6Harness(
            database=database,
            embedding_service=service,
            projects_by_repo=projects,
            scenarios_by_repo=benchmark.scenarios_by_repo,
            strata_by_query=benchmark.strata_by_query,
            formal=True,
        )
        forward = harness.run_matrix(
            repo_order=list(benchmark.repository_ids),
            path_order=[item.path_id for item in FROZEN_PATHS],
        )
        reverse = harness.run_matrix(
            repo_order=list(reversed(benchmark.repository_ids)),
            path_order=["E", "C", "A", "D", "B"],
            reverse_queries=True,
        )
        subset_scenarios = {
            repo: benchmark.answerable_by_repo[repo][:2]
            for repo in benchmark.repository_ids
        }
        subset_strata = {
            item.scenario_id: benchmark.strata_by_query[item.scenario_id]
            for scenarios in subset_scenarios.values()
            for item in scenarios
        }
        subset = Phase6Harness(
            database=database,
            embedding_service=service,
            projects_by_repo=projects,
            scenarios_by_repo=subset_scenarios,
            strata_by_query=subset_strata,
            formal=True,
        ).run_matrix(
            repo_order=list(reversed(benchmark.repository_ids)),
            path_order=["D", "A", "E", "B", "C"],
            reverse_queries=True,
        )
        determinism = phase6_determinism_summary(forward, reverse, subset)
        if not determinism["passed"]:
            raise Phase6RunError("formal replay changed rank, identity, or strict gold match")
        completion = {
            "formal_execution": {
                "network_attempt_count": network_guard.attempt_count,
                "query_encode_calls_total_including_smoke": service.query_encode_calls,
                "query_encode_items_total_including_smoke": service.query_encode_items,
                "document_encode_calls_total": service.document_encode_calls,
                "document_encode_items_total": service.document_encode_items,
                "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
                "normal_repository_order": list(benchmark.repository_ids),
                "reversed_repository_order": list(reversed(benchmark.repository_ids)),
                "normal_path_order": [item.path_id for item in FROZEN_PATHS],
                "interleaved_path_order": ["E", "C", "A", "D", "B"],
                "repeated_subset_query_ids": sorted(subset_strata),
            },
            "dependency_inventory_hash_before": dependency_hash_before,
            "dependency_inventory_hash_after": _dependency_identity(),
            "model_snapshot_metadata_identity_before": snapshot_identity_before,
            "model_snapshot_metadata_identity_after": _snapshot_metadata_identity(config.model_snapshot),
            "source_database_hash_before": source_hash_before,
            "source_database_hash_after": file_hash(config.source_database),
            "formal_database_hash_after_evaluation": file_hash(copied_database),
        }
        if completion["dependency_inventory_hash_after"] != dependency_hash_before:
            raise Phase6RunError("dependency inventory changed during formal evaluation")
        if completion["model_snapshot_metadata_identity_after"] != snapshot_identity_before:
            raise Phase6RunError("formal evaluation modified the model snapshot")
        if completion["source_database_hash_after"] != source_hash_before:
            raise Phase6RunError("formal evaluation modified the source database")
        if network_guard.attempt_count:
            raise Phase6RunError("formal evaluation attempted network access")
        immutable_write_json(final_directory / "runtime_completion.json", completion)
        hashes = write_phase6_artifacts(
            final_directory,
            manifest=manifest,
            records_by_repo_path=forward,
            determinism=determinism,
        )
    return {
        "status": "completed",
        "run_id": run_id,
        "run_directory": str(final_directory),
        "result_hash": hashes["result_hash"],
        "elapsed_seconds": time.perf_counter() - started,
        "repositories": sources,
        "embedding": identity.to_dict(),
        "network_attempt_count": network_guard.attempt_count,
        "determinism": determinism,
    }


def inspect_phase6_source_database(
    path: Path,
    benchmark: Phase6BenchmarkSnapshot,
) -> dict[str, dict[str, Any]]:
    connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        output: dict[str, dict[str, Any]] = {}
        repositories = {item["repository_id"]: item for item in benchmark.repositories}
        for repo in benchmark.repository_ids:
            frozen = repositories[repo]
            row = connection.execute(
                "SELECT id, repo_url, owner, repo, repository_revision FROM projects WHERE id = ?",
                (frozen["project_id"],),
            ).fetchone()
            if row is None:
                raise Phase6RunError(f"source database is missing frozen {repo} project")
            project = dict(row)
            if project["repository_revision"] != frozen["resolved_commit"]:
                raise Phase6RunError(f"source database {repo} revision differs from frozen benchmark")
            chunk_count = int(connection.execute("SELECT COUNT(*) FROM code_chunks WHERE project_id = ?", (project["id"],)).fetchone()[0])
            if chunk_count != int(frozen["chunk_count"]):
                raise Phase6RunError(f"source database {repo} chunk count differs from frozen benchmark")
            graph = relation_graph_identity(path, project["id"])
            if graph["status"] != "complete" or graph["repository_revision"] != project["repository_revision"]:
                raise Phase6RunError(f"source database {repo} relation graph is incomplete or stale")
            expected_graph = (benchmark.manifest.get(repo) or {}).get("graph_identity")
            if expected_graph and graph != expected_graph:
                raise Phase6RunError(f"source database {repo} relation graph identity changed")
            output[repo] = {
                **project,
                "project_id": project["id"],
                "chunk_count": chunk_count,
                "relation_graph": graph,
            }
        return output
    finally:
        connection.close()


def _build_manifest(
    *,
    benchmark: Phase6BenchmarkSnapshot,
    sources: dict[str, dict[str, Any]],
    source_database: Path,
    source_database_hash: str,
    model_snapshot: Path,
    embedding_identity: dict[str, Any],
    pooling: dict[str, Any],
    environment: dict[str, Any],
    index_results: dict[str, Any],
    embedding_rows: dict[str, int],
    historical_rows: int,
    smoke_latency_ms: float,
    timestamp: str,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    frozen = benchmark.manifest
    return {
        "evaluation_version": EVALUATION_VERSION,
        "timestamp": timestamp,
        "repository_commit": _git(root, "rev-parse", "HEAD"),
        "branch": _git(root, "branch", "--show-current"),
        "benchmark_commit": _git(root, "rev-parse", "f559fda248015e8107fb87aa4922ca1483c739b3^{commit}"),
        "benchmark_version": frozen["benchmark_version"],
        "dataset_hash": frozen["dataset_hash"],
        "query_hash": frozen["query_hash"],
        "gold_hash": frozen["gold_hash"],
        "matcher_hash": frozen["matcher_hash"],
        "strata_hash": frozen["strata_hash"],
        "protocol_hash": frozen["protocol_hash"],
        "repository_hash": frozen["repository_hash"],
        "query_count": frozen["total_query_count"],
        "answerable_query_count": frozen["total_answerable_query_count"],
        "unanswerable_query_count": frozen["total_unanswerable_query_count"],
        "repositories": {
            repo: {
                "project_id": source["project_id"],
                "repository_revision": source["repository_revision"],
                "chunk_count": source["chunk_count"],
                "relation_graph": source["relation_graph"],
                "query_count": len(benchmark.scenarios_by_repo[repo]),
                "answerable_query_count": len(benchmark.answerable_by_repo[repo]),
                "stratum_counts": benchmark.stratum_counts(repo),
                "embedding_index": index_results[repo],
                "embedding_row_count": embedding_rows[repo],
            }
            for repo, source in sorted(sources.items())
        },
        "source_database_path": str(source_database.resolve()),
        "source_database_hash": source_database_hash,
        "historical_embedding_rows_removed_from_copy": historical_rows,
        "retrieval_paths": [{"path_id": item.path_id, "label": item.label, **item.request_parameters} for item in FROZEN_PATHS],
        "formal_top_k": 8,
        "top_k_values": [1, 3, 5, 8],
        "metrics": ["hit_at_1", "hit_at_3", "hit_at_5", "hit_at_8", "mrr_at_8", "recall_at_8", "ndcg_at_8"],
        "aggregation": ["per_repository", "micro", "macro", "primary_stratum"],
        "paired_comparisons": ["B-A", "C-B", "D-B", "E-C", "E-D"],
        "random_seed": 20260726,
        "bootstrap_samples": 2_000,
        "bootstrap_method": "repository-stratified paired bootstrap",
        "embedding_identity": embedding_identity,
        "embedding_local_path": str(model_snapshot.resolve()),
        "embedding_revision": BGE_M3_SNAPSHOT_REVISION,
        "pooling": pooling,
        "query_prefix": "",
        "document_prefix": "",
        "model_smoke": {"status": "succeeded", "latency_ms": smoke_latency_ms, "dimension": 1024},
        "environment": environment,
        "network_allowed": False,
        "model_download_allowed": False,
        "dependency_changes_allowed": False,
        "production_retrieval_changes": "none",
    }


def _preflight_paths(config: RuntimeConfig) -> None:
    for label, path in (
        ("Phase 6 benchmark", config.phase6_benchmark_directory),
        ("Click benchmark", config.click_dataset_directory),
        ("source database", config.source_database),
        ("model snapshot", config.model_snapshot),
    ):
        if not path.exists():
            raise Phase6RunError(f"{label} does not exist: {path}")
    if not config.source_database.is_file():
        raise Phase6RunError("source database is not a file")
    required = ("config.json", "modules.json", "1_Pooling/config.json")
    if any(not (config.model_snapshot / value).is_file() for value in required):
        raise Phase6RunError("local BGE-M3 snapshot is incomplete")
    if config.model_snapshot.name != BGE_M3_SNAPSHOT_REVISION:
        raise Phase6RunError("local BGE-M3 snapshot revision differs from the frozen revision")


def _cache_namespace(repo: str, revision: str, cache_identity: str) -> str:
    return f"retrieval-v2-phase6:{repo}:{revision}:{cache_identity}"


def _embedding_rows_by_repository(
    path: Path,
    sources: dict[str, dict[str, Any]],
) -> dict[str, int]:
    connection = sqlite3.connect(path)
    try:
        return {
            repo: int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM code_chunk_embeddings AS e
                    INNER JOIN code_chunks AS c ON c.id = e.code_chunk_id
                    WHERE c.project_id = ?
                    """,
                    (source["project_id"],),
                ).fetchone()[0]
            )
            for repo, source in sorted(sources.items())
        }
    finally:
        connection.close()
