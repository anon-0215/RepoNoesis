from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

from app.database import Database
from app.m5.contracts import RepositorySpec
from app.models import RepoFile, RepositorySnapshot
from app.services.analyzer import analyze_snapshot
from app.services.code_chunker import extract_python_code_chunks_from_files
from app.services.embedding_indexer import EmbeddingIndexer
from app.services.embedding_service import EmbeddingService
from app.services.relation_analysis import index_project_relations


def ingest_repository_snapshot(
    database: Database,
    spec: RepositorySpec,
    repository_root: Path,
    *,
    embedding_service: EmbeddingService | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Read and statically index a fixed checkout without importing or executing it."""
    checkout = (repository_root.resolve() / spec.checkout_name).resolve()
    if repository_root.resolve() not in checkout.parents:
        raise ValueError("repository checkout escapes configured root")
    head = _git(checkout, "rev-parse", "HEAD")
    if head != spec.exact_commit_sha:
        raise ValueError("repository revision changed after dataset validation")
    excluded = tuple(value.rstrip("/") + "/" for value in spec.excluded_paths)
    paths = _git(checkout, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
    selected = [
        value
        for value in paths
        if value.endswith((".py", ".toml", ".md", ".txt"))
        and not any(value.startswith(prefix) for prefix in excluded)
    ][: spec.analysis_configuration.maximum_files]
    files: list[RepoFile] = []
    skipped_large = 0
    for relative in selected:
        path = (checkout / PurePosixPath(relative)).resolve()
        if checkout not in path.parents or path.is_symlink() or not path.is_file():
            continue
        size = path.stat().st_size
        if size > spec.analysis_configuration.maximum_file_bytes:
            skipped_large += 1
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        files.append(RepoFile(path=relative, size=size, content=content))
    snapshot = RepositorySnapshot(
        repo_url=spec.source_url,
        owner=spec.source_url.rstrip("/").split("/")[-2],
        repo=spec.display_name,
        default_branch=spec.default_branch,
        repository_revision=spec.exact_commit_sha,
        files=files,
    )
    project_id = database.create_project(snapshot.to_dict())
    analysis = analyze_snapshot(snapshot)
    chunks = extract_python_code_chunks_from_files(files, spec.exact_commit_sha)
    enriched = [item.to_dict() for item in files]
    by_path = {item["path"]: item for item in enriched}
    for public in analysis["files"]:
        by_path[public["path"]].update(public)
    database.save_analysis(
        project_id,
        analysis,
        list(by_path.values()),
        [],
        [item.to_dict() for item in chunks.chunks],
    )
    relation = index_project_relations(database, project_id)
    embedding_result: dict[str, Any] | None = None
    if embedding_service and embedding_service.settings.enabled:
        embedding_result = EmbeddingIndexer(database, embedding_service).index_project(project_id).to_dict()
    bundle = database.get_bundle(project_id)
    if bundle is None:
        raise RuntimeError("benchmark snapshot was not persisted")
    return project_id, bundle, {
        "file_count": len(files),
        "chunk_count": len(chunks.chunks),
        "chunk_warning_count": len(chunks.warnings),
        "skipped_large_file_count": skipped_large,
        "relation_status": relation.status,
        "relation_node_count": len(relation.nodes),
        "relation_edge_count": len(relation.edges),
        "embedding": embedding_result,
        "target_repository_execution_count": 0,
        "target_repository_import_count": 0,
        "shell_tool_count": 0,
    }


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    return completed.stdout.strip()
