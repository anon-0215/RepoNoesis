from __future__ import annotations

from typing import Any

from app.database import Database
from app.models import RepositorySnapshot
from app.services.analyzer import analyze_snapshot
from app.services.code_chunker import extract_python_code_chunks_from_files
from app.services.embedding_indexer import EmbeddingIndexer
from app.services.learning_agent import build_learning_path
from app.services.relation_analysis import index_project_relations


def build_fixture_snapshot(
    database: Database,
    embedding_service: Any,
    project_id: str,
    snapshot: RepositorySnapshot,
) -> None:
    analysis = analyze_snapshot(snapshot)
    chunks = extract_python_code_chunks_from_files(
        snapshot.files, snapshot.repository_revision
    )
    enriched = [file.to_dict() for file in snapshot.files]
    by_path = {file["path"]: file for file in enriched}
    for public_file in analysis.get("files", []):
        by_path[public_file["path"]].update(public_file)
    project = {"id": project_id, "repo": snapshot.repo, "repo_url": snapshot.repo_url}
    database.save_analysis(
        project_id,
        analysis,
        list(by_path.values()),
        build_learning_path(project, analysis, None),
        [chunk.to_dict() for chunk in chunks.chunks],
    )
    index_project_relations(database, project_id)
    EmbeddingIndexer(database, embedding_service).index_project(project_id)
