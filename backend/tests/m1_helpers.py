from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from app.config import EmbeddingSettings
from app.database import Database
from app.services.embedding_service import EmbeddingService


REVISION = "revision-m1"


def make_project(
    database: Database,
    files_and_chunks: list[tuple[str, str, str]],
) -> tuple[str, dict[str, Any]]:
    project_id = database.create_project(
        {
            "repo_url": "https://github.com/demo/reponoesis-fixture",
            "owner": "demo",
            "repo": "reponoesis-fixture",
            "default_branch": "main",
            "repository_revision": REVISION,
        }
    )
    files = []
    chunks = []
    for index, (path, symbol, content) in enumerate(files_and_chunks, start=1):
        files.append(
            {
                "path": path,
                "extension": Path(path).suffix,
                "language": "Python",
                "size": len(content.encode("utf-8")),
                "content": content,
                "summary": symbol,
                "importance": 100 - index,
                "is_core": True,
                "imports": [],
                "exports": [],
                "symbols": [symbol],
            }
        )
        chunks.append(make_chunk(path, symbol, content))
    database.save_analysis(
        project_id,
        {
            "primary_language": "Python",
            "frameworks": [],
            "files": files,
            "modules": [],
            "overview": "M1 fixture",
        },
        files,
        [],
        chunks,
    )
    bundle = database.get_bundle(project_id)
    assert bundle is not None
    return project_id, bundle


def make_chunk(
    path: str,
    symbol: str,
    content: str,
    *,
    start_line: int = 1,
    revision: str = REVISION,
) -> dict[str, Any]:
    return {
        "repository_revision": revision,
        "language": "python",
        "path": path,
        "chunk_type": "method" if "." in symbol else "function",
        "symbol_name": symbol.rsplit(".", 1)[-1],
        "qualified_name": symbol,
        "parent_symbol": symbol.rsplit(".", 1)[0] if "." in symbol else "",
        "start_line": start_line,
        "end_line": start_line + len(content.splitlines()) - 1,
        "content": content,
        "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }


def disabled_embedding_service() -> EmbeddingService:
    return EmbeddingService(
        EmbeddingSettings(
            enabled=False,
            model_name_or_path="fake-model",
            device="cpu",
            batch_size=4,
            max_length=128,
            normalize=True,
            cache_dir=Path("unused-cache"),
            query_prefix="",
            document_prefix="",
            model_revision="fake-revision",
        ),
        backend_factory=lambda: None,
        cuda_available=lambda: False,
    )
