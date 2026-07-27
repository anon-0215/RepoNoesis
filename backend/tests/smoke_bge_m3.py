from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

os.environ.setdefault("EMBEDDING_ENABLED", "true")
os.environ.setdefault("EMBEDDING_DEVICE", "cpu")
os.environ.setdefault("EMBEDDING_MODEL_NAME_OR_PATH", "BAAI/bge-m3")
os.environ.setdefault("EMBEDDING_BATCH_SIZE", "2")
os.environ.setdefault("EMBEDDING_MAX_LENGTH", "512")
os.environ.setdefault("EMBEDDING_NORMALIZE", "true")
os.environ.setdefault(
    "EMBEDDING_CACHE_DIR",
    str(Path(__file__).resolve().parents[2] / "embedding_cache" / "bge_m3_smoke"),
)

from app.config import get_embedding_settings  # noqa: E402
from app.database import Database  # noqa: E402
from app.services.embedding_indexer import EmbeddingIndexer  # noqa: E402
from app.services.embedding_service import (  # noqa: E402
    EmbeddingService,
    build_code_chunk_document_text,
)
from app.services.semantic_retriever import SemanticRetriever  # noqa: E402


QUERY = "用户身份是如何验证的？"


def main() -> int:
    settings = get_embedding_settings()
    service = EmbeddingService(settings)
    chunks = _smoke_chunks()
    document_texts = [build_code_chunk_document_text(chunk) for chunk in chunks]

    started = perf_counter()
    service.load_model()
    load_ms = _elapsed_ms(started)
    identity = service.get_model_identity()
    backend_id = id(service._backend)  # Smoke assertion for lazy singleton behavior.
    model_id = id(getattr(service._backend, "_model", None))
    service.load_model()
    assert backend_id == id(service._backend)
    assert model_id == id(getattr(service._backend, "_model", None))

    started = perf_counter()
    document_vectors = service.encode_documents(document_texts)
    document_encode_ms = _elapsed_ms(started)
    document_array = np.asarray(document_vectors, dtype=np.float32)
    assert document_array.dtype == np.float32
    assert document_array.shape == (3, 1024)
    assert np.isfinite(document_array).all()
    for vector in document_array:
        assert math.isclose(float(np.linalg.norm(vector)), 1.0, rel_tol=2e-3, abs_tol=2e-3)

    started = perf_counter()
    query_vector = service.encode_query(QUERY)
    query_encode_ms = _elapsed_ms(started)
    query_array = np.asarray(query_vector, dtype=np.float32)
    assert query_array.dtype == np.float32
    assert query_array.shape == (1024,)
    assert np.isfinite(query_array).all()
    assert math.isclose(float(np.linalg.norm(query_array)), 1.0, rel_tol=2e-3, abs_tol=2e-3)

    scores = document_array @ query_array
    ranking = sorted(
        [
            {
                "qualified_name": chunk["qualified_name"],
                "path": chunk["path"],
                "score": float(score),
            }
            for chunk, score in zip(chunks, scores)
        ],
        key=lambda item: (-item["score"], item["path"], item["qualified_name"]),
    )
    assert ranking[0]["qualified_name"] == "authenticate_user", ranking

    integration = _run_integration_smoke(service, chunks)
    report = {
        "model_name": identity.model_name,
        "configured_revision": identity.configured_revision,
        "resolved_revision": identity.resolved_revision,
        "local_snapshot_identity": identity.local_snapshot_identity,
        "model_identity": identity.model_identity,
        "device": identity.device,
        "python": sys.version.split()[0],
        "versions": _versions(),
        "load_ms": load_ms,
        "document_encode_ms": document_encode_ms,
        "query_encode_ms": query_encode_ms,
        "dimension": int(document_array.shape[1]),
        "dtype": str(document_array.dtype),
        "document_count": int(document_array.shape[0]),
        "query_count": 1,
        "document_norms": [float(np.linalg.norm(vector)) for vector in document_array],
        "query_norm": float(np.linalg.norm(query_array)),
        "ranking": ranking,
        "integration": integration,
        "memory_mb": _memory_mb(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _run_integration_smoke(
    service: EmbeddingService,
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        db = Database(Path(directory) / "bge_m3_smoke.sqlite")
        project_id = db.create_project(
            {
                "repo_url": "https://example.com/smoke/repo",
                "owner": "smoke",
                "repo": "repo",
                "default_branch": "main",
            }
        )
        db.save_code_chunks_for_project(project_id, chunks)
        indexer = EmbeddingIndexer(db, service)
        first = indexer.index_project(project_id)
        assert first.generated_chunks == 3, first
        assert first.cached_chunks == 0, first
        second = indexer.index_project(project_id)
        assert second.generated_chunks == 0, second
        assert second.cached_chunks == 3, second

        changed_chunks = [
            _chunk(
                "src/auth.py",
                "function",
                "authenticate_user",
                10,
                "def authenticate_user(username, password):\n"
                "    user = find_user(username)\n"
                "    return user if user and verify_password(password, user.password_hash) else None\n",
            ),
            chunks[1],
            chunks[2],
        ]
        db.save_code_chunks_for_project(project_id, changed_chunks)
        third = indexer.index_project(project_id)
        assert third.generated_chunks == 1, third
        assert third.cached_chunks == 2, third

        outcome = SemanticRetriever(db, service).search(project_id, QUERY, top_k=3)
        assert outcome.status == "ok", outcome
        assert outcome.results[0].qualified_name == "authenticate_user", outcome
        first_result = outcome.results[0]
        assert first_result.path == "src/auth.py"
        assert first_result.start_line == 10
        assert first_result.end_line == 12
        return {
            "first": first.to_dict(),
            "second": second.to_dict(),
            "after_auth_change": third.to_dict(),
            "retrieval_top": first_result.to_dict(),
            "retrieval_ranking": [
                {
                    "qualified_name": result.qualified_name,
                    "path": result.path,
                    "start_line": result.start_line,
                    "end_line": result.end_line,
                    "score": result.semantic_score,
                }
                for result in outcome.results
            ],
        }


def _smoke_chunks() -> list[dict[str, Any]]:
    return [
        _chunk(
            "src/auth.py",
            "function",
            "authenticate_user",
            10,
            "def authenticate_user(username, password):\n"
            "    user = find_user(username)\n"
            "    return user if verify_password(password, user.password_hash) else None\n",
        ),
        _chunk(
            "src/upload.py",
            "function",
            "upload_file",
            20,
            "def upload_file(file):\n"
            "    return storage.save(file)\n",
        ),
        _chunk(
            "src/db.py",
            "function",
            "initialize_database",
            30,
            "def initialize_database():\n"
            "    create_tables()\n",
        ),
    ]


def _chunk(
    path: str,
    chunk_type: str,
    qualified_name: str,
    start_line: int,
    content: str,
) -> dict[str, Any]:
    return {
        "repository_revision": "smoke-revision",
        "language": "python",
        "path": path,
        "chunk_type": chunk_type,
        "symbol_name": qualified_name.rsplit(".", 1)[-1],
        "qualified_name": qualified_name,
        "parent_symbol": qualified_name.rsplit(".", 1)[0] if "." in qualified_name else "",
        "start_line": start_line,
        "end_line": start_line + len(content.splitlines()) - 1,
        "content": content,
        "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }


def _versions() -> dict[str, str]:
    import sentence_transformers
    import tokenizers
    import torch
    import transformers

    return {
        "torch": torch.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
    }


def _memory_mb() -> float | None:
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32.dll")
        psapi = ctypes.WinDLL("psapi.dll")
        handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(
            handle,
            ctypes.byref(counters),
            counters.cb,
        )
        if not ok:
            return None
        return round(counters.WorkingSetSize / (1024 * 1024), 1)
    except Exception:
        return None


def _elapsed_ms(started: float) -> int:
    return int((perf_counter() - started) * 1000)


if __name__ == "__main__":
    raise SystemExit(main())
