from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from app.config import (
    get_agent_limits,
    get_embedding_settings,
    get_llm_settings,
    get_repository_settings,
    load_environment,
)
from app.database import Database
from app.services.agent_core import run_bounded_agent
from app.services.analyzer import analyze_snapshot
from app.services.code_chunker import extract_python_code_chunks_from_files
from app.services.embedding_indexer import EmbeddingIndexer
from app.services.embedding_service import EmbeddingService
from app.services.learning_agent import build_learning_path
from app.services.llm_client import LLMClient, ProviderError
from app.services.relation_analysis import index_project_relations
from app.services.repository_import import (
    RepositoryImportError,
    import_local_repository,
    import_public_git_repository,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="RepoNoesis Local Product smoke gates")
    gate = parser.add_mutually_exclusive_group(required=True)
    gate.add_argument("--gate-a", action="store_true", help="real local BGE-M3 gate")
    gate.add_argument("--gate-b", metavar="PUBLIC_HTTPS_GIT_URL", help="public Git import gate")
    gate.add_argument("--gate-c", action="store_true", help="real configured provider gate")
    args = parser.parse_args()
    load_environment()
    try:
        if args.gate_b:
            imported = import_public_git_repository(args.gate_b, get_repository_settings())
            report = _run_pipeline(imported.snapshot, require_provider=False)
            report["gate"] = "B"
            report["source_type"] = "git_url"
        else:
            with tempfile.TemporaryDirectory(prefix="reponoesis-smoke-") as directory:
                fixture = _create_fixture(Path(directory))
                imported = import_local_repository(str(fixture), get_repository_settings())
                report = _run_pipeline(
                    imported.snapshot,
                    require_provider=bool(args.gate_c),
                )
            report["gate"] = "C" if args.gate_c else "A"
            report["source_type"] = "local"
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (RepositoryImportError, ProviderError) as exc:
        payload = exc.to_safe_dict()
        payload["gate"] = "C" if args.gate_c else ("B" if args.gate_b else "A")
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 2
    except Exception as exc:
        print(
            json.dumps(
                {
                    "code": "smoke_gate_failed",
                    "message": f"Smoke gate failed with {type(exc).__name__}.",
                    "gate": "C" if args.gate_c else ("B" if args.gate_b else "A"),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 1


def _run_pipeline(snapshot: Any, *, require_provider: bool) -> dict[str, Any]:
    llm = LLMClient(get_llm_settings()) if require_provider else None
    if llm is not None:
        llm.require_available()
    settings = get_embedding_settings()
    if not settings.enabled or not settings.offline or settings.provider != "local_bge_m3":
        raise RuntimeError("embedding product configuration is incomplete")
    embedding = EmbeddingService(settings)
    with tempfile.TemporaryDirectory(prefix="reponoesis-smoke-db-") as database_dir:
        database = Database(Path(database_dir) / "smoke.sqlite")
        project_id = database.create_project(snapshot.to_dict())
        analysis = analyze_snapshot(snapshot)
        chunks = extract_python_code_chunks_from_files(
            snapshot.files, snapshot.repository_revision
        )
        if not chunks.chunks:
            raise RuntimeError("no Python chunks were produced")
        files = [file.to_dict() for file in snapshot.files]
        public = {file["path"]: file for file in analysis["files"]}
        for file in files:
            if file["path"] in public:
                file.update(public[file["path"]])
        database.save_analysis(
            project_id,
            analysis,
            files,
            build_learning_path(
                {"id": project_id, "repo": snapshot.repo, "repo_url": snapshot.repo_url},
                analysis,
                None,
            ),
            [chunk.to_dict() for chunk in chunks.chunks],
        )
        relation = index_project_relations(database, project_id)
        embedding_result = EmbeddingIndexer(database, embedding).index_project(project_id)
        bundle = database.get_bundle(project_id)
        assert bundle is not None
        question = (
            "What does repository_summary return?"
            if snapshot.repo == "smoke-fixture"
            else "Which Python functions are central to this repository?"
        )
        result = run_bounded_agent(
            question,
            bundle,
            llm,
            database,
            embedding,
            evidence_count=5,
            limits=get_agent_limits(),
            retrieval_version="v2",
        )
        if not result.get("citations") or not result.get("evidence"):
            raise RuntimeError("validated evidence citations were not produced")
        if require_provider and (
            result.get("answer_mode") != "llm_grounded"
            or result.get("agent_mode") != "bounded"
        ):
            raise RuntimeError("provider answer did not pass the bounded grounding gate")
        identity = embedding.get_effective_embedding_identity()
        return {
            "status": "pass",
            "repository_revision": snapshot.repository_revision,
            "python_file_count": sum(file.path.endswith(".py") for file in snapshot.files),
            "code_chunk_count": len(chunks.chunks),
            "embedding": {
                "provider": settings.provider,
                "model": identity.model_name,
                "device": identity.device,
                "dimension": identity.dimension,
                "offline": settings.offline,
                "is_real": identity.is_real,
                "generated_chunks": embedding_result.generated_chunks,
                "cached_chunks": embedding_result.cached_chunks,
            },
            "relation": {
                "status": relation.status,
                "node_count": len(relation.nodes),
                "edge_count": len(relation.edges),
            },
            "answer": {
                "provider": llm.provider_name if llm else "none",
                "model": llm.model if llm else None,
                "planner_thinking": (
                    (llm.settings.planner_thinking or "omitted")
                    if llm
                    else "omitted"
                ),
                "answer_thinking": (
                    (llm.settings.answer_thinking or "omitted")
                    if llm
                    else "omitted"
                ),
                "agent_mode": result.get("agent_mode"),
                "answer_mode": result.get("answer_mode"),
                "grounding_status": result.get("grounding_status"),
                "citation_count": len(result["citations"]),
                "evidence_count": len(result["evidence"]),
            },
        }


def _create_fixture(parent: Path) -> Path:
    root = parent / "smoke-fixture"
    root.mkdir()
    (root / "app.py").write_text(
        "def repository_summary(files):\n"
        "    \"\"\"Return the number of analyzed source files.\"\"\"\n"
        "    return {\"file_count\": len(files)}\n",
        encoding="utf-8",
    )
    commands = (
        ["git", "init", "-b", "main", str(root)],
        ["git", "-C", str(root), "config", "user.name", "RepoNoesis Smoke"],
        ["git", "-C", str(root), "config", "user.email", "smoke@example.invalid"],
        ["git", "-C", str(root), "add", "app.py"],
        ["git", "-C", str(root), "commit", "-m", "smoke fixture"],
    )
    for command in commands:
        subprocess.run(command, check=True, capture_output=True, timeout=30)
    return root


if __name__ == "__main__":
    raise SystemExit(main())
