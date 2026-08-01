from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.retrieval_phase5.runtime import RuntimeConfig, run_formal_evaluation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run frozen Retrieval v2 Phase 5 offline evaluation")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--source-database", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--resume-database", type=Path)
    args = parser.parse_args(argv)
    result = run_formal_evaluation(
        RuntimeConfig(
            dataset_directory=args.dataset,
            repository_root=args.repository_root,
            source_database=args.source_database,
            model_snapshot=args.model_snapshot,
            artifact_root=args.artifacts,
            resume_database=args.resume_database,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
