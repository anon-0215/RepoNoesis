from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.retrieval_phase6.runtime import RuntimeConfig, run_formal_evaluation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run frozen Retrieval v2 Phase 6 cross-repository evaluation")
    parser.add_argument("--phase6-benchmark", type=Path, required=True)
    parser.add_argument("--click-dataset", type=Path, required=True)
    parser.add_argument("--source-database", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_formal_evaluation(
        RuntimeConfig(
            phase6_benchmark_directory=args.phase6_benchmark,
            click_dataset_directory=args.click_dataset,
            source_database=args.source_database,
            model_snapshot=args.model_snapshot,
            artifact_root=args.artifacts,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
