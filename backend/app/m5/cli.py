from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.m5.contracts import BenchmarkConfig, EXPERIMENT_MODES
from app.m5.dataset import BenchmarkDatasetValidator
from app.m5.runner import BenchmarkRunner, compare_run_records, generate_markdown_report


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "list-modes":
        print(json.dumps({"modes": list(EXPERIMENT_MODES)}, indent=2))
        return 0
    if args.command == "validate":
        report = BenchmarkDatasetValidator(Path(args.dataset), Path(args.repository_root)).validate()
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return 0 if report.valid else 2
    if args.command == "compare":
        result = compare_run_records(Path(args.left), Path(args.right), args.metric, args.seed)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "report":
        markdown = generate_markdown_report(
            Path(args.run_directory), Path(args.output) if args.output else None
        )
        print(markdown)
        return 0
    config = _config(args)
    runner = BenchmarkRunner(config, live=args.live)
    if args.command == "dry-run":
        print(json.dumps(runner.dry_run(), ensure_ascii=False, indent=2))
        return 0
    result = runner.run(resume=args.resume, retry_failed=args.retry_failed)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RepoNoesis M5 benchmark runner")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list-modes")
    validate = sub.add_parser("validate")
    validate.add_argument("--dataset", required=True)
    validate.add_argument("--repository-root", required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.add_argument("--metric", default="hit_at_5")
    compare.add_argument("--seed", type=int, default=20260726)
    report = sub.add_parser("report")
    report.add_argument("--run-directory", required=True)
    report.add_argument("--output")
    for name in ("dry-run", "run"):
        command = sub.add_parser(name)
        command.add_argument("--dataset", required=True)
        command.add_argument("--repository-root", required=True)
        command.add_argument("--artifacts", required=True)
        command.add_argument("--mode", action="append", choices=EXPERIMENT_MODES)
        command.add_argument("--scenario", action="append")
        command.add_argument("--repo", action="append")
        command.add_argument("--seed", type=int, default=20260726)
        command.add_argument("--live", action="store_true")
        command.add_argument("--resume", action="store_true")
        command.add_argument("--retry-failed", action="store_true")
    return parser


def _config(args: argparse.Namespace) -> BenchmarkConfig:
    return BenchmarkConfig(
        dataset_directory=args.dataset,
        repository_root=args.repository_root,
        artifacts_directory=args.artifacts,
        modes=args.mode or ["fixed_lexical_rag"],
        scenario_ids=args.scenario or [],
        repo_ids=args.repo or [],
        random_seed=args.seed,
        dry_run=args.command == "dry-run",
    )


if __name__ == "__main__":
    raise SystemExit(main())
