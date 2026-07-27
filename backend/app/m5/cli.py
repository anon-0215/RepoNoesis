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
        command.add_argument("--cell", action="append", help="exact scenario::mode cell; repeat for a finite plan")
        command.add_argument("--scenario", action="append")
        command.add_argument("--repo", action="append")
        command.add_argument("--batch-index", type=int, default=0)
        command.add_argument("--batch-count", type=int, default=1)
        command.add_argument("--run-purpose", choices=("smoke", "pilot", "full"), default="smoke")
        command.add_argument("--answer-request-limit", type=int, default=24)
        command.add_argument("--evaluator-request-limit", type=int, default=24)
        command.add_argument("--answer-input-token-limit", type=int, default=1_000_000)
        command.add_argument("--answer-output-token-limit", type=int, default=250_000)
        command.add_argument("--evaluator-input-token-limit", type=int, default=100_000)
        command.add_argument("--evaluator-output-token-limit", type=int, default=25_000)
        command.add_argument("--answer-cost-limit-usd", type=float)
        command.add_argument("--evaluator-cost-limit-usd", type=float)
        command.add_argument("--wall-clock-limit-seconds", type=float, default=3_600.0)
        command.add_argument("--request-timeout-seconds", type=float, default=60.0)
        command.add_argument("--seed", type=int, default=20260726)
        command.add_argument("--live", action="store_true")
        command.add_argument("--resume", action="store_true")
        command.add_argument("--retry-failed", action="store_true")
    return parser


def _config(args: argparse.Namespace) -> BenchmarkConfig:
    cell_modes = sorted({cell.rsplit("::", 1)[1] for cell in (args.cell or []) if "::" in cell})
    return BenchmarkConfig(
        dataset_directory=args.dataset,
        repository_root=args.repository_root,
        artifacts_directory=args.artifacts,
        modes=args.mode or cell_modes or ["fixed_lexical_rag"],
        cells=args.cell or [],
        scenario_ids=args.scenario or [],
        repo_ids=args.repo or [],
        batch_index=args.batch_index,
        batch_count=args.batch_count,
        run_purpose=args.run_purpose,
        maximum_answer_requests=args.answer_request_limit,
        maximum_evaluator_requests=args.evaluator_request_limit,
        maximum_answer_input_tokens=args.answer_input_token_limit,
        maximum_answer_output_tokens=args.answer_output_token_limit,
        maximum_evaluator_input_tokens=args.evaluator_input_token_limit,
        maximum_evaluator_output_tokens=args.evaluator_output_token_limit,
        maximum_answer_cost_usd=args.answer_cost_limit_usd,
        maximum_evaluator_cost_usd=args.evaluator_cost_limit_usd,
        maximum_wall_clock_seconds=args.wall_clock_limit_seconds,
        timeout_seconds=args.request_timeout_seconds,
        random_seed=args.seed,
        dry_run=args.command == "dry-run",
    )


if __name__ == "__main__":
    raise SystemExit(main())
