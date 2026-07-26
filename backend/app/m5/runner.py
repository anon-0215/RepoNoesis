from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_agent_limits, get_embedding_settings, get_env_value
from app.database import Database
from app.m5.contracts import BenchmarkConfig, EXPERIMENT_MODES, ProviderIdentity, RepositorySpec
from app.m5.dataset import BenchmarkDatasetValidator, LoadedDataset
from app.m5.embedding import M5EmbeddingProvider, fake_embedding_service
from app.m5.metrics import aggregate_results, paired_delta, scenario_metrics
from app.m5.modes import ModeExecutionError, execute_mode
from app.m5.learning_sequence import run_adaptive_sequence
from app.m5.providers import (
    BudgetedProvider,
    FakeDeterministicProvider,
    OpenAICompatibleProvider,
    OpenAICompatibleSettings,
    ProviderLLMClient,
    StructuredLearningEvaluator,
    redact_secrets,
)
from app.m5.repository import ingest_repository_snapshot
from app.services.embedding_indexer import EmbeddingIndexer


CHECKPOINT_SCHEMA_VERSION = 1


class BenchmarkCheckpointError(ValueError):
    pass


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig, *, live: bool = False) -> None:
        self.config = config
        self.live = live
        self.dataset_validator = BenchmarkDatasetValidator(
            Path(config.dataset_directory), Path(config.repository_root)
        )

    def dry_run(self) -> dict[str, Any]:
        dataset = self.dataset_validator.load_validated()
        scenarios = self._selected_scenarios(dataset)
        return {
            "status": "validated",
            "dry_run": True,
            "dataset_version": dataset.manifest.dataset_version,
            "repositories": sorted({item.repo_id for item in scenarios}),
            "scenario_ids": [item.scenario_id for item in scenarios],
            "modes": list(self.config.modes),
            "planned_answer_runs": sum(mode != "m4_adaptive_sequence" for mode in self.config.modes)
            * len(scenarios),
            "planned_sequence_runs": (
                len(dataset.sequences) if "m4_adaptive_sequence" in self.config.modes else 0
            ),
            "real_provider_calls_allowed": self.live,
        }

    def run(self, *, resume: bool = False, retry_failed: bool = False) -> dict[str, Any]:
        dataset = self.dataset_validator.load_validated()
        scenarios = self._selected_scenarios(dataset)
        provider = self._provider()
        embedding, embedding_kind = self._embedding_service()
        run_id = compute_run_id(self.config, dataset, provider.identity, _code_identity())
        run_dir = Path(self.config.artifacts_directory).resolve() / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = run_dir / "checkpoint.json"
        checkpoint = _load_checkpoint(checkpoint_path) if resume and checkpoint_path.exists() else None
        records = list((checkpoint or {}).get("records", []))
        completed = {
            (item["scenario_id"], item["experiment_mode"]): item
            for item in records
            if item.get("scenario_status") == "succeeded"
            or (not retry_failed and item.get("scenario_status") in {"failed", "timed_out", "cancelled", "degraded"})
        }
        safe_config = _safe_config(self.config)
        manifest = {
            "benchmark_schema_version": 1,
            "metric_schema_version": 1,
            "run_id": run_id,
            "started_at": (checkpoint or {}).get("started_at") or _utc_now(),
            "ended_at": None,
            "host": {"platform": platform.platform(), "python": sys.version},
            "project_code_identity": _code_identity(),
            "dataset_version": dataset.manifest.dataset_version,
            "repository_revisions": {
                item.repo_id: item.exact_commit_sha for item in dataset.repositories
            },
            "modes": list(self.config.modes),
            "safe_config": safe_config,
            "answer_provider": provider.identity.to_dict(),
            "judge_independence": "same_model",
            "embedding_provider": {
                "provider": embedding_kind,
                "model": embedding.settings.model_name_or_path,
                "model_revision": embedding.settings.model_revision,
                "device": embedding.get_model_identity().device,
                "dtype": "float32",
                "normalized": embedding.settings.normalize,
                "max_length": embedding.settings.max_length,
            },
            "prompt_versions": [self.config.prompt_version],
            "metric_versions": [self.config.metric_version],
            "evaluator_version": self.config.evaluator_version,
            "random_seed": self.config.random_seed,
            "live": self.live,
        }
        _atomic_json(run_dir / "manifest.json", manifest)
        database = Database(run_dir / "benchmark.sqlite")
        repo_by_id = {item.repo_id: item for item in dataset.repositories}
        project_map = _load_json_if_exists(run_dir / "projects.json") or {}
        llm = ProviderLLMClient(
            provider,
            maximum_attempts=self.config.maximum_provider_attempts,
            seed=self.config.random_seed,
        )
        for scenario in scenarios:
            spec = repo_by_id[scenario.repo_id]
            project_id, bundle = self._ensure_project(
                database, embedding, spec, project_map, run_dir
            )
            for mode in self.config.modes:
                if mode == "m4_adaptive_sequence":
                    continue
                identity = (scenario.scenario_id, mode)
                if identity in completed:
                    continue
                before = len(llm.results)
                started = time.monotonic()
                try:
                    result = execute_mode(
                        mode,
                        scenario,
                        bundle,
                        database,
                        embedding,
                        llm,
                        limits=get_agent_limits(),
                        deterministic_planner=not self.live,
                        learning_context=_fixed_learning_context(scenario),
                    )
                    status = "succeeded"
                    if result.get("agent_status") in {"budget_exhausted", "cancelled"}:
                        status = "degraded"
                    error = None
                except Exception as exc:
                    result = {}
                    status = "timed_out" if isinstance(exc, TimeoutError) else "failed"
                    error = {"type": type(exc).__name__, "message": str(exc)[:500]}
                latency_ms = int((time.monotonic() - started) * 1000)
                provider_results = [item.safe_dict() for item in llm.results[before:]]
                result["scenario_status"] = status
                result["latency_ms"] = latency_ms
                result["provider_error"] = next(
                    (item.get("error_type") for item in provider_results if item.get("error_type")), None
                )
                record = {
                    "scenario_id": scenario.scenario_id,
                    "repo_id": scenario.repo_id,
                    "repository_revision": scenario.repository_revision,
                    "category": scenario.category,
                    "experiment_mode": mode,
                    "attempt_number": 1 + sum(
                        item.get("scenario_id") == scenario.scenario_id
                        and item.get("experiment_mode") == mode
                        for item in records
                    ),
                    "scenario_status": status,
                    "error": error,
                    "latency_ms": latency_ms,
                    "provider_calls": provider_results,
                    "provider_call_count": len(provider_results),
                    "estimated_cost_usd": _cost(provider_results),
                    "unknown_cost_count": sum(
                        item.get("usage", {}).get("estimated_cost_usd") == "unknown"
                        for item in provider_results
                    ),
                    "degraded_flags": _degraded_flags(result),
                    "result": _bounded_result(result),
                }
                record["metrics"] = scenario_metrics(scenario, {**result, "scenario_status": status})
                records.append(record)
                completed[identity] = record
                _write_checkpoint(checkpoint_path, run_id, manifest["started_at"], records)
        sequence_records: list[dict[str, Any]] = []
        if "m4_adaptive_sequence" in self.config.modes:
            remaining_calls = max(1, self.config.maximum_llm_calls - provider.call_count)
            evaluator_provider = self._evaluator_provider(remaining_calls)
            evaluator = StructuredLearningEvaluator(
                evaluator_provider, seed=self.config.random_seed
            )
            for sequence in sorted(dataset.sequences, key=lambda item: item.sequence_id):
                if self.config.repo_ids and sequence.repo_id not in self.config.repo_ids:
                    continue
                sequence_db_path = run_dir / "sequences" / f"{sequence.sequence_id}.sqlite"
                sequence_db = Database(sequence_db_path)
                sequence_spec = repo_by_id[sequence.repo_id]
                sequence_project_id, _, _ = ingest_repository_snapshot(
                    sequence_db,
                    sequence_spec,
                    Path(self.config.repository_root),
                    embedding_service=None,
                )
                sequence_records.append(
                    run_adaptive_sequence(
                        sequence_db,
                        sequence_project_id,
                        sequence,
                        evaluator,
                    )
                )
            _atomic_json(
                run_dir / "sequence_results.json",
                {"run_id": run_id, "records": sequence_records},
            )
        ordered = sorted(records, key=lambda item: (item["scenario_id"], item["experiment_mode"], item["attempt_number"]))
        summary = aggregate_results(ordered)
        summary["by_mode"] = {
            mode: aggregate_results([item for item in ordered if item["experiment_mode"] == mode])
            for mode in self.config.modes
            if mode != "m4_adaptive_sequence"
        }
        all_calls = [call for item in ordered for call in item.get("provider_calls", [])]
        unknown_usage = sum(
            call.get("usage", {}).get("total_tokens") is None for call in all_calls
        )
        summary["provider_usage"] = {
            "llm_calls": len(all_calls),
            "input_tokens": sum(call.get("usage", {}).get("input_tokens") or 0 for call in all_calls),
            "output_tokens": sum(call.get("usage", {}).get("output_tokens") or 0 for call in all_calls),
            "total_tokens": sum(call.get("usage", {}).get("total_tokens") or 0 for call in all_calls),
            "usage_unknown_count": unknown_usage,
            "estimated_cost_usd": _cost(all_calls),
            "unknown_cost_count": sum(
                call.get("usage", {}).get("estimated_cost_usd") == "unknown" for call in all_calls
            ),
            "evaluator_calls": len(getattr(evaluator if sequence_records else None, "results", [])),
            "embedding_time_ms": sum(
                int((value.get("stats", {}).get("embedding") or {}).get("duration_ms", 0))
                for value in project_map.values()
            ),
        }
        comparisons = _mode_comparisons(ordered, self.config.random_seed)
        _atomic_json(run_dir / "comparisons.json", comparisons)
        summary["adaptive_sequences"] = {
            "count": len(sequence_records),
            "successful_count": sum(item.get("status") == "succeeded" for item in sequence_records),
            "failed_count": sum(item.get("status") != "succeeded" for item in sequence_records),
        }
        final_manifest = {
            **manifest,
            "ended_at": _utc_now(),
            "record_count": len(ordered),
            "sequence_record_count": len(sequence_records),
            "embedding_provider": {
                **manifest["embedding_provider"],
                "model_revision": embedding.get_model_identity().model_revision,
                "device": embedding.get_model_identity().device,
                "dimension": embedding.get_embedding_dimension(),
            },
        }
        _atomic_json(run_dir / "manifest.json", final_manifest)
        _atomic_json(run_dir / "results.json", {"run_id": run_id, "records": ordered})
        _atomic_json(run_dir / "summary.json", summary)
        return {"run_id": run_id, "run_directory": str(run_dir), "summary": summary}

    def _selected_scenarios(self, dataset: LoadedDataset) -> list[Any]:
        selected = [
            item for item in dataset.scenarios
            if (not self.config.scenario_ids or item.scenario_id in self.config.scenario_ids)
            and (not self.config.repo_ids or item.repo_id in self.config.repo_ids)
        ]
        if self.config.scenario_ids:
            missing = sorted(set(self.config.scenario_ids) - {item.scenario_id for item in selected})
            if missing:
                raise ValueError(f"unknown selected scenarios: {missing}")
        return sorted(selected, key=lambda item: item.scenario_id)

    def _provider(self) -> BudgetedProvider:
        if not self.live:
            base: Any = FakeDeterministicProvider()
        else:
            settings = OpenAICompatibleSettings(
                base_url=get_env_value("LLM_BASE_URL", "https://api.deepseek.com"),
                api_key=get_env_value("LLM_API_KEY", ""),
                model=get_env_value("LLM_MODEL", "deepseek-chat"),
                model_revision=get_env_value("M5_LLM_MODEL_REVISION", "provider-managed"),
                allow_network=_env_gate("M5_ALLOW_NETWORK"),
                allow_real_llm=_env_gate("M5_ALLOW_REAL_LLM"),
                allow_paid_eval=_env_gate("M5_ALLOW_PAID_EVAL"),
            )
            base = OpenAICompatibleProvider(settings)
        return BudgetedProvider(
            base,
            maximum_calls=self.config.maximum_llm_calls,
            maximum_input_tokens=self.config.maximum_total_input_tokens,
            maximum_output_tokens=self.config.maximum_total_output_tokens,
        )

    def _embedding_service(self) -> tuple[Any, str]:
        cache_root = Path(self.config.artifacts_directory).resolve() / "cache"
        if not self.live:
            return fake_embedding_service(cache_root / "fake-embedding"), "fake-deterministic"
        wrapper = M5EmbeddingProvider(
            get_embedding_settings(),
            cache_directory=cache_root / "bge-m3",
            allow_model_load=_env_gate("M5_ALLOW_MODEL_LOAD"),
            allow_network=_env_gate("M5_ALLOW_NETWORK"),
        )
        return wrapper.service, "sentence-transformers"

    def _evaluator_provider(self, maximum_calls: int) -> BudgetedProvider:
        if not self.live:
            base: Any = FakeDeterministicProvider(capability="structured_evaluator")
        else:
            settings = OpenAICompatibleSettings(
                base_url=get_env_value("LLM_BASE_URL", "https://api.deepseek.com"),
                api_key=get_env_value("LLM_API_KEY", ""),
                model=get_env_value("M5_EVALUATOR_MODEL", get_env_value("LLM_MODEL", "deepseek-chat")),
                model_revision=get_env_value("M5_EVALUATOR_MODEL_REVISION", "provider-managed"),
                allow_network=_env_gate("M5_ALLOW_NETWORK"),
                allow_real_llm=_env_gate("M5_ALLOW_REAL_LLM"),
                allow_paid_eval=_env_gate("M5_ALLOW_PAID_EVAL"),
            )
            base = OpenAICompatibleProvider(settings, evaluator=True)
        return BudgetedProvider(
            base,
            maximum_calls=maximum_calls,
            maximum_input_tokens=self.config.maximum_total_input_tokens,
            maximum_output_tokens=self.config.maximum_total_output_tokens,
        )

    def _ensure_project(
        self,
        database: Database,
        embedding: Any,
        spec: RepositorySpec,
        project_map: dict[str, Any],
        run_dir: Path,
    ) -> tuple[str, dict[str, Any]]:
        project_id = str(project_map.get(spec.repo_id, {}).get("project_id", ""))
        bundle = database.get_bundle(project_id) if project_id else None
        if bundle is not None:
            return project_id, bundle
        project_id, bundle, stats = ingest_repository_snapshot(
            database,
            spec,
            Path(self.config.repository_root),
            embedding_service=None,
        )
        embedding_stats = EmbeddingIndexer(database, embedding).index_project(project_id).to_dict()
        stats["embedding"] = embedding_stats
        project_map[spec.repo_id] = {"project_id": project_id, "stats": stats}
        _atomic_json(run_dir / "projects.json", project_map)
        return project_id, bundle


def compute_run_id(
    config: BenchmarkConfig,
    dataset: LoadedDataset,
    provider: ProviderIdentity,
    code_identity: dict[str, str],
) -> str:
    config_data = config.model_dump()
    for key in ("dataset_directory", "repository_root", "artifacts_directory", "dry_run"):
        config_data.pop(key, None)
    canonical = {
        "dataset_version": dataset.manifest.dataset_version,
        "repositories": sorted(
            (item.repo_id, item.exact_commit_sha, item.content_fingerprint)
            for item in dataset.repositories
        ),
        "modes": list(config.modes),
        "config": config_data,
        "provider": provider.to_dict(),
        "prompt_version": config.prompt_version,
        "metric_version": config.metric_version,
        "evaluator_version": config.evaluator_version,
        "source_tree_digest": code_identity["source_tree_digest"],
    }
    digest = hashlib.sha256(_canonical(canonical).encode("utf-8")).hexdigest()
    return f"m5-{digest[:24]}"


def compare_run_records(
    left_path: Path,
    right_path: Path,
    metric: str,
    seed: int = 20260726,
) -> dict[str, Any]:
    left = json.loads(left_path.read_text(encoding="utf-8"))["records"]
    right = json.loads(right_path.read_text(encoding="utf-8"))["records"]
    return paired_delta(left, right, metric, seed=seed)


def generate_markdown_report(run_directory: Path, output_path: Path | None = None) -> str:
    run_directory = run_directory.resolve()
    manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_directory / "summary.json").read_text(encoding="utf-8"))
    comparisons = _load_json_if_exists(run_directory / "comparisons.json") or {}
    usage = summary.get("provider_usage", {})
    lines = [
        f"# RepoNoesis M5 run `{manifest['run_id']}`",
        "",
        f"- dataset: `{manifest['dataset_version']}`",
        f"- live: `{str(manifest.get('live', False)).lower()}`",
        f"- provider: `{manifest['answer_provider']['provider']}/{manifest['answer_provider']['model']}`",
        f"- scenarios: {summary['scenario_count']} ({summary['successful_count']} succeeded, {summary['failed_count']} failed)",
        f"- adaptive sequences: {summary.get('adaptive_sequences', {}).get('successful_count', 0)}/{summary.get('adaptive_sequences', {}).get('count', 0)}",
        f"- LLM calls: {usage.get('llm_calls', 0)}; tokens: {usage.get('total_tokens', 'unknown')}; estimated cost: {usage.get('estimated_cost_usd', 'unknown')}",
        "",
        "## Mode results",
        "",
        "| mode | success | Hit@5 | MRR@10 | Evidence F1 | p50 ms | p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, value in summary.get("by_mode", {}).items():
        means = value.get("means", {})
        lines.append(
            f"| {mode} | {value.get('successful_count', 0)}/{value.get('scenario_count', 0)} "
            f"| {_fmt(means.get('hit_at_5'))} | {_fmt(means.get('mrr_at_10'))} "
            f"| {_fmt(means.get('evidence_f1'))} | {_fmt(value.get('p50_latency_ms'))} "
            f"| {_fmt(value.get('p95_latency_ms'))} |"
        )
    lines.extend([
        "", "## Paired comparisons", "", "```json",
        json.dumps(comparisons, ensure_ascii=False, indent=2), "```", "",
        "This report describes a developer-curated pilot. Pending annotations and fake-provider "
        "runs do not establish teaching effectiveness, mastery accuracy, or broad repository generalization.",
    ])
    markdown = "\n".join(lines) + "\n"
    if output_path is not None:
        output_path.write_text(markdown, encoding="utf-8")
    return markdown


def _write_checkpoint(path: Path, run_id: str, started_at: str, records: list[dict[str, Any]]) -> None:
    checksum = hashlib.sha256(_canonical(records).encode("utf-8")).hexdigest()
    _atomic_json(
        path,
        {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "run_id": run_id,
            "started_at": started_at,
            "records_checksum": checksum,
            "records": records,
        },
    )


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkCheckpointError("checkpoint is unreadable or invalid JSON") from exc
    if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise BenchmarkCheckpointError("checkpoint schema version mismatch")
    records = checkpoint.get("records")
    if not isinstance(records, list):
        raise BenchmarkCheckpointError("checkpoint records are invalid")
    checksum = hashlib.sha256(_canonical(records).encode("utf-8")).hexdigest()
    if checksum != checkpoint.get("records_checksum"):
        raise BenchmarkCheckpointError("checkpoint checksum mismatch")
    return checkpoint


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _bounded_result(result: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "answer", "citations", "evidence_schema_version", "evidence", "grounding_status",
        "retrieval_mode", "warnings", "agent_schema_version", "agent_mode", "agent_status",
        "agent_trace", "budget_usage", "relation_schema_version", "analysis_mode",
        "evidence_chains", "relation_summary", "learning_schema_version", "learning_mode",
        "learning_context_summary", "learning_plan_summary", "recommended_next_action",
        "learning_warnings", "experiment_mode", "allowed_tools", "citation_validator_enabled",
        "relation_validator_enabled", "learning_context_is_repository_evidence",
    }
    bounded = {key: value for key, value in result.items() if key in allowed}
    if isinstance(bounded.get("answer"), str):
        bounded["answer"] = bounded["answer"][:10_000]
    return redact_secrets(bounded)


def _fixed_learning_context(scenario: Any) -> dict[str, Any]:
    state = "needs_review" if scenario.difficulty == "hard" else (
        "demonstrated" if scenario.difficulty == "easy" else "practicing"
    )
    depth = "remedial" if state == "needs_review" else ("concise" if state == "demonstrated" else "guided")
    return {
        "learning_schema_version": 1,
        "learning_mode": "profiled",
        "identity_mode": "local_single_user",
        "active_goal": {"goal_id": "benchmark-fixed-goal"},
        "current_plan": {
            "plan_id": "benchmark-fixed-plan",
            "version": 1,
            "status": "active",
            "adapted": False,
            "steps": [{"step_id": "benchmark-step", "status": "active"}],
        },
        "target_states": [{"mastery_status": state, "availability": "current"}],
        "recent_verified_outcomes": [],
        "recommended_explanation_depth": depth,
        "recommended_next_action": "Review the cited source span.",
        "warnings": [],
        "metrics": {
            "target_state_count": 1,
            "demonstrated_target_count": int(state == "demonstrated"),
            "mastered_target_count": 0,
            "needs_review_count": int(state == "needs_review"),
            "max_bytes": 16_384,
        },
    }


def _degraded_flags(result: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if result.get("grounding_status") == "degraded":
        flags.append("grounding_degraded")
    if result.get("agent_status") in {"budget_exhausted", "cancelled", "degraded"}:
        flags.append(f"agent_{result.get('agent_status')}")
    if result.get("learning_mode") == "degraded":
        flags.append("learning_degraded")
    if result.get("provider_error"):
        flags.append("provider_error")
    return flags


def _cost(results: list[dict[str, Any]]) -> float | str:
    costs = [item.get("usage", {}).get("estimated_cost_usd") for item in results]
    if any(value == "unknown" for value in costs):
        return "unknown"
    return sum(float(value) for value in costs if isinstance(value, (int, float)))


def _mode_comparisons(records: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    by_mode = {
        mode: [item for item in records if item.get("experiment_mode") == mode]
        for mode in EXPERIMENT_MODES
    }
    pairs = {
        "dense_minus_lexical": ("fixed_lexical_rag", "fixed_dense_rag"),
        "hybrid_minus_lexical": ("fixed_lexical_rag", "m1_hybrid_rag"),
        "relation_minus_m2": ("m2_bounded_agent", "m3_relation_agent"),
        "profiled_minus_m3": ("m3_relation_agent", "m4_profiled_agent"),
    }
    return {
        name: {
            metric: paired_delta(by_mode[left], by_mode[right], metric, seed=seed)
            for metric in ("hit_at_5", "evidence_f1", "expected_relation_edge_recall")
        }
        for name, (left, right) in pairs.items()
        if by_mode[left] and by_mode[right]
    }


def _fmt(value: Any) -> str:
    return "unknown" if value is None else f"{float(value):.4f}"


def _safe_config(config: BenchmarkConfig) -> dict[str, Any]:
    value = config.model_dump()
    value["dataset_directory"] = Path(value["dataset_directory"]).name
    value["repository_root"] = "[CONFIGURED_REPOSITORY_ROOT]"
    value["artifacts_directory"] = "[CONFIGURED_ARTIFACTS_ROOT]"
    return redact_secrets(value)


def _code_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parents[3]
    head = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    status = _git(root, "status", "--short")
    tracked = _git(root, "ls-files").splitlines()
    untracked = _git(root, "ls-files", "--others", "--exclude-standard").splitlines()
    digest = hashlib.sha256()
    for relative in sorted(set([*tracked, *untracked])):
        normalized = relative.replace("\\", "/")
        if not normalized.startswith((
            "backend/app/",
            "benchmarks/m5/datasets/",
            "benchmarks/m5/schemas/",
        )):
            continue
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            continue
        digest.update(normalized.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    source_digest = digest.hexdigest()
    return {"git_head": head, "git_tree": tree, "source_tree_digest": source_digest, "dirty": str(bool(status)).lower()}


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments], check=True, capture_output=True,
        text=True, encoding="utf-8", timeout=30,
    ).stdout.strip()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _load_json_if_exists(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_gate(name: str) -> bool:
    return get_env_value(name, "").strip().casefold() in {"1", "true", "yes", "on"}
