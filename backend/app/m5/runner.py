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
    PricingConfig,
    ProviderLLMClient,
    StructuredLearningEvaluator,
    redact_secrets,
    normalized_endpoint_identity,
)
from app.m5.repository import ingest_repository_snapshot
from app.services.embedding_indexer import EmbeddingIndexer


CHECKPOINT_SCHEMA_VERSION = 2


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
        cells = self._planned_cells(dataset)
        batch = self._batch_cells(cells)
        sequences = self._selected_sequences(dataset)
        return {
            "status": "validated",
            "dry_run": True,
            "dataset_version": dataset.manifest.dataset_version,
            "run_type": self._run_type,
            "repositories": sorted({item.repo_id for item, _ in cells}),
            "scenario_ids": sorted({item.scenario_id for item, _ in cells}),
            "modes": list(self.config.modes),
            "planned_answer_runs": len(cells),
            "batch_answer_runs": len(batch),
            "batch": {"index": self.config.batch_index, "count": self.config.batch_count},
            "planned_sequence_runs": (
                len(sequences) if "m4_adaptive_sequence" in self.config.modes else 0
            ),
            "real_provider_calls_allowed": self.live,
        }

    def run(self, *, resume: bool = False, retry_failed: bool = False) -> dict[str, Any]:
        dataset = self.dataset_validator.load_validated()
        planned_cells = self._planned_cells(dataset)
        batch_cells = self._batch_cells(planned_cells)
        provider = self._provider()
        embedding, embedding_identity = self._embedding_service()
        evaluator_identity = self._evaluator_identity()
        code_identity = _code_identity()
        run_identity = build_run_identity(
            self.config, dataset, provider.identity, embedding_identity,
            evaluator_identity, code_identity, live=self.live,
        )
        run_identity_digest = hashlib.sha256(_canonical(run_identity).encode()).hexdigest()
        run_id = f"m5-{run_identity_digest[:24]}"
        run_dir = self._artifact_root / "runs" / run_id
        existing_manifest = _load_json_if_exists(run_dir / "manifest.json")
        if existing_manifest:
            _verify_manifest_identity(existing_manifest, run_identity_digest)
        if existing_manifest and not resume:
            raise BenchmarkCheckpointError("existing run requires explicit --resume")
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = run_dir / "checkpoint.json"
        checkpoint = _load_checkpoint(checkpoint_path) if resume and checkpoint_path.exists() else None
        if checkpoint:
            _verify_checkpoint_identity(checkpoint, run_id, run_identity_digest)
            provider.restore(checkpoint.get("answer_budget", {}))
        records = list((checkpoint or {}).get("records", []))
        completed = {
            (item["scenario_id"], item["experiment_mode"]): item
            for item in records
            if item.get("scenario_status") == "succeeded"
            or (not retry_failed and item.get("scenario_status") in {"failed", "timed_out", "cancelled", "degraded"})
        }
        sequence_records = list((checkpoint or {}).get("sequence_records", []))
        safe_config = _safe_config(self.config)
        manifest = {
            "benchmark_schema_version": 2,
            "metric_schema_version": 1,
            "run_id": run_id,
            "started_at": (checkpoint or {}).get("started_at") or _utc_now(),
            "ended_at": None,
            "host": {"platform": platform.platform(), "python": sys.version},
            "run_type": self._run_type,
            "live": self.live,
            "run_purpose": self.config.run_purpose,
            "run_identity": run_identity,
            "run_identity_digest": run_identity_digest,
            "project_code_identity": code_identity,
            "dataset_identity": run_identity["dataset"],
            "dataset_version": dataset.manifest.dataset_version,
            "repository_revisions": {
                item.repo_id: item.exact_commit_sha for item in dataset.repositories
            },
            "modes": list(self.config.modes),
            "planned_cells": [f"{item.scenario_id}::{mode}" for item, mode in planned_cells],
            "current_batch": {"index": self.config.batch_index, "count": self.config.batch_count,
                              "cells": [f"{item.scenario_id}::{mode}" for item, mode in batch_cells]},
            "safe_config": safe_config,
            "answer_provider": provider.identity.to_dict(),
            "evaluator_provider": evaluator_identity.to_dict(),
            "judge_independence": _judge_independence(provider.identity, evaluator_identity),
            "embedding_provider": embedding_identity,
            "budgets": {"answer": provider.ledger(), "evaluator": self._evaluator_limits()},
            "prompt_versions": [self.config.prompt_version],
            "metric_versions": [self.config.metric_version],
            "evaluator_version": self.config.evaluator_version,
            "random_seed": self.config.random_seed,
        }
        _atomic_json(run_dir / "manifest.json", manifest)
        database = Database(run_dir / "benchmark.sqlite")
        repo_by_id = {item.repo_id: item for item in dataset.repositories}
        project_map = _load_json_if_exists(run_dir / "projects.json") or {}
        llm = ProviderLLMClient(
            provider,
            maximum_attempts=self.config.maximum_provider_attempts,
            seed=self.config.random_seed,
            request_timeout_seconds=self.config.timeout_seconds,
            maximum_output_tokens=self.config.maximum_output_tokens,
        )
        stopped_reason: str | None = None
        for scenario, mode in batch_cells:
            if provider.stop_reason:
                stopped_reason = provider.stop_reason
                break
            spec = repo_by_id[scenario.repo_id]
            project_id, bundle = self._ensure_project(
                database, embedding, spec, project_map, run_dir
            )
            identity = (scenario.scenario_id, mode)
            if identity in completed:
                continue
            before = len(llm.results)
            started = time.monotonic()
            try:
                result = execute_mode(
                    mode, scenario, bundle, database, embedding, llm,
                    limits=get_agent_limits(), deterministic_planner=not self.live,
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
            cost = _cost_summary(provider_results)
            record = {
                "scenario_id": scenario.scenario_id, "repo_id": scenario.repo_id,
                "repository_revision": scenario.repository_revision, "category": scenario.category,
                "experiment_mode": mode,
                "attempt_number": 1 + sum(
                    item.get("scenario_id") == scenario.scenario_id
                    and item.get("experiment_mode") == mode for item in records
                ),
                "scenario_status": status, "error": error, "latency_ms": latency_ms,
                "provider_calls": provider_results, "provider_call_count": len(provider_results),
                "cost": cost, "estimated_cost_usd": cost["value"],
                "unknown_cost_count": cost["unknown_count"],
                "degraded_flags": _degraded_flags(result), "result": _bounded_result(result),
            }
            record["metrics"] = scenario_metrics(scenario, {**result, "scenario_status": status})
            records.append(record)
            completed[identity] = record
            _write_checkpoint(
                checkpoint_path, run_id, manifest["started_at"], records,
                run_identity_digest=run_identity_digest, answer_budget=provider.ledger(),
                sequence_records=sequence_records,
            )
            if provider.stop_reason:
                stopped_reason = provider.stop_reason
                break
        evaluator_provider: BudgetedProvider | None = None
        if "m4_adaptive_sequence" in self.config.modes and self.config.batch_index == 0 and not stopped_reason:
            evaluator_provider = self._evaluator_provider()
            if checkpoint and checkpoint.get("evaluator_budget"):
                evaluator_provider.restore(checkpoint["evaluator_budget"])
            evaluator = StructuredLearningEvaluator(
                evaluator_provider, seed=self.config.random_seed,
                maximum_attempts=self.config.maximum_provider_attempts,
                request_timeout_seconds=self.config.timeout_seconds,
                maximum_output_tokens=self.config.maximum_output_tokens,
            )
            completed_sequences = {item["sequence_id"] for item in sequence_records}
            if evaluator_provider.stop_reason:
                stopped_reason = evaluator_provider.stop_reason
            for sequence in self._selected_sequences(dataset):
                if stopped_reason:
                    break
                if sequence.sequence_id in completed_sequences:
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
                _write_checkpoint(
                    checkpoint_path, run_id, manifest["started_at"], records,
                    run_identity_digest=run_identity_digest, answer_budget=provider.ledger(),
                    evaluator_budget=evaluator_provider.ledger(), sequence_records=sequence_records,
                )
                if evaluator_provider.stop_reason:
                    stopped_reason = evaluator_provider.stop_reason
                    break
            _atomic_json(
                run_dir / "sequence_results.json",
                {"run_id": run_id, "records": sequence_records},
            )
        ordered = _latest_records(records)
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
        answer_ledger = provider.ledger()
        evaluator_ledger = evaluator_provider.ledger() if evaluator_provider else (checkpoint or {}).get("evaluator_budget") or None
        cost_summary = _ledger_cost_summary(answer_ledger, evaluator_ledger)
        token_usage_complete = answer_ledger["usage_complete"] and (
            evaluator_ledger is None or evaluator_ledger.get("usage_complete", True)
        )
        combined_input = answer_ledger["input_tokens"] + int((evaluator_ledger or {}).get("input_tokens", 0))
        combined_output = answer_ledger["output_tokens"] + int((evaluator_ledger or {}).get("output_tokens", 0))
        summary["provider_usage"] = {
            "answer": answer_ledger,
            "evaluator": evaluator_ledger,
            "llm_calls": provider.request_count + int((evaluator_ledger or {}).get("request_count", 0)),
            "input_tokens": combined_input if token_usage_complete else "unknown",
            "output_tokens": combined_output if token_usage_complete else "unknown",
            "total_tokens": (combined_input + combined_output) if token_usage_complete else "unknown",
            "token_usage_complete": token_usage_complete,
            "usage_unknown_count": unknown_usage,
            "cost": cost_summary,
            "estimated_cost_usd": cost_summary["value"],
            "unknown_cost_count": cost_summary["unknown_count"],
            "evaluator_calls": int((evaluator_ledger or {}).get("request_count", 0)),
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
        completed_cell_count = len({(item["scenario_id"], item["experiment_mode"]) for item in ordered})
        sequences_expected = len(self._selected_sequences(dataset)) if "m4_adaptive_sequence" in self.config.modes else 0
        partial = completed_cell_count < len(planned_cells) or len(sequence_records) < sequences_expected
        effective_stop_reason = stopped_reason if partial else None
        summary.update({
            "run_type": self._run_type, "partial_run": partial,
            "completion_status": "partial" if partial else "completed",
            "stop_reason": effective_stop_reason or ("batch_complete" if self.config.batch_count > 1 and partial else "completed"),
            "planned_cell_count": len(planned_cells), "completed_cell_count": completed_cell_count,
        })
        final_manifest = {
            **manifest,
            "ended_at": _utc_now(),
            "record_count": len(ordered),
            "sequence_record_count": len(sequence_records),
            "budgets": summary["provider_usage"],
            "partial_run": partial, "completion_status": summary["completion_status"],
            "stop_reason": summary["stop_reason"],
        }
        _atomic_json(run_dir / "manifest.json", final_manifest)
        _atomic_json(run_dir / "results.json", {"run_id": run_id, "records": ordered})
        _atomic_json(run_dir / "summary.json", summary)
        return {"run_id": run_id, "run_directory": str(run_dir), "summary": summary}

    @property
    def _run_type(self) -> str:
        return "live" if self.live else "fake"

    @property
    def _artifact_root(self) -> Path:
        return Path(self.config.artifacts_directory).resolve() / self._run_type

    def _planned_cells(self, dataset: LoadedDataset) -> list[tuple[Any, str]]:
        scenarios = {item.scenario_id: item for item in self._selected_scenarios(dataset)}
        if self.config.cells:
            result = []
            for cell in self.config.cells:
                scenario_id, mode = cell.rsplit("::", 1)
                scenario = next((item for item in dataset.scenarios if item.scenario_id == scenario_id), None)
                if scenario is None:
                    raise ValueError(f"unknown selected scenario: {scenario_id}")
                result.append((scenario, mode))
            return sorted(result, key=lambda item: (item[0].scenario_id, item[1]))
        modes = [mode for mode in self.config.modes if mode != "m4_adaptive_sequence"]
        return [(scenario, mode) for scenario in scenarios.values() for mode in modes]

    def _batch_cells(self, cells: list[tuple[Any, str]]) -> list[tuple[Any, str]]:
        return [cell for index, cell in enumerate(cells) if index % self.config.batch_count == self.config.batch_index]

    def _selected_sequences(self, dataset: LoadedDataset) -> list[Any]:
        return sorted(
            [item for item in dataset.sequences if not self.config.repo_ids or item.repo_id in self.config.repo_ids],
            key=lambda item: item.sequence_id,
        )

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
            pricing = _pricing_from_env("M5_ANSWER")
            self._validate_paid_run(pricing, self.config.maximum_answer_cost_usd, "answer")
            settings = OpenAICompatibleSettings(
                base_url=get_env_value("LLM_BASE_URL", ""),
                api_key=get_env_value("LLM_API_KEY", ""),
                model=get_env_value("LLM_MODEL", ""),
                model_revision=get_env_value("M5_LLM_MODEL_REVISION", ""),
                allow_network=_env_gate("M5_ALLOW_NETWORK"),
                allow_real_llm=_env_gate("M5_ALLOW_REAL_LLM"),
                allow_paid_eval=_env_gate("M5_ALLOW_PAID_EVAL"),
                pricing=pricing,
            )
            base = OpenAICompatibleProvider(settings)
        return BudgetedProvider(
            base,
            maximum_requests=self.config.maximum_answer_requests,
            maximum_input_tokens=self.config.maximum_answer_input_tokens,
            maximum_output_tokens=self.config.maximum_answer_output_tokens,
            maximum_cost_usd=self.config.maximum_answer_cost_usd,
            maximum_wall_clock_seconds=self.config.maximum_wall_clock_seconds,
        )

    def _embedding_service(self) -> tuple[Any, dict[str, Any]]:
        cache_root = self._artifact_root / "cache"
        if not self.live:
            service = fake_embedding_service(cache_root / "fake-embedding")
            return service, {
                "provider": "fake-deterministic", "model": "fake-bge-m3",
                "model_revision": "fixture-v1", "dimension": 16, "normalize": True,
                "max_length": 512, "batch_size": 8,
                "query_prefix_identity": _text_identity(""),
                "document_prefix_identity": _text_identity(""),
                "device": "cpu", "dtype": "float32",
            }
        settings = get_embedding_settings()
        if not get_env_value("EMBEDDING_MODEL_NAME_OR_PATH", "").strip():
            raise ValueError("live embedding model name/path must be explicitly configured")
        if not settings.model_revision:
            raise ValueError("live embedding model revision must be explicit")
        dimension_text = get_env_value("M5_EMBEDDING_DIMENSION", "").strip()
        if not dimension_text.isdigit() or int(dimension_text) <= 0:
            raise ValueError("M5_EMBEDDING_DIMENSION must be an explicit positive integer")
        wrapper = M5EmbeddingProvider(
            settings,
            cache_directory=cache_root / "bge-m3",
            allow_model_load=_env_gate("M5_ALLOW_MODEL_LOAD"),
            allow_network=_env_gate("M5_ALLOW_NETWORK"),
        )
        resolved_model = wrapper.ensure_model_identity()
        actual_dimension = wrapper.get_embedding_dimension()
        if actual_dimension != int(dimension_text):
            raise ValueError(
                f"configured embedding dimension {dimension_text} does not match loaded model dimension {actual_dimension}"
            )
        model_name = resolved_model.model_name
        if Path(model_name).is_absolute():
            model_name = f"local:{Path(model_name).name}"
        return wrapper, {
            "provider": "sentence-transformers", "model": model_name,
            "model_revision": resolved_model.model_revision,
            "configured_model_revision": settings.model_revision,
            "dimension": int(dimension_text),
            "normalize": settings.normalize, "max_length": settings.max_length,
            "batch_size": settings.batch_size,
            "query_prefix_identity": _text_identity(settings.query_prefix),
            "document_prefix_identity": _text_identity(settings.document_prefix),
            "device": settings.device,
            "dtype": "float32",
        }

    def _evaluator_identity(self) -> ProviderIdentity:
        if not self.live:
            return FakeDeterministicProvider(capability="structured_evaluator").identity
        base_url = get_env_value("M5_EVALUATOR_BASE_URL", "").strip() or get_env_value("LLM_BASE_URL", "")
        model = get_env_value("M5_EVALUATOR_MODEL", "")
        revision = get_env_value("M5_EVALUATOR_MODEL_REVISION", "")
        if not base_url or not model.strip() or not revision.strip():
            raise ValueError("live evaluator base URL, model, and model revision/identity must be explicit")
        pricing = _pricing_from_env("M5_EVALUATOR")
        pricing_identity = ({"status": "unknown", "reason": "pricing_not_configured"}
                            if pricing is None else {
                                "status": "configured", "currency": pricing.currency,
                                "model": pricing.model,
                                "input_price_per_unit": pricing.input_price_per_unit,
                                "output_price_per_unit": pricing.output_price_per_unit,
                                "unit_tokens": pricing.unit_tokens, "source": pricing.source,
                            })
        return ProviderIdentity(
            "openai-compatible", model, revision, "structured_evaluator", True,
            normalized_endpoint_identity(base_url),
            pricing_identity,
        )

    def _evaluator_provider(self) -> BudgetedProvider:
        if not self.live:
            base: Any = FakeDeterministicProvider(capability="structured_evaluator")
        else:
            pricing = _pricing_from_env("M5_EVALUATOR")
            self._validate_paid_run(pricing, self.config.maximum_evaluator_cost_usd, "evaluator")
            settings = OpenAICompatibleSettings(
                base_url=get_env_value("M5_EVALUATOR_BASE_URL", "").strip() or get_env_value("LLM_BASE_URL", ""),
                api_key=get_env_value("M5_EVALUATOR_API_KEY", "") or get_env_value("LLM_API_KEY", ""),
                model=get_env_value("M5_EVALUATOR_MODEL", ""),
                model_revision=get_env_value("M5_EVALUATOR_MODEL_REVISION", ""),
                allow_network=_env_gate("M5_ALLOW_NETWORK"),
                allow_real_llm=_env_gate("M5_ALLOW_REAL_LLM"),
                allow_paid_eval=_env_gate("M5_ALLOW_PAID_EVAL"),
                pricing=pricing,
            )
            base = OpenAICompatibleProvider(settings, evaluator=True)
        return BudgetedProvider(
            base,
            maximum_requests=self.config.maximum_evaluator_requests,
            maximum_input_tokens=self.config.maximum_evaluator_input_tokens,
            maximum_output_tokens=self.config.maximum_evaluator_output_tokens,
            maximum_cost_usd=self.config.maximum_evaluator_cost_usd,
            maximum_wall_clock_seconds=self.config.maximum_wall_clock_seconds,
        )

    def _evaluator_limits(self) -> dict[str, Any]:
        return {
            "maximum_requests": self.config.maximum_evaluator_requests,
            "maximum_input_tokens": self.config.maximum_evaluator_input_tokens,
            "maximum_output_tokens": self.config.maximum_evaluator_output_tokens,
            "maximum_cost_usd": self.config.maximum_evaluator_cost_usd,
            "maximum_wall_clock_seconds": self.config.maximum_wall_clock_seconds,
        }

    def _validate_paid_run(self, pricing: PricingConfig | None, maximum_cost: float | None, role: str) -> None:
        if self.config.run_purpose in {"pilot", "full"}:
            if pricing is None:
                raise ValueError(f"paid {self.config.run_purpose} requires explicit {role} pricing")
            if maximum_cost is None:
                raise ValueError(f"paid {self.config.run_purpose} requires an explicit {role} cost budget")

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
    *,
    live: bool = False,
    embedding_identity: dict[str, Any] | None = None,
    evaluator: ProviderIdentity | None = None,
) -> str:
    identity = build_run_identity(
        config, dataset, provider,
        embedding_identity or {"provider": "unspecified", "model_revision": "unspecified"},
        evaluator or ProviderIdentity("unspecified", "unspecified", "unspecified", "structured_evaluator", False),
        code_identity, live=live,
    )
    digest = hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest()
    return f"m5-{digest[:24]}"


def build_run_identity(
    config: BenchmarkConfig,
    dataset: LoadedDataset,
    provider: ProviderIdentity,
    embedding_identity: dict[str, Any],
    evaluator: ProviderIdentity,
    code_identity: dict[str, str],
    *,
    live: bool,
) -> dict[str, Any]:
    config_data = config.model_dump()
    for key in (
        "dataset_directory", "repository_root", "artifacts_directory", "dry_run",
        "batch_index", "batch_count",
    ):
        config_data.pop(key, None)
    return {
        "run_type": "live" if live else "fake",
        "live": live,
        "dataset": {
            "dataset_version": dataset.manifest.dataset_version,
            "manifest_revision": getattr(dataset.manifest, "dataset_revision", None),
            "repositories": sorted(
                (item.repo_id, item.exact_commit_sha, item.content_fingerprint)
                for item in dataset.repositories
            ),
            "content_identity": _dataset_content_identity(Path(config.dataset_directory)),
        },
        "modes": list(config.modes),
        "config": config_data,
        "answer_provider": provider.to_dict(),
        "embedding_provider": embedding_identity,
        "evaluator_provider": evaluator.to_dict(),
        "prompt_version": config.prompt_version,
        "metric_version": config.metric_version,
        "evaluator_version": config.evaluator_version,
        "source_tree_digest": code_identity["source_tree_digest"],
    }


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
        f"- run type: `{manifest.get('run_type', 'unknown')}`",
        f"- completion: `{summary.get('completion_status', 'unknown')}`; partial: `{str(summary.get('partial_run', True)).lower()}`; stop reason: `{summary.get('stop_reason', 'unknown')}`",
        f"- provider: `{manifest['answer_provider']['provider']}/{manifest['answer_provider']['model']}`",
        f"- scenarios: {summary['scenario_count']} ({summary['successful_count']} succeeded, {summary['failed_count']} failed)",
        f"- adaptive sequences: {summary.get('adaptive_sequences', {}).get('successful_count', 0)}/{summary.get('adaptive_sequences', {}).get('count', 0)}",
        f"- LLM requests: {usage.get('llm_calls', 0)}; tokens: {usage.get('total_tokens', 'unknown')}; cost: {usage.get('cost', {}).get('status', 'unknown')} / {usage.get('cost', {}).get('value', 'unknown')} USD",
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


def _write_checkpoint(
    path: Path, run_id: str, started_at: str, records: list[dict[str, Any]], *,
    run_identity_digest: str = "legacy", answer_budget: dict[str, Any] | None = None,
    evaluator_budget: dict[str, Any] | None = None,
    sequence_records: list[dict[str, Any]] | None = None,
) -> None:
    protected = {"records": records, "sequence_records": sequence_records or [],
                 "answer_budget": answer_budget or {}, "evaluator_budget": evaluator_budget or {}}
    checksum = hashlib.sha256(_canonical(protected).encode("utf-8")).hexdigest()
    _atomic_json(
        path,
        {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "run_id": run_id,
            "run_identity_digest": run_identity_digest,
            "started_at": started_at,
            "checkpoint_checksum": checksum,
            **protected,
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
    protected = {"records": records, "sequence_records": checkpoint.get("sequence_records", []),
                 "answer_budget": checkpoint.get("answer_budget", {}),
                 "evaluator_budget": checkpoint.get("evaluator_budget", {})}
    checksum = hashlib.sha256(_canonical(protected).encode("utf-8")).hexdigest()
    if checksum != checkpoint.get("checkpoint_checksum"):
        raise BenchmarkCheckpointError("checkpoint checksum mismatch")
    return checkpoint


def _verify_checkpoint_identity(checkpoint: dict[str, Any], run_id: str, digest: str) -> None:
    if checkpoint.get("run_id") != run_id or checkpoint.get("run_identity_digest") != digest:
        raise BenchmarkCheckpointError("checkpoint run identity mismatch")


def _verify_manifest_identity(manifest: dict[str, Any], digest: str) -> None:
    if manifest.get("run_identity_digest") != digest:
        raise BenchmarkCheckpointError("run directory manifest identity mismatch")


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
    return _cost_summary(results)["value"]


def _cost_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    usages = [item.get("usage", {}) for item in results]
    unknown = [usage for usage in usages if usage.get("cost_status", "unknown") == "unknown"]
    if unknown:
        reasons = sorted({str(item.get("cost_unknown_reason") or "unspecified") for item in unknown})
        return {"status": "unknown", "value": "unknown", "currency": "USD",
                "unknown_count": len(unknown), "unknown_reasons": reasons}
    value = sum(float(item.get("estimated_cost_usd", 0.0)) for item in usages)
    status = "known_zero" if value == 0 else "calculated"
    return {"status": status, "value": value, "currency": "USD",
            "unknown_count": 0, "unknown_reasons": []}


def _ledger_cost_summary(answer: dict[str, Any], evaluator: dict[str, Any] | None) -> dict[str, Any]:
    ledgers = [answer, *([evaluator] if evaluator else [])]
    unknown = [item for item in ledgers if not item.get("cost_complete", True)]
    if unknown:
        reasons = sorted({reason for item in unknown for reason in item.get("cost_unknown_reasons", ["unspecified"])})
        return {"status": "unknown", "value": "unknown", "currency": "USD",
                "unknown_count": len(unknown), "unknown_reasons": reasons}
    value = sum(float(item.get("cost_usd", 0.0)) for item in ledgers)
    return {"status": "known_zero" if value == 0 else "calculated", "value": value,
            "currency": "USD", "unknown_count": 0, "unknown_reasons": []}


def _latest_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for item in records:
        key = (item["scenario_id"], item["experiment_mode"])
        if key not in latest or int(item.get("attempt_number", 0)) > int(latest[key].get("attempt_number", 0)):
            latest[key] = item
    return sorted(latest.values(), key=lambda item: (item["scenario_id"], item["experiment_mode"]))


def _dataset_content_identity(directory: Path) -> str:
    digest = hashlib.sha256()
    for name in ("manifest.json", "repositories.json", "scenarios.jsonl", "sequences.jsonl"):
        path = directory / name
        if not path.is_file():
            return "unavailable:test-fixture"
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return f"sha256:{digest.hexdigest()}"


def _text_identity(value: str) -> dict[str, Any]:
    return {"sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(), "length": len(value)}


def _judge_independence(answer: ProviderIdentity, evaluator: ProviderIdentity) -> str:
    if answer.endpoint_identity != evaluator.endpoint_identity:
        return "independent_endpoint"
    if answer.model != evaluator.model or answer.model_revision != evaluator.model_revision:
        return "shared_endpoint_independent_model"
    return "shared_endpoint_same_model"


def _pricing_from_env(prefix: str) -> PricingConfig | None:
    names = {
        "model": f"{prefix}_PRICING_MODEL",
        "currency": f"{prefix}_PRICING_CURRENCY",
        "input": f"{prefix}_INPUT_PRICE_PER_UNIT",
        "output": f"{prefix}_OUTPUT_PRICE_PER_UNIT",
        "unit": f"{prefix}_PRICING_UNIT_TOKENS",
        "source": f"{prefix}_PRICING_SOURCE",
    }
    values = {key: get_env_value(name, "").strip() for key, name in names.items()}
    if not any(values.values()):
        return None
    if not all(values.values()):
        missing = [names[key] for key, value in values.items() if not value]
        raise ValueError(f"incomplete pricing configuration; missing: {', '.join(missing)}")
    try:
        pricing = PricingConfig(
            model=values["model"], currency=values["currency"], input_price_per_unit=float(values["input"]),
            output_price_per_unit=float(values["output"]), unit_tokens=int(values["unit"]),
            source=values["source"],
        )
    except ValueError as exc:
        raise ValueError(f"invalid explicit pricing configuration for {prefix}") from exc
    pricing.validate()
    return pricing


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
