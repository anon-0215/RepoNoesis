from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from typing import Any

from app.retrieval_phase5.metrics import aggregate_path_records


def aggregate_cross_repository(
    records_by_repo_path: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    repositories = sorted(records_by_repo_path)
    paths = sorted({path for values in records_by_repo_path.values() for path in values})
    per_repository = {
        repo: {
            path: aggregate_path_records(records_by_repo_path[repo].get(path, []))
            for path in paths
        }
        for repo in repositories
    }
    micro = {
        path: aggregate_path_records(
            [item for repo in repositories for item in records_by_repo_path[repo].get(path, [])]
        )
        for path in paths
    }
    macro: dict[str, Any] = {}
    for path in paths:
        metric_names = sorted({name for repo in repositories for name in per_repository[repo][path]["metrics"]})
        macro[path] = {
            "repository_count": len(repositories),
            "metrics": {
                name: statistics.fmean(
                    float(per_repository[repo][path]["metrics"].get(name, 0.0))
                    for repo in repositories
                )
                for name in metric_names
            },
        }
    strata_records: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for repo in repositories:
        for path, records in records_by_repo_path[repo].items():
            for item in records:
                strata_records[str(item.get("primary_stratum", "unknown"))][path].append(item)
    strata = {
        stratum: {path: aggregate_path_records(values.get(path, [])) for path in paths}
        for stratum, values in sorted(strata_records.items())
    }
    return {
        "per_repository": per_repository,
        "micro": micro,
        "macro": macro,
        "strata": strata,
    }


def repository_stratified_compare(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    left_path: str,
    right_path: str,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    left_by_key = {
        (str(item["repository_id"]), str(item["query_id"])): item
        for item in left if item.get("valid") and not item.get("skipped")
    }
    right_by_key = {
        (str(item["repository_id"]), str(item["query_id"])): item
        for item in right if item.get("valid") and not item.get("skipped")
    }
    keys = sorted(set(left_by_key).intersection(right_by_key))
    by_repo: dict[str, list[float]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    gain = loss = improved = unchanged = regressed = 0
    for repo, query_id in keys:
        first = float(left_by_key[(repo, query_id)]["metrics"]["mrr_at_8"])
        second = float(right_by_key[(repo, query_id)]["metrics"]["mrr_at_8"])
        delta = second - first
        if not math.isfinite(delta):
            raise ValueError("paired MRR delta contains NaN or Infinity")
        by_repo[repo].append(delta)
        improved += delta > 0
        unchanged += delta == 0
        regressed += delta < 0
        left_hit = bool(left_by_key[(repo, query_id)]["metrics"]["hit_at_8"])
        right_hit = bool(right_by_key[(repo, query_id)]["metrics"]["hit_at_8"])
        gain += right_hit and not left_hit
        loss += left_hit and not right_hit
        rows.append({"repository_id": repo, "query_id": query_id, "left": first, "right": second, "delta": delta})
    sample_count = max(2_000, min(20_000, int(samples))) if keys else 0
    micro_ci, macro_ci = _stratified_bootstrap(by_repo, seed=seed, samples=sample_count)
    all_values = [value for repo in sorted(by_repo) for value in by_repo[repo]]
    repo_means = [statistics.fmean(by_repo[repo]) for repo in sorted(by_repo)]
    return {
        "left_path": left_path,
        "right_path": right_path,
        "paired_count": len(keys),
        "repository_counts": {repo: len(values) for repo, values in sorted(by_repo.items())},
        "micro_mean_mrr_at_8_delta": statistics.fmean(all_values) if all_values else None,
        "macro_mean_mrr_at_8_delta": statistics.fmean(repo_means) if repo_means else None,
        "improved_queries": improved,
        "unchanged_queries": unchanged,
        "regressed_queries": regressed,
        "new_gold_gain_at_8": gain,
        "gold_loss_at_8": loss,
        "micro_bootstrap_95_ci": micro_ci,
        "macro_bootstrap_95_ci": macro_ci,
        "seed": seed,
        "samples": sample_count,
        "query_deltas": rows,
    }


def _stratified_bootstrap(
    by_repo: dict[str, list[float]], *, seed: int, samples: int
) -> tuple[list[float] | None, list[float] | None]:
    if not by_repo:
        return None, None
    randomizer = random.Random(seed)
    micro_values: list[float] = []
    macro_values: list[float] = []
    repos = sorted(by_repo)
    for _ in range(samples):
        sampled = {
            repo: [randomizer.choice(by_repo[repo]) for _ in by_repo[repo]]
            for repo in repos
        }
        flat = [value for repo in repos for value in sampled[repo]]
        micro_values.append(statistics.fmean(flat))
        macro_values.append(statistics.fmean(statistics.fmean(sampled[repo]) for repo in repos))
    micro_values.sort()
    macro_values.sort()
    return _ci(micro_values), _ci(macro_values)


def _ci(values: list[float]) -> list[float]:
    return [_percentile(values, 0.025), _percentile(values, 0.975)]


def _percentile(values: list[float], fraction: float) -> float:
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)
