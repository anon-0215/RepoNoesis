from __future__ import annotations

import math
import random
import statistics
from collections import Counter
from typing import Any, Iterable

from app.m5.contracts import Scenario
from app.retrieval_phase5.contracts import BOOTSTRAP_SAMPLES, RANDOM_SEED, TOP_K_VALUES


def _candidate_symbol(candidate: dict[str, Any]) -> str:
    return str(candidate.get("qualified_name") or candidate.get("qualified_symbol") or "")


def _gold_identity(span: Any, revision: str) -> tuple[Any, ...]:
    return (
        revision,
        span.path,
        span.qualified_symbol,
        int(span.start_line),
        int(span.end_line),
        span.content_hash,
    )


def strict_gold_match(
    candidate: dict[str, Any],
    scenario: Scenario,
) -> tuple[bool, list[tuple[Any, ...]]]:
    if candidate.get("validation_status") != "valid":
        return False, []
    identity = (
        str(candidate.get("repository_revision", "")),
        str(candidate.get("path", "")),
        _candidate_symbol(candidate),
        int(candidate.get("start_line", 0)),
        int(candidate.get("end_line", 0)),
        str(candidate.get("content_hash", "")),
    )
    matches = [
        _gold_identity(span, scenario.repository_revision)
        for span in scenario.expected_source_spans
        if identity == _gold_identity(span, scenario.repository_revision)
    ]
    return bool(matches), matches


def containment_match(
    candidate: dict[str, Any],
    scenario: Scenario,
) -> tuple[bool, list[tuple[Any, ...]]]:
    if candidate.get("validation_status") != "valid":
        return False, []
    revision = str(candidate.get("repository_revision", ""))
    path = str(candidate.get("path", ""))
    start = int(candidate.get("start_line", 0))
    end = int(candidate.get("end_line", 0))
    matches = [
        _gold_identity(span, scenario.repository_revision)
        for span in scenario.expected_source_spans
        if revision == scenario.repository_revision
        and path == span.path
        and start <= span.start_line
        and end >= span.end_line
    ]
    return bool(matches), matches


def evaluate_query(
    scenario: Scenario,
    candidates: list[dict[str, Any]],
    *,
    top_k: int,
) -> dict[str, Any]:
    if scenario.unanswerable:
        return {
            "query_id": scenario.scenario_id,
            "query_text": scenario.question,
            "valid": True,
            "skipped": True,
            "skip_reason": "unanswerable_query_skipped_by_frozen_protocol",
            "metrics": None,
            "candidates": [],
            "relation_origin_gold_hits": 0,
            "relation_assisted_gold_hits": 0,
        }
    if top_k < 1:
        raise ValueError("top_k must be positive")
    limited = candidates[:top_k]
    strict_sets: list[set[tuple[Any, ...]]] = []
    rows: list[dict[str, Any]] = []
    relation_origin_hits = 0
    relation_assisted_hits = 0
    for rank, candidate in enumerate(limited, 1):
        strict, identities = strict_gold_match(candidate, scenario)
        contained, containment_identities = containment_match(candidate, scenario)
        strict_identity_set = set(identities)
        strict_sets.append(strict_identity_set)
        sources = list(candidate.get("retrieval_sources") or [])
        if strict and sources == ["relation"]:
            relation_origin_hits += 1
        elif strict and any("relation" in str(value) for value in sources):
            relation_assisted_hits += 1
        rows.append(
            {
                **candidate,
                "rank": rank,
                "gold_match": strict,
                "matched_gold_identities": [list(item) for item in sorted(strict_identity_set)],
                "containment_diagnostic": contained and not strict,
                "contained_gold_identities": [list(item) for item in sorted(set(containment_identities))],
                "match_reason": "strict_identity" if strict else (
                    "containing_chunk_diagnostic_only" if contained else "no_match"
                ),
            }
        )
    relevances = [bool(item) for item in strict_sets]
    first_rank = next((rank for rank, value in enumerate(relevances, 1) if value), None)
    declared = {
        _gold_identity(span, scenario.repository_revision)
        for span in scenario.expected_source_spans
    }
    metrics: dict[str, Any] = {}
    for cutoff in TOP_K_VALUES:
        observed = set().union(*strict_sets[:cutoff]) if strict_sets[:cutoff] else set()
        metrics[f"hit_at_{cutoff}"] = float(any(relevances[:cutoff]))
        metrics[f"recall_at_{cutoff}"] = len(observed.intersection(declared)) / len(declared) if declared else 0.0
        metrics[f"ndcg_at_{cutoff}"] = _ndcg(relevances[:cutoff], min(len(declared), cutoff))
    metrics["mrr_at_8"] = 1.0 / first_rank if first_rank is not None and first_rank <= 8 else 0.0
    metrics["hit_at_10"] = metrics["hit_at_8"]
    metrics["mrr_at_10"] = metrics["mrr_at_8"]
    metrics["hit_at_10_disclosure"] = "computed-from-at-most-top-8-not-ten-retrieved"
    return {
        "query_id": scenario.scenario_id,
        "query_text": scenario.question,
        "category": scenario.category,
        "valid": True,
        "skipped": False,
        "skip_reason": None,
        "top_k": top_k,
        "metrics": metrics,
        "candidates": rows,
        "relation_origin_gold_hits": relation_origin_hits,
        "relation_assisted_gold_hits": relation_assisted_hits,
    }


def aggregate_path_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(records, key=lambda item: str(item.get("query_id", "")))
    valid = [item for item in ordered if item.get("valid") and not item.get("skipped")]
    skipped = [item for item in ordered if item.get("skipped")]
    metric_names = sorted(
        {
            key
            for item in valid
            for key, value in (item.get("metrics") or {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    means: dict[str, float] = {}
    for name in metric_names:
        values = [float(item["metrics"][name]) for item in valid]
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"metric {name} contains NaN or Infinity")
        means[name] = statistics.fmean(values) if values else 0.0
    latencies = sorted(
        float(item["latency_ms"])
        for item in valid
        if isinstance(item.get("latency_ms"), (int, float))
        and math.isfinite(float(item["latency_ms"]))
    )
    return {
        "query_count": len(ordered),
        "valid_answerable_count": len(valid),
        "skipped_count": len(skipped),
        "failed_count": len(ordered) - len(valid) - len(skipped),
        "metrics": means,
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
    }


def compare_paths(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    left_path: str,
    right_path: str,
    seed: int = RANDOM_SEED,
    samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    left_by_id = {
        item["query_id"]: item
        for item in left
        if item.get("valid") and not item.get("skipped")
    }
    right_by_id = {
        item["query_id"]: item
        for item in right
        if item.get("valid") and not item.get("skipped")
    }
    query_ids = sorted(set(left_by_id).intersection(right_by_id))
    deltas: list[float] = []
    improved = unchanged = regressed = gain = loss = 0
    query_deltas: list[dict[str, Any]] = []
    for query_id in query_ids:
        first = float(left_by_id[query_id]["metrics"]["mrr_at_10"])
        second = float(right_by_id[query_id]["metrics"]["mrr_at_10"])
        delta = second - first
        deltas.append(delta)
        improved += delta > 0
        unchanged += delta == 0
        regressed += delta < 0
        left_hit = bool(left_by_id[query_id]["metrics"]["hit_at_8"])
        right_hit = bool(right_by_id[query_id]["metrics"]["hit_at_8"])
        gain += right_hit and not left_hit
        loss += left_hit and not right_hit
        query_deltas.append(
            {"query_id": query_id, "left": first, "right": second, "delta": delta}
        )
    confidence = _paired_bootstrap(deltas, seed=seed, samples=samples)
    return {
        "left_path": left_path,
        "right_path": right_path,
        "paired_count": len(query_ids),
        "mean_mrr_at_10_delta": statistics.fmean(deltas) if deltas else None,
        "improved_queries": improved,
        "unchanged_queries": unchanged,
        "regressed_queries": regressed,
        "relation_new_gold_gain_at_8": gain,
        "relation_gold_loss_at_8": loss,
        "bootstrap_95_ci": confidence,
        "seed": seed,
        "samples": max(100, min(20_000, samples)) if deltas else 0,
        "query_deltas": query_deltas,
    }


def classify_failures(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_path = {str(item.get("path_id")): item for item in records}
    query_id = next((str(item.get("query_id")) for item in records), "")
    hits = {
        path: bool((item.get("metrics") or {}).get("hit_at_8"))
        for path, item in by_path.items()
    }
    categories: set[str] = set()
    if len(hits) == 5 and all(hits.values()):
        categories.add("all paths hit")
    if len(hits) == 5 and not any(hits.values()):
        categories.add("all paths miss")
    if hits.get("B") and not hits.get("A"):
        categories.add("v2 fixes v1")
    if hits.get("A") and not hits.get("B"):
        categories.add("v2 regresses v1")
    if hits.get("C") and not hits.get("B"):
        categories.add("hierarchy gain")
    if hits.get("B") and not hits.get("C"):
        categories.add("hierarchy noise")
    if hits.get("D") and not hits.get("B"):
        categories.add("relation gain")
    if hits.get("B") and not hits.get("D"):
        categories.add("relation noise")
    if hits.get("E") and not hits.get("C") and not hits.get("D"):
        categories.add("hierarchy + relation complementary")
    if (hits.get("C") or hits.get("D")) and not hits.get("E"):
        categories.add("hierarchy + relation conflict")
    for item in records:
        candidates = item.get("candidates") or []
        if any(value.get("containment_diagnostic") for value in candidates):
            categories.add("matcher limitation")
        warnings = " ".join(str(value) for value in item.get("warnings", [])).casefold()
        audit = item.get("retrieval_audit") or {}
        relation = audit.get("relation") or {}
        if relation.get("controlled_unavailable"):
            categories.add("relation unavailable")
        if relation.get("truncated"):
            categories.add("budget truncation")
        if "ambiguous" in warnings:
            categories.add("ambiguous relation target")
        if "external" in warnings:
            categories.add("external relation")
        if "stale" in warnings or "scope" in warnings:
            categories.add("stale relation graph" if "stale" in warnings else "scope conflict")
        suppressed = ((relation.get("selection") or {}).get("suppressed_relation_candidates") or [])
        reasons = {str(value.get("reason", "")) for value in suppressed if isinstance(value, dict)}
        if "slot_cap" in reasons:
            categories.add("slot-cap suppression")
    return {
        "query_id": query_id,
        "categories": sorted(categories),
        "path_hits": dict(sorted(hits.items())),
    }


def _ndcg(relevance: list[bool], ideal_count: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 1) for rank, value in enumerate(relevance, 1) if value)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal if ideal else 0.0


def _paired_bootstrap(values: list[float], *, seed: int, samples: int) -> list[float] | None:
    if not values:
        return None
    count = max(100, min(20_000, int(samples)))
    randomizer = random.Random(seed)
    means = sorted(
        statistics.fmean(randomizer.choice(values) for _ in values)
        for _ in range(count)
    )
    return [_percentile(means, 0.025) or 0.0, _percentile(means, 0.975) or 0.0]


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def warning_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                str(warning)
                for record in records
                for warning in record.get("warnings", [])
            ).items()
        )
    )
