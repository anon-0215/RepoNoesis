from __future__ import annotations

import math
import random
import statistics
from collections import Counter, defaultdict
from typing import Any, Callable, Iterable

from app.m5 import METRIC_SCHEMA_VERSION
from app.m5.contracts import Scenario


def scenario_metrics(scenario: Scenario, result: dict[str, Any]) -> dict[str, Any]:
    evidence = [item for item in result.get("evidence", []) if isinstance(item, dict)]
    relevance = [_evidence_relevant(item, scenario) for item in evidence]
    gold_count = max(
        len(set(scenario.expected_files)),
        len(set(scenario.expected_symbols)),
        len(scenario.expected_source_spans),
        1 if not scenario.unanswerable else 0,
    )
    retrieved_relevant = sum(relevance)
    reciprocal_rank = next((1.0 / rank for rank, relevant in enumerate(relevance[:10], 1) if relevant), 0.0)
    dcg = sum((1.0 / math.log2(rank + 1)) for rank, relevant in enumerate(relevance[:10], 1) if relevant)
    # File-level gold labels can legitimately match several retrieved code chunks.
    # Normalise against every observed relevant gain as well as the declared gold
    # cardinality so binary nDCG remains within its mathematical [0, 1] range.
    ideal_relevant = min(max(gold_count, retrieved_relevant), 10)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_relevant + 1))
    paths = {str(item.get("path", "")) for item in evidence}
    symbols = {
        str(item.get("qualified_name") or item.get("symbol_name") or "") for item in evidence
    }
    spans = {
        (
            str(item.get("path", "")),
            int(item.get("start_line", 0)),
            int(item.get("end_line", 0)),
            str(item.get("content_hash", "")),
        )
        for item in evidence
    }
    gold_spans = {
        (item.path, item.start_line, item.end_line, item.content_hash)
        for item in scenario.expected_source_spans
    }
    precision = retrieved_relevant / len(evidence) if evidence else (1.0 if scenario.unanswerable else 0.0)
    recall = min(1.0, retrieved_relevant / gold_count) if gold_count else (1.0 if not evidence else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    answer = str(result.get("answer", ""))
    abstained = result.get("grounding_status") == "insufficient_evidence" or "证据不足" in answer
    key_hits = sum(point.casefold() in answer.casefold() for point in scenario.expected_key_points)
    chains = [item for item in result.get("evidence_chains", []) if isinstance(item, dict)]
    relation_calls = _tool_count(result, "expand_relations")
    relation_gold = len(scenario.expected_relation_edges)
    relation_hits = sum(
        _relation_matches(gold, chains, evidence) for gold in scenario.expected_relation_edges
    )
    invalid_evidence = sum(item.get("validation_status") != "valid" for item in evidence)
    cross_revision = sum(item.get("repository_revision") != scenario.repository_revision for item in evidence)
    referenced_ids = set(_answer_evidence_ids(answer))
    valid_ids = {str(item.get("evidence_id")) for item in evidence if item.get("validation_status") == "valid"}
    return {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "hit_at_1": float(any(relevance[:1])),
        "hit_at_5": float(any(relevance[:5])),
        "mrr_at_10": reciprocal_rank,
        "ndcg_at_10": dcg / ideal if ideal else (1.0 if scenario.unanswerable and not evidence else 0.0),
        "expected_file_recall": _set_recall(set(scenario.expected_files), paths),
        "expected_symbol_recall": _set_recall(set(scenario.expected_symbols), symbols),
        "expected_span_recall": _set_recall(gold_spans, spans),
        "evidence_precision": precision,
        "evidence_recall": recall,
        "evidence_f1": f1,
        "citation_validation_pass_rate": (
            len(referenced_ids.intersection(valid_ids)) / len(referenced_ids) if referenced_ids else 0.0
        ),
        "current_revision_citation_rate": (
            (len(evidence) - cross_revision) / len(evidence) if evidence else 1.0
        ),
        "cross_revision_citation_count": cross_revision,
        "unsupported_citation_count": len(referenced_ids - valid_ids),
        "invalid_evidence_count": invalid_evidence,
        "expected_relation_edge_recall": relation_hits / relation_gold if relation_gold else None,
        "valid_evidence_chain_rate": (
            sum(item.get("resolution_status") in {"resolved", "ambiguous", "unresolved", "external"} for item in chains) / len(chains)
            if chains else (0.0 if relation_gold else None)
        ),
        "relation_validator_pass_rate": 1.0 if result.get("relation_validator_enabled") else None,
        "invalid_chain_count": sum(item.get("validation_status") == "invalid" for item in chains),
        "unnecessary_relation_expansion": int(relation_calls > 0 and scenario.category not in {"relation", "impact"}),
        "relation_budget_exhausted": int(relation_calls > 0 and result.get("agent_status") == "budget_exhausted"),
        "answer_produced": int(bool(answer.strip()) and not abstained),
        "correct_abstention": int(abstained) if scenario.unanswerable else None,
        "required_citation_present": int(bool(referenced_ids.intersection(valid_ids))) if not scenario.unanswerable else None,
        "forbidden_citation_absent": int(not evidence) if scenario.unanswerable else None,
        "expected_key_point_coverage": key_hits / len(scenario.expected_key_points) if scenario.expected_key_points else 1.0,
        "unsupported_claim_count": len(referenced_ids - valid_ids),
        "tool_calls": _total_tool_calls(result),
        "planner_steps": len(result.get("agent_trace", [])),
        "search_calls": _tool_count(result, "search_code"),
        "lookup_calls": _tool_count(result, "lookup_symbol"),
        "read_calls": _tool_count(result, "read_source"),
        "relation_calls": relation_calls,
        "learning_context_calls": _tool_count(result, "get_learning_context"),
        "budget_exhausted": int(result.get("agent_status") == "budget_exhausted"),
        "fallback": int("fallback" in str(result.get("agent_mode", ""))),
        "timeout": int(result.get("scenario_status") == "timed_out"),
        "provider_error": int(bool(result.get("provider_error"))),
        "latency_ms": _finite_number(result.get("latency_ms", 0.0)),
    }


def aggregate_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    succeeded = [item for item in records if item.get("scenario_status") == "succeeded"]
    status_counts = Counter(str(item.get("scenario_status", "unknown")) for item in records)
    numeric: dict[str, list[float]] = defaultdict(list)
    for record in succeeded:
        for key, value in (record.get("metrics") or {}).items():
            if key.endswith("schema_version"):
                continue
            if isinstance(value, bool):
                numeric[key].append(float(value))
            elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                numeric[key].append(float(value))
    means = {key: statistics.fmean(values) for key, values in numeric.items() if values}
    latencies = sorted(numeric.get("latency_ms", []))
    by_repo = _group_aggregate(succeeded, lambda item: str(item.get("repo_id", "")))
    by_category = _group_aggregate(succeeded, lambda item: str(item.get("category", "")))
    return {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "scenario_count": len(records),
        "successful_count": len(succeeded),
        "failed_count": len(records) - len(succeeded),
        "status_counts": dict(status_counts),
        "means": means,
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "by_repository": by_repo,
        "by_category": by_category,
        "missing_value_policy": "failed scenarios excluded from quality means and counted separately",
    }


def paired_delta(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    metric: str,
    *,
    seed: int = 20260726,
    samples: int = 2_000,
) -> dict[str, Any]:
    left_by_id = {item["scenario_id"]: item for item in left if item.get("scenario_status") == "succeeded"}
    right_by_id = {item["scenario_id"]: item for item in right if item.get("scenario_status") == "succeeded"}
    values: list[float] = []
    for scenario_id in sorted(set(left_by_id).intersection(right_by_id)):
        first = (left_by_id[scenario_id].get("metrics") or {}).get(metric)
        second = (right_by_id[scenario_id].get("metrics") or {}).get(metric)
        if _is_finite_number(first) and _is_finite_number(second):
            values.append(float(second) - float(first))
    if not values:
        return {"metric": metric, "paired_count": 0, "mean_delta": None, "bootstrap_95_ci": None}
    randomizer = random.Random(seed)
    boot = [
        statistics.fmean(randomizer.choice(values) for _ in values)
        for _ in range(max(100, min(20_000, samples)))
    ]
    boot.sort()
    return {
        "metric": metric,
        "paired_count": len(values),
        "mean_delta": statistics.fmean(values),
        "bootstrap_95_ci": [_percentile(boot, 0.025), _percentile(boot, 0.975)],
        "seed": seed,
        "samples": len(boot),
    }


def _evidence_relevant(item: dict[str, Any], scenario: Scenario) -> bool:
    path = str(item.get("path", ""))
    symbol = str(item.get("qualified_name") or item.get("symbol_name") or "")
    span = (
        path,
        int(item.get("start_line", 0)),
        int(item.get("end_line", 0)),
        str(item.get("content_hash", "")),
    )
    gold_spans = {
        (value.path, value.start_line, value.end_line, value.content_hash)
        for value in scenario.expected_source_spans
    }
    return path in scenario.expected_files or symbol in scenario.expected_symbols or span in gold_spans


def _set_recall(gold: set[Any], observed: set[Any]) -> float:
    return len(gold.intersection(observed)) / len(gold) if gold else 1.0


def _answer_evidence_ids(answer: str) -> list[str]:
    import re

    return re.findall(r"\[(E\d+)\]", answer)


def _total_tool_calls(result: dict[str, Any]) -> int:
    return sum(len(step.get("tool_calls", [])) for step in result.get("agent_trace", []) if isinstance(step, dict))


def _tool_count(result: dict[str, Any], name: str) -> int:
    return sum(
        call.get("tool_name") == name
        for step in result.get("agent_trace", [])
        if isinstance(step, dict)
        for call in step.get("tool_calls", [])
        if isinstance(call, dict)
    )


def _relation_matches(gold: Any, chains: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> bool:
    relation_type_seen = any(gold.relation_type in item.get("relation_types", []) for item in chains)
    if not relation_type_seen:
        return False
    identities = {
        (str(item.get("path", "")), str(item.get("qualified_name") or item.get("symbol_name") or ""))
        for item in evidence
    }
    source_seen = (gold.source_path, gold.source_symbol) in identities or any(
        path == gold.source_path and symbol.endswith("." + gold.source_symbol)
        for path, symbol in identities
    )
    target_seen = (
        (gold.target_path, gold.target_symbol) in identities
        if gold.target_path
        else any(symbol == gold.target_symbol or symbol.endswith("." + gold.target_symbol) for _, symbol in identities)
    )
    return source_seen and target_seen


def _group_aggregate(records: list[dict[str, Any]], key: Callable[[dict[str, Any]], str]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        groups[key(item)].append(item)
    output: dict[str, Any] = {}
    for name, items in sorted(groups.items()):
        hit_values = [float(item["metrics"]["hit_at_5"]) for item in items]
        output[name] = {"count": len(items), "hit_at_5": statistics.fmean(hit_values)}
    return output


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _finite_number(value: Any) -> float:
    if not _is_finite_number(value):
        raise ValueError("metric contains NaN, Infinity, or a non-numeric value")
    return float(value)
