from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from app.retrieval_phase5.contracts import (
    FORMAL_TOP_K,
    FROZEN_PATHS,
    canonical_hash,
    canonical_json,
    immutable_write_json,
)
from app.retrieval_phase5.metrics import (
    aggregate_path_records,
    classify_failures,
    compare_paths,
    warning_counts,
)


COMPARISON_PAIRS = (
    ("A", "B"),
    ("B", "C"),
    ("B", "D"),
    ("C", "E"),
    ("D", "E"),
)


def validate_result_matrix(records_by_path: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    expected_paths = [item.path_id for item in FROZEN_PATHS]
    if sorted(records_by_path) != sorted(expected_paths):
        raise ValueError("result matrix must contain exactly paths A through E")
    ids_by_path = {
        path: [str(item.get("query_id", "")) for item in records]
        for path, records in records_by_path.items()
    }
    canonical_ids = sorted(ids_by_path[expected_paths[0]])
    if any(sorted(values) != canonical_ids for values in ids_by_path.values()):
        raise ValueError("result matrix paths do not cover the same query identities")
    for path, records in records_by_path.items():
        if len({item.get("query_id") for item in records}) != len(records):
            raise ValueError(f"path {path} contains duplicate query identities")
        for item in records:
            if not item.get("skipped") and int(item.get("top_k", 0)) != FORMAL_TOP_K:
                raise ValueError("result matrix changed the frozen top-k")
            _require_finite(item)
    return {
        "paths": expected_paths,
        "query_count_per_path": len(canonical_ids),
        "query_ids": canonical_ids,
        "same_query_set": True,
        "same_top_k": True,
    }


def build_relation_diagnostics(records_by_path: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    enabled = [
        item
        for path in ("D", "E")
        for item in records_by_path.get(path, [])
        if item.get("valid") and not item.get("skipped")
    ]
    denominator = len(enabled)
    triggered = candidate_queries = selected_queries = 0
    candidate_count = edge_count = truncated_queries = 0
    external = unresolved = ambiguous = stale_scope = 0
    duplicate_support = multi_seed = multi_path = 0
    relation_types: dict[str, Counter[str]] = defaultdict(Counter)
    suppression_reasons: Counter[str] = Counter()
    direct_backfill = 0
    warning_counter: Counter[str] = Counter()
    relation_origin_gold_hits = relation_assisted_gold_hits = 0
    for item in enabled:
        relation = ((item.get("retrieval_audit") or {}).get("relation") or {})
        if relation and not relation.get("controlled_unavailable"):
            triggered += 1
        priorities = relation.get("candidate_priorities") or {}
        candidate_count += len(priorities)
        candidate_queries += bool(priorities)
        edge_count += int(relation.get("edges_accepted") or 0)
        truncated_queries += bool(relation.get("truncated"))
        selection = relation.get("selection") or {}
        selected = selection.get("selected_relation_candidates") or []
        selected_queries += bool(selected)
        direct_backfill += int(selection.get("direct_backfill") or 0)
        for suppressed in selection.get("suppressed_relation_candidates") or []:
            if isinstance(suppressed, dict):
                suppression_reasons[str(suppressed.get("reason", "unknown"))] += 1
        paths = [value for value in relation.get("relation_paths") or [] if isinstance(value, dict)]
        selected_paths = [
            value for value in selection.get("selected_relation_paths") or [] if isinstance(value, dict)
        ]
        by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for value in paths:
            by_target[str(value.get("target_chunk_identity", ""))].append(value)
            relation_types[str(value.get("relation_type", "unknown"))]["edge"] += 1
        for value in selected_paths:
            relation_types[str(value.get("relation_type", "unknown"))]["selected"] += 1
        for target, values in by_target.items():
            if target:
                relation_types[str(values[0].get("relation_type", "unknown"))]["candidate"] += 1
            multi_seed += len({value.get("seed_chunk_identity") for value in values}) > 1
            multi_path += len(values) > 1
        support = relation.get("existing_candidate_support") or {}
        duplicate_support += sum(bool(values) for values in support.values())
        for resolution in relation.get("node_resolutions") or []:
            status = str((resolution or {}).get("status", ""))
            unresolved += status in {"not_found", "unresolved"}
            ambiguous += status == "ambiguous"
            external += status == "external"
            stale_scope += status in {"stale", "scope_conflict"}
        for warning in [*relation.get("warnings", []), *item.get("warnings", [])]:
            warning_counter[str(warning)] += 1
        relation_origin_gold_hits += int(item.get("relation_origin_gold_hits") or 0)
        relation_assisted_gold_hits += int(item.get("relation_assisted_gold_hits") or 0)
    gain = loss = rr_improved = rr_unchanged = rr_regressed = 0
    for off_path, on_path in (("B", "D"), ("C", "E")):
        comparison = compare_paths(
            records_by_path.get(off_path, []),
            records_by_path.get(on_path, []),
            left_path=off_path,
            right_path=on_path,
        )
        gain += comparison["relation_new_gold_gain_at_8"]
        loss += comparison["relation_gold_loss_at_8"]
        rr_improved += comparison["improved_queries"]
        rr_unchanged += comparison["unchanged_queries"]
        rr_regressed += comparison["regressed_queries"]
    return {
        "valid_relation_enabled_query_count": denominator,
        "relation_enabled_query_count": denominator,
        "relation_triggered_query_count": triggered,
        "relation_candidate_query_count": candidate_queries,
        "relation_selected_query_count": selected_queries,
        "relation_candidate_count": candidate_count,
        "accepted_edge_count": edge_count,
        "truncated_query_count": truncated_queries,
        "external_target_count": external,
        "unresolved_target_count": unresolved,
        "ambiguous_target_count": ambiguous,
        "stale_or_scope_conflict_count": stale_scope,
        "duplicate_target_support_count": duplicate_support,
        "multi_seed_target_count": multi_seed,
        "multi_path_target_count": multi_path,
        "relation_type_counts": {
            key: dict(sorted(value.items())) for key, value in sorted(relation_types.items())
        },
        "relation_origin_gold_hit_count": relation_origin_gold_hits,
        "relation_assisted_gold_hit_count": relation_assisted_gold_hits,
        "relation_new_gold_gain_at_8": gain,
        "relation_gold_loss_at_8": loss,
        "reciprocal_rank_improved_queries": rr_improved,
        "reciprocal_rank_unchanged_queries": rr_unchanged,
        "reciprocal_rank_regressed_queries": rr_regressed,
        "suppression_reasons": dict(sorted(suppression_reasons.items())),
        "direct_backfill_count": direct_backfill,
        "warning_counts": dict(sorted(warning_counter.items())),
        "relation_trigger_rate": triggered / denominator if denominator else 0.0,
        "relation_candidate_rate": candidate_queries / denominator if denominator else 0.0,
        "relation_selected_rate": selected_queries / denominator if denominator else 0.0,
        "relation_new_gold_gain_rate_at_8": gain / denominator if denominator else 0.0,
        "relation_gold_loss_rate_at_8": loss / denominator if denominator else 0.0,
        "denominator_definition": "valid answerable path/query cells across D and E",
    }


def write_result_artifacts(
    run_directory: Path,
    *,
    manifest: dict[str, Any],
    records_by_path: dict[str, list[dict[str, Any]]],
    determinism: dict[str, Any],
) -> dict[str, Any]:
    root = run_directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    matrix = validate_result_matrix(records_by_path)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if canonical_json(existing) != canonical_json(manifest):
            raise FileExistsError("existing manifest differs and cannot be overwritten")
    else:
        immutable_write_json(manifest_path, manifest)
    ordered_records = [
        item
        for path in [value.path_id for value in FROZEN_PATHS]
        for item in sorted(records_by_path[path], key=lambda value: str(value.get("query_id", "")))
    ]
    aggregates = {
        path: aggregate_path_records(records_by_path[path])
        for path in [value.path_id for value in FROZEN_PATHS]
    }
    comparisons = [
        compare_paths(
            records_by_path[left],
            records_by_path[right],
            left_path=left,
            right_path=right,
        )
        for left, right in COMPARISON_PAIRS
    ]
    relation = build_relation_diagnostics(records_by_path)
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in ordered_records:
        by_query[str(item.get("query_id", ""))].append(item)
    failures = [classify_failures(values) for _, values in sorted(by_query.items())]
    invalid_evidence = sum(
        int((item.get("citation_validation") or {}).get("invalid") or 0)
        for item in ordered_records
    )
    invalid_chains = sum(
        int((item.get("relation_validation") or {}).get("invalid") or 0)
        for item in ordered_records
    )
    validation = {
        **matrix,
        "invalid_evidence_count": invalid_evidence,
        "invalid_relation_chain_count": invalid_chains,
        "warning_counts": warning_counts(ordered_records),
        "determinism": determinism,
        "no_nan_or_infinity": True,
        "all_paths_present": True,
        "passed": invalid_evidence == 0 and invalid_chains == 0 and bool(determinism.get("passed")),
    }
    aggregate_value = {
        "evaluation_version": manifest.get("evaluation_version"),
        "paths": aggregates,
        "top_10_disclosure": "computed-from-at-most-top-8-not-ten-retrieved",
    }
    result_value = {
        "manifest_identity": canonical_hash(manifest),
        "aggregate": aggregate_value,
        "comparisons": comparisons,
        "relation_diagnostics": relation,
        "validation_summary": validation,
        "failure_cases": failures,
    }
    immutable_write_json(root / "aggregate.json", aggregate_value)
    immutable_write_json(root / "paired_comparisons.json", comparisons)
    immutable_write_json(root / "relation_diagnostics.json", relation)
    immutable_write_json(root / "validation_summary.json", validation)
    immutable_write_json(root / "failure_cases.json", failures)
    immutable_write_json(root / "results.json", result_value)
    _write_jsonl(root / "query_results.jsonl", ordered_records)
    _write_csv(root / "query_results.csv", ordered_records)
    result_hashes = {
        name: _sha256(root / name)
        for name in (
            "manifest.json", "aggregate.json", "paired_comparisons.json",
            "relation_diagnostics.json", "validation_summary.json",
            "failure_cases.json", "results.json", "query_results.jsonl",
            "query_results.csv",
        )
    }
    result_hash = canonical_hash(result_hashes)
    hashes = {"files": result_hashes, "result_hash": result_hash}
    immutable_write_json(root / "result_hashes.json", hashes)
    _write_text(root / "report.md", _markdown_report(manifest, aggregate_value, comparisons, relation, validation, failures, result_hash))
    return hashes


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for item in records:
            handle.write(canonical_json(item) + "\n")


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "query_id", "query_text", "path_id", "valid", "skipped", "skip_reason",
        "latency_ms", "top_k", "rank", "chunk_identity", "path", "qualified_name",
        "start_line", "end_line", "content_hash", "gold_match", "match_reason",
        "candidate_origin", "retrieval_sources", "edge_id", "relation_type", "direction",
        "relation_priority", "fusion_score", "hierarchy_priority", "selection_reason",
        "citation_validation", "relation_validation", "warnings",
    ]
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            candidates = record.get("candidates") or [None]
            for candidate in candidates:
                candidate = candidate or {}
                writer.writerow(
                    {
                        **{key: record.get(key) for key in fields},
                        **{key: candidate.get(key) for key in fields if key not in {"query_id", "query_text", "path_id"}},
                        "query_id": record.get("query_id"),
                        "query_text": record.get("query_text"),
                        "path_id": record.get("path_id"),
                        "retrieval_sources": json.dumps(candidate.get("retrieval_sources", []), ensure_ascii=False),
                        "warnings": json.dumps(record.get("warnings", []), ensure_ascii=False),
                    }
                )


def _markdown_report(
    manifest: dict[str, Any],
    aggregate: dict[str, Any],
    comparisons: list[dict[str, Any]],
    relation: dict[str, Any],
    validation: dict[str, Any],
    failures: list[dict[str, Any]],
    result_hash: str,
) -> str:
    labels = {item.path_id: item.label for item in FROZEN_PATHS}
    lines = [
        "# Retrieval v2 Phase 5 offline evaluation",
        "",
        "## Acceptance",
        "",
        "Completed and passed" if validation.get("passed") else "Completed but failed",
        "",
        "## Frozen run",
        "",
        f"- Repository commit: `{manifest.get('repository_commit', 'unknown')}`",
        f"- Corpus revision: `{manifest.get('corpus_repository_revision', 'unknown')}`",
        f"- Dataset/query count: `{manifest.get('dataset_version', 'unknown')}` / {manifest.get('query_count', 'unknown')}",
        f"- Real embedding: `{manifest.get('embedding_model', 'unknown')}` @ `{manifest.get('embedding_revision', 'unknown')}`",
        f"- Device / dimension: `{manifest.get('device', 'unknown')}` / {manifest.get('embedding_dimension', 'unknown')}",
        f"- Result hash: `{result_hash}`",
        "",
        "## Five-path results",
        "",
        "| Path | Hit@1 | Hit@3 | Hit@5 | Hit@8 | Hit@10 lower bound | MRR@10 lower bound | Queries | Skipped | Mean latency ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for path, value in aggregate["paths"].items():
        metrics = value["metrics"]
        lines.append(
            f"| {labels[path]} | {_fmt(metrics.get('hit_at_1'))} | {_fmt(metrics.get('hit_at_3'))} | "
            f"{_fmt(metrics.get('hit_at_5'))} | {_fmt(metrics.get('hit_at_8'))} | "
            f"{_fmt(metrics.get('hit_at_10'))} | {_fmt(metrics.get('mrr_at_10'))} | "
            f"{value['valid_answerable_count']} | {value['skipped_count']} | {_fmt(value['latency_ms']['mean'])} |"
        )
    lines.extend(["", "Hit@10 and MRR@10 are computed from at most eight returned candidates; production top-k was not changed.", "", "## Paired comparisons", ""])
    for item in comparisons:
        lines.append(
            f"- {item['left_path']} -> {item['right_path']}: mean MRR delta {_fmt(item['mean_mrr_at_10_delta'])}; "
            f"improved/unchanged/regressed {item['improved_queries']}/{item['unchanged_queries']}/{item['regressed_queries']}; "
            f"95% CI `{item['bootstrap_95_ci']}`."
        )
    lines.extend(
        [
            "",
            "## Relation diagnostics",
            "",
            f"- Trigger/candidate/selected rates: {_fmt(relation['relation_trigger_rate'])} / {_fmt(relation['relation_candidate_rate'])} / {_fmt(relation['relation_selected_rate'])}.",
            f"- New strict gold gain / strict gold loss: {relation['relation_new_gold_gain_at_8']} / {relation['relation_gold_loss_at_8']} paired cells.",
            f"- Relation-origin / relation-assisted strict hits: {relation['relation_origin_gold_hit_count']} / {relation['relation_assisted_gold_hit_count']}.",
            f"- Truncated queries / direct backfill: {relation['truncated_query_count']} / {relation['direct_backfill_count']}.",
            "",
            "## Evidence and validation",
            "",
            f"- Invalid final Evidence: {validation['invalid_evidence_count']}.",
            f"- Invalid relation chains: {validation['invalid_relation_chain_count']}.",
            f"- Determinism passed: {bool((validation.get('determinism') or {}).get('passed'))}.",
            "- These are Evidence/Citation Contract Validity results, not answer correctness or citation sufficiency.",
            "",
            "## Failure taxonomy",
            "",
        ]
    )
    for item in failures:
        lines.append(f"- `{item['query_id']}`: {', '.join(item['categories']) or 'no frozen failure category'}")
    lines.extend(
        [
            "",
            "## Conclusion boundary",
            "",
            "The results describe one fixed Click revision and eleven answerable queries. They do not prove statistical stability across repositories, general relation effectiveness, teaching effectiveness, answer correctness, or universal code-understanding quality.",
            "",
        ]
    )
    return "\n".join(lines)


def _require_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("result contains NaN or Infinity")
    if isinstance(value, dict):
        for item in value.values():
            _require_finite(item)
    elif isinstance(value, list):
        for item in value:
            _require_finite(item)


def _sha256(path: Path) -> str:
    digest = __import__("hashlib").sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_text(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value.rstrip() + "\n")


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
