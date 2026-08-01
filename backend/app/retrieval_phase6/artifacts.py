from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from app.retrieval_phase5.contracts import (
    FORMAL_TOP_K,
    FROZEN_PATHS,
    canonical_hash,
    canonical_json,
    immutable_write_json,
)
from app.retrieval_phase5.metrics import classify_failures
from app.retrieval_phase6.metrics import aggregate_cross_repository
from app.retrieval_phase6.metrics import repository_stratified_compare


COMPARISON_PAIRS = (("A", "B"), ("B", "C"), ("B", "D"), ("C", "E"), ("D", "E"))


def validate_cross_repository_matrix(
    records_by_repo_path: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    expected_paths = {item.path_id for item in FROZEN_PATHS}
    global_ids: set[str] = set()
    query_count = 0
    for repo, matrix in sorted(records_by_repo_path.items()):
        if set(matrix) != expected_paths:
            raise ValueError(f"repository {repo} must contain exactly paths A through E")
        ids_by_path = {
            path: [str(item.get("query_id", "")) for item in records]
            for path, records in matrix.items()
        }
        canonical = sorted(ids_by_path["A"])
        if any(sorted(values) != canonical for values in ids_by_path.values()):
            raise ValueError(f"repository {repo} paths do not cover the same query identities")
        if len(canonical) != len(set(canonical)) or global_ids.intersection(canonical):
            raise ValueError("query identities must be unique within and across repositories")
        global_ids.update(canonical)
        query_count += len(canonical)
        for path, records in matrix.items():
            for item in records:
                if item.get("repository_id") != repo or item.get("path_id") != path:
                    raise ValueError("record repository/path attribution is inconsistent")
                if not item.get("skipped") and int(item.get("top_k", 0)) != FORMAL_TOP_K:
                    raise ValueError("result matrix changed the frozen top-k")
    return {
        "repository_count": len(records_by_repo_path),
        "query_count": query_count,
        "query_count_per_path": query_count,
        "same_query_set_within_repository": True,
        "globally_unique_query_ids": True,
        "same_top_k": True,
    }


def build_applicability_diagnostics(
    records_by_repo_path: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    relation_enabled = [
        item
        for matrix in records_by_repo_path.values()
        for path in ("D", "E")
        for item in matrix[path]
        if item.get("valid") and not item.get("skipped")
    ]
    hierarchy_enabled = [
        item
        for matrix in records_by_repo_path.values()
        for path in ("C", "E")
        for item in matrix[path]
        if item.get("valid") and not item.get("skipped")
    ]
    relation = _grouped_diagnostics(relation_enabled, _relation_summary)
    hierarchy = _grouped_diagnostics(hierarchy_enabled, _hierarchy_summary)
    gains_by_key: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
    paired: list[dict[str, Any]] = []
    for repo, matrix in sorted(records_by_repo_path.items()):
        for left, right in (("B", "D"), ("C", "E")):
            comparison = repository_stratified_compare(
                matrix[left], matrix[right], left_path=left, right_path=right,
                seed=20260726, samples=2_000,
            )
            paired.append({"repository_id": repo, **comparison})
            left_by_id = {item["query_id"]: item for item in matrix[left] if not item.get("skipped")}
            right_by_id = {item["query_id"]: item for item in matrix[right] if not item.get("skipped")}
            for query_id in sorted(set(left_by_id).intersection(right_by_id)):
                gain = bool(right_by_id[query_id]["metrics"]["hit_at_8"]) and not bool(left_by_id[query_id]["metrics"]["hit_at_8"])
                loss = bool(left_by_id[query_id]["metrics"]["hit_at_8"]) and not bool(right_by_id[query_id]["metrics"]["hit_at_8"])
                gains_by_key[(repo, query_id)][f"{left}->{right}:gain"] = gain
                gains_by_key[(repo, query_id)][f"{left}->{right}:loss"] = loss
    relation["overall"]["new_gold_gain_at_8"] = sum(any(v for k, v in flags.items() if k.endswith(":gain")) for flags in gains_by_key.values())
    relation["overall"]["gold_loss_at_8"] = sum(any(v for k, v in flags.items() if k.endswith(":loss")) for flags in gains_by_key.values())
    relation["paired_comparisons"] = paired
    relation["effects"] = _paired_effects(records_by_repo_path, (("B", "D"), ("C", "E")))
    hierarchy["effects"] = _paired_effects(records_by_repo_path, (("B", "C"), ("D", "E")))
    return {"relation": relation, "hierarchy": hierarchy}


def _paired_effects(
    records_by_repo_path: dict[str, dict[str, list[dict[str, Any]]]],
    contrasts: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for repo, matrix in sorted(records_by_repo_path.items()):
        for left, right in contrasts:
            left_by_id = {
                str(item["query_id"]): item
                for item in matrix[left]
                if item.get("valid") and not item.get("skipped")
            }
            right_by_id = {
                str(item["query_id"]): item
                for item in matrix[right]
                if item.get("valid") and not item.get("skipped")
            }
            for query_id in sorted(set(left_by_id).intersection(right_by_id)):
                first = left_by_id[query_id]
                second = right_by_id[query_id]
                first_mrr = float(first["metrics"]["mrr_at_8"])
                second_mrr = float(second["metrics"]["mrr_at_8"])
                first_hit = bool(first["metrics"]["hit_at_8"])
                second_hit = bool(second["metrics"]["hit_at_8"])
                first_ranked = _ranked_identity(first)
                second_ranked = _ranked_identity(second)
                rows.append(
                    {
                        "repository_id": repo,
                        "query_id": query_id,
                        "primary_stratum": str(first.get("primary_stratum", "unknown")),
                        "contrast": f"{left}->{right}",
                        "left_path": left,
                        "right_path": right,
                        "left_mrr_at_8": first_mrr,
                        "right_mrr_at_8": second_mrr,
                        "mrr_at_8_delta": second_mrr - first_mrr,
                        "strict_gain": second_hit and not first_hit,
                        "strict_loss": first_hit and not second_hit,
                        "left_first_gold_rank": _first_gold_rank(first),
                        "right_first_gold_rank": _first_gold_rank(second),
                        "ranked_identity_changed": first_ranked != second_ranked,
                        "candidate_set_changed": set(first_ranked) != set(second_ranked),
                    }
                )
    by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_contrast: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in rows:
        by_repo[item["repository_id"]].append(item)
        by_stratum[item["primary_stratum"]].append(item)
        by_contrast[item["contrast"]].append(item)
    return {
        "combined": _effect_summary(rows),
        "by_repository": {key: _effect_summary(value) for key, value in sorted(by_repo.items())},
        "by_stratum": {key: _effect_summary(value) for key, value in sorted(by_stratum.items())},
        "by_contrast": {key: _effect_summary(value) for key, value in sorted(by_contrast.items())},
        "query_effects": rows,
    }


def _effect_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [float(item["mrr_at_8_delta"]) for item in rows]
    return {
        "paired_cell_count": len(rows),
        "strict_gain_cell_count": sum(bool(item["strict_gain"]) for item in rows),
        "strict_loss_cell_count": sum(bool(item["strict_loss"]) for item in rows),
        "rank_improved_cell_count": sum(value > 0 for value in deltas),
        "rank_unchanged_cell_count": sum(value == 0 for value in deltas),
        "rank_regressed_cell_count": sum(value < 0 for value in deltas),
        "ranked_identity_changed_cell_count": sum(bool(item["ranked_identity_changed"]) for item in rows),
        "candidate_set_changed_cell_count": sum(bool(item["candidate_set_changed"]) for item in rows),
        "mean_mrr_at_8_delta": sum(deltas) / len(deltas) if deltas else None,
    }


def _ranked_identity(record: dict[str, Any]) -> list[str]:
    return [str(item.get("chunk_identity", "")) for item in record.get("candidates") or []]


def _first_gold_rank(record: dict[str, Any]) -> int | None:
    return next(
        (int(item.get("rank", 0)) for item in record.get("candidates") or [] if item.get("gold_match")),
        None,
    )


def write_phase6_artifacts(
    root: Path,
    *,
    manifest: dict[str, Any],
    records_by_repo_path: dict[str, dict[str, list[dict[str, Any]]]],
    determinism: dict[str, Any],
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    validation = validate_cross_repository_matrix(records_by_repo_path)
    aggregate = aggregate_cross_repository(records_by_repo_path)
    comparisons = []
    for left, right in COMPARISON_PAIRS:
        left_records = [item for repo in sorted(records_by_repo_path) for item in records_by_repo_path[repo][left]]
        right_records = [item for repo in sorted(records_by_repo_path) for item in records_by_repo_path[repo][right]]
        comparisons.append(
            repository_stratified_compare(
                left_records,
                right_records,
                left_path=left,
                right_path=right,
                seed=int(manifest.get("random_seed", 20260726)),
                samples=int(manifest.get("bootstrap_samples", 2_000)),
            )
        )
    diagnostics = build_applicability_diagnostics(records_by_repo_path)
    flat = [
        item
        for repo in sorted(records_by_repo_path)
        for path in "ABCDE"
        for item in records_by_repo_path[repo][path]
    ]
    by_query: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in flat:
        by_query[(str(item["repository_id"]), str(item["query_id"]))].append(item)
    failures = [
        {"repository_id": repo, **classify_failures(records)}
        for (repo, _query), records in sorted(by_query.items())
    ]
    invalid_evidence = sum(int((item.get("citation_validation") or {}).get("invalid") or 0) for item in flat)
    invalid_relations = sum(int((item.get("relation_validation") or {}).get("invalid") or 0) for item in flat)
    validation.update(
        {
            "determinism": determinism,
            "invalid_evidence_count": invalid_evidence,
            "invalid_relation_chain_count": invalid_relations,
            "passed": bool(determinism.get("passed")) and invalid_evidence == 0 and invalid_relations == 0,
        }
    )
    _require_finite({"aggregate": aggregate, "comparisons": comparisons, "diagnostics": diagnostics})
    immutable_write_json(root / "aggregate.json", aggregate)
    immutable_write_json(root / "paired_comparisons.json", comparisons)
    immutable_write_json(root / "applicability_diagnostics.json", diagnostics)
    immutable_write_json(root / "validation_summary.json", validation)
    immutable_write_json(root / "failure_cases.json", failures)
    immutable_write_json(
        root / "results.json",
        {
            "evaluation_version": manifest.get("evaluation_version"),
            "aggregate": aggregate,
            "paired_comparisons": comparisons,
            "applicability_diagnostics": diagnostics,
            "validation": validation,
            "failure_cases": failures,
        },
    )
    _write_jsonl(root / "query_results.jsonl", flat)
    _write_csv(root / "query_results.csv", flat)
    names = (
        "aggregate.json", "paired_comparisons.json", "applicability_diagnostics.json",
        "validation_summary.json", "failure_cases.json", "results.json",
        "query_results.jsonl", "query_results.csv",
    )
    hashes = {name: _sha256(root / name) for name in names}
    value = {"files": hashes, "result_hash": canonical_hash(hashes)}
    immutable_write_json(root / "result_hashes.json", value)
    _write_text(root / "report.md", _report(manifest, aggregate, comparisons, diagnostics, validation, value["result_hash"]))
    return value


def _grouped_diagnostics(
    records: list[dict[str, Any]],
    summarize: Callable[[list[dict[str, Any]]], dict[str, Any]],
) -> dict[str, Any]:
    by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        by_repo[str(item.get("repository_id"))].append(item)
        by_stratum[str(item.get("primary_stratum"))].append(item)
    return {
        "overall": summarize(records),
        "by_repository": {key: summarize(value) for key, value in sorted(by_repo.items())},
        "by_stratum": {key: summarize(value) for key, value in sorted(by_stratum.items())},
    }


def _relation_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    selected = candidates = triggered = truncated = direct_backfill = 0
    edges = 0
    suppressions: Counter[str] = Counter()
    types: Counter[str] = Counter()
    for item in records:
        relation = ((item.get("retrieval_audit") or {}).get("relation") or {})
        triggered += bool(relation) and not relation.get("controlled_unavailable")
        candidates += bool(relation.get("candidate_priorities"))
        selection = relation.get("selection") or {}
        selected_paths = selection.get("selected_relation_paths") or []
        selected += bool(selection.get("selected_relation_candidates"))
        edges += int(relation.get("edges_accepted") or 0)
        truncated += bool(relation.get("truncated"))
        direct_backfill += int(selection.get("direct_backfill") or 0)
        for path in selected_paths:
            if isinstance(path, dict):
                types[str(path.get("relation_type", "unknown"))] += 1
        for value in selection.get("suppressed_relation_candidates") or []:
            if isinstance(value, dict):
                suppressions[str(value.get("reason", "unknown"))] += 1
    return {
        "enabled_query_count": len(records),
        "triggered_query_count": triggered,
        "candidate_query_count": candidates,
        "selected_query_count": selected,
        "accepted_edge_count": edges,
        "truncated_query_count": truncated,
        "direct_backfill_count": direct_backfill,
        "selected_relation_types": dict(sorted(types.items())),
        "suppression_reasons": dict(sorted(suppressions.items())),
    }


def _hierarchy_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    executed = derived = retained_hierarchy = warnings = truncated = 0
    hierarchy_candidate_count = retained_hierarchy_candidate_count = 0
    retained_origin: Counter[str] = Counter()
    for item in records:
        hierarchy = ((item.get("retrieval_audit") or {}).get("hierarchy") or {})
        executed += bool(hierarchy)
        audit_candidates = [
            value for value in hierarchy.get("candidates") or []
            if isinstance(value, dict)
        ]
        hierarchy_candidates = [
            value for value in audit_candidates
            if value.get("origin") == "hierarchy"
        ]
        retained_candidates = [
            value for value in hierarchy_candidates
            if value.get("decision") == "retained"
        ]
        # The production hierarchy audit exposes candidate origin/decision rows.
        # The count fields remain a compatibility fallback for small synthetic fixtures.
        derived_count = len(hierarchy_candidates) or int(hierarchy.get("derived_candidate_count") or 0)
        retained_count = len(retained_candidates) or int(hierarchy.get("selected_candidate_count") or 0)
        derived += derived_count > 0
        retained_hierarchy += retained_count > 0
        hierarchy_candidate_count += derived_count
        retained_hierarchy_candidate_count += retained_count
        warnings += len(hierarchy.get("warnings") or [])
        truncated += bool(hierarchy.get("truncated"))
        for candidate in item.get("candidates") or []:
            retained_origin[str(candidate.get("candidate_origin", "unknown"))] += 1
    return {
        "enabled_query_count": len(records),
        "executed_query_count": executed,
        "derived_candidate_query_count": derived,
        "hierarchy_candidate_count": hierarchy_candidate_count,
        "retained_hierarchy_query_count": retained_hierarchy,
        "retained_hierarchy_candidate_count": retained_hierarchy_candidate_count,
        "selected_candidate_query_count": retained_hierarchy,
        "truncated_query_count": truncated,
        "warning_count": warnings,
        "retained_candidate_origins": dict(sorted(retained_origin.items())),
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for item in records:
            handle.write(canonical_json(item) + "\n")


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = (
        "repository_id", "primary_stratum", "query_id", "query_text", "path_id",
        "valid", "skipped", "latency_ms", "top_k", "hit_at_1", "hit_at_3",
        "hit_at_5", "hit_at_8", "mrr_at_8", "recall_at_8", "ndcg_at_8",
        "rank", "chunk_identity", "path", "qualified_name", "start_line", "end_line",
        "content_hash", "gold_match", "candidate_origin", "retrieval_sources", "warnings",
    )
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            metrics = record.get("metrics") or {}
            candidates = record.get("candidates") or [None]
            for candidate in candidates:
                candidate = candidate or {}
                writer.writerow(
                    {
                        "repository_id": record.get("repository_id"),
                        "primary_stratum": record.get("primary_stratum"),
                        "query_id": record.get("query_id"),
                        "query_text": record.get("query_text"),
                        "path_id": record.get("path_id"),
                        "valid": record.get("valid"),
                        "skipped": record.get("skipped"),
                        "latency_ms": record.get("latency_ms"),
                        "top_k": record.get("top_k"),
                        **{name: metrics.get(name) for name in ("hit_at_1", "hit_at_3", "hit_at_5", "hit_at_8", "mrr_at_8", "recall_at_8", "ndcg_at_8")},
                        **{name: candidate.get(name) for name in ("rank", "chunk_identity", "path", "qualified_name", "start_line", "end_line", "content_hash", "gold_match", "candidate_origin")},
                        "retrieval_sources": json.dumps(candidate.get("retrieval_sources", []), ensure_ascii=False),
                        "warnings": json.dumps(record.get("warnings", []), ensure_ascii=False),
                    }
                )


def _report(
    manifest: dict[str, Any],
    aggregate: dict[str, Any],
    comparisons: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    validation: dict[str, Any],
    result_hash: str,
) -> str:
    lines = [
        "# Retrieval v2 Phase 6 cross-repository offline evaluation",
        "",
        "## Acceptance",
        "",
        "Completed and passed" if validation["passed"] else "Completed but failed",
        "",
        f"- Result hash: `{result_hash}`",
        f"- Determinism: `{bool((validation.get('determinism') or {}).get('passed'))}`",
        f"- Repositories: `{', '.join(sorted(aggregate['per_repository']))}`",
        "",
        "## Micro five-path results",
        "",
        "| Path | Hit@1 | Hit@3 | Hit@5 | Hit@8 | MRR@8 | Recall@8 | nDCG@8 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for path in "ABCDE":
        metrics = aggregate["micro"][path]["metrics"]
        lines.append("| " + path + " | " + " | ".join(_fmt(metrics.get(name)) for name in ("hit_at_1", "hit_at_3", "hit_at_5", "hit_at_8", "mrr_at_8", "recall_at_8", "ndcg_at_8")) + " |")
    lines.extend(["", "## Repository-stratified paired comparisons", ""])
    for item in comparisons:
        lines.append(
            f"- {item['left_path']} -> {item['right_path']}: micro/macro MRR@8 delta "
            f"{_fmt(item['micro_mean_mrr_at_8_delta'])}/{_fmt(item['macro_mean_mrr_at_8_delta'])}; "
            f"gain/loss {item['new_gold_gain_at_8']}/{item['gold_loss_at_8']}; "
            f"micro CI `{item['micro_bootstrap_95_ci']}`."
        )
    relation = diagnostics["relation"]["overall"]
    hierarchy = diagnostics["hierarchy"]["overall"]
    lines.extend(
        [
            "",
            "## Applicability diagnostics",
            "",
            f"- Relation selected queries: {relation['selected_query_count']}; unique strict gain/loss queries: {relation['new_gold_gain_at_8']}/{relation['gold_loss_at_8']}.",
            f"- Hierarchy executed/derived/selected queries: {hierarchy['executed_query_count']}/{hierarchy['derived_candidate_query_count']}/{hierarchy['selected_candidate_query_count']}.",
            f"- Invalid final Evidence/relation chains: {validation['invalid_evidence_count']}/{validation['invalid_relation_chain_count']}.",
            "",
            "## Boundary",
            "",
            "This frozen offline result diagnoses two fixed Python repositories. It does not establish universal retrieval, teaching, or answer-generation quality.",
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
    return "n/a" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value)
