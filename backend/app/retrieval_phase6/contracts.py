from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.m5.contracts import AllowedEvidenceScope, RelationIdentity, Scenario, SourceSpan
from app.retrieval_phase5.contracts import MATCHER_VERSION, canonical_hash, file_hash
from app.retrieval_phase5.runner import ClickBenchmarkSnapshot, load_click_benchmark


class Phase6ContractError(ValueError):
    pass


@dataclass(frozen=True)
class Phase6BenchmarkSnapshot:
    phase6_directory: Path
    click: ClickBenchmarkSnapshot
    manifest: dict[str, Any]
    repositories: tuple[dict[str, Any], ...]
    scenarios_by_repo: dict[str, tuple[Scenario, ...]]
    strata_by_query: dict[str, str]
    source_records_by_query: dict[str, dict[str, Any]]

    @property
    def repository_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.scenarios_by_repo))

    @property
    def new_repository_ids(self) -> tuple[str, ...]:
        return tuple(item for item in self.repository_ids if item != "click")

    @property
    def answerable_by_repo(self) -> dict[str, tuple[Scenario, ...]]:
        return {
            repo: tuple(item for item in scenarios if not item.unanswerable)
            for repo, scenarios in self.scenarios_by_repo.items()
        }

    @property
    def query_ids(self) -> tuple[str, ...]:
        return tuple(
            item.scenario_id
            for repo in self.repository_ids
            for item in self.scenarios_by_repo[repo]
        )

    @property
    def matcher_hash(self) -> str:
        return str(self.manifest["matcher_hash"])

    @property
    def matcher_version(self) -> str:
        return MATCHER_VERSION

    def stratum_counts(self, repository_id: str) -> dict[str, int]:
        return dict(
            sorted(
                Counter(
                    self.strata_by_query[item.scenario_id]
                    for item in self.scenarios_by_repo[repository_id]
                ).items()
            )
        )


def load_phase6_benchmark(
    phase6_directory: Path,
    click_dataset_directory: Path,
) -> Phase6BenchmarkSnapshot:
    directory = phase6_directory.resolve()
    manifest = _read_json(directory / "manifest.json")
    if manifest.get("evaluation_version") != "retrieval-v2-phase6@1" or not manifest.get("frozen"):
        raise Phase6ContractError("Phase 6 manifest is not the frozen evaluation contract")
    for name, expected in (manifest.get("file_hashes_before_manifest") or {}).items():
        path = directory / str(name)
        if not path.is_file() or file_hash(path) != expected:
            raise Phase6ContractError(f"frozen benchmark file hash mismatch: {name}")
    matcher = _read_json(directory / "matcher.json")
    if canonical_hash(matcher) != manifest.get("matcher_hash"):
        raise Phase6ContractError("frozen matcher identity differs from the manifest")
    click = load_click_benchmark(click_dataset_directory)
    if any(
        (
            click.dataset_hash != manifest["click"]["dataset_hash"],
            click.query_hash != manifest["click"]["query_hash"],
            click.gold_hash != manifest["click"]["gold_hash"],
            click.matcher_hash != manifest["matcher_hash"],
        )
    ):
        raise Phase6ContractError("Phase 5 Click benchmark identity changed")
    repositories_raw = _read_json(directory / "repositories.json")
    if not isinstance(repositories_raw, list):
        raise Phase6ContractError("repositories.json must be an array")
    repositories = tuple(sorted(repositories_raw, key=lambda item: item["repository_id"]))
    if tuple(item["repository_id"] for item in repositories) != ("click", "httpx"):
        raise Phase6ContractError("Phase 6 must contain frozen Click plus exactly one HTTPX repository")
    httpx_records = tuple(_read_jsonl(directory / "httpx_queries.jsonl"))
    httpx = tuple(sorted((_adapt_httpx_query(item) for item in httpx_records), key=lambda item: item.scenario_id))
    click_strata = _read_json(directory / "click_strata.json")
    strata = {**click_strata, **{item["query_id"]: item["primary_stratum"] for item in httpx_records}}
    scenarios_by_repo = {"click": click.scenarios, "httpx": httpx}
    ids = [item.scenario_id for values in scenarios_by_repo.values() for item in values]
    if len(ids) != len(set(ids)) or len(ids) != manifest.get("total_query_count"):
        raise Phase6ContractError("Phase 6 query identities are not globally unique and complete")
    if set(strata) != set(ids):
        raise Phase6ContractError("every frozen query must have exactly one primary stratum")
    snapshot = Phase6BenchmarkSnapshot(
        phase6_directory=directory,
        click=click,
        manifest=manifest,
        repositories=repositories,
        scenarios_by_repo=scenarios_by_repo,
        strata_by_query=strata,
        source_records_by_query={item["query_id"]: item for item in httpx_records},
    )
    if snapshot.stratum_counts("httpx") != {
        "direct_behavior_location": 6,
        "hierarchy_sensitive": 4,
        "relation_dependent": 6,
        "symbol_focused": 4,
        "unanswerable": 2,
    }:
        raise Phase6ContractError("HTTPX source-first stratum minima changed")
    return snapshot


def _adapt_httpx_query(value: dict[str, Any]) -> Scenario:
    unanswerable = value["answerability"] == "unanswerable"
    span = value.get("gold_span")
    source_spans = [] if unanswerable else [
        SourceSpan(
            path=value["gold_path"],
            qualified_symbol=value["gold_qualified_symbol"],
            start_line=span["start_line"],
            end_line=span["end_line"],
            content_hash=value["gold_content_hash"],
        )
    ]
    relation_edges: list[RelationIdentity] = []
    if value["primary_stratum"] == "relation_dependent":
        evidence = next(
            item for item in value["source_review_evidence"]
            if item.get("relation_type") in {"imports", "calls", "references", "defines"}
        )
        relation_edges.append(
            RelationIdentity(
                relation_type=evidence["relation_type"],
                source_path=evidence["source_path"],
                source_symbol=evidence["source_symbol"],
                target_path=evidence.get("target_path"),
                target_symbol=evidence["target_symbol"],
            )
        )
    category = {
        "direct_behavior_location": "explain",
        "symbol_focused": "locate",
        "relation_dependent": "relation",
        "hierarchy_sensitive": "locate",
        "unanswerable": "unanswerable",
    }[value["primary_stratum"]]
    return Scenario(
        scenario_id=value["query_id"],
        dataset_version="cross-repo-v1",
        repo_id="httpx",
        repository_revision=value["gold_revision"],
        language="python",
        question=value["query_text"],
        category=category,
        difficulty="hard" if value["primary_stratum"] in {"relation_dependent", "hierarchy_sensitive"} else "medium",
        expected_target_type="none" if unanswerable else ("relation" if relation_edges else "symbol"),
        expected_files=[] if unanswerable else [value["gold_path"]],
        expected_symbols=[] if unanswerable else [value["gold_qualified_symbol"]],
        expected_source_spans=source_spans,
        expected_content_hashes=[] if unanswerable else [value["gold_content_hash"]],
        expected_relation_edges=relation_edges,
        expected_key_points=[value["gold_reason"]],
        unanswerable=unanswerable,
        allowed_evidence_scope=AllowedEvidenceScope(paths=[] if unanswerable else [value["gold_path"]]),
        maximum_steps=5,
        maximum_tool_calls=8,
        annotation_provenance="agent_assisted_developer_curation",
        annotation_status="agent_curated_pending_human_review",
        annotation_note="Phase 6 source-first static review frozen before production retrieval.",
    )


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase6ContractError(f"unable to read frozen benchmark file: {path.name}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase6ContractError(f"unable to read frozen benchmark file: {path.name}") from exc
