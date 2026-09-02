from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.m5.contracts import AllowedEvidenceScope, RelationIdentity, Scenario, SourceSpan
from app.retrieval_phase5.contracts import (
    FROZEN_PATHS,
    FROZEN_TEXT_HASH_POLICY,
    MATCHER_VERSION,
    ManifestError,
    canonical_hash,
    frozen_text_hash,
    read_frozen_text,
)
from app.retrieval_phase5.runner import ClickBenchmarkSnapshot, load_click_benchmark


class Phase6ContractError(ValueError):
    pass


PHASE6_FORMAL_TOP_K = 8
PHASE6_PATH_E_LABEL = "hierarchy + relation"
_PHASE6_COMPONENTS = (
    "click_strata.json",
    "httpx_queries.jsonl",
    "matcher.json",
    "protocol.json",
    "repositories.json",
    "selection.json",
    "strata.json",
)


def phase6_path_label(path_id: str, phase5_label: str) -> str:
    return PHASE6_PATH_E_LABEL if path_id == "E" else phase5_label


@dataclass(frozen=True)
class Phase6BenchmarkSnapshot:
    phase6_directory: Path
    click: ClickBenchmarkSnapshot
    manifest: dict[str, Any]
    protocol: dict[str, Any]
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

    @property
    def formal_top_k(self) -> int:
        return int(self.manifest["formal_top_k"])

    @property
    def failure_taxonomy(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self.protocol["failure_taxonomy"])

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
    manifest_content = _load_frozen_text(directory / "manifest.json")
    manifest = _parse_json("manifest.json", manifest_content)
    if manifest.get("evaluation_version") != "retrieval-v2-phase6@1" or not manifest.get("frozen"):
        raise Phase6ContractError("Phase 6 manifest is not the frozen evaluation contract")
    declared_hashes = manifest.get("file_hashes_before_manifest") or {}
    if set(declared_hashes) != set(_PHASE6_COMPONENTS):
        raise Phase6ContractError("Phase 6 frozen component set changed")
    contents: dict[str, bytes] = {}
    for name in _PHASE6_COMPONENTS:
        content = _load_frozen_text(directory / name)
        if frozen_text_hash(content) != declared_hashes[name]:
            raise Phase6ContractError(f"frozen benchmark file hash mismatch: {name}")
        contents[name] = content
    matcher = _parse_json("matcher.json", contents["matcher.json"])
    protocol = _parse_json("protocol.json", contents["protocol.json"])
    click_strata = _parse_json("click_strata.json", contents["click_strata.json"])
    repositories_raw = _parse_json("repositories.json", contents["repositories.json"])
    _parse_json("selection.json", contents["selection.json"])
    _parse_json("strata.json", contents["strata.json"])
    httpx_records = tuple(_parse_jsonl("httpx_queries.jsonl", contents["httpx_queries.jsonl"]))
    if canonical_hash(matcher) != manifest.get("matcher_hash"):
        raise Phase6ContractError("frozen matcher identity differs from the manifest")
    _validate_phase6_protocol(manifest, protocol)
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
    expected_dataset_hash = canonical_hash({
        "benchmark_version": manifest["benchmark_version"],
        "repository_hash": manifest["repository_hash"],
        "query_hash": manifest["query_hash"],
        "gold_hash": manifest["gold_hash"],
        "matcher_hash": manifest["matcher_hash"],
        "strata_hash": manifest["strata_hash"],
        "protocol_hash": manifest["protocol_hash"],
        "click_phase5_dataset_hash": click.dataset_hash,
    })
    if manifest.get("dataset_hash") != expected_dataset_hash:
        raise Phase6ContractError("Phase 6 dataset identity differs from its frozen inputs")
    if not isinstance(repositories_raw, list):
        raise Phase6ContractError("repositories.json must be an array")
    repositories = tuple(sorted(repositories_raw, key=lambda item: item["repository_id"]))
    if tuple(item["repository_id"] for item in repositories) != ("click", "httpx"):
        raise Phase6ContractError("Phase 6 must contain frozen Click plus exactly one HTTPX repository")
    _validate_revision_bindings(manifest, repositories, httpx_records, click)
    _validate_repository_paths(repositories, httpx_records, click)
    httpx = tuple(sorted((_adapt_httpx_query(item) for item in httpx_records), key=lambda item: item.scenario_id))
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
        protocol=protocol,
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
    gold_candidates = [] if unanswerable else _httpx_gold_candidates(value)
    source_spans = [
        SourceSpan(
            path=item["gold_path"],
            qualified_symbol=item["gold_qualified_symbol"],
            start_line=item["gold_span"]["start_line"],
            end_line=item["gold_span"]["end_line"],
            content_hash=item["gold_content_hash"],
        )
        for item in gold_candidates
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
        expected_files=list(dict.fromkeys(item.path for item in source_spans)),
        expected_symbols=list(dict.fromkeys(item.qualified_symbol for item in source_spans)),
        expected_source_spans=source_spans,
        expected_content_hashes=list(dict.fromkeys(item.content_hash for item in source_spans)),
        expected_relation_edges=relation_edges,
        expected_key_points=[value["gold_reason"]],
        unanswerable=unanswerable,
        allowed_evidence_scope=AllowedEvidenceScope(paths=list(dict.fromkeys(item.path for item in source_spans))),
        maximum_steps=5,
        maximum_tool_calls=8,
        annotation_provenance="agent_assisted_developer_curation",
        annotation_status="agent_curated_pending_human_review",
        annotation_note="Phase 6 source-first static review frozen before production retrieval.",
    )


def _httpx_gold_candidates(value: dict[str, Any]) -> list[dict[str, Any]]:
    alternatives = value.get("acceptable_multi_gold")
    if not isinstance(alternatives, list) or any(not isinstance(item, dict) for item in alternatives):
        raise Phase6ContractError("acceptable_multi_gold must be an array of complete gold candidates")
    return [value, *alternatives]


def _validate_phase6_protocol(manifest: dict[str, Any], protocol: dict[str, Any]) -> None:
    if manifest.get("frozen_text_hash_policy") != FROZEN_TEXT_HASH_POLICY:
        raise Phase6ContractError("Phase 6 frozen text hash policy must be utf8-lf-v1")
    if (
        manifest.get("formal_top_k") != PHASE6_FORMAL_TOP_K
        or protocol.get("formal_top_k") != PHASE6_FORMAL_TOP_K
    ):
        raise Phase6ContractError("Phase 6 formal_top_k must equal 8")
    expected_paths = {
        item.path_id: {
            "label": phase6_path_label(item.path_id, item.label),
            **item.request_parameters,
        }
        for item in FROZEN_PATHS
    }
    if protocol.get("paths") != expected_paths:
        raise Phase6ContractError("Phase 6 path declarations differ from the executable contract")
    taxonomy = protocol.get("failure_taxonomy")
    if (
        not isinstance(taxonomy, list)
        or not taxonomy
        or any(not isinstance(item, str) or not item for item in taxonomy)
        or len(taxonomy) != len(set(taxonomy))
    ):
        raise Phase6ContractError("Phase 6 failure taxonomy must be a unique non-empty string list")


def _validate_revision_bindings(
    manifest: dict[str, Any],
    repositories: tuple[dict[str, Any], ...],
    httpx_records: tuple[dict[str, Any], ...],
    click: ClickBenchmarkSnapshot,
) -> None:
    revisions = {
        item["repository_id"]: _required_revision(item.get("resolved_commit"), "repository")
        for item in repositories
    }
    click_manifest = manifest.get("click") or {}
    httpx_manifest = manifest.get("httpx") or {}
    click_revision = _required_revision(click.repository_revision, "Click benchmark")
    httpx_revision = revisions["httpx"]
    if revisions["click"] != click_revision or click_manifest.get("repository_revision") != click_revision:
        raise Phase6ContractError("Click repository revisions are not cross-bound")
    if httpx_manifest.get("repository_revision") != httpx_revision:
        raise Phase6ContractError("HTTPX manifest and repository revisions differ")
    graph = httpx_manifest.get("graph_identity") or {}
    if graph.get("repository_revision") != httpx_revision:
        raise Phase6ContractError("HTTPX graph and repository revisions differ")
    for record in httpx_records:
        if record.get("repository_id") != "httpx" or record.get("gold_revision") != httpx_revision:
            raise Phase6ContractError("HTTPX query/gold revision differs from the repository revision")
        for candidate in _httpx_gold_candidates(record):
            if candidate.get("gold_revision") != httpx_revision:
                raise Phase6ContractError("HTTPX multi-gold revision differs from the repository revision")


def _required_revision(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Phase6ContractError(f"{context} revision must be non-empty")
    return value


def _validate_repository_paths(
    repositories: tuple[dict[str, Any], ...],
    httpx_records: tuple[dict[str, Any], ...],
    click: ClickBenchmarkSnapshot,
) -> None:
    for repository in repositories:
        if repository.get("license_path") is not None:
            _require_repository_path(repository["license_path"], "repository license_path")
    for scenario in click.scenarios:
        for path in (
            *scenario.expected_files,
            *(item.path for item in scenario.expected_source_spans),
            *scenario.allowed_evidence_scope.paths,
            *(item.source_path for item in scenario.expected_relation_edges),
            *(item.target_path for item in scenario.expected_relation_edges if item.target_path),
        ):
            _require_repository_path(path, f"Click scenario {scenario.scenario_id}")
    for record in httpx_records:
        if record.get("answerability") == "unanswerable":
            if record.get("acceptable_multi_gold") != []:
                raise Phase6ContractError("unanswerable query cannot declare multi-gold")
        else:
            for candidate in _httpx_gold_candidates(record):
                _require_repository_path(candidate.get("gold_path"), "HTTPX gold path")
        for evidence in record.get("source_review_evidence") or []:
            if isinstance(evidence, dict):
                for key in ("path", "source_path", "target_path"):
                    if evidence.get(key) is not None:
                        _require_repository_path(evidence[key], f"source_review_evidence.{key}")


def _require_repository_path(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise Phase6ContractError(f"{context} must be a non-empty repository-relative path")
    if value.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", value):
        raise Phase6ContractError(f"{context} must be repository-relative")
    if ".." in re.split(r"[\\/]", value):
        raise Phase6ContractError(f"{context} must not contain a parent escape segment")
    return value


def _load_frozen_text(path: Path) -> bytes:
    try:
        return read_frozen_text(path)
    except ManifestError as exc:
        raise Phase6ContractError(f"unable to read frozen benchmark file: {path.name}") from exc


def _parse_json(name: str, content: bytes) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise Phase6ContractError(f"unable to parse frozen benchmark file: {name}") from exc


def _parse_jsonl(name: str, content: bytes) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in content.splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise Phase6ContractError(f"unable to parse frozen benchmark file: {name}") from exc
