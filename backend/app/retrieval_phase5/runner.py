from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from app.database import Database
from app.m5.contracts import RepositorySpec, Scenario
from app.retrieval_phase5.contracts import (
    FORMAL_TOP_K,
    FROZEN_PATHS,
    MATCHER_DEFINITION,
    FrozenPath,
    canonical_hash,
    ensure_formal_embedding_identity,
    file_hash,
)
from app.retrieval_phase5.metrics import evaluate_query
from app.services.agent_contracts import AgentLimits, CancellationToken, SearchCodeInput
from app.services.agent_tools import EvidenceStore, build_m2_tool_registry, build_tool_context
from app.services.embedding_service import EmbeddingService
from app.services.evidence import CitationValidator, Evidence
from app.services.relation_graph import EvidenceChain, RelationValidator


class Phase5RunError(RuntimeError):
    pass


@dataclass(frozen=True)
class ClickBenchmarkSnapshot:
    dataset_directory: Path
    repository: RepositorySpec
    scenarios: tuple[Scenario, ...]
    dataset_hash: str
    query_hash: str
    gold_hash: str
    matcher_hash: str

    @property
    def answerable(self) -> tuple[Scenario, ...]:
        return tuple(item for item in self.scenarios if not item.unanswerable)

    @property
    def repository_revision(self) -> str:
        return self.repository.exact_commit_sha


class CountingEmbeddingService:
    def __init__(self, service: EmbeddingService) -> None:
        self.service = service
        self.query_encode_calls = 0
        self.query_encode_items = 0
        self.document_encode_calls = 0
        self.document_encode_items = 0

    @property
    def settings(self) -> Any:
        return self.service.settings

    def encode_query(self, text: str, local_files_only: bool = False) -> list[float]:
        self.query_encode_calls += 1
        self.query_encode_items += 1
        return self.service.encode_query(text, local_files_only=local_files_only)

    def encode_documents(
        self,
        texts: list[str],
        local_files_only: bool = False,
    ) -> list[list[float]]:
        self.document_encode_calls += 1
        self.document_encode_items += len(texts)
        return self.service.encode_documents(texts, local_files_only=local_files_only)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.service, name)


def load_click_benchmark(dataset_directory: Path) -> ClickBenchmarkSnapshot:
    directory = dataset_directory.resolve()
    manifest_path = directory / "manifest.json"
    repositories_path = directory / "repositories.json"
    scenarios_path = directory / "scenarios.jsonl"
    manifest = _read_json(manifest_path)
    repositories_raw = _read_json(repositories_path)
    if not isinstance(repositories_raw, list):
        raise Phase5RunError("benchmark repositories must be a JSON array")
    repositories = [RepositorySpec.model_validate(item) for item in repositories_raw]
    click = [item for item in repositories if item.repo_id == "click"]
    if len(click) != 1:
        raise Phase5RunError("benchmark must contain exactly one Click repository")
    scenarios = tuple(
        sorted(
            (
                Scenario.model_validate(json.loads(line))
                for line in scenarios_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
                and json.loads(line).get("repo_id") == "click"
            ),
            key=lambda item: item.scenario_id,
        )
    )
    if len(scenarios) != 12 or sum(not item.unanswerable for item in scenarios) != 11:
        raise Phase5RunError("Click benchmark shape differs from the frozen 12/11 protocol")
    if any(item.repository_revision != click[0].exact_commit_sha for item in scenarios):
        raise Phase5RunError("Click query/gold revisions do not match the repository revision")
    dataset_identity = {
        "manifest_sha256": file_hash(manifest_path),
        "repositories_sha256": file_hash(repositories_path),
        "scenarios_sha256": file_hash(scenarios_path),
        "dataset_version": manifest.get("dataset_version"),
        "selection": {"repo_id": "click", "scenario_ids": [item.scenario_id for item in scenarios]},
    }
    query_identity = [
        {
            "scenario_id": item.scenario_id,
            "question": item.question,
            "category": item.category,
            "unanswerable": item.unanswerable,
        }
        for item in scenarios
    ]
    gold_identity = [
        {
            "scenario_id": item.scenario_id,
            "repository_revision": item.repository_revision,
            "expected_target_type": item.expected_target_type,
            "expected_files": list(item.expected_files),
            "expected_symbols": list(item.expected_symbols),
            "expected_source_spans": [span.model_dump() for span in item.expected_source_spans],
            "expected_content_hashes": list(item.expected_content_hashes),
            "expected_relation_edges": [edge.model_dump() for edge in item.expected_relation_edges],
            "unanswerable": item.unanswerable,
        }
        for item in scenarios
    ]
    return ClickBenchmarkSnapshot(
        dataset_directory=directory,
        repository=click[0],
        scenarios=scenarios,
        dataset_hash=canonical_hash(dataset_identity),
        query_hash=canonical_hash(query_identity),
        gold_hash=canonical_hash(gold_identity),
        matcher_hash=canonical_hash(MATCHER_DEFINITION),
    )


def relation_graph_identity(database_path: Path, project_id: str) -> dict[str, Any]:
    path = Path(database_path).resolve()
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        run = connection.execute(
            """
            SELECT project_id, repository_revision, status, parsed_files, failed_files,
                   unsupported_files, node_count, edge_count, warnings_json
            FROM relation_index_runs WHERE project_id = ?
            """,
            (project_id,),
        ).fetchone()
        if run is None:
            return {
                "status": "missing",
                "repository_revision": None,
                "node_count": 0,
                "edge_count": 0,
                "graph_hash": canonical_hash({"project_id": project_id, "status": "missing"}),
            }
        nodes = [
            dict(item)
            for item in connection.execute(
                "SELECT * FROM relation_nodes WHERE project_id = ? ORDER BY node_id",
                (project_id,),
            )
        ]
        edges = [
            dict(item)
            for item in connection.execute(
                "SELECT * FROM code_relations WHERE project_id = ? ORDER BY edge_id",
                (project_id,),
            )
        ]
        run_value = dict(run)
        return {
            "status": str(run_value["status"]),
            "repository_revision": str(run_value["repository_revision"]),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "graph_hash": canonical_hash({"run": run_value, "nodes": nodes, "edges": edges}),
        }
    finally:
        connection.close()


class Phase5Harness:
    def __init__(
        self,
        *,
        database: Database,
        embedding_service: EmbeddingService | CountingEmbeddingService,
        project_id: str,
        scenarios: Iterable[Scenario],
        formal: bool,
    ) -> None:
        self.database = database
        self.embedding_service = (
            embedding_service
            if isinstance(embedding_service, CountingEmbeddingService)
            else CountingEmbeddingService(embedding_service)
        )
        self.project_id = project_id
        self.scenarios = tuple(sorted(scenarios, key=lambda item: item.scenario_id))
        self.formal = formal
        self.bundle = database.get_bundle(project_id)
        if self.bundle is None:
            raise Phase5RunError("corpus project is absent from the evaluation database")
        project = self.bundle.get("project") or {}
        revision = str(project.get("repository_revision", ""))
        chunk_revisions = {
            str(item.get("repository_revision", ""))
            for item in self.bundle.get("code_chunks", [])
        }
        if not revision or chunk_revisions != {revision}:
            raise Phase5RunError("corpus must bind exactly one repository revision")
        if any(item.repository_revision != revision for item in self.scenarios):
            raise Phase5RunError("benchmark and corpus repository revisions differ")
        self.repository_revision = revision
        graph = relation_graph_identity(database.path, project_id)
        self.relation_graph = graph
        if formal and (
            graph["status"] != "complete"
            or graph["repository_revision"] != revision
        ):
            raise Phase5RunError("formal run requires a complete relation graph for the corpus revision")
        if formal:
            identity = self.embedding_service.ensure_effective_embedding_identity(
                local_files_only=True
            )
            backend = getattr(self.embedding_service.service, "_backend", None)
            try:
                ensure_formal_embedding_identity(identity, backend=backend)
            except ValueError as exc:
                raise Phase5RunError(str(exc)) from exc

    def run_matrix(
        self,
        *,
        path_order: list[str],
        scenario_order: list[str] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        path_by_id = {item.path_id: item for item in FROZEN_PATHS}
        if len(path_order) != len(set(path_order)) or any(item not in path_by_id for item in path_order):
            raise Phase5RunError("path order contains duplicates or unknown paths")
        scenario_by_id = {item.scenario_id: item for item in self.scenarios}
        order = scenario_order or sorted(scenario_by_id)
        if len(order) != len(set(order)) or set(order) != set(scenario_by_id):
            raise Phase5RunError("scenario order must cover the same frozen query set exactly once")
        output: dict[str, list[dict[str, Any]]] = {}
        for path_id in path_order:
            output[path_id] = [
                self.run_query(path_by_id[path_id], scenario_by_id[scenario_id])
                for scenario_id in order
            ]
        counts = {path_id: len(records) for path_id, records in output.items()}
        if len(set(counts.values())) > 1:
            raise Phase5RunError("retrieval paths evaluated different query counts")
        return output

    def run_query(self, path: FrozenPath, scenario: Scenario) -> dict[str, Any]:
        if scenario.unanswerable:
            record = evaluate_query(scenario, [], top_k=FORMAL_TOP_K)
            return {
                **record,
                "path_id": path.path_id,
                "path_label": path.label,
                "request_parameters": path.request_parameters,
                "latency_ms": 0.0,
                "warnings": [],
                "retrieval_audit": {},
                "query_encode_count": 0,
                "citation_validation": {"valid": 0, "invalid": 0, "warnings": []},
                "relation_validation": {"valid": 0, "invalid": 0, "warnings": []},
            }
        request_id = f"phase5:{path.path_id}:{scenario.scenario_id}"
        context = build_tool_context(
            request_id=request_id,
            bundle=self.bundle,
            database=self.database,
            embedding_service=self.embedding_service,
            evidence_store=EvidenceStore(),
            limits=AgentLimits(max_search_results=20),
            cancellation=CancellationToken(),
            deadline_monotonic=time.monotonic() + 60.0,
            retrieval_version=path.retrieval_version,
            hierarchy_mode=path.hierarchy_mode,
            relation_mode=path.relation_mode,
        )
        spec = build_m2_tool_registry(context.limits).get("search_code")
        before_queries = self.embedding_service.query_encode_calls
        started = time.perf_counter()
        try:
            payload, warnings, truncated = spec.handler(
                context,
                SearchCodeInput(query=scenario.question, top_k=FORMAL_TOP_K),
            )
        except Exception as exc:
            raise Phase5RunError(
                f"{path.path_id}/{scenario.scenario_id} retrieval failed with {type(exc).__name__}"
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000
        query_encode_count = self.embedding_service.query_encode_calls - before_queries
        if truncated:
            warnings = [*warnings, "search handler reported truncated output"]
        evidence = context.evidence_store.all(request_id)
        valid_evidence, citation_warnings = CitationValidator(self.database).validate_all(evidence)
        invalid_evidence = [item for item in evidence if item.validation_status != "valid"]
        chains = context.chain_store.all(request_id)
        evidence_by_chunk_id = {item.code_chunk_id: item.evidence_id for item in valid_evidence}
        valid_chains, relation_warnings = RelationValidator(self.database).validate_chains(
            owner_id=request_id,
            project_id=self.project_id,
            repository_revision=self.repository_revision,
            chains=chains,
            valid_evidence_ids={item.evidence_id for item in valid_evidence},
            evidence_by_chunk_id=evidence_by_chunk_id,
        )
        invalid_chain_count = len(chains) - len(valid_chains)
        audit = payload.get("retrieval_audit", {}) if isinstance(payload, dict) else {}
        relation_audit = audit.get("relation", {}) if isinstance(audit, dict) else {}
        candidates = [
            self._candidate_record(
                item,
                audit=audit,
                valid_chain_ids={chain.chain_id for chain in valid_chains},
                chains=chains,
            )
            for item in evidence
        ]
        self._validate_cell(
            path,
            payload=payload,
            audit=audit,
            relation_audit=relation_audit,
            warnings=[*warnings, *citation_warnings, *relation_warnings],
            query_encode_count=query_encode_count,
            invalid_evidence_count=len(invalid_evidence),
            invalid_chain_count=invalid_chain_count,
        )
        evaluated = evaluate_query(scenario, candidates, top_k=FORMAL_TOP_K)
        return {
            **evaluated,
            "path_id": path.path_id,
            "path_label": path.label,
            "request_parameters": path.request_parameters,
            "latency_ms": latency_ms,
            "warnings": list(dict.fromkeys([*warnings, *citation_warnings, *relation_warnings])),
            "retrieval_mode": payload.get("retrieval_mode"),
            "retrieval_audit": audit,
            "query_encode_count": query_encode_count,
            "citation_validation": {
                "valid": len(valid_evidence),
                "invalid": len(invalid_evidence),
                "warnings": citation_warnings,
            },
            "relation_validation": {
                "valid": len(valid_chains),
                "invalid": invalid_chain_count,
                "warnings": relation_warnings,
                "chains": [chain.public_summary() for chain in chains],
            },
        }

    def _validate_cell(
        self,
        path: FrozenPath,
        *,
        payload: dict[str, Any],
        audit: dict[str, Any],
        relation_audit: dict[str, Any],
        warnings: list[str],
        query_encode_count: int,
        invalid_evidence_count: int,
        invalid_chain_count: int,
    ) -> None:
        if not self.formal:
            return
        if query_encode_count != 1:
            raise Phase5RunError("formal answerable query must encode exactly once")
        warning_text = " ".join(warnings).casefold()
        forbidden = ("dense retrieval unavailable", "semantic retrieval unavailable", "embeddings are disabled")
        if any(value in warning_text for value in forbidden):
            raise Phase5RunError("formal retrieval silently fell back from the real dense provider")
        if payload.get("retrieval_mode") != "hybrid":
            raise Phase5RunError("formal v1/v2 comparison requires active lexical+dense retrieval")
        if invalid_evidence_count or invalid_chain_count:
            raise Phase5RunError("formal final Evidence or relation provenance failed validation")
        if path.retrieval_version == "v2":
            if audit.get("retrieval_version") != "v2":
                raise Phase5RunError("v2 retrieval audit is missing or mismatched")
            dense = ((audit.get("sources") or {}).get("dense") or {})
            if dense.get("status") != "ok":
                raise Phase5RunError("formal v2 dense source was not available")
        if path.hierarchy_mode == "off" and "hierarchy" in audit:
            raise Phase5RunError("hierarchy-off path executed hierarchy normalization")
        if path.hierarchy_mode == "normalize_v1" and "hierarchy" not in audit:
            raise Phase5RunError("hierarchy-on path did not execute normalization")
        if path.relation_mode == "off" and "relation" in audit:
            raise Phase5RunError("relation-off path executed relation expansion")
        if path.relation_mode == "expand_v1":
            if not relation_audit or relation_audit.get("controlled_unavailable"):
                raise Phase5RunError("relation-on path did not execute the frozen expansion")
            if relation_audit.get("unexpected_error"):
                raise Phase5RunError("relation expansion failed safely but formal evaluation cannot continue")

    @staticmethod
    def _candidate_record(
        evidence: Evidence,
        *,
        audit: dict[str, Any],
        valid_chain_ids: set[str],
        chains: list[EvidenceChain],
    ) -> dict[str, Any]:
        value = evidence.to_dict()
        relation = audit.get("relation", {}) if isinstance(audit, dict) else {}
        selected_paths = [
            item
            for item in relation.get("selected_relation_paths", [])
            if isinstance(item, dict)
            and str(item.get("target_chunk_identity", "")) == evidence.chunk_identity
        ]
        support_paths = [
            item
            for item in (relation.get("existing_candidate_support", {}) or {}).get(
                evidence.chunk_identity, []
            )
            if isinstance(item, dict)
        ]
        related_chains = [
            chain
            for chain in chains
            if evidence.evidence_id in chain.seed_evidence_ids
            or evidence.evidence_id in chain.supporting_evidence_ids
        ]
        sources = list(evidence.retrieval_sources)
        candidate_origin = (
            "relation"
            if sources == ["relation"]
            else "hierarchy"
            if sources == ["hierarchy"]
            else "direct"
        )
        if support_paths and candidate_origin != "relation":
            sources = [*sources, "relation_support"]
        return {
            **value,
            "retrieval_sources": sources,
            "candidate_origin": candidate_origin,
            "direct_provenance": {
                "retrieval_sources": list(evidence.retrieval_sources),
                "lexical_rank": evidence.lexical_rank,
                "semantic_rank": evidence.semantic_rank,
                "fusion_rank": evidence.fusion_rank,
                "fusion_score": evidence.fusion_score,
            },
            "hierarchy_provenance": (
                {"normalization": audit.get("hierarchy"), "selected_identity": evidence.chunk_identity}
                if candidate_origin == "hierarchy"
                else None
            ),
            "relation_provenance": [*selected_paths, *support_paths],
            "seed_identity": next(
                (item.get("seed_chunk_identity") for item in selected_paths), None
            ),
            "edge_id": next((item.get("edge_id") for item in selected_paths), None),
            "relation_type": next((item.get("relation_type") for item in selected_paths), None),
            "direction": next((item.get("direction") for item in selected_paths), None),
            "relation_priority": next((item.get("path_priority") for item in selected_paths), None),
            "hierarchy_priority": None,
            "citation_validation": evidence.validation_status,
            "relation_validation": (
                "valid"
                if related_chains and all(chain.chain_id in valid_chain_ids for chain in related_chains)
                else "not_applicable"
                if not related_chains
                else "invalid"
            ),
        }


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase5RunError(f"unable to read benchmark file: {path.name}") from exc
