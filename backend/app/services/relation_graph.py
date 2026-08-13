from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import time
from typing import Any, Callable

from app.database import Database
from app.services.relation_analysis import RELATION_TYPES


RELATION_API_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RelationTraversalLimits:
    max_depth: int = 2
    per_node_limit: int = 20
    max_nodes: int = 64
    max_edges: int = 128
    max_paths: int = 24
    max_output_bytes: int = 65_536


@dataclass(frozen=True)
class RelationPath:
    node_ids: list[str]
    edge_ids: list[str]
    resolution_status: str

    @property
    def depth(self) -> int:
        return len(self.edge_ids)


@dataclass(frozen=True)
class RelationTraversalResult:
    seed_node_ids: list[str]
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    paths: list[RelationPath]
    unresolved_count: int
    ambiguous_count: int
    external_count: int
    truncated: bool
    warnings: list[str]
    duration_ms: int


@dataclass
class EvidenceChain:
    chain_id: str
    owner_id: str
    project_id: str
    repository_revision: str
    seed_evidence_ids: list[str]
    supporting_evidence_ids: list[str]
    ordered_node_ids: list[str]
    ordered_edge_ids: list[str]
    relation_types: list[str]
    resolution_status: str
    truncated: bool
    warnings: list[str] = field(default_factory=list)
    contract_version: str = "m3_agent_relation_v1@1"
    ordered_directions: list[str] = field(default_factory=list)

    def public_summary(self) -> dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "relation_types": list(self.relation_types),
            "path_length": len(self.ordered_edge_ids),
            "seed_evidence_ids": list(self.seed_evidence_ids),
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "resolution_status": self.resolution_status,
            "truncated": self.truncated,
        }


@dataclass
class EvidenceChainStore:
    _items: dict[str, EvidenceChain] = field(default_factory=dict)

    def add(
        self,
        *,
        owner_id: str,
        project_id: str,
        repository_revision: str,
        seed_evidence_ids: list[str],
        supporting_evidence_ids: list[str],
        path: RelationPath,
        edges_by_id: dict[str, dict[str, Any]],
        truncated: bool,
        warnings: list[str],
        contract_version: str = "m3_agent_relation_v1@1",
        ordered_directions: list[str] | None = None,
    ) -> EvidenceChain:
        chain_id = _chain_id(
            owner_id=owner_id,
            project_id=project_id,
            repository_revision=repository_revision,
            seed_evidence_ids=seed_evidence_ids,
            supporting_evidence_ids=supporting_evidence_ids,
            node_ids=path.node_ids,
            edge_ids=path.edge_ids,
        )
        relation_types = sorted(
            {
                str(edges_by_id[edge_id]["relation_type"])
                for edge_id in path.edge_ids
                if edge_id in edges_by_id
            }
        )
        directions = list(ordered_directions or [])
        if not directions:
            for index, edge_id in enumerate(path.edge_ids):
                if index >= len(path.node_ids):
                    break
                current = path.node_ids[index]
                next_node = (
                    path.node_ids[index + 1]
                    if index + 1 < len(path.node_ids)
                    else None
                )
                edge = edges_by_id.get(edge_id)
                if edge is None or next_node is None:
                    directions.append("unknown")
                elif (
                    str(edge.get("source_node_id", "")) == current
                    and str(edge.get("target_node_id", "")) == next_node
                ):
                    directions.append("outgoing")
                elif (
                    str(edge.get("target_node_id", "")) == current
                    and str(edge.get("source_node_id", "")) == next_node
                ):
                    directions.append("incoming")
                else:
                    directions.append("unknown")
        chain = EvidenceChain(
            chain_id=chain_id,
            owner_id=owner_id,
            project_id=project_id,
            repository_revision=repository_revision,
            seed_evidence_ids=sorted(set(seed_evidence_ids)),
            supporting_evidence_ids=sorted(set(supporting_evidence_ids)),
            ordered_node_ids=list(path.node_ids),
            ordered_edge_ids=list(path.edge_ids),
            relation_types=relation_types,
            resolution_status=path.resolution_status,
            truncated=truncated,
            warnings=list(dict.fromkeys(warnings)),
            contract_version=contract_version,
            ordered_directions=directions,
        )
        self._items[chain_id] = chain
        return chain

    def all(self, owner_id: str) -> list[EvidenceChain]:
        return sorted(
            (
                chain
                for chain in self._items.values()
                if chain.owner_id == owner_id
            ),
            key=lambda item: item.chain_id,
        )

    def supporting_ids(self, owner_id: str) -> set[str]:
        return {
            evidence_id
            for chain in self.all(owner_id)
            for evidence_id in chain.supporting_evidence_ids
        }


class RelationGraphService:
    def __init__(self, database: Database) -> None:
        self.database = database

    def expand(
        self,
        *,
        project_id: str,
        repository_revision: str,
        seed_node_ids: list[str],
        relation_types: list[str],
        direction: str,
        max_depth: int,
        limits: RelationTraversalLimits,
        check_active: Callable[[], None] | None = None,
    ) -> RelationTraversalResult:
        started = time.monotonic()
        if direction not in {"outbound", "inbound", "both"}:
            raise ValueError("invalid relation direction")
        normalized_types = sorted(set(relation_types))
        if not normalized_types or not set(normalized_types).issubset(RELATION_TYPES):
            raise ValueError("unknown relation type")
        effective_depth = min(max(1, int(max_depth)), limits.max_depth)
        seeds = self.database.get_relation_nodes(
            project_id,
            repository_revision,
            node_ids=sorted(set(seed_node_ids)),
        )
        seed_ids = [str(item["node_id"]) for item in seeds]
        if len(seed_ids) != len(set(seed_node_ids)):
            raise ValueError("one or more relation seeds are outside the bound snapshot")

        nodes_by_id = {str(item["node_id"]): item for item in seeds}
        edges_by_id: dict[str, dict[str, Any]] = {}
        paths_by_node: dict[str, RelationPath] = {
            node_id: RelationPath([node_id], [], "resolved") for node_id in seed_ids
        }
        paths: list[RelationPath] = []
        frontier = list(seed_ids)
        truncated = False
        warnings: list[str] = []
        unresolved_count = 0
        ambiguous_count = 0
        external_count = 0

        for _depth in range(1, effective_depth + 1):
            if check_active:
                check_active()
            if not frontier:
                break
            candidates = self._neighbor_edges(
                project_id,
                repository_revision,
                frontier,
                normalized_types,
                direction,
            )
            grouped: dict[str, list[dict[str, Any]]] = {node_id: [] for node_id in frontier}
            for edge in candidates:
                for current in frontier:
                    if _edge_touches(edge, current, direction):
                        grouped[current].append(edge)
            next_frontier: list[str] = []
            for current in sorted(frontier):
                ordered = sorted(grouped[current], key=_edge_sort_key)
                if len(ordered) > limits.per_node_limit:
                    ordered = ordered[: limits.per_node_limit]
                    truncated = True
                for edge in ordered:
                    edge_id = str(edge["edge_id"])
                    if edge_id in edges_by_id:
                        continue
                    if len(edges_by_id) >= limits.max_edges:
                        truncated = True
                        break
                    status = str(edge["resolution_status"])
                    unresolved_count += int(status == "unresolved")
                    ambiguous_count += int(status == "ambiguous")
                    external_count += int(status == "external")
                    edges_by_id[edge_id] = edge
                    neighbor = _neighbor_id(edge, current, direction)
                    base_path = paths_by_node[current]
                    if neighbor is None:
                        if len(paths) < limits.max_paths:
                            paths.append(
                                RelationPath(
                                    list(base_path.node_ids),
                                    [*base_path.edge_ids, edge_id],
                                    status,
                                )
                            )
                        else:
                            truncated = True
                        continue
                    if neighbor in base_path.node_ids:
                        # The edge remains auditable, but a cyclic path is not expanded.
                        continue
                    node_rows = self.database.get_relation_nodes(
                        project_id,
                        repository_revision,
                        node_ids=[neighbor],
                    )
                    if len(node_rows) != 1:
                        continue
                    if neighbor not in nodes_by_id and len(nodes_by_id) >= limits.max_nodes:
                        truncated = True
                        continue
                    nodes_by_id[neighbor] = node_rows[0]
                    path_status = (
                        "ambiguous"
                        if "ambiguous"
                        in {base_path.resolution_status, status}
                        else status
                    )
                    candidate_path = RelationPath(
                        [*base_path.node_ids, neighbor],
                        [*base_path.edge_ids, edge_id],
                        path_status,
                    )
                    if neighbor not in paths_by_node:
                        paths_by_node[neighbor] = candidate_path
                        next_frontier.append(neighbor)
                    if len(paths) < limits.max_paths:
                        paths.append(candidate_path)
                    else:
                        truncated = True
                if len(edges_by_id) >= limits.max_edges:
                    break
            frontier = sorted(set(next_frontier))

        selected_paths = sorted(
            {tuple(path.edge_ids): path for path in paths}.values(),
            key=lambda item: (item.depth, item.edge_ids, item.node_ids),
        )[: limits.max_paths]
        if len(paths) > len(selected_paths):
            truncated = True
        if truncated:
            warnings.append("Relation traversal was truncated by server limits.")
        return RelationTraversalResult(
            seed_node_ids=seed_ids,
            nodes=sorted(nodes_by_id.values(), key=_node_sort_key),
            edges=sorted(edges_by_id.values(), key=_edge_sort_key),
            paths=selected_paths,
            unresolved_count=unresolved_count,
            ambiguous_count=ambiguous_count,
            external_count=external_count,
            truncated=truncated,
            warnings=warnings,
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    def _neighbor_edges(
        self,
        project_id: str,
        repository_revision: str,
        frontier: list[str],
        relation_types: list[str],
        direction: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if direction in {"outbound", "both"}:
            rows.extend(
                self.database.get_relations(
                    project_id,
                    repository_revision,
                    relation_types=relation_types,
                    source_node_ids=frontier,
                )
            )
        if direction in {"inbound", "both"}:
            rows.extend(
                self.database.get_relations(
                    project_id,
                    repository_revision,
                    relation_types=relation_types,
                    target_node_ids=frontier,
                )
            )
        return sorted(
            {str(item["edge_id"]): item for item in rows}.values(),
            key=_edge_sort_key,
        )


class RelationValidator:
    def __init__(self, database: Database) -> None:
        self.database = database

    def validate_chains(
        self,
        *,
        owner_id: str,
        project_id: str,
        repository_revision: str,
        chains: list[EvidenceChain],
        valid_evidence_ids: set[str],
        evidence_by_chunk_id: dict[int, str] | None = None,
    ) -> tuple[list[EvidenceChain], list[str]]:
        valid: list[EvidenceChain] = []
        warnings: list[str] = []
        for chain in chains:
            reason = self._invalid_reason(
                chain,
                owner_id,
                project_id,
                repository_revision,
                valid_evidence_ids,
                evidence_by_chunk_id or {},
            )
            if reason is None:
                valid.append(chain)
            else:
                warnings.append(
                    f"Evidence chain {chain.chain_id} was rejected: {reason}."
                )
        return valid, warnings

    def _invalid_reason(
        self,
        chain: EvidenceChain,
        owner_id: str,
        project_id: str,
        revision: str,
        valid_evidence_ids: set[str],
        evidence_by_chunk_id: dict[int, str],
    ) -> str | None:
        if chain.owner_id != owner_id:
            return "chain is not owned by this request"
        if chain.project_id != project_id or chain.repository_revision != revision:
            return "chain identity mismatch"
        expected_chain_id = _chain_id(
            owner_id=chain.owner_id,
            project_id=chain.project_id,
            repository_revision=chain.repository_revision,
            seed_evidence_ids=chain.seed_evidence_ids,
            supporting_evidence_ids=chain.supporting_evidence_ids,
            node_ids=chain.ordered_node_ids,
            edge_ids=chain.ordered_edge_ids,
        )
        if chain.chain_id != expected_chain_id:
            return "chain identity is invalid"
        if not set(chain.seed_evidence_ids).issubset(valid_evidence_ids):
            return "seed Evidence is invalid"
        if not set(chain.supporting_evidence_ids).issubset(valid_evidence_ids):
            return "supporting Evidence is invalid"
        if len(chain.ordered_node_ids) > 64 or len(chain.ordered_edge_ids) > 128:
            return "chain exceeds validator limits"
        nodes = self.database.get_relation_nodes_bounded(
            project_id,
            revision,
            node_ids=chain.ordered_node_ids,
            limit=max(1, len(chain.ordered_node_ids) + 1),
        )
        if {str(item["node_id"]) for item in nodes} != set(chain.ordered_node_ids):
            return "relation node no longer exists"
        if any(str(item["node_id"]) != _expected_node_id(item) for item in nodes):
            return "relation node identity is invalid"
        edges = self.database.get_relations(
            project_id,
            revision,
            edge_ids=chain.ordered_edge_ids,
            limit=max(1, len(chain.ordered_edge_ids) + 1),
        )
        edges_by_id = {str(item["edge_id"]): item for item in edges}
        if not set(chain.ordered_edge_ids).issubset(edges_by_id):
            return "relation edge no longer exists"
        if any(
            edge_id != _expected_edge_id(edges_by_id[edge_id])
            for edge_id in chain.ordered_edge_ids
        ):
            return "relation edge identity is invalid"
        expected_types = sorted(
            {str(edges_by_id[edge_id]["relation_type"]) for edge_id in chain.ordered_edge_ids}
        )
        if chain.relation_types != expected_types:
            return "relation types do not match persisted edges"
        if len(chain.ordered_node_ids) not in {
            len(chain.ordered_edge_ids),
            len(chain.ordered_edge_ids) + 1,
        }:
            return "invalid chain shape"
        for index, edge_id in enumerate(chain.ordered_edge_ids):
            edge = edges_by_id[edge_id]
            if edge["resolution_status"] not in {"resolved", "ambiguous"}:
                return "chain contains a non-traversable edge"
            if index >= len(chain.ordered_node_ids):
                return "invalid chain ordering"
            source = chain.ordered_node_ids[index]
            target = (
                chain.ordered_node_ids[index + 1]
                if index + 1 < len(chain.ordered_node_ids)
                else None
            )
            direction = (
                chain.ordered_directions[index]
                if index < len(chain.ordered_directions)
                else "unknown"
            )
            if target is None:
                return "invalid chain ordering"
            if direction == "outgoing":
                if not (
                    str(edge["source_node_id"]) == source
                    and str(edge["target_node_id"]) == target
                ):
                    return "outgoing relation direction is invalid"
            elif direction == "incoming":
                if not (
                    str(edge["target_node_id"]) == source
                    and str(edge["source_node_id"]) == target
                ):
                    return "incoming relation direction is invalid"
            else:
                endpoints = {str(edge["source_node_id"]), str(edge["target_node_id"])}
                if source not in endpoints or target not in endpoints:
                    return "edge ordering does not match nodes"
        if chain.contract_version == "relation_expansion_v1@1":
            if len(chain.ordered_edge_ids) != 1 or len(chain.ordered_node_ids) != 2:
                return "Phase 4 relation chain must contain exactly one hop"
            if chain.resolution_status != "resolved":
                return "Phase 4 relation chain must be resolved"
            nodes_by_id = {str(item["node_id"]): item for item in nodes}
            seed_node = nodes_by_id.get(chain.ordered_node_ids[0])
            target_node = nodes_by_id.get(chain.ordered_node_ids[1])
            if seed_node is None or target_node is None:
                return "Phase 4 relation endpoints are missing"
            if seed_node.get("code_chunk_id") is None or target_node.get("code_chunk_id") is None:
                return "Phase 4 relation endpoints must map to code chunks"
            seed_evidence = evidence_by_chunk_id.get(int(seed_node["code_chunk_id"]))
            target_evidence = evidence_by_chunk_id.get(int(target_node["code_chunk_id"]))
            if seed_evidence not in chain.seed_evidence_ids:
                return "Phase 4 seed Evidence does not match the relation node"
            if target_evidence not in chain.supporting_evidence_ids:
                return "Phase 4 target Evidence does not match the relation node"
            if not self._chunk_nodes_match_snapshot(project_id, revision, nodes):
                return "relation node content is stale"
        elif not self._nodes_match_snapshot(project_id, revision, nodes):
            return "relation node content is stale"
        return None

    def _chunk_nodes_match_snapshot(
        self,
        project_id: str,
        revision: str,
        nodes: list[dict[str, Any]],
    ) -> bool:
        chunk_ids = [int(item["code_chunk_id"]) for item in nodes]
        chunks = self.database.get_code_chunks_by_ids_bounded(
            project_id,
            revision,
            chunk_ids,
            limit=min(65, len(chunk_ids) + 1),
        )
        chunks_by_id = {int(item["id"]): item for item in chunks}
        if len(chunks_by_id) != len(set(chunk_ids)):
            return False
        return all(
            int(node["code_chunk_id"]) in chunks_by_id
            and str(node["path"])
            == str(chunks_by_id[int(node["code_chunk_id"])]["path"])
            and str(node["qualified_name"])
            == str(chunks_by_id[int(node["code_chunk_id"])]["qualified_name"])
            and int(node["start_line"])
            == int(chunks_by_id[int(node["code_chunk_id"])]["start_line"])
            and int(node["end_line"])
            == int(chunks_by_id[int(node["code_chunk_id"])]["end_line"])
            and str(node["content_hash"])
            == str(chunks_by_id[int(node["code_chunk_id"])]["content_hash"])
            for node in nodes
        )

    def _nodes_match_snapshot(
        self,
        project_id: str,
        revision: str,
        nodes: list[dict[str, Any]],
    ) -> bool:
        bundle = self.database.get_bundle(project_id)
        if bundle is None:
            return False
        files = {str(item["path"]): item for item in bundle.get("files", [])}
        chunks = {int(item["id"]): item for item in bundle.get("code_chunks", [])}
        if {
            str(item.get("repository_revision", ""))
            for item in chunks.values()
        } != {revision}:
            return False
        for node in nodes:
            chunk_id = node.get("code_chunk_id")
            if chunk_id is None:
                file = files.get(str(node["path"]))
                if file is None:
                    return False
                digest = hashlib.sha256(
                    str(file.get("content", "")).encode("utf-8")
                ).hexdigest()
            else:
                chunk = chunks.get(int(chunk_id))
                if chunk is None:
                    return False
                digest = str(chunk["content_hash"])
            if digest != str(node["content_hash"]):
                return False
        return True


def _edge_touches(edge: dict[str, Any], node_id: str, direction: str) -> bool:
    return (
        direction in {"outbound", "both"}
        and str(edge["source_node_id"]) == node_id
    ) or (
        direction in {"inbound", "both"}
        and str(edge.get("target_node_id")) == node_id
    )


def _neighbor_id(
    edge: dict[str, Any], current: str, direction: str
) -> str | None:
    source = str(edge["source_node_id"])
    target = edge.get("target_node_id")
    target = str(target) if target is not None else None
    if direction == "outbound":
        return target if source == current else None
    if direction == "inbound":
        return source if target == current else None
    if source == current:
        return target
    if target == current:
        return source
    return None


def _node_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(item.get("path", "")),
        int(item.get("start_line", 0)),
        str(item.get("qualified_name", "")),
        str(item.get("node_id", "")),
    )


def _edge_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(item.get("source_path", "")),
        int(item.get("source_start_line", 0)),
        str(item.get("relation_type", "")),
        str(item.get("raw_target_name", "")),
        str(item.get("target_path") or ""),
        int(item.get("target_start_line") or 0),
        str(item.get("edge_id", "")),
    )


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _chain_id(
    *,
    owner_id: str,
    project_id: str,
    repository_revision: str,
    seed_evidence_ids: list[str],
    supporting_evidence_ids: list[str],
    node_ids: list[str],
    edge_ids: list[str],
) -> str:
    identity = {
        "owner_id": owner_id,
        "project_id": project_id,
        "repository_revision": repository_revision,
        "seed_evidence_ids": sorted(set(seed_evidence_ids)),
        "supporting_evidence_ids": sorted(set(supporting_evidence_ids)),
        "node_ids": node_ids,
        "edge_ids": edge_ids,
    }
    return "C" + hashlib.sha256(_canonical(identity)).hexdigest()


def _expected_node_id(node: dict[str, Any]) -> str:
    identity = {
        "project_id": str(node["project_id"]),
        "revision": str(node["repository_revision"]),
        "language": str(node["language"]),
        "node_type": str(node["node_type"]),
        "path": str(node["path"]),
        "qualified_name": str(node["qualified_name"]),
        "start_line": int(node["start_line"]),
        "end_line": int(node["end_line"]),
        "content_hash": str(node["content_hash"]),
    }
    return "N" + hashlib.sha256(_canonical(identity)).hexdigest()


def _expected_edge_id(edge: dict[str, Any]) -> str:
    identity = {
        "project_id": str(edge["project_id"]),
        "revision": str(edge["repository_revision"]),
        "relation_type": str(edge["relation_type"]),
        "source_node_id": str(edge["source_node_id"]),
        "source_line": int(edge["source_start_line"]),
        "target_node_id": (
            str(edge["target_node_id"])
            if edge.get("target_node_id") is not None
            else None
        ),
        "raw_target_name": str(edge["raw_target_name"]),
        "resolution_status": str(edge["resolution_status"]),
        "resolution_rule": str(edge["resolution_rule"]),
    }
    return "R" + hashlib.sha256(_canonical(identity)).hexdigest()
