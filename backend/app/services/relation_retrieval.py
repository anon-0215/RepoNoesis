from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any, Literal

from app.database import Database
from app.services.hybrid_retriever import HybridSearchResult
from app.services.relation_analysis import RELATION_SCHEMA_VERSION, RELATION_TYPES


RelationMode = Literal["off", "expand_v1"]
RelationDirection = Literal["outgoing", "incoming"]

RELATION_MODE_OFF = "off"
RELATION_MODE_EXPAND_V1 = "expand_v1"
SUPPORTED_RELATION_MODES = frozenset({RELATION_MODE_OFF, RELATION_MODE_EXPAND_V1})
RELATION_EXPANSION_VERSION = "relation_expansion_v1@1"
RELATION_SELECTION_VERSION = "relation_selection_v1@1"
RELATION_WHITELIST_VERSION = "relation_whitelist_v1@1"
RELATION_PRIORITY_VERSION = "relation_priority_v1@1"
RELATION_GRAPH_VERSION = f"relation_schema_v{RELATION_SCHEMA_VERSION}@1"


class RelationRetrievalContractError(ValueError):
    """A persisted relation row violated the Phase 4 retrieval contract."""


@dataclass(frozen=True)
class RelationTypePolicy:
    database_type: str
    outgoing_view: str
    incoming_view: str
    weight: float

    def __post_init__(self) -> None:
        if self.database_type not in RELATION_TYPES:
            raise ValueError("relation policy must use a real M3 relation type")
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, (int, float))
            or not math.isfinite(float(self.weight))
            or not 0.0 < float(self.weight) <= 1.0
        ):
            raise ValueError("relation type weight must be finite and in (0, 1]")


RELATION_TYPE_POLICIES: dict[str, RelationTypePolicy] = {
    "calls": RelationTypePolicy("calls", "calls", "called_by", 1.0),
    "imports": RelationTypePolicy("imports", "imports", "imported_by", 0.9),
    "references": RelationTypePolicy(
        "references", "references", "referenced_by", 0.8
    ),
    "defines": RelationTypePolicy("defines", "defines", "defined_by", 0.7),
}


@dataclass(frozen=True)
class RelationExpansionLimits:
    max_relation_seeds: int = 12
    max_edges_per_seed: int = 8
    max_relation_rows_total: int = 96
    max_unique_relation_targets: int = 24
    max_relation_depth: int = 1
    max_relation_paths_per_target: int = 8
    max_relation_warnings: int = 16
    configured_max_relation_slots: int = 3
    relation_fraction_cap: float = 0.30
    max_relation_slots_per_seed: int = 1
    max_relation_slots_per_family: int = 2
    max_final_top_k: int = 8

    def __post_init__(self) -> None:
        _bounded_int("max_relation_seeds", self.max_relation_seeds, 1, 12)
        _bounded_int("max_edges_per_seed", self.max_edges_per_seed, 1, 8)
        _bounded_int(
            "max_relation_rows_total", self.max_relation_rows_total, 1, 96
        )
        _bounded_int(
            "max_unique_relation_targets",
            self.max_unique_relation_targets,
            1,
            24,
        )
        if self.max_relation_depth != 1:
            raise ValueError("Phase 4 relation depth must be exactly one")
        _bounded_int(
            "max_relation_paths_per_target",
            self.max_relation_paths_per_target,
            1,
            8,
        )
        _bounded_int("max_relation_warnings", self.max_relation_warnings, 1, 16)
        _bounded_int(
            "configured_max_relation_slots",
            self.configured_max_relation_slots,
            0,
            3,
        )
        if (
            isinstance(self.relation_fraction_cap, bool)
            or not isinstance(self.relation_fraction_cap, (int, float))
            or not math.isfinite(float(self.relation_fraction_cap))
            or not 0.0 < float(self.relation_fraction_cap) <= 0.5
        ):
            raise ValueError("relation_fraction_cap must be finite and in (0, 0.5]")
        _bounded_int(
            "max_relation_slots_per_seed", self.max_relation_slots_per_seed, 1, 3
        )
        _bounded_int(
            "max_relation_slots_per_family",
            self.max_relation_slots_per_family,
            1,
            3,
        )
        _bounded_int("max_final_top_k", self.max_final_top_k, 1, 8)


@dataclass(frozen=True)
class RelationNodeResolution:
    status: str
    chunk: dict[str, Any] | None
    candidate_identities: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class RelationSeed:
    candidate: HybridSearchResult
    chunk_identity: str
    selection_rank: int
    origin: str
    fused_score: float | None
    hierarchy_priority: float | None


@dataclass(frozen=True)
class RelationPathProvenance:
    path_identity: str
    seed_chunk_identity: str
    seed_node_id: str
    target_node_id: str
    target_chunk_identity: str
    edge_id: str
    relation_type: str
    relation_view: str
    direction: str
    project_id: str
    repository_revision: str
    depth: int
    seed_selection_rank: int
    seed_origin: str
    seed_fused_score: float | None
    seed_hierarchy_priority: float | None
    relation_type_weight: float
    depth_decay: float
    path_priority: float
    resolution_status: str


@dataclass
class RelationCandidate:
    candidate: HybridSearchResult
    paths: list[RelationPathProvenance]
    priority: float
    resolution_status: str = "unique"


@dataclass(frozen=True)
class RelationExpansionResult:
    candidates: list[RelationCandidate]
    supporting_paths: dict[str, list[RelationPathProvenance]]
    warnings: list[str]
    truncated: bool
    audit: dict[str, Any]


@dataclass(frozen=True)
class RelationSelectionResult:
    results: list[HybridSearchResult]
    selected_paths: list[RelationPathProvenance]
    warnings: list[str]
    audit: dict[str, Any]


def validate_relation_mode(
    value: Any,
    *,
    retrieval_version: str | None = None,
) -> RelationMode:
    if not isinstance(value, str) or value not in SUPPORTED_RELATION_MODES:
        raise ValueError("relation_mode must be exactly 'off' or 'expand_v1'")
    if value == RELATION_MODE_EXPAND_V1 and retrieval_version != "v2":
        raise ValueError("relation_mode='expand_v1' requires retrieval_version='v2'")
    return value  # type: ignore[return-value]


def canonical_relation_view(relation_type: str, direction: str) -> str:
    policy = RELATION_TYPE_POLICIES.get(relation_type)
    if policy is None:
        raise ValueError("unknown relation type")
    if direction == "outgoing":
        return policy.outgoing_view
    if direction == "incoming":
        return policy.incoming_view
    raise ValueError("relation direction must be outgoing or incoming")


def relation_path_priority(
    seed_selection_rank: Any,
    relation_type_weight: Any,
    *,
    depth: Any = 1,
) -> float:
    _bounded_int("seed_selection_rank", seed_selection_rank, 1, 1_000_000)
    _bounded_int("depth", depth, 1, 32)
    weight = _require_finite_weight(relation_type_weight)
    seed_rank_signal = 1.0 / (1.0 + seed_selection_rank)
    depth_decay = 1.0 / (2 ** (depth - 1))
    value = seed_rank_signal * weight * depth_decay
    if not math.isfinite(value):
        raise ValueError("relation path priority must be finite")
    return value


def resolve_relation_node_to_chunk(
    node: dict[str, Any],
    chunks: list[dict[str, Any]],
    *,
    project_id: str,
    repository_revision: str,
) -> RelationNodeResolution:
    if (
        str(node.get("project_id", "")) != project_id
        or str(node.get("repository_revision", "")) != repository_revision
    ):
        return RelationNodeResolution("scope_conflict", None, reason="node scope mismatch")
    chunk_id = node.get("code_chunk_id")
    if chunk_id is None:
        return RelationNodeResolution(
            "unsupported", None, reason="relation node is not backed by a code chunk"
        )
    try:
        authoritative_id = int(chunk_id)
    except (TypeError, ValueError):
        return RelationNodeResolution("invalid_metadata", None)
    matches = [
        row
        for row in chunks
        if str(row.get("project_id", "")) == project_id
        and str(row.get("repository_revision", "")) == repository_revision
        and int(row.get("id", -1)) == authoritative_id
    ]
    identities = tuple(sorted(_chunk_identity_from_row(row) for row in matches))
    if not matches:
        return RelationNodeResolution("not_found", None)
    if len(matches) != 1:
        return RelationNodeResolution("ambiguous", None, identities)
    chunk = matches[0]
    exact_fields = (
        ("path", str),
        ("qualified_name", str),
        ("start_line", int),
        ("end_line", int),
        ("content_hash", str),
    )
    for key, caster in exact_fields:
        try:
            if caster(node.get(key)) != caster(chunk.get(key)):
                return RelationNodeResolution("stale", None, identities)
        except (TypeError, ValueError):
            return RelationNodeResolution("invalid_metadata", None, identities)
    if str(node.get("path", "")) != _normalize_path(str(node.get("path", ""))):
        return RelationNodeResolution("invalid_metadata", None, identities)
    return RelationNodeResolution("unique", dict(chunk), identities)


class RelationRetrievalExpander:
    """Perform one request-local, bounded, one-hop expansion over persisted M3 edges."""

    def __init__(
        self,
        database: Database,
        *,
        limits: RelationExpansionLimits | None = None,
    ) -> None:
        self.database = database
        self.limits = limits or RelationExpansionLimits()

    def expand(
        self,
        *,
        project_id: str,
        repository_revision: str,
        base_candidates: list[HybridSearchResult],
        hierarchy_mode: str,
        hierarchy_audit: dict[str, Any] | None,
    ) -> RelationExpansionResult:
        limits = self.limits
        warnings: list[str] = []
        index_status = self.database.get_relation_index_status(
            project_id, repository_revision
        )
        audit = _empty_expansion_audit(limits, hierarchy_mode)
        if index_status is None:
            warning = (
                "No relation index exists for the bound revision; the frozen base "
                "retrieval candidates were preserved."
            )
            audit["controlled_unavailable"] = True
            audit["warnings"] = [warning]
            return RelationExpansionResult([], {}, [warning], False, audit)
        audit["index_status"] = str(index_status.get("status", "unknown"))
        if index_status.get("status") == "partial":
            _append_warning(
                warnings,
                "The bound relation index is partial; only persisted resolved edges were used.",
                limits,
            )

        seeds = _relation_seeds(base_candidates, hierarchy_audit, limits)
        audit["seeds"] = [_seed_audit(item) for item in seeds]
        if not seeds:
            audit["warnings"] = list(warnings)
            return RelationExpansionResult([], {}, warnings, False, audit)
        chunk_ids = [item.candidate.code_chunk_id for item in seeds]
        seed_nodes = self.database.get_relation_nodes_bounded(
            project_id,
            repository_revision,
            code_chunk_ids=chunk_ids,
            limit=min(65, len(chunk_ids) + 1),
        )
        nodes_by_chunk: dict[int, list[dict[str, Any]]] = {}
        for node in seed_nodes:
            if node.get("code_chunk_id") is not None:
                nodes_by_chunk.setdefault(int(node["code_chunk_id"]), []).append(node)
        valid_seeds: list[tuple[RelationSeed, dict[str, Any]]] = []
        for seed in seeds:
            rows = nodes_by_chunk.get(seed.candidate.code_chunk_id, [])
            exact = [row for row in rows if _node_matches_candidate(row, seed.candidate)]
            if len(exact) == 1 and _expected_node_id(exact[0]) == str(exact[0]["node_id"]):
                valid_seeds.append((seed, exact[0]))
            else:
                reason = "ambiguous" if len(exact) > 1 else "not_found_or_stale"
                audit["seed_rejections"].append(
                    {"chunk_identity": seed.chunk_identity, "reason": reason}
                )
                _append_warning(
                    warnings,
                    "A relation seed did not map uniquely to the bound relation graph; "
                    "its direct Evidence was preserved.",
                    limits,
                )
        if not valid_seeds:
            audit["warnings"] = list(warnings)
            return RelationExpansionResult([], {}, warnings, False, audit)

        seed_node_ids = [str(node["node_id"]) for _, node in valid_seeds]
        rows = self.database.get_relation_neighbors_bounded(
            project_id,
            repository_revision,
            seed_node_ids=seed_node_ids,
            relation_types=sorted(RELATION_TYPE_POLICIES),
            direction="both",
            limit=limits.max_relation_rows_total + 1,
        )
        truncated = len(rows) > limits.max_relation_rows_total
        rows = rows[: limits.max_relation_rows_total]
        audit["rows_inspected"] = len(rows)
        audit["query"] = {
            "project_id": project_id,
            "repository_revision": repository_revision,
            "seed_node_ids": sorted(seed_node_ids),
            "directions": ["incoming", "outgoing"],
            "relation_types": sorted(RELATION_TYPE_POLICIES),
            "row_limit": limits.max_relation_rows_total,
            "truncated": truncated,
        }
        if truncated:
            _append_warning(
                warnings, "Relation rows were truncated by the total request budget.", limits
            )

        node_by_id = {
            str(node["node_id"]): node for _, node in valid_seeds
        }
        neighbor_ids: set[str] = set()
        inspected: list[tuple[RelationSeed, dict[str, Any], dict[str, Any], str, str]] = []
        edges_per_seed: dict[str, int] = {}
        for seed, seed_node in sorted(
            valid_seeds, key=lambda item: (item[0].selection_rank, item[0].chunk_identity)
        ):
            seed_node_id = str(seed_node["node_id"])
            touching = [row for row in rows if _edge_touches(row, seed_node_id)]
            touching = sorted(touching, key=_edge_sort_key)
            if len(touching) > limits.max_edges_per_seed:
                truncated = True
                _append_warning(
                    warnings,
                    "A relation seed exceeded its per-seed edge budget.",
                    limits,
                )
            for edge in touching[: limits.max_edges_per_seed]:
                edges_per_seed[seed.chunk_identity] = (
                    edges_per_seed.get(seed.chunk_identity, 0) + 1
                )
                reason = _edge_rejection_reason(
                    edge, project_id, repository_revision, seed_node_id
                )
                if reason is not None:
                    audit["edge_rejections"].append(
                        {"edge_id": str(edge.get("edge_id", "")), "reason": reason}
                    )
                    continue
                direction = (
                    "outgoing"
                    if str(edge["source_node_id"]) == seed_node_id
                    else "incoming"
                )
                target_node_id = (
                    str(edge["target_node_id"])
                    if direction == "outgoing"
                    else str(edge["source_node_id"])
                )
                if target_node_id == seed_node_id:
                    audit["edge_rejections"].append(
                        {"edge_id": str(edge["edge_id"]), "reason": "self_loop"}
                    )
                    continue
                neighbor_ids.add(target_node_id)
                inspected.append((seed, seed_node, edge, direction, target_node_id))

        if neighbor_ids:
            neighbor_rows = self.database.get_relation_nodes_bounded(
                project_id,
                repository_revision,
                node_ids=sorted(neighbor_ids),
                limit=min(65, len(neighbor_ids) + 1),
            )
            node_by_id.update({str(row["node_id"]): row for row in neighbor_rows})
        chunk_ids = sorted(
            {
                int(node_by_id[node_id]["code_chunk_id"])
                for node_id in neighbor_ids
                if node_id in node_by_id
                and node_by_id[node_id].get("code_chunk_id") is not None
            }
        )
        chunk_rows = (
            self.database.get_code_chunks_by_ids_bounded(
                project_id,
                repository_revision,
                chunk_ids,
                limit=min(65, len(chunk_ids) + 1),
            )
            if chunk_ids
            else []
        )
        base_identities = {_chunk_identity(item) for item in base_candidates}
        candidates_by_identity: dict[str, RelationCandidate] = {}
        supporting_paths: dict[str, list[RelationPathProvenance]] = {}
        for seed, seed_node, edge, direction, target_node_id in inspected:
            target_node = node_by_id.get(target_node_id)
            if target_node is None:
                audit["node_resolutions"].append(
                    {"node_id": target_node_id, "status": "not_found"}
                )
                continue
            resolution = resolve_relation_node_to_chunk(
                target_node,
                chunk_rows,
                project_id=project_id,
                repository_revision=repository_revision,
            )
            audit["node_resolutions"].append(
                {
                    "node_id": target_node_id,
                    "status": resolution.status,
                    "candidate_identities": list(resolution.candidate_identities),
                }
            )
            if resolution.status != "unique" or resolution.chunk is None:
                continue
            target = _hybrid_from_relation_chunk(resolution.chunk)
            target_identity = _chunk_identity(target)
            relation_type = str(edge["relation_type"])
            policy = RELATION_TYPE_POLICIES[relation_type]
            priority = relation_path_priority(
                seed.selection_rank, policy.weight, depth=1
            )
            path = _relation_path(
                seed=seed,
                seed_node_id=str(seed_node["node_id"]),
                target_node_id=target_node_id,
                target_chunk_identity=target_identity,
                edge=edge,
                direction=direction,
                weight=policy.weight,
                priority=priority,
                project_id=project_id,
                repository_revision=repository_revision,
            )
            if target_identity in base_identities:
                _append_path(
                    supporting_paths,
                    target_identity,
                    path,
                    limits.max_relation_paths_per_target,
                )
                continue
            state = candidates_by_identity.get(target_identity)
            if state is None:
                if len(candidates_by_identity) >= limits.max_unique_relation_targets:
                    truncated = True
                    _append_warning(
                        warnings,
                        "Relation target candidates were truncated by the unique-target budget.",
                        limits,
                    )
                    continue
                state = RelationCandidate(target, [], priority)
                candidates_by_identity[target_identity] = state
            if len(state.paths) < limits.max_relation_paths_per_target:
                if path.path_identity not in {item.path_identity for item in state.paths}:
                    state.paths.append(path)
                    state.paths.sort(key=_path_sort_key)
            else:
                truncated = True
            state.priority = max(item.path_priority for item in state.paths)

        candidates = sorted(candidates_by_identity.values(), key=_relation_candidate_sort_key)
        for paths in supporting_paths.values():
            paths.sort(key=_path_sort_key)
        audit.update(
            {
                "edges_accepted": sum(len(item.paths) for item in candidates)
                + sum(len(value) for value in supporting_paths.values()),
                "edges_per_seed": dict(sorted(edges_per_seed.items())),
                "relation_paths": [
                    asdict(path)
                    for item in candidates
                    for path in sorted(item.paths, key=_path_sort_key)
                ],
                "existing_candidate_support": {
                    identity: [asdict(path) for path in paths]
                    for identity, paths in sorted(supporting_paths.items())
                },
                "candidate_priorities": {
                    _chunk_identity(item.candidate): item.priority for item in candidates
                },
                "truncated": truncated,
                "warnings": list(warnings),
            }
        )
        return RelationExpansionResult(
            candidates, supporting_paths, warnings, truncated, audit
        )


def select_relation_aware_candidates(
    base_candidates: list[HybridSearchResult],
    relation_candidates: list[RelationCandidate],
    *,
    top_k: int,
    limits: RelationExpansionLimits | None = None,
    base_order: dict[str, int] | None = None,
) -> RelationSelectionResult:
    limits = limits or RelationExpansionLimits()
    _bounded_int("top_k", top_k, 1, limits.max_final_top_k)
    if base_order is None:
        ordered_base = sorted(base_candidates, key=_base_candidate_sort_key)
    else:
        ordered_base = sorted(
            base_candidates,
            key=lambda item: (
                base_order.get(_chunk_identity(item), 10**9),
                *_base_candidate_sort_key(item),
            ),
        )
    ordered_base = ordered_base[:top_k]
    if top_k < 3:
        relation_slot_cap = 0
    else:
        relation_slot_cap = min(
            limits.configured_max_relation_slots,
            max(1, math.floor(top_k * limits.relation_fraction_cap)),
        )
    minimum_direct_slots = top_k - relation_slot_cap
    retained_base = ordered_base[:minimum_direct_slots]
    retained_identities = {_chunk_identity(item) for item in retained_base}
    all_base_identities = {_chunk_identity(item) for item in ordered_base}

    eligible: list[tuple[RelationCandidate, RelationPathProvenance]] = []
    suppressed: list[dict[str, Any]] = []
    for item in relation_candidates:
        identity = _chunk_identity(item.candidate)
        if item.resolution_status != "unique":
            suppressed.append({"chunk_identity": identity, "reason": item.resolution_status})
            continue
        if identity in all_base_identities:
            suppressed.append({"chunk_identity": identity, "reason": "already_base"})
            continue
        retained_paths = [
            path for path in item.paths if path.seed_chunk_identity in retained_identities
        ]
        if not retained_paths:
            suppressed.append(
                {"chunk_identity": identity, "reason": "seed_not_retained"}
            )
            continue
        anchor = sorted(retained_paths, key=_path_sort_key)[0]
        eligible.append((item, anchor))
    eligible.sort(key=_selection_sort_key)

    selected: list[tuple[RelationCandidate, RelationPathProvenance]] = []
    seed_occupancy: dict[str, int] = {}
    family_occupancy: dict[str, int] = {}
    for item, anchor in eligible:
        if len(selected) >= relation_slot_cap:
            suppressed.append(
                {"chunk_identity": _chunk_identity(item.candidate), "reason": "slot_cap"}
            )
            continue
        family = f"{anchor.relation_type}:{anchor.direction}"
        if seed_occupancy.get(anchor.seed_chunk_identity, 0) >= limits.max_relation_slots_per_seed:
            suppressed.append(
                {"chunk_identity": _chunk_identity(item.candidate), "reason": "seed_cap"}
            )
            continue
        if family_occupancy.get(family, 0) >= limits.max_relation_slots_per_family:
            suppressed.append(
                {"chunk_identity": _chunk_identity(item.candidate), "reason": "family_cap"}
            )
            continue
        selected.append((item, anchor))
        seed_occupancy[anchor.seed_chunk_identity] = (
            seed_occupancy.get(anchor.seed_chunk_identity, 0) + 1
        )
        family_occupancy[family] = family_occupancy.get(family, 0) + 1

    direct_needed = min(top_k - len(selected), len(ordered_base))
    selected_base = ordered_base[:direct_needed]
    selected_by_seed: dict[str, list[tuple[RelationCandidate, RelationPathProvenance]]] = {}
    for pair in selected:
        selected_by_seed.setdefault(pair[1].seed_chunk_identity, []).append(pair)
    for values in selected_by_seed.values():
        values.sort(key=_selection_sort_key)
    results: list[HybridSearchResult] = []
    selected_paths: list[RelationPathProvenance] = []
    for base in selected_base:
        results.append(base)
        for item, anchor in selected_by_seed.get(_chunk_identity(base), []):
            if len(results) >= top_k:
                break
            results.append(item.candidate)
            selected_paths.extend(
                sorted(
                    [
                        path
                        for path in item.paths
                        if path.seed_chunk_identity in {
                            _chunk_identity(value) for value in selected_base
                        }
                    ],
                    key=_path_sort_key,
                )
            )
    selected_relation_identities = {
        _chunk_identity(item.candidate) for item, _ in selected
    }
    audit = {
        "selection_version": RELATION_SELECTION_VERSION,
        "priority_version": RELATION_PRIORITY_VERSION,
        "final_top_k": top_k,
        "relation_slot_cap": relation_slot_cap,
        "minimum_direct_slots": minimum_direct_slots,
        "selected_direct_candidates": [
            _chunk_identity(item) for item in results if item.retrieval_sources != ["relation"]
        ],
        "selected_relation_candidates": [
            _chunk_identity(item) for item in results if item.retrieval_sources == ["relation"]
        ],
        "selected_relation_paths": [asdict(item) for item in selected_paths],
        "suppressed_relation_candidates": sorted(
            suppressed, key=lambda item: (item["reason"], item["chunk_identity"])
        ),
        "direct_backfill": max(0, direct_needed - minimum_direct_slots),
        "seed_occupancy": dict(sorted(seed_occupancy.items())),
        "family_occupancy": dict(sorted(family_occupancy.items())),
        "final_ordering": [_chunk_identity(item) for item in results[:top_k]],
        "selected_relation_identity_set": sorted(selected_relation_identities),
    }
    return RelationSelectionResult(results[:top_k], selected_paths, [], audit)


def _relation_seeds(
    base_candidates: list[HybridSearchResult],
    hierarchy_audit: dict[str, Any] | None,
    limits: RelationExpansionLimits,
) -> list[RelationSeed]:
    hierarchy_by_identity = {
        str(item.get("chunk_identity", "")): item
        for item in (hierarchy_audit or {}).get("candidates", [])
        if isinstance(item, dict)
    }
    hierarchy_order = {
        str(identity): rank
        for rank, identity in enumerate(
            (hierarchy_audit or {}).get("selection_order", []), start=1
        )
    }
    ordered = sorted(
        base_candidates,
        key=lambda item: (
            hierarchy_order.get(_chunk_identity(item), 10**9),
            *_base_candidate_sort_key(item),
        ),
    )
    seeds: list[RelationSeed] = []
    for ordinal, item in enumerate(ordered, start=1):
        identity = _chunk_identity(item)
        metadata = hierarchy_by_identity.get(identity, {})
        if metadata and metadata.get("decision") != "retained":
            continue
        origin = str(metadata.get("origin", "direct"))
        real_rank = hierarchy_order.get(identity)
        if real_rank is None:
            real_rank = int(item.fusion_rank) if int(item.fusion_rank) > 0 else ordinal
        fused_score = (
            float(metadata["original_fused_score"])
            if metadata.get("original_fused_score") is not None
            else (float(item.fusion_score) if int(item.fusion_rank) > 0 else None)
        )
        hierarchy_priority = (
            float(metadata["group_priority"])
            if metadata.get("group_priority") is not None
            else None
        )
        seeds.append(
            RelationSeed(
                item,
                identity,
                real_rank,
                origin,
                fused_score,
                hierarchy_priority,
            )
        )
    seeds.sort(key=lambda item: (item.selection_rank, item.chunk_identity))
    return seeds[: limits.max_relation_seeds]


def _hybrid_from_relation_chunk(chunk: dict[str, Any]) -> HybridSearchResult:
    return HybridSearchResult(
        project_id=str(chunk["project_id"]),
        repository_revision=str(chunk["repository_revision"]),
        code_chunk_id=int(chunk["id"]),
        language=str(chunk["language"]),
        path=str(chunk["path"]),
        chunk_type=str(chunk["chunk_type"]),
        symbol_name=str(chunk["symbol_name"]),
        qualified_name=str(chunk["qualified_name"]),
        start_line=int(chunk["start_line"]),
        end_line=int(chunk["end_line"]),
        content=str(chunk["content"]),
        content_hash=str(chunk["content_hash"]),
        retrieval_sources=["relation"],
        lexical_score=None,
        lexical_rank=None,
        semantic_score=None,
        semantic_rank=None,
        fusion_score=0.0,
        fusion_rank=0,
    )


def _relation_path(
    *,
    seed: RelationSeed,
    seed_node_id: str,
    target_node_id: str,
    target_chunk_identity: str,
    edge: dict[str, Any],
    direction: str,
    weight: float,
    priority: float,
    project_id: str,
    repository_revision: str,
) -> RelationPathProvenance:
    identity = {
        "version": RELATION_EXPANSION_VERSION,
        "seed_chunk_identity": seed.chunk_identity,
        "seed_node_id": seed_node_id,
        "target_node_id": target_node_id,
        "target_chunk_identity": target_chunk_identity,
        "edge_id": str(edge["edge_id"]),
        "direction": direction,
        "project_id": project_id,
        "repository_revision": repository_revision,
    }
    path_id = "P" + hashlib.sha256(_canonical(identity)).hexdigest()
    return RelationPathProvenance(
        path_identity=path_id,
        seed_chunk_identity=seed.chunk_identity,
        seed_node_id=seed_node_id,
        target_node_id=target_node_id,
        target_chunk_identity=target_chunk_identity,
        edge_id=str(edge["edge_id"]),
        relation_type=str(edge["relation_type"]),
        relation_view=canonical_relation_view(str(edge["relation_type"]), direction),
        direction=direction,
        project_id=project_id,
        repository_revision=repository_revision,
        depth=1,
        seed_selection_rank=seed.selection_rank,
        seed_origin=seed.origin,
        seed_fused_score=seed.fused_score,
        seed_hierarchy_priority=seed.hierarchy_priority,
        relation_type_weight=weight,
        depth_decay=1.0,
        path_priority=priority,
        resolution_status="resolved",
    )


def _edge_rejection_reason(
    edge: dict[str, Any],
    project_id: str,
    repository_revision: str,
    seed_node_id: str,
) -> str | None:
    if str(edge.get("project_id", "")) != project_id:
        return "project_scope_conflict"
    if str(edge.get("repository_revision", "")) != repository_revision:
        return "revision_scope_conflict"
    relation_type = str(edge.get("relation_type", ""))
    if relation_type not in RELATION_TYPE_POLICIES:
        return "unsupported_relation_type"
    if str(edge.get("resolution_status", "")) != "resolved":
        return str(edge.get("resolution_status", "invalid"))
    if edge.get("target_node_id") is None:
        return "missing_target"
    if not _edge_touches(edge, seed_node_id):
        return "edge_does_not_touch_seed"
    if str(edge.get("edge_id", "")) != _expected_edge_id(edge):
        return "invalid_edge_identity"
    return None


def _node_matches_candidate(
    node: dict[str, Any], candidate: HybridSearchResult
) -> bool:
    return (
        int(node.get("code_chunk_id") or -1) == candidate.code_chunk_id
        and str(node.get("project_id", "")) == candidate.project_id
        and str(node.get("repository_revision", ""))
        == candidate.repository_revision
        and str(node.get("path", "")) == candidate.path
        and str(node.get("qualified_name", "")) == candidate.qualified_name
        and int(node.get("start_line", 0)) == candidate.start_line
        and int(node.get("end_line", 0)) == candidate.end_line
        and str(node.get("content_hash", "")) == candidate.content_hash
    )


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


def _chunk_identity(item: HybridSearchResult) -> str:
    return "|".join(
        [
            item.project_id,
            item.repository_revision,
            item.path,
            str(item.start_line),
            str(item.end_line),
            item.content_hash,
            str(item.code_chunk_id),
        ]
    )


def _chunk_identity_from_row(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("project_id", "")),
            str(row.get("repository_revision", "")),
            str(row.get("path", "")),
            str(row.get("start_line", "")),
            str(row.get("end_line", "")),
            str(row.get("content_hash", "")),
            str(row.get("id", "")),
        ]
    )


def _base_candidate_sort_key(item: HybridSearchResult) -> tuple[Any, ...]:
    return (
        int(item.fusion_rank) if int(item.fusion_rank) > 0 else 10**9,
        -float(item.fusion_score),
        item.qualified_name.casefold(),
        _normalize_path(item.path),
        item.start_line,
        item.end_line,
        _chunk_identity(item),
    )


def _relation_candidate_sort_key(item: RelationCandidate) -> tuple[Any, ...]:
    best = sorted(item.paths, key=_path_sort_key)[0]
    return (
        -float(item.priority),
        -float(best.relation_type_weight),
        0 if best.direction == "outgoing" else 1,
        best.seed_selection_rank,
        item.candidate.qualified_name.casefold(),
        _normalize_path(item.candidate.path),
        item.candidate.start_line,
        item.candidate.end_line,
        _chunk_identity(item.candidate),
        best.path_identity,
    )


def _selection_sort_key(
    value: tuple[RelationCandidate, RelationPathProvenance]
) -> tuple[Any, ...]:
    item, anchor = value
    return (
        -float(anchor.path_priority),
        -float(anchor.relation_type_weight),
        0 if anchor.direction == "outgoing" else 1,
        anchor.seed_selection_rank,
        item.candidate.qualified_name.casefold(),
        _normalize_path(item.candidate.path),
        item.candidate.start_line,
        item.candidate.end_line,
        _chunk_identity(item.candidate),
        anchor.path_identity,
    )


def _path_sort_key(item: RelationPathProvenance) -> tuple[Any, ...]:
    return (
        -float(item.path_priority),
        item.seed_selection_rank,
        item.relation_type,
        0 if item.direction == "outgoing" else 1,
        item.seed_chunk_identity,
        item.target_chunk_identity,
        item.edge_id,
        item.path_identity,
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


def _edge_touches(edge: dict[str, Any], node_id: str) -> bool:
    return str(edge.get("source_node_id", "")) == node_id or str(
        edge.get("target_node_id", "")
    ) == node_id


def _append_path(
    values: dict[str, list[RelationPathProvenance]],
    identity: str,
    path: RelationPathProvenance,
    limit: int,
) -> None:
    paths = values.setdefault(identity, [])
    if path.path_identity not in {item.path_identity for item in paths} and len(paths) < limit:
        paths.append(path)


def _append_warning(
    warnings: list[str], value: str, limits: RelationExpansionLimits
) -> None:
    if value not in warnings and len(warnings) < limits.max_relation_warnings:
        warnings.append(value)


def _empty_expansion_audit(
    limits: RelationExpansionLimits, hierarchy_mode: str
) -> dict[str, Any]:
    return {
        "expansion_version": RELATION_EXPANSION_VERSION,
        "selection_version": RELATION_SELECTION_VERSION,
        "priority_version": RELATION_PRIORITY_VERSION,
        "whitelist_version": RELATION_WHITELIST_VERSION,
        "graph_version": RELATION_GRAPH_VERSION,
        "hierarchy_mode": hierarchy_mode,
        "relation_whitelist": sorted(RELATION_TYPE_POLICIES),
        "relation_type_weights": {
            key: value.weight for key, value in sorted(RELATION_TYPE_POLICIES.items())
        },
        "budgets": asdict(limits),
        "seeds": [],
        "seed_rejections": [],
        "edge_rejections": [],
        "node_resolutions": [],
        "rows_inspected": 0,
        "edges_accepted": 0,
        "relation_paths": [],
        "existing_candidate_support": {},
        "candidate_priorities": {},
        "truncated": False,
        "controlled_unavailable": False,
        "warnings": [],
    }


def _seed_audit(seed: RelationSeed) -> dict[str, Any]:
    return {
        "chunk_identity": seed.chunk_identity,
        "code_chunk_id": seed.candidate.code_chunk_id,
        "selection_rank": seed.selection_rank,
        "origin": seed.origin,
        "fused_score": seed.fused_score,
        "hierarchy_priority": seed.hierarchy_priority,
    }


def _normalize_path(value: str) -> str:
    return value.replace("\\", "/").strip("/")


def _bounded_int(name: str, value: Any, minimum: int, maximum: int) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
        or value > maximum
    ):
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")


def _require_finite_weight(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 < float(value) <= 1.0
    ):
        raise ValueError("relation type weight must be finite and in (0, 1]")
    return float(value)


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
