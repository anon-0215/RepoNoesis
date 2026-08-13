from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from typing import Any, Literal

from app.database import Database
from app.services.hybrid_retriever import HybridSearchResult


HierarchyMode = Literal["off", "normalize_v1"]
HierarchyAuthority = Literal[
    "explicit_structural_metadata",
    "span_inference",
    "validated_existing_index",
]

HIERARCHY_MODE_OFF = "off"
HIERARCHY_MODE_NORMALIZE_V1 = "normalize_v1"
SUPPORTED_HIERARCHY_MODES = frozenset(
    {HIERARCHY_MODE_OFF, HIERARCHY_MODE_NORMALIZE_V1}
)
HIERARCHY_RESOLVER_VERSION = "hierarchy_resolver_v1@1"
HIERARCHY_NORMALIZATION_VERSION = "hierarchy_normalization_v1@1"


class HierarchyContractError(ValueError):
    """Stored chunk hierarchy metadata violated the frozen chunk contract."""


@dataclass(frozen=True)
class CanonicalSpan:
    """The repository's canonical one-based, inclusive source-line span."""

    start_line: int
    end_line: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.start_line, int)
            or isinstance(self.start_line, bool)
            or not isinstance(self.end_line, int)
            or isinstance(self.end_line, bool)
            or self.start_line < 1
            or self.end_line < self.start_line
        ):
            raise HierarchyContractError(
                "hierarchy spans must be positive one-based inclusive integer ranges"
            )

    @property
    def size(self) -> int:
        return self.end_line - self.start_line + 1

    def strictly_contains(self, other: "CanonicalSpan") -> bool:
        return (
            self.start_line <= other.start_line
            and self.end_line >= other.end_line
            and self != other
        )

    def overlaps(self, other: "CanonicalSpan") -> bool:
        return max(self.start_line, other.start_line) <= min(
            self.end_line, other.end_line
        )

    def partial_overlap(self, other: "CanonicalSpan") -> bool:
        return (
            self.overlaps(other)
            and not self.strictly_contains(other)
            and not other.strictly_contains(self)
            and self != other
        )

    def disjoint(self, other: "CanonicalSpan") -> bool:
        return not self.overlaps(other)


@dataclass(frozen=True)
class HierarchyScope:
    project_id: str
    repository_revision: str
    normalized_path: str

    def __post_init__(self) -> None:
        normalized = _normalize_path(self.normalized_path)
        if not self.project_id or not self.repository_revision or not normalized:
            raise HierarchyContractError(
                "hierarchy scope requires project, revision, and normalized path"
            )
        object.__setattr__(self, "normalized_path", normalized)


@dataclass(frozen=True)
class HierarchyChunkMetadata:
    chunk_identity: str
    code_chunk_id: int
    scope: HierarchyScope
    language: str
    chunk_type: str
    symbol_name: str
    qualified_name: str
    parent_symbol: str
    span: CanonicalSpan
    content: str
    content_hash: str
    metadata_source: HierarchyAuthority = "validated_existing_index"

    def __post_init__(self) -> None:
        if not self.chunk_identity:
            raise HierarchyContractError("chunk identity must not be empty")
        if (
            not isinstance(self.code_chunk_id, int)
            or isinstance(self.code_chunk_id, bool)
            or self.code_chunk_id < 1
        ):
            raise HierarchyContractError(
                "hierarchy metadata requires an authoritative code chunk ID"
            )
        if not isinstance(self.content, str) or not self.content_hash:
            raise HierarchyContractError("chunk content and content hash are required")

    @property
    def span_size(self) -> int:
        return self.span.size


@dataclass(frozen=True)
class HierarchyRelation:
    relation_type: str
    direction: str
    depth: int
    direct: bool
    authority: str
    ambiguous: bool = False
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class HierarchyDerivedCandidate:
    candidate: HierarchyChunkMetadata
    derived_from_identity: str
    relation_type: str
    direction: str
    depth: int
    authority: str


@dataclass(frozen=True)
class HierarchyLimits:
    max_direct_candidates: int = 24
    max_paths: int = 8
    max_rows_per_path: int = 128
    max_total_rows: int = 512
    max_depth: int = 2
    max_neighbors_per_candidate: int = 8
    max_derived_candidates: int = 24
    max_normalization_groups: int = 72
    max_family_members: int = 2
    max_final_top_k: int = 8

    def __post_init__(self) -> None:
        _bounded_int("max_direct_candidates", self.max_direct_candidates, 1, 72)
        _bounded_int("max_paths", self.max_paths, 1, 32)
        _bounded_int("max_rows_per_path", self.max_rows_per_path, 1, 512)
        _bounded_int("max_total_rows", self.max_total_rows, 1, 2_048)
        _bounded_int("max_depth", self.max_depth, 1, 4)
        _bounded_int(
            "max_neighbors_per_candidate",
            self.max_neighbors_per_candidate,
            1,
            32,
        )
        _bounded_int(
            "max_derived_candidates", self.max_derived_candidates, 0, 72
        )
        _bounded_int(
            "max_normalization_groups", self.max_normalization_groups, 1, 256
        )
        _bounded_int("max_family_members", self.max_family_members, 1, 8)
        _bounded_int("max_final_top_k", self.max_final_top_k, 1, 8)


@dataclass(frozen=True)
class HierarchyResolution:
    metadata_by_identity: dict[str, HierarchyChunkMetadata]
    parent_by_child: dict[str, str]
    parent_authority: dict[str, str]
    ambiguous_identities: set[str]
    links: list[HierarchyDerivedCandidate]
    warnings: list[str]
    truncated: bool
    audit: dict[str, Any]


@dataclass(frozen=True)
class HierarchyNormalizationResult:
    results: list[HybridSearchResult]
    warnings: list[str]
    audit: dict[str, Any]


def validate_hierarchy_mode(
    value: str,
    *,
    retrieval_version: str | None = None,
) -> HierarchyMode:
    if not isinstance(value, str) or value not in SUPPORTED_HIERARCHY_MODES:
        raise ValueError("hierarchy_mode must be exactly 'off' or 'normalize_v1'")
    if value == HIERARCHY_MODE_NORMALIZE_V1 and retrieval_version != "v2":
        raise ValueError("hierarchy_mode='normalize_v1' requires retrieval_version='v2'")
    return value  # type: ignore[return-value]


def classify_hierarchy_relation(
    left: HierarchyChunkMetadata,
    right: HierarchyChunkMetadata,
    parent_by_child: dict[str, str] | None = None,
    ambiguous_identities: set[str] | None = None,
) -> HierarchyRelation:
    parents = parent_by_child or {}
    ambiguous = ambiguous_identities or set()
    if left.scope != right.scope:
        return HierarchyRelation("cross_scope", "none", 0, False, "scope_contract")
    if left.chunk_identity == right.chunk_identity:
        return HierarchyRelation("same_identity", "same", 0, True, "chunk_identity")
    if left.chunk_identity in ambiguous or right.chunk_identity in ambiguous:
        return HierarchyRelation(
            "ambiguous", "none", 0, False, "controlled_unavailable", True
        )
    if left.span == right.span:
        relation = (
            "exact_span_duplicate"
            if _compatible_exact_span(left, right)
            else "exact_span_conflict"
        )
        return HierarchyRelation(relation, "peer", 0, True, "validated_existing_index")

    right_depth = _ancestor_depth(
        child_identity=right.chunk_identity,
        ancestor_identity=left.chunk_identity,
        parent_by_child=parents,
    )
    if right_depth is not None:
        return HierarchyRelation(
            "parent" if right_depth == 1 else "ancestor",
            "down",
            right_depth,
            right_depth == 1,
            "resolved_hierarchy",
        )
    left_depth = _ancestor_depth(
        child_identity=left.chunk_identity,
        ancestor_identity=right.chunk_identity,
        parent_by_child=parents,
    )
    if left_depth is not None:
        return HierarchyRelation(
            "child" if left_depth == 1 else "descendant",
            "up",
            left_depth,
            left_depth == 1,
            "resolved_hierarchy",
        )
    left_parent = parents.get(left.chunk_identity)
    if left_parent and left_parent == parents.get(right.chunk_identity):
        return HierarchyRelation("sibling", "peer", 1, True, "resolved_hierarchy")
    if left.span.strictly_contains(right.span):
        return HierarchyRelation(
            "strict_containment", "down", 0, False, "validated_span"
        )
    if right.span.strictly_contains(left.span):
        return HierarchyRelation(
            "strict_containment", "up", 0, False, "validated_span"
        )
    if left.span.partial_overlap(right.span):
        return HierarchyRelation(
            "partial_overlap", "peer", 0, False, "validated_span"
        )
    return HierarchyRelation("disjoint", "none", 0, False, "validated_span")


class HierarchyResolver:
    """Resolve exact, bounded hierarchy only within direct-candidate file scopes."""

    def __init__(
        self,
        database: Database,
        limits: HierarchyLimits | None = None,
    ) -> None:
        self.database = database
        self.limits = limits or HierarchyLimits()

    def resolve(self, direct_candidates: list[Any]) -> HierarchyResolution:
        ordered_direct = sorted(direct_candidates, key=_direct_sort_key)
        metadata_by_identity = {
            str(item.chunk_identity): _metadata_from_candidate(item)
            for item in ordered_direct
        }
        warnings: list[str] = []
        ambiguous: set[str] = set()
        parent_by_child: dict[str, str] = {}
        parent_authority: dict[str, str] = {}
        links: list[HierarchyDerivedCandidate] = []
        query_audit: list[dict[str, Any]] = []
        truncated = False

        seeds = ordered_direct[: self.limits.max_direct_candidates]
        if len(seeds) < len(ordered_direct):
            truncated = True
            ambiguous.update(
                str(item.chunk_identity)
                for item in ordered_direct[self.limits.max_direct_candidates :]
            )
            warnings.append(
                "Hierarchy direct candidate budget was reached; later direct candidates "
                "were preserved without hierarchy normalization."
            )
        scopes: list[HierarchyScope] = []
        seen_scopes: set[HierarchyScope] = set()
        for seed in seeds:
            scope = metadata_by_identity[str(seed.chunk_identity)].scope
            if scope not in seen_scopes:
                seen_scopes.add(scope)
                scopes.append(scope)
        allowed_scopes = scopes[: self.limits.max_paths]
        if len(allowed_scopes) < len(scopes):
            truncated = True
            skipped_scopes = set(scopes[self.limits.max_paths :])
            ambiguous.update(
                identity
                for identity, item in metadata_by_identity.items()
                if item.scope in skipped_scopes
            )
            warnings.append(
                "Hierarchy path budget was reached; later paths were preserved without "
                "hierarchy normalization."
            )

        total_rows = 0
        direct_identities = {str(item.chunk_identity) for item in ordered_direct}
        derived_identities: set[str] = set()
        for scope in allowed_scopes:
            remaining = self.limits.max_total_rows - total_rows
            if remaining <= 0:
                truncated = True
                remaining_scopes = set(
                    allowed_scopes[allowed_scopes.index(scope) :]
                )
                ambiguous.update(
                    identity
                    for identity, item in metadata_by_identity.items()
                    if item.scope in remaining_scopes
                )
                warnings.append(
                    "Hierarchy total row budget was reached; remaining paths were "
                    "preserved without hierarchy normalization."
                )
                break
            allowed_rows = min(self.limits.max_rows_per_path, remaining)
            rows = self.database.get_code_chunks_for_hierarchy(
                scope.project_id,
                scope.repository_revision,
                scope.normalized_path,
                limit=allowed_rows + 1,
            )
            path_truncated = len(rows) > allowed_rows
            query_audit.append(
                {
                    "project_id": scope.project_id,
                    "repository_revision": scope.repository_revision,
                    "normalized_path": scope.normalized_path,
                    "row_limit": allowed_rows,
                    "returned_rows": min(len(rows), allowed_rows),
                    "truncated": path_truncated,
                }
            )
            if path_truncated:
                truncated = True
                ambiguous.update(
                    identity
                    for identity, item in metadata_by_identity.items()
                    if item.scope == scope
                )
                warnings.append(
                    f"Hierarchy rows for {scope.normalized_path} were truncated; "
                    "direct candidates were preserved and authoritative parent "
                    "resolution was skipped for that path."
                )
                continue
            total_rows += len(rows)
            try:
                path_metadata = [_metadata_from_row(row) for row in rows]
            except HierarchyContractError as exc:
                warnings.append(
                    f"Invalid hierarchy metadata for {scope.normalized_path}: {exc}. "
                    "Direct candidates were preserved."
                )
                ambiguous.update(
                    identity
                    for identity, item in metadata_by_identity.items()
                    if item.scope == scope
                )
                continue
            if any(item.scope != scope for item in path_metadata):
                raise HierarchyContractError(
                    "bounded hierarchy query returned a row outside its scope"
                )
            by_identity = {item.chunk_identity: item for item in path_metadata}
            if len(by_identity) != len(path_metadata):
                raise HierarchyContractError(
                    "bounded hierarchy query returned duplicate chunk identities"
                )
            by_id = {item.code_chunk_id: item for item in path_metadata}
            scoped_seeds = [
                item
                for item in seeds
                if metadata_by_identity[str(item.chunk_identity)].scope == scope
            ]
            invalid_seed = False
            for seed in scoped_seeds:
                stored = by_id.get(int(seed.code_chunk_id))
                if stored is None or stored.chunk_identity != str(seed.chunk_identity):
                    invalid_seed = True
                    ambiguous.add(str(seed.chunk_identity))
                    warnings.append(
                        "A direct retrieval candidate did not match the bounded chunk "
                        "snapshot; it was preserved without hierarchy normalization."
                    )
                else:
                    metadata_by_identity[stored.chunk_identity] = stored
            if invalid_seed:
                ambiguous.update(str(item.chunk_identity) for item in scoped_seeds)
                continue
            metadata_by_identity.update(by_identity)

            path_parents, path_authority, path_ambiguous, path_warnings = (
                _resolve_parent_map(path_metadata)
            )
            parent_by_child.update(path_parents)
            parent_authority.update(path_authority)
            ambiguous.update(path_ambiguous)
            warnings.extend(path_warnings)
            children: dict[str, list[str]] = {}
            for child_identity, parent_identity in path_parents.items():
                children.setdefault(parent_identity, []).append(child_identity)
            for values in children.values():
                values.sort(key=lambda identity: _metadata_sort_key(by_identity[identity]))

            for seed in scoped_seeds:
                seed_identity = str(seed.chunk_identity)
                if seed_identity in ambiguous:
                    continue
                neighbor_count = 0
                current = seed_identity
                for depth in range(1, self.limits.max_depth + 1):
                    parent_identity = path_parents.get(current)
                    if parent_identity is None:
                        break
                    if neighbor_count >= self.limits.max_neighbors_per_candidate:
                        truncated = True
                        warnings.append(
                            "Hierarchy per-candidate neighbor budget was reached."
                        )
                        break
                    if _allow_derived_link(
                        parent_identity,
                        direct_identities,
                        derived_identities,
                        self.limits.max_derived_candidates,
                    ):
                        links.append(
                            HierarchyDerivedCandidate(
                                candidate=by_identity[parent_identity],
                                derived_from_identity=seed_identity,
                                relation_type="parent" if depth == 1 else "ancestor",
                                direction="up",
                                depth=depth,
                                authority=path_authority.get(current, "span_inference"),
                            )
                        )
                        if parent_identity not in direct_identities:
                            derived_identities.add(parent_identity)
                        neighbor_count += 1
                    else:
                        truncated = True
                        warnings.append(
                            "Hierarchy derived candidate budget was reached; "
                            "remaining derived candidates were omitted."
                        )
                    current = parent_identity

                frontier = [(seed_identity, 0)]
                visited = {seed_identity}
                while frontier:
                    current, current_depth = frontier.pop(0)
                    if current_depth >= self.limits.max_depth:
                        continue
                    for child_identity in children.get(current, []):
                        if child_identity in visited:
                            continue
                        visited.add(child_identity)
                        depth = current_depth + 1
                        if neighbor_count >= self.limits.max_neighbors_per_candidate:
                            truncated = True
                            warnings.append(
                                "Hierarchy per-candidate neighbor budget was reached."
                            )
                            frontier = []
                            break
                        if _allow_derived_link(
                            child_identity,
                            direct_identities,
                            derived_identities,
                            self.limits.max_derived_candidates,
                        ):
                            links.append(
                                HierarchyDerivedCandidate(
                                    candidate=by_identity[child_identity],
                                    derived_from_identity=seed_identity,
                                    relation_type="child" if depth == 1 else "descendant",
                                    direction="down",
                                    depth=depth,
                                    authority=path_authority.get(
                                        child_identity, "span_inference"
                                    ),
                                )
                            )
                            if child_identity not in direct_identities:
                                derived_identities.add(child_identity)
                            neighbor_count += 1
                        else:
                            truncated = True
                            warnings.append(
                                "Hierarchy derived candidate budget was reached; "
                                "remaining derived candidates were omitted."
                            )
                        frontier.append((child_identity, depth))

        links = _deduplicate_links(links)
        return HierarchyResolution(
            metadata_by_identity=metadata_by_identity,
            parent_by_child=parent_by_child,
            parent_authority=parent_authority,
            ambiguous_identities=ambiguous,
            links=links,
            warnings=_deduplicate(warnings),
            truncated=truncated,
            audit={
                "resolver_version": HIERARCHY_RESOLVER_VERSION,
                "queries": query_audit,
                "budgets": _limits_dict(self.limits),
                "direct_candidate_count": len(ordered_direct),
                "processed_direct_candidate_count": len(seeds),
                "metadata_row_count": total_rows,
                "derived_candidate_count": len(derived_identities),
                "resolved_parents": [
                    {
                        "child_identity": child_identity,
                        "parent_identity": parent_by_child[child_identity],
                        "authority": parent_authority.get(
                            child_identity, "span_inference"
                        ),
                    }
                    for child_identity in sorted(parent_by_child)
                ],
                "links": [
                    {
                        "candidate_identity": item.candidate.chunk_identity,
                        "derived_from_identity": item.derived_from_identity,
                        "relation_type": item.relation_type,
                        "direction": item.direction,
                        "depth": item.depth,
                        "authority": item.authority,
                    }
                    for item in links
                ],
                "ambiguous_identities": sorted(ambiguous),
                "truncated": truncated,
            },
        )


@dataclass
class _NormalizationCandidate:
    metadata: HierarchyChunkMetadata
    direct: Any | None = None
    derived_from: set[str] = field(default_factory=set)
    relations: list[HierarchyDerivedCandidate] = field(default_factory=list)
    group_priority: float = 0.0
    family_root: str = ""
    group_id: str = ""
    decision: str = "pending"
    selection_reason: str = ""

    @property
    def origin(self) -> str:
        return "direct" if self.direct is not None else "hierarchy"


def normalize_hierarchy_candidates(
    direct_candidates: list[Any],
    resolution: HierarchyResolution,
    *,
    final_top_k: int,
    limits: HierarchyLimits | None = None,
    normalization_version: str = HIERARCHY_NORMALIZATION_VERSION,
) -> HierarchyNormalizationResult:
    limits = limits or HierarchyLimits()
    if normalization_version != HIERARCHY_NORMALIZATION_VERSION:
        raise ValueError("unknown hierarchy normalization version")
    _bounded_int("final_top_k", final_top_k, 1, limits.max_final_top_k)

    states: dict[str, _NormalizationCandidate] = {}
    direct_by_identity: dict[str, Any] = {}
    for direct in direct_candidates:
        identity = str(direct.chunk_identity)
        if not math.isfinite(float(direct.fused_score)):
            raise ValueError("direct fused scores must be finite")
        metadata = resolution.metadata_by_identity.get(identity) or _metadata_from_candidate(
            direct
        )
        states[identity] = _NormalizationCandidate(metadata=metadata, direct=direct)
        direct_by_identity[identity] = direct
    for link in resolution.links:
        identity = link.candidate.chunk_identity
        state = states.get(identity)
        if state is None:
            state = _NormalizationCandidate(metadata=link.candidate)
            states[identity] = state
        state.derived_from.add(link.derived_from_identity)
        state.relations.append(link)

    if len(states) > limits.max_normalization_groups * max(
        1, limits.max_family_members
    ):
        raise ValueError("hierarchy normalization candidate pool exceeds its hard limit")
    for state in states.values():
        if state.direct is not None:
            state.group_priority = float(state.direct.fused_score)
        else:
            scores = [
                float(direct_by_identity[value].fused_score)
                for value in state.derived_from
                if value in direct_by_identity
            ]
            state.group_priority = max(scores, default=0.0)
        if not math.isfinite(state.group_priority):
            raise ValueError("hierarchy group priority must be finite")
        state.family_root = _family_root(
            state.metadata.chunk_identity,
            resolution.parent_by_child,
            resolution.ambiguous_identities,
        )
        state.group_id = _group_id(state.metadata.scope, state.family_root)

    exact_span_suppressed: dict[str, str] = {}
    exact_span_audit: list[dict[str, Any]] = []
    span_groups: dict[tuple[Any, ...], list[_NormalizationCandidate]] = {}
    for state in states.values():
        key = (
            state.metadata.scope,
            state.metadata.span.start_line,
            state.metadata.span.end_line,
        )
        span_groups.setdefault(key, []).append(state)
    for peers in span_groups.values():
        if len(peers) < 2:
            continue
        ordered = sorted(peers, key=_normalization_sort_key)
        exact_group_id = _exact_span_group_id(
            ordered[0].metadata.scope,
            ordered[0].metadata.span,
        )
        if any(
            item.metadata.chunk_identity in resolution.ambiguous_identities
            for item in ordered
        ):
            exact_span_audit.append(
                {
                    "group_identity": exact_group_id,
                    "members": [item.metadata.chunk_identity for item in ordered],
                    "compatible": False,
                    "ambiguous": True,
                    "representative": None,
                    "suppressed_members": [],
                }
            )
            continue
        compatible = all(
            _compatible_exact_span(ordered[0].metadata, item.metadata)
            for item in ordered[1:]
        )
        representative: str | None = None
        suppressed_members: list[str] = []
        if compatible:
            representative = ordered[0].metadata.chunk_identity
            for peer in ordered[1:]:
                exact_span_suppressed[peer.metadata.chunk_identity] = representative
                suppressed_members.append(peer.metadata.chunk_identity)
        exact_span_audit.append(
            {
                "group_identity": exact_group_id,
                "members": [item.metadata.chunk_identity for item in ordered],
                "compatible": compatible,
                "ambiguous": False,
                "representative": representative,
                "suppressed_members": suppressed_members,
            }
        )

    ordered_states = sorted(states.values(), key=_normalization_sort_key)
    selected: list[_NormalizationCandidate] = []
    family_counts: dict[str, int] = {}
    for state in ordered_states:
        identity = state.metadata.chunk_identity
        if identity in exact_span_suppressed:
            state.decision = "suppressed"
            state.selection_reason = "compatible_exact_span_representative_selected"
            continue
        ambiguous_direct = (
            state.direct is not None
            and identity in resolution.ambiguous_identities
        )
        if (
            not ambiguous_direct
            and family_counts.get(state.group_id, 0) >= limits.max_family_members
        ):
            state.decision = "suppressed"
            state.selection_reason = "hierarchy_family_occupancy_limit"
            continue
        if len(selected) >= final_top_k:
            state.decision = "suppressed"
            state.selection_reason = "final_top_k_limit"
            continue
        state.decision = "retained"
        if ambiguous_direct:
            state.selection_reason = "ambiguous_hierarchy_preserved_direct_candidate"
        elif state.direct is not None:
            state.selection_reason = "direct_candidate_retained"
        else:
            state.selection_reason = "bounded_hierarchy_candidate_selected"
        selected.append(state)
        family_counts[state.group_id] = family_counts.get(state.group_id, 0) + 1

    results = [_to_hybrid_result(state) for state in selected]
    candidate_audit = [_candidate_audit(state) for state in ordered_states]
    group_audit: list[dict[str, Any]] = []
    by_group: dict[str, list[_NormalizationCandidate]] = {}
    for state in ordered_states:
        by_group.setdefault(state.group_id, []).append(state)
    if len(by_group) > limits.max_normalization_groups:
        raise ValueError("hierarchy normalization group budget was exceeded")
    for group_id in sorted(by_group):
        members = by_group[group_id]
        group_audit.append(
            {
                "group_identity": group_id,
                "family_root": members[0].family_root,
                "group_members": [item.metadata.chunk_identity for item in members],
                "direct_candidates": [
                    item.metadata.chunk_identity
                    for item in members
                    if item.direct is not None
                ],
                "hierarchy_derived_candidates": [
                    item.metadata.chunk_identity
                    for item in members
                    if item.direct is None
                ],
                "group_priority": max(item.group_priority for item in members),
                "selected_members": [
                    item.metadata.chunk_identity
                    for item in members
                    if item.decision == "retained"
                ],
                "suppressed_members": [
                    item.metadata.chunk_identity
                    for item in members
                    if item.decision == "suppressed"
                ],
                "ambiguous": any(
                    item.metadata.chunk_identity in resolution.ambiguous_identities
                    for item in members
                ),
            }
        )
    audit = {
        "normalization_version": normalization_version,
        "resolver_version": HIERARCHY_RESOLVER_VERSION,
        "budgets": _limits_dict(limits),
        "resolver": dict(resolution.audit),
        "truncated": resolution.truncated,
        "warnings": list(resolution.warnings),
        "candidates": candidate_audit,
        "groups": group_audit,
        "exact_span_groups": sorted(
            exact_span_audit,
            key=lambda item: item["group_identity"],
        ),
        "selection_order": [item.metadata.chunk_identity for item in selected],
        "final_top_k": final_top_k,
    }
    return HierarchyNormalizationResult(
        results=results,
        warnings=list(resolution.warnings),
        audit=audit,
    )


def _resolve_parent_map(
    items: list[HierarchyChunkMetadata],
) -> tuple[dict[str, str], dict[str, str], set[str], list[str]]:
    parents: dict[str, str] = {}
    authority: dict[str, str] = {}
    ambiguous: set[str] = set()
    warnings: list[str] = []
    for child in sorted(items, key=_metadata_sort_key):
        candidates: list[HierarchyChunkMetadata]
        source: str
        if child.parent_symbol:
            named = [
                item
                for item in items
                if item.chunk_identity != child.chunk_identity
                and item.qualified_name == child.parent_symbol
            ]
            candidates = [
                item for item in named if item.span.strictly_contains(child.span)
            ]
            source = "explicit_structural_metadata"
            if not candidates:
                ambiguous.add(child.chunk_identity)
                warnings.append(
                    f"Hierarchy metadata for {child.qualified_name} is ambiguous: "
                    "its explicit parent is missing, out of scope, or does not contain it."
                )
                continue
        else:
            candidates = [
                item
                for item in items
                if item.chunk_identity != child.chunk_identity
                and item.span.strictly_contains(child.span)
            ]
            source = "span_inference"
            if not candidates:
                continue
        minimum_size = min(item.span.size for item in candidates)
        nearest = [item for item in candidates if item.span.size == minimum_size]
        if len(nearest) != 1:
            ambiguous.add(child.chunk_identity)
            warnings.append(
                f"Hierarchy metadata for {child.qualified_name} is ambiguous: "
                "multiple indistinguishable nearest parents exist."
            )
            continue
        parents[child.chunk_identity] = nearest[0].chunk_identity
        authority[child.chunk_identity] = source

    cyclic = _cycle_identities(parents)
    if cyclic:
        ambiguous.update(cyclic)
        for identity in cyclic:
            parents.pop(identity, None)
            authority.pop(identity, None)
        warnings.append(
            "Hierarchy parent metadata formed a cycle; affected direct candidates "
            "were preserved without destructive normalization."
        )
    return parents, authority, ambiguous, warnings


def _cycle_identities(parents: dict[str, str]) -> set[str]:
    cyclic: set[str] = set()
    for start in parents:
        order: list[str] = []
        positions: dict[str, int] = {}
        current = start
        while current in parents:
            if current in positions:
                cyclic.update(order[positions[current] :])
                break
            positions[current] = len(order)
            order.append(current)
            current = parents[current]
    return cyclic


def _allow_derived_link(
    identity: str,
    direct_identities: set[str],
    derived_identities: set[str],
    maximum: int,
) -> bool:
    return (
        identity in direct_identities
        or identity in derived_identities
        or len(derived_identities) < maximum
    )


def _deduplicate_links(
    links: list[HierarchyDerivedCandidate],
) -> list[HierarchyDerivedCandidate]:
    values = {
        (
            item.candidate.chunk_identity,
            item.derived_from_identity,
            item.relation_type,
            item.direction,
            item.depth,
            item.authority,
        ): item
        for item in links
    }
    return [
        values[key]
        for key in sorted(
            values,
            key=lambda key: (
                key[1], key[4], key[2], key[3], key[0], key[5]
            ),
        )
    ]


def _metadata_from_candidate(item: Any) -> HierarchyChunkMetadata:
    return HierarchyChunkMetadata(
        chunk_identity=str(item.chunk_identity),
        code_chunk_id=int(item.code_chunk_id),
        scope=HierarchyScope(
            str(item.project_id),
            str(item.repository_revision),
            str(item.path),
        ),
        language=str(item.language),
        chunk_type=str(item.chunk_type),
        symbol_name=str(item.symbol_name),
        qualified_name=str(item.qualified_name),
        parent_symbol=str(getattr(item, "parent_symbol", "")),
        span=CanonicalSpan(int(item.start_line), int(item.end_line)),
        content=str(item.content),
        content_hash=str(item.content_hash),
    )


def _metadata_from_row(row: dict[str, Any]) -> HierarchyChunkMetadata:
    identity = "|".join(
        [
            str(row["project_id"]),
            str(row["repository_revision"]),
            str(row["path"]),
            str(row["start_line"]),
            str(row["end_line"]),
            str(row["content_hash"]),
            str(row["id"]),
        ]
    )
    return HierarchyChunkMetadata(
        chunk_identity=identity,
        code_chunk_id=int(row["id"]),
        scope=HierarchyScope(
            str(row["project_id"]),
            str(row["repository_revision"]),
            str(row["path"]),
        ),
        language=str(row["language"]),
        chunk_type=str(row["chunk_type"]),
        symbol_name=str(row["symbol_name"]),
        qualified_name=str(row["qualified_name"]),
        parent_symbol=str(row.get("parent_symbol") or ""),
        span=CanonicalSpan(row["start_line"], row["end_line"]),
        content=str(row["content"]),
        content_hash=str(row["content_hash"]),
    )


def _to_hybrid_result(state: _NormalizationCandidate) -> HybridSearchResult:
    if state.direct is not None:
        return state.direct.to_hybrid_result()
    item = state.metadata
    return HybridSearchResult(
        project_id=item.scope.project_id,
        repository_revision=item.scope.repository_revision,
        code_chunk_id=item.code_chunk_id,
        language=item.language,
        path=item.scope.normalized_path,
        chunk_type=item.chunk_type,
        symbol_name=item.symbol_name,
        qualified_name=item.qualified_name,
        start_line=item.span.start_line,
        end_line=item.span.end_line,
        content=item.content,
        content_hash=item.content_hash,
        retrieval_sources=["hierarchy"],
        lexical_score=None,
        lexical_rank=None,
        semantic_score=None,
        semantic_rank=None,
        fusion_score=0.0,
        fusion_rank=0,
    )


def _candidate_audit(state: _NormalizationCandidate) -> dict[str, Any]:
    direct = state.direct
    source_records: dict[str, Any] = {}
    contributions: dict[str, float] = {}
    if direct is not None:
        source_records = {
            str(source): {
                "rank": record.rank,
                "raw_score": record.raw_score,
                "reasons": list(record.reasons),
                "metadata": dict(record.metadata),
            }
            for source, record in sorted(direct.source_records.items())
        }
        contributions = {
            str(source): float(value)
            for source, value in sorted(direct.fusion_contributions.items())
        }
    return {
        "chunk_identity": state.metadata.chunk_identity,
        "code_chunk_id": state.metadata.code_chunk_id,
        "project_id": state.metadata.scope.project_id,
        "repository_revision": state.metadata.scope.repository_revision,
        "normalized_path": state.metadata.scope.normalized_path,
        "symbol_name": state.metadata.symbol_name,
        "qualified_name": state.metadata.qualified_name,
        "symbol_kind": state.metadata.chunk_type,
        "start_line": state.metadata.span.start_line,
        "end_line": state.metadata.span.end_line,
        "span_size": state.metadata.span.size,
        "content_hash": state.metadata.content_hash,
        "origin": state.origin,
        "derived_from": sorted(state.derived_from),
        "relations": [
            {
                "derived_from": item.derived_from_identity,
                "relation_type": item.relation_type,
                "direction": item.direction,
                "depth": item.depth,
                "authority": item.authority,
            }
            for item in sorted(
                state.relations,
                key=lambda item: (
                    item.derived_from_identity,
                    item.depth,
                    item.relation_type,
                    item.direction,
                ),
            )
        ],
        "metadata_source": state.metadata.metadata_source,
        "source_records": source_records,
        "fusion_contributions": contributions,
        "original_fused_score": (
            float(direct.fused_score) if direct is not None else None
        ),
        "original_fusion_rank": (
            int(direct.fusion_rank) if direct is not None else None
        ),
        "group_priority": state.group_priority,
        "normalization_score": state.group_priority,
        "group_identity": state.group_id,
        "family_root": state.family_root,
        "decision": state.decision,
        "selection_reason": state.selection_reason,
    }


def _normalization_sort_key(state: _NormalizationCandidate) -> tuple[Any, ...]:
    if state.direct is not None:
        ranks = [record.rank for record in state.direct.source_records.values()]
        return (
            0,
            -float(state.direct.fused_score),
            min(ranks, default=10**9),
            -len(state.direct.source_records),
            state.metadata.qualified_name.casefold(),
            state.metadata.scope.normalized_path,
            state.metadata.span.start_line,
            state.metadata.span.end_line,
            state.metadata.chunk_identity,
        )
    depth = min((item.depth for item in state.relations), default=10**9)
    relation_order = min(
        (
            {"parent": 0, "child": 1, "ancestor": 2, "descendant": 3}.get(
                item.relation_type, 9
            )
            for item in state.relations
        ),
        default=9,
    )
    return (
        1,
        -state.group_priority,
        depth,
        relation_order,
        state.metadata.qualified_name.casefold(),
        state.metadata.scope.normalized_path,
        state.metadata.span.start_line,
        state.metadata.span.end_line,
        state.metadata.chunk_identity,
    )


def _direct_sort_key(item: Any) -> tuple[Any, ...]:
    records = list(getattr(item, "source_records", {}).values())
    return (
        int(getattr(item, "fusion_rank", 0)) or 10**9,
        -float(getattr(item, "fused_score", 0.0)),
        min((record.rank for record in records), default=10**9),
        -len(records),
        str(item.qualified_name).casefold(),
        _normalize_path(str(item.path)),
        int(item.start_line),
        int(item.end_line),
        str(item.chunk_identity),
    )


def _metadata_sort_key(item: HierarchyChunkMetadata) -> tuple[Any, ...]:
    return (
        item.scope.normalized_path,
        item.span.start_line,
        -item.span.end_line,
        item.qualified_name.casefold(),
        item.chunk_type,
        item.chunk_identity,
    )


def _family_root(
    identity: str,
    parent_by_child: dict[str, str],
    ambiguous_identities: set[str],
) -> str:
    if identity in ambiguous_identities:
        return identity
    current = identity
    seen = {identity}
    while current in parent_by_child:
        parent = parent_by_child[current]
        if parent in seen or parent in ambiguous_identities:
            return identity
        seen.add(parent)
        current = parent
    return current


def _group_id(scope: HierarchyScope, family_root: str) -> str:
    value = "\0".join(
        [
            scope.project_id,
            scope.repository_revision,
            scope.normalized_path,
            family_root,
        ]
    )
    return "H" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _exact_span_group_id(scope: HierarchyScope, span: CanonicalSpan) -> str:
    value = "\0".join(
        [
            scope.project_id,
            scope.repository_revision,
            scope.normalized_path,
            str(span.start_line),
            str(span.end_line),
        ]
    )
    return "X" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _compatible_exact_span(
    left: HierarchyChunkMetadata,
    right: HierarchyChunkMetadata,
) -> bool:
    return (
        left.scope == right.scope
        and left.span == right.span
        and left.content_hash == right.content_hash
        and left.symbol_name == right.symbol_name
        and left.qualified_name == right.qualified_name
        and left.chunk_type == right.chunk_type
    )


def _ancestor_depth(
    *,
    child_identity: str,
    ancestor_identity: str,
    parent_by_child: dict[str, str],
) -> int | None:
    current = child_identity
    seen = {current}
    for depth in range(1, len(parent_by_child) + 2):
        current = parent_by_child.get(current, "")
        if not current or current in seen:
            return None
        if current == ancestor_identity:
            return depth
        seen.add(current)
    return None


def _limits_dict(limits: HierarchyLimits) -> dict[str, int]:
    return {
        "max_direct_candidates": limits.max_direct_candidates,
        "max_paths": limits.max_paths,
        "max_rows_per_path": limits.max_rows_per_path,
        "max_total_rows": limits.max_total_rows,
        "max_depth": limits.max_depth,
        "max_neighbors_per_candidate": limits.max_neighbors_per_candidate,
        "max_derived_candidates": limits.max_derived_candidates,
        "max_normalization_groups": limits.max_normalization_groups,
        "max_family_members": limits.max_family_members,
        "max_final_top_k": limits.max_final_top_k,
    }


def _normalize_path(value: str) -> str:
    return str(value).replace("\\", "/").lstrip("/")


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _bounded_int(name: str, value: int, minimum: int, maximum: int) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
        or value > maximum
    ):
        raise ValueError(
            f"{name} must be an integer between {minimum} and {maximum}"
        )
