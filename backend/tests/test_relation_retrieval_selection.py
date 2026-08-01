from __future__ import annotations

import unittest

from app.services.hybrid_retriever import HybridSearchResult
from app.services.relation_retrieval import (
    RelationCandidate,
    RelationExpansionLimits,
    RelationPathProvenance,
    select_relation_aware_candidates,
)


def candidate(identity: str, rank: int, *, source: str = "lexical") -> HybridSearchResult:
    code_chunk_id = max(1, sum(ord(value) for value in identity))
    return HybridSearchResult(
        project_id="project",
        repository_revision="revision",
        code_chunk_id=code_chunk_id,
        language="python",
        path=f"src/{identity}.py",
        chunk_type="function",
        symbol_name=identity,
        qualified_name=identity,
        start_line=1,
        end_line=1,
        content=f"def {identity}(): pass\n",
        content_hash=(str(code_chunk_id % 10) * 64),
        retrieval_sources=[source],
        lexical_score=1.0 if source == "lexical" else None,
        lexical_rank=rank if source == "lexical" else None,
        semantic_score=None,
        semantic_rank=None,
        fusion_score=(1.0 / (60 + rank)) if source != "relation" else 0.0,
        fusion_rank=rank if source != "relation" else 0,
    )


def chunk_identity(item: HybridSearchResult) -> str:
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


def relation_candidate(
    identity: str,
    *,
    seed: str,
    seed_rank: int,
    priority: float,
    relation_type: str = "calls",
    direction: str = "outgoing",
) -> RelationCandidate:
    target = candidate(identity, 100 + seed_rank, source="relation")
    seed_item = candidate(seed, seed_rank)
    path = RelationPathProvenance(
        path_identity=f"path-{seed}-{identity}-{relation_type}-{direction}",
        seed_chunk_identity=chunk_identity(seed_item),
        seed_node_id=f"node-{seed}",
        target_node_id=f"node-{identity}",
        target_chunk_identity=chunk_identity(target),
        edge_id=f"edge-{seed}-{identity}",
        relation_type=relation_type,
        relation_view=relation_type,
        direction=direction,
        project_id="project",
        repository_revision="revision",
        depth=1,
        seed_selection_rank=seed_rank,
        seed_origin="direct",
        seed_fused_score=0.1,
        seed_hierarchy_priority=None,
        relation_type_weight=1.0,
        depth_decay=1.0,
        path_priority=priority,
        resolution_status="resolved",
    )
    return RelationCandidate(
        candidate=target,
        paths=[path],
        priority=priority,
    )


class RelationSelectionTests(unittest.TestCase):
    def test_direct_minimum_relation_cap_and_direct_backfill(self):
        direct = [candidate(f"d{i}", i) for i in range(1, 6)]
        relations = [
            relation_candidate("r1", seed="d1", seed_rank=1, priority=0.5),
            relation_candidate("r2", seed="d2", seed_rank=2, priority=0.4),
        ]
        selected = select_relation_aware_candidates(direct, relations, top_k=5)
        self.assertEqual(len(selected.results), 5)
        self.assertEqual(sum(item.retrieval_sources == ["relation"] for item in selected.results), 1)
        self.assertGreaterEqual(sum(item.retrieval_sources != ["relation"] for item in selected.results), 4)
        self.assertEqual(selected.audit["relation_slot_cap"], 1)

        backfilled = select_relation_aware_candidates(direct, [], top_k=5)
        self.assertEqual(
            [item.qualified_name for item in backfilled.results],
            [item.qualified_name for item in direct],
        )

    def test_small_top_k_zero_relation_and_invalid_top_k_rejected(self):
        direct = [candidate("d1", 1), candidate("d2", 2)]
        relation = relation_candidate("r1", seed="d1", seed_rank=1, priority=0.5)
        for top_k in (1, 2):
            result = select_relation_aware_candidates(direct, [relation], top_k=top_k)
            self.assertFalse(any(item.retrieval_sources == ["relation"] for item in result.results))
        for invalid in (0, -1, 9, True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                select_relation_aware_candidates(direct, [relation], top_k=invalid)

    def test_per_seed_and_family_caps_prevent_hub_occupancy(self):
        direct = [candidate(f"d{i}", i) for i in range(1, 9)]
        relations = [
            relation_candidate("r1", seed="d1", seed_rank=1, priority=0.9),
            relation_candidate("r2", seed="d1", seed_rank=1, priority=0.8),
            relation_candidate("r3", seed="d2", seed_rank=2, priority=0.7),
            relation_candidate("r4", seed="d3", seed_rank=3, priority=0.6),
        ]
        result = select_relation_aware_candidates(direct, relations, top_k=8)
        relation_names = [
            item.qualified_name
            for item in result.results
            if item.retrieval_sources == ["relation"]
        ]
        self.assertLessEqual(len(relation_names), 2)
        self.assertFalse({"r1", "r2"}.issubset(relation_names))
        self.assertLessEqual(max(result.audit["family_occupancy"].values(), default=0), 2)

    def test_duplicate_direct_target_and_ambiguous_candidate_do_not_take_slots(self):
        direct = [candidate(f"d{i}", i) for i in range(1, 6)]
        duplicate = relation_candidate("d2", seed="d1", seed_rank=1, priority=1.0)
        ambiguous = relation_candidate("bad", seed="d1", seed_rank=1, priority=0.9)
        ambiguous.resolution_status = "ambiguous"
        result = select_relation_aware_candidates(
            direct, [duplicate, ambiguous], top_k=5
        )
        self.assertEqual([item.qualified_name for item in result.results], [f"d{i}" for i in range(1, 6)])

    def test_complete_ties_and_input_order_are_deterministic(self):
        direct = [candidate(f"d{i}", i) for i in range(1, 9)]
        first_relations = [
            relation_candidate("z", seed="d1", seed_rank=1, priority=0.5),
            relation_candidate("a", seed="d2", seed_rank=2, priority=0.5),
            relation_candidate("m", seed="d3", seed_rank=3, priority=0.5),
        ]
        first = select_relation_aware_candidates(direct, first_relations, top_k=8)
        second = select_relation_aware_candidates(
            list(reversed(direct)), list(reversed(first_relations)), top_k=8
        )
        self.assertEqual(
            [item.qualified_name for item in first.results],
            [item.qualified_name for item in second.results],
        )
        self.assertEqual(first.audit["final_ordering"], second.audit["final_ordering"])


if __name__ == "__main__":
    unittest.main()
