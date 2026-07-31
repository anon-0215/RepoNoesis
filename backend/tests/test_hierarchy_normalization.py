from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

from app.services.hierarchy_normalization import (
    CanonicalSpan,
    HIERARCHY_NORMALIZATION_VERSION,
    HierarchyChunkMetadata,
    HierarchyDerivedCandidate,
    HierarchyLimits,
    HierarchyResolution,
    HierarchyScope,
    normalize_hierarchy_candidates,
)
from app.services.hybrid_retriever import HybridSearchResult


def meta(
    number: int,
    start: int,
    end: int,
    *,
    name: str | None = None,
    chunk_type: str = "function",
    content_hash: str | None = None,
) -> HierarchyChunkMetadata:
    qualified = name or f"item{number}"
    return HierarchyChunkMetadata(
        chunk_identity=f"identity-{number}",
        code_chunk_id=number,
        scope=HierarchyScope("project", "revision", "src/app.py"),
        language="python",
        chunk_type=chunk_type,
        symbol_name=qualified.rsplit(".", 1)[-1],
        qualified_name=qualified,
        parent_symbol="",
        span=CanonicalSpan(start, end),
        content=f"# {qualified}\n",
        content_hash=content_hash or f"{number:064x}",
    )


class FakeMergedCandidate:
    def __init__(self, item: HierarchyChunkMetadata, rank: int, score: float | None = None):
        self.chunk_identity = item.chunk_identity
        self.project_id = item.scope.project_id
        self.repository_revision = item.scope.repository_revision
        self.code_chunk_id = item.code_chunk_id
        self.language = item.language
        self.path = item.scope.normalized_path
        self.chunk_type = item.chunk_type
        self.symbol_name = item.symbol_name
        self.qualified_name = item.qualified_name
        self.start_line = item.span.start_line
        self.end_line = item.span.end_line
        self.content = item.content
        self.content_hash = item.content_hash
        self.fused_score = score if score is not None else 1.0 / (60 + rank)
        self.fusion_rank = rank
        self.source_records = {
            "lexical": SimpleNamespace(
                rank=rank,
                raw_score=float(100 - rank),
                reasons=["bm25_match"],
                metadata={"scoring": "bm25"},
            )
        }
        self.fusion_contributions = {"lexical": self.fused_score}

    def to_hybrid_result(self) -> HybridSearchResult:
        record = self.source_records["lexical"]
        return HybridSearchResult(
            project_id=self.project_id,
            repository_revision=self.repository_revision,
            code_chunk_id=self.code_chunk_id,
            language=self.language,
            path=self.path,
            chunk_type=self.chunk_type,
            symbol_name=self.symbol_name,
            qualified_name=self.qualified_name,
            start_line=self.start_line,
            end_line=self.end_line,
            content=self.content,
            content_hash=self.content_hash,
            retrieval_sources=["lexical"],
            lexical_score=record.raw_score,
            lexical_rank=record.rank,
            fusion_score=self.fused_score,
            fusion_rank=self.fusion_rank,
        )


def resolution(
    items: list[HierarchyChunkMetadata],
    *,
    parents: dict[str, str] | None = None,
    links: list[HierarchyDerivedCandidate] | None = None,
    ambiguous: set[str] | None = None,
) -> HierarchyResolution:
    return HierarchyResolution(
        metadata_by_identity={item.chunk_identity: item for item in items},
        parent_by_child=dict(parents or {}),
        parent_authority={key: "span_inference" for key in (parents or {})},
        ambiguous_identities=set(ambiguous or set()),
        links=list(links or []),
        warnings=[],
        truncated=False,
        audit={"queries": []},
    )


class HierarchyNormalizationTests(unittest.TestCase):
    def test_parent_and_child_direct_candidates_keep_original_scores_and_identity(self):
        parent = meta(1, 1, 10, name="outer")
        child = meta(2, 2, 4, name="outer.inner")
        direct = [FakeMergedCandidate(parent, 1), FakeMergedCandidate(child, 2)]

        result = normalize_hierarchy_candidates(
            direct,
            resolution([parent, child], parents={child.chunk_identity: parent.chunk_identity}),
            final_top_k=2,
        )

        self.assertEqual([item.code_chunk_id for item in result.results], [1, 2])
        self.assertEqual([item.fusion_rank for item in result.results], [1, 2])
        self.assertEqual([item.fusion_score for item in result.results], [direct[0].fused_score, direct[1].fused_score])
        self.assertTrue(all(item["origin"] == "direct" for item in result.audit["candidates"]))

    def test_child_direct_parent_derived_has_no_fabricated_source_or_rrf_score(self):
        parent = meta(1, 1, 10, name="outer")
        child = meta(2, 2, 4, name="outer.inner")
        link = HierarchyDerivedCandidate(
            candidate=parent,
            derived_from_identity=child.chunk_identity,
            relation_type="parent",
            direction="up",
            depth=1,
            authority="span_inference",
        )
        result = normalize_hierarchy_candidates(
            [FakeMergedCandidate(child, 1)],
            resolution(
                [parent, child],
                parents={child.chunk_identity: parent.chunk_identity},
                links=[link],
            ),
            final_top_k=2,
        )

        derived = next(item for item in result.results if item.code_chunk_id == 1)
        self.assertEqual(derived.retrieval_sources, ["hierarchy"])
        self.assertIsNone(derived.lexical_rank)
        self.assertIsNone(derived.semantic_rank)
        self.assertEqual(derived.fusion_score, 0.0)
        self.assertEqual(derived.fusion_rank, 0)
        audit = next(item for item in result.audit["candidates"] if item["code_chunk_id"] == 1)
        self.assertEqual(audit["origin"], "hierarchy")
        self.assertEqual(audit["source_records"], {})
        self.assertEqual(audit["fusion_contributions"], {})
        self.assertIsNone(audit["original_fused_score"])
        self.assertEqual(audit["derived_from"], [child.chunk_identity])

    def test_parent_direct_child_derived_is_supported(self):
        parent = meta(1, 1, 10, name="outer")
        child = meta(2, 2, 4, name="outer.inner")
        link = HierarchyDerivedCandidate(
            candidate=child,
            derived_from_identity=parent.chunk_identity,
            relation_type="child",
            direction="down",
            depth=1,
            authority="span_inference",
        )
        result = normalize_hierarchy_candidates(
            [FakeMergedCandidate(parent, 1)],
            resolution(
                [parent, child],
                parents={child.chunk_identity: parent.chunk_identity},
                links=[link],
            ),
            final_top_k=2,
        )
        self.assertEqual([item.code_chunk_id for item in result.results], [1, 2])

    def test_compatible_exact_span_peer_uses_auditable_representative_not_identity_merge(self):
        digest = "a" * 64
        first = meta(1, 1, 3, name="target", content_hash=digest)
        second = meta(2, 1, 3, name="target", content_hash=digest)

        result = normalize_hierarchy_candidates(
            [FakeMergedCandidate(second, 2), FakeMergedCandidate(first, 1)],
            resolution([first, second]),
            final_top_k=2,
        )

        self.assertEqual([item.code_chunk_id for item in result.results], [1])
        self.assertNotEqual(first.chunk_identity, second.chunk_identity)
        suppressed = next(item for item in result.audit["candidates"] if item["code_chunk_id"] == 2)
        self.assertEqual(suppressed["decision"], "suppressed")
        self.assertEqual(suppressed["selection_reason"], "compatible_exact_span_representative_selected")
        exact_group = result.audit["exact_span_groups"][0]
        self.assertTrue(exact_group["compatible"])
        self.assertEqual(exact_group["representative"], first.chunk_identity)
        self.assertEqual(exact_group["suppressed_members"], [second.chunk_identity])

    def test_exact_span_metadata_conflict_partial_overlap_and_siblings_are_preserved(self):
        conflicting_a = meta(1, 1, 3, name="target", content_hash="a" * 64)
        conflicting_b = meta(2, 1, 3, name="other", content_hash="a" * 64)
        overlap = meta(3, 3, 5, name="overlap")
        sibling = meta(4, 7, 8, name="sibling")
        common_parent = "outside-parent"
        parents = {conflicting_a.chunk_identity: common_parent, sibling.chunk_identity: common_parent}

        result = normalize_hierarchy_candidates(
            [
                FakeMergedCandidate(conflicting_a, 1),
                FakeMergedCandidate(conflicting_b, 2),
                FakeMergedCandidate(overlap, 3),
                FakeMergedCandidate(sibling, 4),
            ],
            resolution([conflicting_a, conflicting_b, overlap, sibling], parents=parents),
            final_top_k=4,
            limits=HierarchyLimits(max_family_members=4),
        )

        self.assertEqual({item.code_chunk_id for item in result.results}, {1, 2, 3, 4})

    def test_ambiguous_family_preserves_direct_candidates_and_skips_destructive_normalization(self):
        first = meta(1, 1, 10)
        second = meta(2, 2, 3)
        result = normalize_hierarchy_candidates(
            [FakeMergedCandidate(first, 1), FakeMergedCandidate(second, 2)],
            resolution([first, second], ambiguous={first.chunk_identity, second.chunk_identity}),
            final_top_k=2,
        )

        self.assertEqual([item.code_chunk_id for item in result.results], [1, 2])
        self.assertTrue(all(item["decision"] == "retained" for item in result.audit["candidates"]))

    def test_ambiguous_exact_span_peers_are_not_suppressed(self):
        digest = "a" * 64
        first = meta(1, 1, 3, name="target", content_hash=digest)
        second = meta(2, 1, 3, name="target", content_hash=digest)
        result = normalize_hierarchy_candidates(
            [FakeMergedCandidate(first, 1), FakeMergedCandidate(second, 2)],
            resolution(
                [first, second],
                ambiguous={first.chunk_identity, second.chunk_identity},
            ),
            final_top_k=2,
        )

        self.assertEqual([item.code_chunk_id for item in result.results], [1, 2])

    def test_family_occupancy_is_bounded_and_suppressed_candidate_remains_in_audit(self):
        root = meta(1, 1, 20)
        middle = meta(2, 2, 15)
        child = meta(3, 3, 5)
        parents = {middle.chunk_identity: root.chunk_identity, child.chunk_identity: middle.chunk_identity}
        result = normalize_hierarchy_candidates(
            [FakeMergedCandidate(root, 1), FakeMergedCandidate(middle, 2), FakeMergedCandidate(child, 3)],
            resolution([root, middle, child], parents=parents),
            final_top_k=3,
            limits=HierarchyLimits(max_family_members=2),
        )

        self.assertEqual([item.code_chunk_id for item in result.results], [1, 2])
        suppressed = next(item for item in result.audit["candidates"] if item["code_chunk_id"] == 3)
        self.assertEqual(suppressed["selection_reason"], "hierarchy_family_occupancy_limit")

    def test_input_order_and_complete_ties_are_deterministic(self):
        first = meta(1, 1, 2, name="a")
        second = meta(2, 4, 5, name="b")
        a = FakeMergedCandidate(first, 1, score=0.5)
        b = FakeMergedCandidate(second, 2, score=0.5)
        expected = normalize_hierarchy_candidates(
            [a, b], resolution([first, second]), final_top_k=2
        )
        reversed_result = normalize_hierarchy_candidates(
            [b, a], resolution([second, first]), final_top_k=2
        )

        self.assertEqual(
            [item.code_chunk_id for item in expected.results],
            [item.code_chunk_id for item in reversed_result.results],
        )
        self.assertEqual(expected.audit["selection_order"], reversed_result.audit["selection_order"])

    def test_versions_and_numeric_limits_are_strict(self):
        item = meta(1, 1, 2)
        with self.assertRaises(ValueError):
            normalize_hierarchy_candidates(
                [FakeMergedCandidate(item, 1)],
                resolution([item]),
                final_top_k=1,
                normalization_version="unknown",
            )
        for kwargs in (
            {"max_direct_candidates": 0},
            {"max_rows_per_path": -1},
            {"max_family_members": 0},
            {"max_derived_candidates": 10_000},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HierarchyLimits(**kwargs)
        bad = FakeMergedCandidate(item, 1, score=math.nan)
        with self.assertRaises(ValueError):
            normalize_hierarchy_candidates([bad], resolution([item]), final_top_k=1)
        infinite = FakeMergedCandidate(item, 1, score=math.inf)
        with self.assertRaises(ValueError):
            normalize_hierarchy_candidates(
                [infinite], resolution([item]), final_top_k=1
            )
        for invalid_top_k in (0, -1, 9):
            with self.subTest(final_top_k=invalid_top_k), self.assertRaises(ValueError):
                normalize_hierarchy_candidates(
                    [FakeMergedCandidate(item, 1)],
                    resolution([item]),
                    final_top_k=invalid_top_k,
                )
        self.assertEqual(HIERARCHY_NORMALIZATION_VERSION, "hierarchy_normalization_v1@1")


if __name__ == "__main__":
    unittest.main()
