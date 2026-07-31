from __future__ import annotations

import unittest

from app.services.hierarchy_normalization import (
    CanonicalSpan,
    HierarchyChunkMetadata,
    HierarchyContractError,
    HierarchyScope,
    classify_hierarchy_relation,
)


def metadata(
    identity: str,
    start: int,
    end: int,
    *,
    project_id: str = "project",
    revision: str = "revision",
    path: str = "src/app.py",
    qualified_name: str | None = None,
    chunk_type: str = "function",
    content_hash: str | None = None,
) -> HierarchyChunkMetadata:
    name = qualified_name or identity
    return HierarchyChunkMetadata(
        chunk_identity=identity,
        code_chunk_id=int(identity.removeprefix("chunk") or "1"),
        scope=HierarchyScope(project_id, revision, path),
        language="python",
        chunk_type=chunk_type,
        symbol_name=name.rsplit(".", 1)[-1],
        qualified_name=name,
        parent_symbol="",
        span=CanonicalSpan(start, end),
        content=f"# {identity}\n",
        content_hash=content_hash or identity.rjust(64, "0")[-64:],
    )


class CanonicalSpanTests(unittest.TestCase):
    def test_inclusive_span_equality_containment_overlap_and_size(self):
        outer = CanonicalSpan(1, 10)
        inner = CanonicalSpan(2, 3)

        self.assertEqual(outer.size, 10)
        self.assertEqual(CanonicalSpan(2, 3), inner)
        self.assertTrue(outer.strictly_contains(inner))
        self.assertFalse(inner.strictly_contains(outer))
        self.assertTrue(CanonicalSpan(1, 2).overlaps(CanonicalSpan(2, 3)))
        self.assertTrue(CanonicalSpan(1, 2).partial_overlap(CanonicalSpan(2, 3)))
        self.assertTrue(CanonicalSpan(1, 2).disjoint(CanonicalSpan(3, 4)))

    def test_invalid_spans_are_contract_errors_not_disjoint(self):
        for start, end in ((0, 1), (-1, 2), (3, 2), (True, 2), (1, None), ("1", 2)):
            with self.subTest(start=start, end=end), self.assertRaises(
                HierarchyContractError
            ):
                CanonicalSpan(start, end)  # type: ignore[arg-type]


class HierarchyRelationTests(unittest.TestCase):
    def test_same_identity_and_exact_span_are_distinct_contracts(self):
        first = metadata("chunk1", 1, 3, qualified_name="target")
        same = metadata("chunk1", 1, 3, qualified_name="target")
        peer = metadata("chunk2", 1, 3, qualified_name="target", content_hash=first.content_hash)
        conflict = metadata("chunk3", 1, 3, qualified_name="other", content_hash=first.content_hash)

        self.assertEqual(classify_hierarchy_relation(first, same).relation_type, "same_identity")
        self.assertEqual(
            classify_hierarchy_relation(first, peer).relation_type,
            "exact_span_duplicate",
        )
        self.assertEqual(
            classify_hierarchy_relation(first, conflict).relation_type,
            "exact_span_conflict",
        )

    def test_direct_indirect_parent_child_and_sibling_are_distinguished(self):
        root = metadata("chunk1", 1, 20, qualified_name="root")
        parent = metadata("chunk2", 2, 15, qualified_name="root.parent")
        child = metadata("chunk3", 3, 5, qualified_name="root.parent.child")
        sibling = metadata("chunk4", 7, 9, qualified_name="root.parent.sibling")
        parents = {
            parent.chunk_identity: root.chunk_identity,
            child.chunk_identity: parent.chunk_identity,
            sibling.chunk_identity: parent.chunk_identity,
        }

        self.assertEqual(classify_hierarchy_relation(parent, child, parents).relation_type, "parent")
        self.assertEqual(classify_hierarchy_relation(child, parent, parents).relation_type, "child")
        ancestor = classify_hierarchy_relation(root, child, parents)
        descendant = classify_hierarchy_relation(child, root, parents)
        self.assertEqual((ancestor.relation_type, ancestor.depth), ("ancestor", 2))
        self.assertEqual((descendant.relation_type, descendant.depth), ("descendant", 2))
        self.assertEqual(
            classify_hierarchy_relation(child, sibling, parents).relation_type,
            "sibling",
        )

    def test_partial_overlap_disjoint_and_cross_scope_do_not_create_hierarchy(self):
        first = metadata("chunk1", 1, 5)
        overlap = metadata("chunk2", 4, 8)
        disjoint = metadata("chunk3", 9, 12)
        other_path = metadata("chunk4", 1, 5, path="src/other.py")
        other_revision = metadata("chunk5", 1, 5, revision="other")
        other_project = metadata("chunk6", 1, 5, project_id="other")

        self.assertEqual(classify_hierarchy_relation(first, overlap).relation_type, "partial_overlap")
        self.assertEqual(classify_hierarchy_relation(first, disjoint).relation_type, "disjoint")
        for other in (other_path, other_revision, other_project):
            with self.subTest(scope=other.scope):
                self.assertEqual(
                    classify_hierarchy_relation(first, other).relation_type,
                    "cross_scope",
                )

    def test_nested_function_containment_never_becomes_identity(self):
        outer = metadata("chunk1", 1, 10, qualified_name="outer")
        inner = metadata("chunk2", 2, 4, qualified_name="outer.inner")

        relation = classify_hierarchy_relation(outer, inner)
        self.assertEqual(relation.relation_type, "strict_containment")
        self.assertNotEqual(outer.chunk_identity, inner.chunk_identity)


if __name__ == "__main__":
    unittest.main()
