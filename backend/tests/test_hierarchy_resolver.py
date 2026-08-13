from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from app.database import Database
from app.services.code_chunker import extract_python_code_chunks
from app.services.hierarchy_normalization import (
    HierarchyLimits,
    HierarchyResolver,
)


REVISION = "revision-hierarchy"


def candidate(chunk: dict, project_id: str, rank: int = 1) -> SimpleNamespace:
    identity = "|".join(
        str(value)
        for value in (
            project_id,
            chunk["repository_revision"],
            chunk["path"],
            chunk["start_line"],
            chunk["end_line"],
            chunk["content_hash"],
            chunk["id"],
        )
    )
    return SimpleNamespace(
        **{key: chunk[key] for key in (
            "project_id", "repository_revision", "language", "path", "chunk_type",
            "symbol_name", "qualified_name", "start_line", "end_line", "content",
            "content_hash",
        )},
        code_chunk_id=int(chunk["id"]),
        chunk_identity=identity,
        fused_score=1.0 / (60 + rank),
        fusion_rank=rank,
        source_records={},
        fusion_contributions={},
    )


def raw_chunk(
    qualified_name: str,
    start: int,
    end: int,
    *,
    chunk_type: str,
    parent_symbol: str = "",
) -> dict:
    content = f"# {qualified_name}\n"
    return {
        "repository_revision": REVISION,
        "language": "python",
        "path": "src/app.py",
        "chunk_type": chunk_type,
        "symbol_name": qualified_name.rsplit(".", 1)[-1],
        "qualified_name": qualified_name,
        "parent_symbol": parent_symbol,
        "start_line": start,
        "end_line": end,
        "content": content,
        "content_hash": hashlib.sha256(content.encode()).hexdigest(),
    }


class HierarchyResolverTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.directory.name) / "hierarchy.sqlite")
        self.project_id = self.database.create_project(
            {
                "repo_url": "https://github.com/demo/hierarchy",
                "owner": "demo",
                "repo": "hierarchy",
                "default_branch": "main",
                "repository_revision": REVISION,
            }
        )

    def tearDown(self):
        self.directory.cleanup()

    def _save_source(self, source: str, path: str = "src/app.py") -> dict[str, dict]:
        extracted = extract_python_code_chunks(path, source, REVISION)
        self.assertEqual(extracted.warnings, [])
        self.database.save_code_chunks_for_project(
            self.project_id, [item.to_dict() for item in extracted.chunks]
        )
        return {
            item["qualified_name"]: item
            for item in self.database.get_code_chunks(self.project_id)
        }

    def test_explicit_parent_metadata_and_bounded_ancestor_descendant_resolution(self):
        chunks = self._save_source(
            "class Outer:\n"
            "    def method(self):\n"
            "        def inner():\n"
            "            return 1\n"
            "        return inner()\n"
        )
        direct = candidate(chunks["Outer.method.inner"], self.project_id)

        result = HierarchyResolver(self.database).resolve([direct])

        links = {(item.relation_type, item.depth, item.candidate.qualified_name) for item in result.links}
        self.assertIn(("parent", 1, "Outer.method"), links)
        self.assertIn(("ancestor", 2, "Outer"), links)
        self.assertEqual(
            result.parent_by_child[direct.chunk_identity],
            next(
                identity
                for identity, item in result.metadata_by_identity.items()
                if item.qualified_name == "Outer.method"
            ),
        )
        self.assertTrue(all(item.authority == "explicit_structural_metadata" for item in result.links))

    def test_span_fallback_selects_unique_nearest_parent_and_direct_children(self):
        chunks = [
            raw_chunk("outer", 1, 20, chunk_type="function"),
            raw_chunk("middle", 2, 15, chunk_type="function"),
            raw_chunk("child", 3, 5, chunk_type="function"),
        ]
        self.database.save_code_chunks_for_project(self.project_id, chunks)
        stored = {item["qualified_name"]: item for item in self.database.get_code_chunks(self.project_id)}

        result = HierarchyResolver(self.database).resolve(
            [candidate(stored["middle"], self.project_id)]
        )
        relations = {(item.relation_type, item.candidate.qualified_name, item.authority) for item in result.links}

        self.assertIn(("parent", "outer", "span_inference"), relations)
        self.assertIn(("child", "child", "span_inference"), relations)
        self.assertNotIn(("child", "outer", "span_inference"), relations)

    def test_equal_nearest_parents_are_ambiguous_and_preserve_direct_candidate(self):
        chunks = [
            raw_chunk("ParentA", 1, 10, chunk_type="class"),
            raw_chunk("ParentB", 1, 10, chunk_type="function"),
            raw_chunk("child", 2, 3, chunk_type="function"),
        ]
        self.database.save_code_chunks_for_project(self.project_id, chunks)
        stored = {item["qualified_name"]: item for item in self.database.get_code_chunks(self.project_id)}
        direct = candidate(stored["child"], self.project_id)

        result = HierarchyResolver(self.database).resolve([direct])

        self.assertIn(direct.chunk_identity, result.ambiguous_identities)
        self.assertNotIn(direct.chunk_identity, result.parent_by_child)
        self.assertEqual(result.links, [])
        self.assertTrue(any("ambiguous" in warning.casefold() for warning in result.warnings))

    def test_explicit_parent_conflict_is_ambiguous_without_span_fallback(self):
        chunks = [
            raw_chunk("claimed", 20, 25, chunk_type="function"),
            raw_chunk("actual_container", 1, 10, chunk_type="function"),
            raw_chunk(
                "child",
                2,
                3,
                chunk_type="function",
                parent_symbol="claimed",
            ),
        ]
        self.database.save_code_chunks_for_project(self.project_id, chunks)
        stored = {
            item["qualified_name"]: item
            for item in self.database.get_code_chunks(self.project_id)
        }
        direct = candidate(stored["child"], self.project_id)

        result = HierarchyResolver(self.database).resolve([direct])

        self.assertIn(direct.chunk_identity, result.ambiguous_identities)
        self.assertNotIn(direct.chunk_identity, result.parent_by_child)
        self.assertFalse(
            any(item.candidate.qualified_name == "actual_container" for item in result.links)
        )

    def test_symbol_name_and_equal_content_do_not_guess_parent(self):
        same_content = "# copied\n"
        digest = hashlib.sha256(same_content.encode()).hexdigest()
        first = raw_chunk("parent.child", 1, 2, chunk_type="function")
        second = raw_chunk("parent", 10, 11, chunk_type="function")
        first.update(content=same_content, content_hash=digest)
        second.update(content=same_content, content_hash=digest)
        self.database.save_code_chunks_for_project(self.project_id, [first, second])
        stored = {
            item["qualified_name"]: item
            for item in self.database.get_code_chunks(self.project_id)
        }

        result = HierarchyResolver(self.database).resolve(
            [candidate(stored["parent.child"], self.project_id)]
        )

        self.assertEqual(result.parent_by_child, {})
        self.assertEqual(result.links, [])

    def test_depth_and_derived_budgets_are_hard_and_auditable(self):
        chunks = [
            raw_chunk("root", 1, 20, chunk_type="function"),
            raw_chunk("middle", 2, 15, chunk_type="function"),
            raw_chunk("child", 3, 5, chunk_type="function"),
        ]
        self.database.save_code_chunks_for_project(self.project_id, chunks)
        stored = {
            item["qualified_name"]: item
            for item in self.database.get_code_chunks(self.project_id)
        }
        direct = candidate(stored["child"], self.project_id)

        depth_one = HierarchyResolver(
            self.database,
            HierarchyLimits(max_depth=1),
        ).resolve([direct])
        self.assertEqual(
            {(item.relation_type, item.depth) for item in depth_one.links},
            {("parent", 1)},
        )

        no_derived = HierarchyResolver(
            self.database,
            HierarchyLimits(max_derived_candidates=0),
        ).resolve([direct])
        self.assertEqual(no_derived.links, [])
        self.assertTrue(no_derived.truncated)
        self.assertTrue(
            any("derived candidate budget" in warning.casefold() for warning in no_derived.warnings)
        )

    def test_project_revision_and_path_scopes_never_cross(self):
        first_path = raw_chunk("outer", 1, 10, chunk_type="function")
        child = raw_chunk("child", 2, 3, chunk_type="function")
        other_path = dict(
            raw_chunk("other_outer", 1, 20, chunk_type="function"),
            path="src/other.py",
        )
        other_revision = dict(
            raw_chunk("old_outer", 1, 30, chunk_type="function"),
            repository_revision="old-revision",
        )
        self.database.save_code_chunks_for_project(
            self.project_id,
            [first_path, child, other_path, other_revision],
        )
        stored = {
            (item["repository_revision"], item["path"], item["qualified_name"]): item
            for item in self.database.get_code_chunks(self.project_id)
        }
        direct = candidate(
            stored[(REVISION, "src/app.py", "child")],
            self.project_id,
        )

        result = HierarchyResolver(self.database).resolve([direct])

        names = {item.candidate.qualified_name for item in result.links}
        self.assertIn("outer", names)
        self.assertNotIn("other_outer", names)
        self.assertNotIn("old_outer", names)

    def test_truncated_path_does_not_claim_authoritative_parent(self):
        chunks = [
            raw_chunk(f"item{index}", index, 20 - index, chunk_type="function")
            for index in range(1, 6)
        ]
        self.database.save_code_chunks_for_project(self.project_id, chunks)
        direct_row = self.database.get_code_chunks(self.project_id)[-1]
        direct = candidate(direct_row, self.project_id)

        result = HierarchyResolver(
            self.database,
            HierarchyLimits(max_rows_per_path=2, max_total_rows=2),
        ).resolve([direct])

        self.assertTrue(result.truncated)
        self.assertEqual(result.links, [])
        self.assertTrue(any("truncated" in warning.casefold() for warning in result.warnings))

    def test_queries_are_scope_bound_limited_and_never_use_relation_graph(self):
        chunks = self._save_source("def outer():\n    def inner():\n        return 1\n    return inner\n")
        direct = candidate(chunks["outer.inner"], self.project_id)
        real = self.database.get_code_chunks_for_hierarchy

        with (
            patch.object(self.database, "get_code_chunks_for_hierarchy", wraps=real) as query,
            patch.object(self.database, "get_relations", side_effect=AssertionError("relation graph used")),
        ):
            HierarchyResolver(self.database).resolve([direct])

        query.assert_called_once_with(
            self.project_id,
            REVISION,
            "src/app.py",
            limit=129,
        )

    def test_unexpected_database_error_is_not_silently_reported_as_no_hierarchy(self):
        chunks = self._save_source("def target():\n    return 1\n")
        direct = candidate(chunks["target"], self.project_id)

        with patch.object(
            self.database,
            "get_code_chunks_for_hierarchy",
            side_effect=RuntimeError("database failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "database failed"):
                HierarchyResolver(self.database).resolve([direct])


if __name__ == "__main__":
    unittest.main()
