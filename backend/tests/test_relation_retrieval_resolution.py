from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from app.database import Database
from app.services.relation_retrieval import resolve_relation_node_to_chunk
from tests.m3_helpers import call_chain_sources, make_relation_project


class RelationNodeResolutionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.directory.name) / "resolution.sqlite")
        self.project_id, _ = make_relation_project(self.database, call_chain_sources())
        self.revision = "revision-m3"
        self.chunks = self.database.get_code_chunks(self.project_id)
        self.nodes = self.database.get_relation_nodes(self.project_id, self.revision)

    def tearDown(self):
        self.directory.cleanup()

    def test_authoritative_chunk_id_maps_to_one_exact_complete_chunk(self):
        node = next(item for item in self.nodes if item["qualified_name"] == "a")
        result = resolve_relation_node_to_chunk(
            node,
            self.chunks,
            project_id=self.project_id,
            repository_revision=self.revision,
        )
        self.assertEqual(result.status, "unique")
        self.assertEqual(result.chunk["id"], node["code_chunk_id"])
        self.assertEqual(result.chunk["content_hash"], node["content_hash"])
        self.assertEqual(result.chunk["path"], node["path"])

    def test_file_external_missing_ambiguous_stale_and_scope_conflicts_are_rejected(self):
        chunk_node = next(item for item in self.nodes if item["qualified_name"] == "a")
        file_node = next(item for item in self.nodes if item["node_type"] == "file")
        cases = []
        cases.append((file_node, self.chunks, "unsupported"))
        cases.append(({**chunk_node, "code_chunk_id": 999999}, self.chunks, "not_found"))
        cases.append(({**chunk_node, "project_id": "other"}, self.chunks, "scope_conflict"))
        cases.append(
            ({**chunk_node, "repository_revision": "old"}, self.chunks, "scope_conflict")
        )
        cases.append(({**chunk_node, "content_hash": "0" * 64}, self.chunks, "stale"))
        matching = next(
            item for item in self.chunks if item["id"] == chunk_node["code_chunk_id"]
        )
        cases.append((chunk_node, [matching, dict(matching)], "ambiguous"))
        for node, rows, expected in cases:
            with self.subTest(expected=expected):
                result = resolve_relation_node_to_chunk(
                    node,
                    rows,
                    project_id=self.project_id,
                    repository_revision=self.revision,
                )
                self.assertEqual(result.status, expected)
                self.assertIsNone(result.chunk)

    def test_same_symbol_or_same_span_never_substitutes_for_chunk_identity(self):
        node = next(item for item in self.nodes if item["qualified_name"] == "a")
        exact = next(item for item in self.chunks if item["id"] == node["code_chunk_id"])
        impostors = [
            {**exact, "id": exact["id"] + 1000},
            {**exact, "id": exact["id"] + 1001, "path": "pkg/other.py"},
        ]
        result = resolve_relation_node_to_chunk(
            node,
            impostors,
            project_id=self.project_id,
            repository_revision=self.revision,
        )
        self.assertEqual(result.status, "not_found")
        self.assertIsNone(result.chunk)


if __name__ == "__main__":
    unittest.main()
