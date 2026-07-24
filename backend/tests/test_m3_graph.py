from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from app.database import Database
from app.services.relation_graph import (
    EvidenceChainStore,
    RelationGraphService,
    RelationTraversalLimits,
    RelationValidator,
)
from tests.m3_helpers import call_chain_sources, make_relation_project


class M3GraphTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.directory.name) / "graph.sqlite")
        self.project_id, self.bundle = make_relation_project(
            self.db, call_chain_sources()
        )
        self.nodes = self.db.get_relation_nodes(
            self.project_id, "revision-m3"
        )
        self.by_symbol = {
            item["qualified_name"]: item
            for item in self.nodes
            if item["code_chunk_id"] is not None
        }
        self.service = RelationGraphService(self.db)

    def tearDown(self):
        self.directory.cleanup()

    def _expand(self, **kwargs):
        return self.service.expand(
            project_id=self.project_id,
            repository_revision="revision-m3",
            seed_node_ids=[self.by_symbol["a"]["node_id"]],
            relation_types=["calls"],
            direction=kwargs.pop("direction", "outbound"),
            max_depth=kwargs.pop("max_depth", 2),
            limits=kwargs.pop("limits", RelationTraversalLimits()),
            **kwargs,
        )

    def test_stable_bfs_one_and_two_hop(self):
        one = self._expand(max_depth=1)
        two = self._expand(max_depth=2)
        self.assertIn("b", [item["qualified_name"] for item in one.nodes])
        self.assertNotIn("c", [item["qualified_name"] for item in one.nodes])
        self.assertIn("c", [item["qualified_name"] for item in two.nodes])
        again = self._expand(max_depth=2)
        self.assertEqual(
            [item["edge_id"] for item in two.edges],
            [item["edge_id"] for item in again.edges],
        )

    def test_inbound_both_cycle_and_dedup_terminate(self):
        inbound = self.service.expand(
            project_id=self.project_id,
            repository_revision="revision-m3",
            seed_node_ids=[self.by_symbol["a"]["node_id"]],
            relation_types=["calls"],
            direction="inbound",
            max_depth=2,
            limits=RelationTraversalLimits(),
        )
        self.assertIn("c", [item["qualified_name"] for item in inbound.nodes])
        both = self.service.expand(
            project_id=self.project_id,
            repository_revision="revision-m3",
            seed_node_ids=[self.by_symbol["a"]["node_id"]],
            relation_types=["calls"],
            direction="both",
            max_depth=2,
            limits=RelationTraversalLimits(),
        )
        self.assertLessEqual(len(both.nodes), 3)
        self.assertEqual(len(both.edges), len({item["edge_id"] for item in both.edges}))

    def test_depth_neighbor_node_edge_and_path_limits_truncate(self):
        result = self._expand(
            max_depth=2,
            limits=RelationTraversalLimits(
                max_depth=1,
                per_node_limit=1,
                max_nodes=2,
                max_edges=1,
                max_paths=1,
                max_output_bytes=1024,
            ),
        )
        self.assertLessEqual(len(result.nodes), 2)
        self.assertLessEqual(len(result.edges), 1)
        self.assertLessEqual(len(result.paths), 1)
        self.assertTrue(result.truncated or len(result.edges) == 1)

    def test_invalid_seed_revision_direction_and_empty_result(self):
        with self.assertRaises(ValueError):
            self._expand(direction="sideways")
        with self.assertRaises(ValueError):
            self.service.expand(
                project_id=self.project_id,
                repository_revision="other",
                seed_node_ids=[self.by_symbol["a"]["node_id"]],
                relation_types=["calls"],
                direction="outbound",
                max_depth=1,
                limits=RelationTraversalLimits(),
            )
        empty = self.service.expand(
            project_id=self.project_id,
            repository_revision="revision-m3",
            seed_node_ids=[self.by_symbol["a"]["node_id"]],
            relation_types=["references"],
            direction="inbound",
            max_depth=1,
            limits=RelationTraversalLimits(),
        )
        self.assertEqual(empty.edges, [])

    def test_relation_validator_rejects_deleted_edge_and_other_request(self):
        result = self._expand(max_depth=1)
        path = next(path for path in result.paths if path.resolution_status == "resolved")
        store = EvidenceChainStore()
        chain = store.add(
            owner_id="request-1",
            project_id=self.project_id,
            repository_revision="revision-m3",
            seed_evidence_ids=[],
            supporting_evidence_ids=[],
            path=path,
            edges_by_id={item["edge_id"]: item for item in result.edges},
            truncated=False,
            warnings=[],
        )
        validator = RelationValidator(self.db)
        valid, _warnings = validator.validate_chains(
            owner_id="request-1",
            project_id=self.project_id,
            repository_revision="revision-m3",
            chains=[chain],
            valid_evidence_ids=set(),
        )
        self.assertEqual(len(valid), 1)
        foreign, _warnings = validator.validate_chains(
            owner_id="request-2",
            project_id=self.project_id,
            repository_revision="revision-m3",
            chains=[chain],
            valid_evidence_ids=set(),
        )
        self.assertEqual(foreign, [])
        original_chain_id = chain.chain_id
        chain.chain_id = "C" + "0" * 64
        forged, _warnings = validator.validate_chains(
            owner_id="request-1",
            project_id=self.project_id,
            repository_revision="revision-m3",
            chains=[chain],
            valid_evidence_ids=set(),
        )
        self.assertEqual(forged, [])
        chain.chain_id = original_chain_id
        with self.db.connect() as conn:
            conn.execute(
                "DELETE FROM code_relations WHERE edge_id = ?",
                (path.edge_ids[0],),
            )
        stale, _warnings = validator.validate_chains(
            owner_id="request-1",
            project_id=self.project_id,
            repository_revision="revision-m3",
            chains=[chain],
            valid_evidence_ids=set(),
        )
        self.assertEqual(stale, [])


if __name__ == "__main__":
    unittest.main()
