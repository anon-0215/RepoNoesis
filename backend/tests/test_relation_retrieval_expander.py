from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from app.database import Database
from app.services.hybrid_retriever import HybridSearchResult
from app.services.relation_retrieval import (
    RELATION_EXPANSION_VERSION,
    RELATION_GRAPH_VERSION,
    RelationExpansionLimits,
    RelationRetrievalExpander,
)
from tests.m3_helpers import call_chain_sources, make_relation_project


def hybrid(chunk: dict, rank: int = 1, source: str = "lexical") -> HybridSearchResult:
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
        retrieval_sources=[source],
        lexical_score=1.0 if source == "lexical" else None,
        lexical_rank=rank if source == "lexical" else None,
        semantic_score=None,
        semantic_rank=None,
        fusion_score=1.0 / (60 + rank) if source != "hierarchy" else 0.0,
        fusion_rank=rank if source != "hierarchy" else 0,
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


class RecordingDatabase(Database):
    def __init__(self, path):
        super().__init__(path)
        self.neighbor_calls = []
        self.chunk_calls = []

    def get_relation_neighbors_bounded(self, *args, **kwargs):
        self.neighbor_calls.append((args, dict(kwargs)))
        return super().get_relation_neighbors_bounded(*args, **kwargs)

    def get_code_chunks_by_ids_bounded(self, *args, **kwargs):
        self.chunk_calls.append((args, dict(kwargs)))
        return super().get_code_chunks_by_ids_bounded(*args, **kwargs)


class RelationRetrievalExpanderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.database = RecordingDatabase(Path(self.directory.name) / "expander.sqlite")
        self.project_id, _ = make_relation_project(self.database, call_chain_sources())
        self.revision = "revision-m3"
        self.chunks = {
            item["qualified_name"]: item
            for item in self.database.get_code_chunks(self.project_id)
        }

    def tearDown(self):
        self.directory.cleanup()

    def test_one_hop_outgoing_and_incoming_calls_use_real_edges_and_chunks(self):
        result = RelationRetrievalExpander(self.database).expand(
            project_id=self.project_id,
            repository_revision=self.revision,
            base_candidates=[hybrid(self.chunks["a"])],
            hierarchy_mode="off",
            hierarchy_audit=None,
        )
        names = {item.candidate.qualified_name for item in result.candidates}
        self.assertIn("b", names)
        self.assertIn("c", names)
        paths = [path for item in result.candidates for path in item.paths]
        self.assertTrue(all(path.depth == 1 for path in paths))
        self.assertEqual({path.direction for path in paths}, {"incoming", "outgoing"})
        self.assertTrue(all(path.edge_id.startswith("R") for path in paths))
        self.assertTrue(all(path.target_chunk_identity for path in paths))
        self.assertEqual(result.audit["expansion_version"], RELATION_EXPANSION_VERSION)
        self.assertEqual(result.audit["graph_version"], RELATION_GRAPH_VERSION)

    def test_queries_are_batch_scoped_ordered_limited_and_depth_is_one(self):
        limits = RelationExpansionLimits(max_relation_rows_total=4)
        result = RelationRetrievalExpander(self.database, limits=limits).expand(
            project_id=self.project_id,
            repository_revision=self.revision,
            base_candidates=[hybrid(self.chunks["a"]), hybrid(self.chunks["b"], 2)],
            hierarchy_mode="off",
            hierarchy_audit=None,
        )
        self.assertEqual(len(self.database.neighbor_calls), 1)
        args, kwargs = self.database.neighbor_calls[0]
        self.assertEqual(args[:2], (self.project_id, self.revision))
        self.assertEqual(kwargs["limit"], 5)
        self.assertEqual(kwargs["direction"], "both")
        self.assertLessEqual(result.audit["rows_inspected"], 4)
        self.assertEqual(result.audit["budgets"]["max_relation_depth"], 1)
        self.assertTrue(self.database.chunk_calls)
        self.assertTrue(all(call[1]["limit"] >= 1 for call in self.database.chunk_calls))

    def test_existing_direct_target_is_support_only_and_keeps_scores(self):
        direct_a = hybrid(self.chunks["a"])
        direct_b = hybrid(self.chunks["b"], 2)
        original = (direct_b.fusion_score, direct_b.fusion_rank, list(direct_b.retrieval_sources))
        result = RelationRetrievalExpander(self.database).expand(
            project_id=self.project_id,
            repository_revision=self.revision,
            base_candidates=[direct_a, direct_b],
            hierarchy_mode="off",
            hierarchy_audit=None,
        )
        self.assertNotIn("b", [item.candidate.qualified_name for item in result.candidates])
        self.assertIn(chunk_identity(direct_b), result.supporting_paths)
        self.assertEqual(
            (direct_b.fusion_score, direct_b.fusion_rank, direct_b.retrieval_sources),
            original,
        )

    def test_multiple_seeds_merge_target_paths_without_fake_source_hits(self):
        result = RelationRetrievalExpander(self.database).expand(
            project_id=self.project_id,
            repository_revision=self.revision,
            base_candidates=[hybrid(self.chunks["a"]), hybrid(self.chunks["c"], 2)],
            hierarchy_mode="off",
            hierarchy_audit=None,
        )
        by_name = {item.candidate.qualified_name: item for item in result.candidates}
        target = by_name["b"]
        self.assertEqual(target.candidate.retrieval_sources, ["relation"])
        self.assertIsNone(target.candidate.lexical_rank)
        self.assertIsNone(target.candidate.semantic_rank)
        self.assertEqual(target.candidate.fusion_score, 0.0)
        self.assertEqual(target.candidate.fusion_rank, 0)
        self.assertGreaterEqual(len(target.paths), 1)
        self.assertEqual(target.priority, max(path.path_priority for path in target.paths))

    def test_external_unresolved_file_nodes_and_stale_graph_never_create_candidates(self):
        no_index_database = Database(Path(self.directory.name) / "no-index.sqlite")
        project_id, _ = make_relation_project(
            no_index_database, call_chain_sources(), index_relations=False
        )
        chunk = next(
            item for item in no_index_database.get_code_chunks(project_id)
            if item["qualified_name"] == "a"
        )
        result = RelationRetrievalExpander(no_index_database).expand(
            project_id=project_id,
            repository_revision=self.revision,
            base_candidates=[hybrid(chunk)],
            hierarchy_mode="off",
            hierarchy_audit=None,
        )
        self.assertEqual(result.candidates, [])
        self.assertTrue(result.audit["controlled_unavailable"])
        self.assertTrue(result.warnings)

    def test_seed_rows_edge_order_and_repeated_runs_are_deterministic(self):
        seeds = [hybrid(self.chunks["a"]), hybrid(self.chunks["b"], 2)]
        first = RelationRetrievalExpander(self.database).expand(
            project_id=self.project_id,
            repository_revision=self.revision,
            base_candidates=seeds,
            hierarchy_mode="off",
            hierarchy_audit=None,
        )
        second = RelationRetrievalExpander(self.database).expand(
            project_id=self.project_id,
            repository_revision=self.revision,
            base_candidates=list(reversed(seeds)),
            hierarchy_mode="off",
            hierarchy_audit=None,
        )
        self.assertEqual(first.audit["relation_paths"], second.audit["relation_paths"])
        self.assertEqual(
            [(chunk_identity(item.candidate), item.priority) for item in first.candidates],
            [(chunk_identity(item.candidate), item.priority) for item in second.candidates],
        )


if __name__ == "__main__":
    unittest.main()
