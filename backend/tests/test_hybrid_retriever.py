from types import SimpleNamespace
import tempfile
import unittest
from pathlib import Path

from app.database import Database
from app.services.hybrid_retriever import HybridRetriever, RRF_K
from app.services.semantic_retriever import SemanticSearchOutcome, SemanticSearchResult
from tests.m1_helpers import disabled_embedding_service, make_project


class FakeSemanticRetriever:
    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error

    def search(self, *_args, **_kwargs):
        if self.error:
            raise self.error
        return SemanticSearchOutcome(
            status="ok",
            results=self.results,
            model_name="fake",
            total_candidates=len(self.results),
        )


def semantic_result(chunk, project_id, score):
    return SemanticSearchResult(
        project_id=project_id,
        repository_revision=chunk["repository_revision"],
        code_chunk_id=chunk["id"],
        language=chunk["language"],
        path=chunk["path"],
        chunk_type=chunk["chunk_type"],
        symbol_name=chunk["symbol_name"],
        qualified_name=chunk["qualified_name"],
        start_line=chunk["start_line"],
        end_line=chunk["end_line"],
        content=chunk["content"],
        content_hash=chunk["content_hash"],
        semantic_score=score,
        model_name="fake",
    )


class HybridRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.directory.name) / "hybrid.sqlite")
        self.project_id, _bundle = make_project(
            self.db,
            [
                ("src/auth.py", "authenticate_user", "def authenticate_user():\n    return verify_password()\n"),
                ("src/upload.py", "upload_file", "def upload_file():\n    return save_blob()\n"),
                ("src/db.py", "initialize_database", "def initialize_database():\n    return create_tables()\n"),
            ],
        )
        self.chunks = {
            chunk["qualified_name"]: chunk
            for chunk in self.db.get_code_chunks(self.project_id)
        }
        self.enabled_service = SimpleNamespace(settings=SimpleNamespace(enabled=True))

    def tearDown(self):
        self.directory.cleanup()

    def test_weighted_rrf_and_duplicate_chunk_merge(self):
        semantic = FakeSemanticRetriever(
            [
                semantic_result(self.chunks["upload_file"], self.project_id, 0.9),
                semantic_result(self.chunks["authenticate_user"], self.project_id, 0.8),
            ]
        )
        outcome = HybridRetriever(
            self.db,
            self.enabled_service,
            semantic_retriever=semantic,
        ).search(self.project_id, "authenticate_user")

        auth = next(item for item in outcome.results if item.qualified_name == "authenticate_user")
        self.assertEqual(outcome.retrieval_mode, "hybrid")
        self.assertEqual(auth.retrieval_sources, ["lexical", "semantic"])
        self.assertAlmostEqual(
            auth.fusion_score,
            1 / (RRF_K + auth.lexical_rank) + 1 / (RRF_K + auth.semantic_rank),
        )
        self.assertEqual(
            len({item.code_chunk_id for item in outcome.results}),
            len(outcome.results),
        )

    def test_semantic_only_and_lexical_only_candidates_are_kept(self):
        semantic = FakeSemanticRetriever(
            [semantic_result(self.chunks["upload_file"], self.project_id, 0.9)]
        )
        outcome = HybridRetriever(
            self.db,
            self.enabled_service,
            semantic_retriever=semantic,
        ).search(self.project_id, "authenticate_user")
        sources = {item.qualified_name: item.retrieval_sources for item in outcome.results}
        self.assertEqual(sources["authenticate_user"], ["lexical"])
        self.assertEqual(sources["upload_file"], ["semantic"])

    def test_disabled_and_exception_fall_back_to_lexical(self):
        disabled = HybridRetriever(self.db, disabled_embedding_service()).search(
            self.project_id,
            "authenticate_user",
        )
        self.assertEqual(disabled.retrieval_mode, "lexical")
        self.assertIn("disabled", disabled.warnings[0].lower())

        failed = HybridRetriever(
            self.db,
            self.enabled_service,
            semantic_retriever=FakeSemanticRetriever(error=RuntimeError("boom")),
        ).search(self.project_id, "authenticate_user")
        self.assertEqual(failed.retrieval_mode, "lexical")
        self.assertIn("RuntimeError", failed.warnings[0])

    def test_wrong_project_or_revision_semantic_result_is_rejected(self):
        wrong = semantic_result(self.chunks["upload_file"], "other-project", 1.0)
        outcome = HybridRetriever(
            self.db,
            self.enabled_service,
            semantic_retriever=FakeSemanticRetriever([wrong]),
        ).search(self.project_id, "authenticate_user")
        self.assertEqual(outcome.retrieval_mode, "lexical")
        self.assertNotIn("upload_file", [item.qualified_name for item in outcome.results])
        self.assertTrue(any("different project" in warning for warning in outcome.warnings))

    def test_stable_ties_and_evidence_count_bounds(self):
        semantic = FakeSemanticRetriever(
            [
                semantic_result(self.chunks["upload_file"], self.project_id, 0.5),
                semantic_result(self.chunks["initialize_database"], self.project_id, 0.5),
            ]
        )
        retriever = HybridRetriever(
            self.db,
            self.enabled_service,
            semantic_retriever=semantic,
        )
        first = retriever.search(self.project_id, "unmatched", evidence_count=99)
        second = retriever.search(self.project_id, "unmatched", evidence_count=99)
        self.assertLessEqual(len(first.results), 8)
        self.assertEqual(
            [(item.path, item.fusion_rank) for item in first.results],
            [(item.path, item.fusion_rank) for item in second.results],
        )


if __name__ == "__main__":
    unittest.main()
