import tempfile
import unittest
from pathlib import Path

from app.database import Database
from app.services.evidence import CitationValidator, EvidenceBuilder
from app.services.hybrid_retriever import HybridRetriever
from tests.m1_helpers import disabled_embedding_service, make_project


class EvidenceTests(unittest.TestCase):
    def _fixture(self, excerpt_limit=2000):
        directory = tempfile.TemporaryDirectory()
        db = Database(Path(directory.name) / "evidence.sqlite")
        project_id, bundle = make_project(
            db,
            [
                (
                    "src/auth.py",
                    "authenticate_user",
                    "def authenticate_user(password):\n    return verify_password(password)\n",
                )
            ],
        )
        outcome = HybridRetriever(db, disabled_embedding_service()).search(
            project_id,
            "authenticate_user",
        )
        evidence = EvidenceBuilder(excerpt_limit).build(
            outcome.results,
            bundle["project"],
        )[0]
        return directory, db, evidence

    def test_builder_serializes_source_scores_and_truncates_excerpt(self):
        directory, db, evidence = self._fixture(excerpt_limit=12)
        self.addCleanup(directory.cleanup)
        data = evidence.to_dict()
        self.assertEqual(len(data["excerpt"]), 12)
        self.assertEqual(data["retrieval_sources"], ["lexical"])
        self.assertEqual(data["lexical_rank"], 1)
        self.assertIsNone(data["semantic_score"])
        self.assertEqual(len(data["content_hash"]), 64)
        self.assertIn("fusion rank 1", data["selection_reason"])
        self.assertEqual(CitationValidator(db).validate(evidence).validation_status, "valid")

    def test_valid_evidence(self):
        directory, db, evidence = self._fixture()
        self.addCleanup(directory.cleanup)
        validated = CitationValidator(db).validate(evidence)
        self.assertEqual(validated.validation_status, "valid")
        self.assertIsNone(validated.invalid_reason)

    def test_missing_file_is_rejected_without_source_in_warning(self):
        directory, db, evidence = self._fixture()
        self.addCleanup(directory.cleanup)
        with db.connect() as conn:
            conn.execute(
                "DELETE FROM repo_files WHERE project_id = ?",
                (evidence.project_id,),
            )
        valid, warnings = CitationValidator(db).validate_all([evidence])
        self.assertEqual(valid, [])
        self.assertIn("no longer exists", warnings[0])
        self.assertNotIn("verify_password", warnings[0])
        self.assertEqual(evidence.excerpt, "")

    def test_unsafe_paths_are_rejected(self):
        for unsafe in ("../auth.py", "/src/auth.py", r"C:\src\auth.py"):
            with self.subTest(path=unsafe):
                directory, db, evidence = self._fixture()
                try:
                    evidence.path = unsafe
                    validated = CitationValidator(db).validate(evidence)
                    self.assertEqual(validated.invalid_reason, "unsafe repository path")
                finally:
                    directory.cleanup()

    def test_line_range_and_chunk_identity_are_rejected(self):
        cases = [
            ("start_line", 0, "invalid line range"),
            ("end_line", 999, "line range exceeds stored source"),
            ("chunk_identity", "wrong", "chunk identity mismatch"),
            ("qualified_name", "other", "qualified symbol identity mismatch"),
        ]
        for field, value, reason in cases:
            with self.subTest(field=field):
                directory, db, evidence = self._fixture()
                try:
                    setattr(evidence, field, value)
                    self.assertEqual(
                        CitationValidator(db).validate(evidence).invalid_reason,
                        reason,
                    )
                finally:
                    directory.cleanup()

    def test_hash_revision_and_repository_mismatch_are_rejected(self):
        cases = [
            ("content_hash", "0" * 64, "content hash mismatch"),
            ("repository_revision", "old", "repository revision mismatch"),
            ("repository_id", "other/repo", "repository identity mismatch"),
            ("project_id", "other", "source file or code chunk no longer exists"),
        ]
        for field, value, reason in cases:
            with self.subTest(field=field):
                directory, db, evidence = self._fixture()
                try:
                    setattr(evidence, field, value)
                    self.assertEqual(
                        CitationValidator(db).validate(evidence).invalid_reason,
                        reason,
                    )
                finally:
                    directory.cleanup()

    def test_changed_source_and_excerpt_mismatch_are_rejected(self):
        directory, db, evidence = self._fixture()
        self.addCleanup(directory.cleanup)
        with db.connect() as conn:
            conn.execute(
                "UPDATE repo_files SET content = ? WHERE project_id = ?",
                ("def changed():\n    return False\n", evidence.project_id),
            )
        self.assertEqual(
            CitationValidator(db).validate(evidence).invalid_reason,
            "stored source no longer matches code chunk",
        )

        directory2, db2, evidence2 = self._fixture()
        self.addCleanup(directory2.cleanup)
        evidence2.excerpt = "not in source"
        self.assertEqual(
            CitationValidator(db2).validate(evidence2).invalid_reason,
            "excerpt mismatch",
        )


if __name__ == "__main__":
    unittest.main()
