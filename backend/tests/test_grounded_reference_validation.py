from __future__ import annotations

import unittest

from app.services.evidence import Evidence
from app.services.qa_agent import _validate_grounded_answer_references


def _evidence(
    evidence_id: str,
    path: str,
    start_line: int,
    end_line: int,
    *,
    retrieval_sources: list[str] | None = None,
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        project_id="project-1",
        repository_id="demo/repository",
        repository_url="https://example.invalid/demo/repository",
        repository_revision="revision-1",
        path=path,
        language="python",
        code_chunk_id=int(evidence_id[1:]),
        chunk_identity=f"identity-{evidence_id}",
        chunk_type="function",
        symbol_name=f"symbol_{evidence_id}",
        qualified_name=f"module.symbol_{evidence_id}",
        start_line=start_line,
        end_line=end_line,
        content_hash=f"hash-{evidence_id}",
        excerpt="def example():\n    return True\n",
        retrieval_sources=list(retrieval_sources or ["lexical"]),
        lexical_score=1.0,
        lexical_rank=1,
        semantic_score=None,
        semantic_rank=None,
        fusion_score=1.0,
        fusion_rank=1,
        selection_reason="offline validation fixture",
        validation_status="valid",
    )


class GroundedReferenceValidationTests(unittest.TestCase):
    def setUp(self):
        self.auth = _evidence("E1", "src/auth.py", 10, 20)
        self.upload = _evidence("E2", "src/upload.py", 30, 40)

    def test_correct_id_and_exact_relative_location_are_accepted(self):
        self.assertEqual(
            _validate_grounded_answer_references(
                "Grounded fact [E1] src/auth.py:10-20.", [self.auth]
            ),
            (True, None, 1),
        )

    def test_missing_exact_location_has_specific_reason(self):
        self.assertEqual(
            _validate_grounded_answer_references("Grounded fact [E1].", [self.auth]),
            (False, "citation_location_missing", 1),
        )

    def test_wrong_path_has_specific_reason(self):
        self.assertEqual(
            _validate_grounded_answer_references(
                "Grounded fact [E1] src/missing.py:10-20.", [self.auth]
            ),
            (False, "citation_path_mismatch", 1),
        )

    def test_wrong_line_range_has_specific_reason(self):
        self.assertEqual(
            _validate_grounded_answer_references(
                "Grounded fact [E1] src/auth.py:11-20.", [self.auth]
            ),
            (False, "citation_line_range_mismatch", 1),
        )

    def test_windows_separator_is_rejected_while_posix_separator_is_accepted(self):
        accepted = _validate_grounded_answer_references(
            "Grounded fact [E1] src/auth.py:10-20.", [self.auth]
        )
        rejected = _validate_grounded_answer_references(
            r"Grounded fact [E1] src\auth.py:10-20.", [self.auth]
        )

        self.assertEqual(accepted, (True, None, 1))
        self.assertEqual(rejected, (False, "citation_path_mismatch", 1))

    def test_absolute_path_is_rejected_while_repository_relative_path_is_accepted(self):
        accepted = _validate_grounded_answer_references(
            "Grounded fact [E1] src/auth.py:10-20.", [self.auth]
        )
        rejected = _validate_grounded_answer_references(
            "Grounded fact [E1] /repo/src/auth.py:10-20.", [self.auth]
        )

        self.assertEqual(accepted, (True, None, 1))
        self.assertEqual(rejected, (False, "citation_path_mismatch", 1))

    def test_location_from_another_evidence_does_not_bind_to_cited_id(self):
        self.assertEqual(
            _validate_grounded_answer_references(
                "Grounded fact [E1] src/upload.py:30-40.",
                [self.auth, self.upload],
            ),
            (False, "citation_evidence_binding_failed", 1),
        )

    def test_relation_expanded_evidence_uses_the_current_id_location_mapping(self):
        relation_evidence = _evidence(
            "E2",
            "src/related.py",
            50,
            60,
            retrieval_sources=["relation"],
        )

        self.assertEqual(
            _validate_grounded_answer_references(
                "Related fact [E2] src/related.py:50-60.",
                [self.auth, relation_evidence],
            ),
            (True, None, 1),
        )


if __name__ == "__main__":
    unittest.main()
