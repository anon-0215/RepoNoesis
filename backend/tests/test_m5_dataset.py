from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from app.m5.dataset import BenchmarkDatasetValidator
from m5_helpers import (
    make_dataset,
    mocked_git,
    mutate_first_jsonl,
    mutate_json,
)


class M5DatasetTests(unittest.TestCase):
    def validate(self, mutate=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        dataset, repositories = make_dataset(Path(temporary.name))
        if mutate:
            mutate(dataset, repositories)
        with mocked_git():
            return BenchmarkDatasetValidator(dataset, repositories).validate()

    def assert_invalid(self, mutate, text):
        report = self.validate(mutate)
        self.assertFalse(report.valid)
        self.assertIn(text, " ".join(report.errors))

    def test_valid_offline_dataset(self):
        report = self.validate()
        self.assertTrue(report.valid, report.errors)
        self.assertEqual(report.scenario_count, 36)
        self.assertEqual(report.category_counts["relation"], 9)

    def test_invalid_schema_version(self):
        self.assert_invalid(
            lambda d, _: mutate_json(
                d / "manifest.json", lambda value: value.update(benchmark_schema_version=3)
            ),
            "dataset load failed",
        )

    def test_duplicate_scenario(self):
        def mutate(dataset, _):
            lines = (dataset / "scenarios.jsonl").read_text().splitlines()
            first = json.loads(lines[0]); second = json.loads(lines[1]); second["scenario_id"] = first["scenario_id"]
            lines[1] = json.dumps(second)
            (dataset / "scenarios.jsonl").write_text("\n".join(lines) + "\n")
        self.assert_invalid(mutate, "duplicate scenario_id")

    def test_duplicate_repo(self):
        def mutate(dataset, _):
            mutate_json(dataset / "repositories.json", lambda value: value[1].update(repo_id=value[0]["repo_id"]))
        self.assert_invalid(mutate, "duplicate repo_id")

    def test_invalid_sha(self):
        def mutate(dataset, _):
            mutate_json(dataset / "repositories.json", lambda value: value[0].update(exact_commit_sha="bad"))
        self.assert_invalid(mutate, "dataset load failed")

    def test_unknown_category(self):
        self.assert_invalid(
            lambda d, _: mutate_first_jsonl(d / "scenarios.jsonl", lambda value: value.update(category="unknown")),
            "dataset load failed",
        )

    def test_unknown_field(self):
        self.assert_invalid(
            lambda d, _: mutate_first_jsonl(d / "scenarios.jsonl", lambda value: value.update(skip_validator=True)),
            "dataset load failed",
        )

    def test_path_traversal(self):
        def mutate(dataset, _):
            mutate_first_jsonl(dataset / "scenarios.jsonl", lambda value: value["expected_files"].__setitem__(0, "../app.py"))
        self.assert_invalid(mutate, "path traversal")

    def test_missing_file(self):
        def mutate(dataset, _):
            def change(value):
                value["expected_files"] = ["missing.py"]
                value["allowed_evidence_scope"]["paths"] = ["missing.py"]
            mutate_first_jsonl(dataset / "scenarios.jsonl", change)
        self.assert_invalid(mutate, "expected file is missing")

    def test_invalid_symbol(self):
        self.assert_invalid(
            lambda d, _: mutate_first_jsonl(d / "scenarios.jsonl", lambda value: value.update(expected_symbols=["missing"])),
            "expected symbol is missing",
        )

    def test_span_out_of_range(self):
        def mutate(dataset, _):
            mutate_first_jsonl(dataset / "scenarios.jsonl", lambda value: value["expected_source_spans"][0].update(end_line=99))
        self.assert_invalid(mutate, "source span exceeds file")

    def test_content_hash_mismatch(self):
        def mutate(dataset, _):
            mutate_first_jsonl(dataset / "scenarios.jsonl", lambda value: value["expected_source_spans"][0].update(content_hash="0" * 64))
        self.assert_invalid(mutate, "content hash mismatch")

    def test_non_relation_span_must_cover_complete_ast_symbol(self):
        def mutate(dataset, _):
            def change(value):
                digest = hashlib.sha256(b"def target():\n").hexdigest()
                value["expected_source_spans"][0].update(end_line=1, content_hash=digest)
                value["expected_content_hashes"] = [digest]
            mutate_first_jsonl(dataset / "scenarios.jsonl", change)
        self.assert_invalid(mutate, "source span does not match AST symbol identity")

    def test_cross_revision_target(self):
        self.assert_invalid(
            lambda d, _: mutate_first_jsonl(d / "scenarios.jsonl", lambda value: value.update(repository_revision="f" * 40)),
            "cross-contamination",
        )

    def test_excessive_budget(self):
        self.assert_invalid(
            lambda d, _: mutate_first_jsonl(d / "scenarios.jsonl", lambda value: value.update(maximum_tool_calls=99)),
            "dataset load failed",
        )

    def test_unanswerable_cannot_declare_gold(self):
        def mutate(dataset, _):
            def change(value):
                value["unanswerable"] = True
                value["expected_target_type"] = "none"
            mutate_first_jsonl(dataset / "scenarios.jsonl", change)
        self.assert_invalid(mutate, "unanswerable scenario cannot declare source gold")

    def test_outer_function_cannot_claim_nested_wrapper_call(self):
        def mutate(dataset, _):
            def change(value):
                value["expected_relation_edges"][0]["source_symbol"] = "outer"
            lines = [json.loads(line) for line in (dataset / "scenarios.jsonl").read_text().splitlines()]
            relation = next(item for item in lines if item["category"] == "relation")
            change(relation)
            (dataset / "scenarios.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in lines), encoding="utf-8"
            )
        self.assert_invalid(mutate, "declared call relation is not present in source")

    def test_nested_wrapper_owns_its_direct_call(self):
        def mutate(dataset, _):
            lines = [json.loads(line) for line in (dataset / "scenarios.jsonl").read_text().splitlines()]
            relation = next(item for item in lines if item["category"] == "relation")
            relation["expected_relation_edges"][0]["source_symbol"] = "outer.new_func"
            (dataset / "scenarios.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in lines), encoding="utf-8"
            )
        report = self.validate(mutate)
        self.assertTrue(report.valid, report.errors)

    def test_placeholder_sequence_answer_is_rejected(self):
        self.assert_invalid(
            lambda d, _: mutate_first_jsonl(
                d / "sequences.jsonl",
                lambda value: value["steps"][0].update(answer_text="pass controlled benchmark answer"),
            ),
            "placeholder answer_text is forbidden",
        )

    def test_human_review_requires_complete_provenance(self):
        def mutate(dataset, _):
            def change(value):
                value.update(annotation_provenance="user_confirmed", annotation_status="human_reviewed")
            mutate_first_jsonl(dataset / "scenarios.jsonl", change)
        self.assert_invalid(mutate, "dataset load failed")


if __name__ == "__main__":
    unittest.main()
