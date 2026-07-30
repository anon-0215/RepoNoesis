from __future__ import annotations

import copy
import json
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from app.m5.identity import identity_digest
from app.m5.live_dense_protocol import (
    DEFAULT_LIVE_DENSE_PROTOCOL_PATH,
    LiveDenseAcceptanceProtocol,
    LiveDenseProtocolError,
    evaluate_answerable_scenario,
    evaluate_overall_acceptance,
    evaluate_unanswerable_scenario,
    load_live_dense_protocol,
    partition_chunk_records,
    validate_protocol_repository_coverage,
    validate_stage_c_physical_noop,
    validate_stage_statistics,
    validate_stage_target,
)


REVISION = "a" * 40
CONTENT_IDENTITY = "sha256:" + "b" * 64
OTHER_CONTENT_IDENTITY = "sha256:" + "c" * 64


def _persistent(label: str, revision: str = REVISION) -> dict:
    return {
        "repository_revision": revision,
        "path": f"src/{label}.py",
        "chunk_type": "function",
        "qualified_name": f"module.{label}",
        "start_line": 1,
        "end_line": 3,
        "content_hash": identity_digest({"source": label}),
    }


def _record(
    label: str,
    *,
    repo_id: str = "synthetic",
    revision: str = REVISION,
    content_identity: str = CONTENT_IDENTITY,
) -> dict:
    persistent = _persistent(label, revision)
    return {
        "chunk_identity": f"chunk-sha256:{identity_digest(persistent)}",
        "repository_id": repo_id,
        "repository_revision": revision,
        "repository_content_identity": content_identity,
        "persistent_identity": persistent,
    }


def _trace(label: str = "target", **changes) -> dict:
    persistent = _persistent(label)
    value = {
        "repository_id": "synthetic",
        "repository_revision": REVISION,
        "repository_content_identity": CONTENT_IDENTITY,
        "path": persistent["path"],
        "qualified_symbol": persistent["qualified_name"],
        "start_line": persistent["start_line"],
        "end_line": persistent["end_line"],
        "content_hash": persistent["content_hash"],
        "chunk_identity": f"chunk-sha256:{identity_digest(persistent)}",
        "full_index_membership": True,
    }
    value.update(changes)
    return value


def _candidate(label: str = "target", **changes) -> dict:
    value = _trace(label, **changes)
    value["validation_status"] = "valid"
    return value


class M5LiveDenseProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_live_dense_protocol()

    def _partition(self, count: int, *, reverse: bool = False):
        records = [_record(f"item_{index}") for index in range(count)]
        if reverse:
            records.reverse()
        return partition_chunk_records(
            records,
            repository_id="synthetic",
            repository_revision=REVISION,
            repository_content_identity=CONTENT_IDENTITY,
        )

    def _mutated_protocol(self, mutate) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        raw = json.loads(DEFAULT_LIVE_DENSE_PROTOCOL_PATH.read_text(encoding="utf-8"))
        mutate(raw)
        path = Path(temporary.name) / "protocol.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        return path

    def test_authoritative_protocol_loads_without_network_or_torch(self):
        with patch.object(socket, "create_connection", side_effect=AssertionError("network attempted")):
            protocol = load_live_dense_protocol()
        self.assertEqual(protocol.protocol_id, "m5-live-dense-acceptance")
        self.assertEqual(protocol.protocol_version, 1)
        self.assertNotIn("torch", sys.modules)

    def test_schema_has_no_top_level_defaults_and_rejects_unknown_fields(self):
        schema = LiveDenseAcceptanceProtocol.model_json_schema()
        self.assertEqual(set(schema["required"]), set(LiveDenseAcceptanceProtocol.model_fields))
        exported_path = (
            DEFAULT_LIVE_DENSE_PROTOCOL_PATH.parents[1]
            / "schemas"
            / "live-dense-acceptance.schema.json"
        )
        self.assertEqual(
            json.loads(exported_path.read_text(encoding="utf-8")),
            schema,
        )
        raw = json.loads(DEFAULT_LIVE_DENSE_PROTOCOL_PATH.read_text(encoding="utf-8"))
        raw["unexpected"] = True
        with self.assertRaises(ValidationError):
            LiveDenseAcceptanceProtocol.model_validate(raw)

    def test_partition_counts_for_n_1_through_4(self):
        for count, expected in ((1, (1, 0)), (2, (1, 1)), (3, (2, 1)), (4, (2, 2))):
            with self.subTest(count=count):
                partition = self._partition(count)
                self.assertEqual((len(partition.stage_a), len(partition.stage_b)), expected)
                self.assertEqual(len(partition.full), count)

    def test_input_order_does_not_change_utf8_byte_order(self):
        forward = self._partition(4)
        reverse = self._partition(4, reverse=True)
        forward_ids = [item.chunk_identity for item in forward.ordered]
        self.assertEqual(forward_ids, [item.chunk_identity for item in reverse.ordered])
        self.assertEqual(forward_ids, sorted(forward_ids, key=lambda value: value.encode("utf-8")))
        self.assertEqual(self.protocol.stable_ordering.encoding, "utf-8-bytes")

    def test_partitions_are_disjoint_and_cover_full(self):
        partition = self._partition(4)
        a = {item.chunk_identity for item in partition.stage_a}
        b = {item.chunk_identity for item in partition.stage_b}
        self.assertFalse(a.intersection(b))
        self.assertEqual(a.union(b), {item.chunk_identity for item in partition.full})

    def test_duplicate_and_empty_stable_identity_are_rejected(self):
        duplicate = _record("same")
        with self.assertRaisesRegex(LiveDenseProtocolError, "globally unique"):
            partition_chunk_records(
                [duplicate, copy.deepcopy(duplicate)],
                repository_id="synthetic",
                repository_revision=REVISION,
                repository_content_identity=CONTENT_IDENTITY,
            )
        empty = _record("empty")
        empty["chunk_identity"] = ""
        with self.assertRaises(ValidationError):
            partition_chunk_records(
                [empty],
                repository_id="synthetic",
                repository_revision=REVISION,
                repository_content_identity=CONTENT_IDENTITY,
            )

    def test_cross_repository_revision_and_content_identity_are_rejected(self):
        variants = (
            _record("repo", repo_id="other"),
            _record("revision", revision="d" * 40),
            _record("content", content_identity=OTHER_CONTENT_IDENTITY),
        )
        for record in variants:
            with self.subTest(record=record["persistent_identity"]["path"]):
                with self.assertRaisesRegex(LiveDenseProtocolError, "crosses repository"):
                    partition_chunk_records(
                        [record],
                        repository_id="synthetic",
                        repository_revision=REVISION,
                        repository_content_identity=CONTENT_IDENTITY,
                    )

    def test_stable_identity_must_match_existing_inventory_contract(self):
        record = _record("identity")
        record["chunk_identity"] = "chunk-sha256:" + "f" * 64
        with self.assertRaisesRegex(LiveDenseProtocolError, "build_chunk_inventory"):
            partition_chunk_records(
                [record],
                repository_id="synthetic",
                repository_revision=REVISION,
                repository_content_identity=CONTENT_IDENTITY,
            )

    def test_stage_statistics_freeze_a_b_and_c(self):
        partition = self._partition(3)
        validate_stage_statistics(
            partition,
            stage="A",
            generated=2,
            cached=0,
            document_encode_calls=1,
            document_encode_batches=1,
            document_encode_items=2,
        )
        validate_stage_statistics(
            partition,
            stage="B",
            generated=1,
            cached=2,
            document_encode_calls=1,
            document_encode_batches=1,
            document_encode_items=1,
        )
        validate_stage_statistics(
            partition,
            stage="C",
            generated=0,
            cached=3,
            document_encode_calls=0,
            document_encode_batches=0,
            document_encode_items=0,
        )
        validate_stage_target(
            partition,
            stage="C",
            target_chunk_identities=[item.chunk_identity for item in partition.full],
        )
        with self.assertRaisesRegex(LiveDenseProtocolError, "frozen partition"):
            validate_stage_target(
                partition,
                stage="C",
                target_chunk_identities=[item.chunk_identity for item in partition.stage_a],
            )

    def test_stage_c_nonzero_encode_fails(self):
        with self.assertRaises(LiveDenseProtocolError):
            validate_stage_statistics(
                self._partition(1),
                stage="C",
                generated=0,
                cached=1,
                document_encode_calls=1,
                document_encode_batches=1,
                document_encode_items=1,
            )

    def test_stage_c_manifest_or_checkpoint_physical_change_fails(self):
        original = {
            "manifest": {"byte_length": 10, "sha256": "1" * 64, "mtime_ns": 100},
            "checkpoint": {"byte_length": 20, "sha256": "2" * 64, "mtime_ns": 200},
        }
        validate_stage_c_physical_noop(original, copy.deepcopy(original))
        for filename in ("manifest", "checkpoint"):
            for field in ("byte_length", "sha256", "mtime_ns"):
                changed = copy.deepcopy(original)
                changed[filename][field] = (
                    "3" * 64 if field == "sha256" else changed[filename][field] + 1
                )
                with self.subTest(filename=filename, field=field):
                    with self.assertRaises(LiveDenseProtocolError):
                        validate_stage_c_physical_noop(original, changed)

    def test_gold_at_ranks_one_two_and_three_passes(self):
        gold = _trace()
        miss = _candidate("miss")
        for rank in (1, 2, 3):
            candidates = [copy.deepcopy(miss) for _ in range(rank - 1)] + [_candidate()]
            result = evaluate_answerable_scenario(
                self.protocol,
                scenario_id=f"rank-{rank}",
                query_encode_count=1,
                gold=[gold],
                candidates=candidates,
            )
            self.assertTrue(result.passed)
            self.assertEqual(result.gold_rank, rank)

    def test_gold_after_top_three_or_missing_fails(self):
        misses = [_candidate(f"miss_{index}") for index in range(3)]
        rank_four = evaluate_answerable_scenario(
            self.protocol,
            scenario_id="rank-four",
            query_encode_count=1,
            gold=[_trace()],
            candidates=[*misses, _candidate()],
        )
        missing = evaluate_answerable_scenario(
            self.protocol,
            scenario_id="missing",
            query_encode_count=1,
            gold=[_trace()],
            candidates=misses,
        )
        self.assertFalse(rank_four.passed)
        self.assertIsNone(rank_four.gold_rank)
        self.assertFalse(missing.passed)

    def test_any_complete_gold_may_pass(self):
        result = evaluate_answerable_scenario(
            self.protocol,
            scenario_id="multi-gold",
            query_encode_count=1,
            gold=[_trace("first"), _trace("second")],
            candidates=[_candidate("second")],
        )
        self.assertTrue(result.passed)

    def test_partial_path_only_match_and_identity_mismatches_fail(self):
        mutations = (
            {"start_line": 2},
            {"content_hash": "e" * 64},
            {"repository_id": "other"},
            {"repository_revision": "d" * 40},
            {"repository_content_identity": OTHER_CONTENT_IDENTITY},
            {"qualified_symbol": "module.other"},
            {"full_index_membership": False},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                candidate = _candidate()
                candidate.update(mutation)
                if mutation == {"full_index_membership": False}:
                    with self.assertRaises(ValidationError):
                        evaluate_answerable_scenario(
                            self.protocol,
                            scenario_id="invalid-membership",
                            query_encode_count=1,
                            gold=[_trace()],
                            candidates=[candidate],
                        )
                    continue
                result = evaluate_answerable_scenario(
                    self.protocol,
                    scenario_id="identity-mismatch",
                    query_encode_count=1,
                    gold=[_trace()],
                    candidates=[candidate],
                )
                self.assertFalse(result.passed)

    def test_answerable_query_must_encode_exactly_once(self):
        for count in (0, 2):
            with self.subTest(count=count):
                with self.assertRaisesRegex(LiveDenseProtocolError, "exactly once"):
                    evaluate_answerable_scenario(
                        self.protocol,
                        scenario_id="query-count",
                        query_encode_count=count,
                        gold=[_trace()],
                        candidates=[_candidate()],
                    )

    def test_all_answerable_pass_is_required_and_unanswerable_is_excluded(self):
        passed = evaluate_answerable_scenario(
            self.protocol,
            scenario_id="pass",
            query_encode_count=1,
            gold=[_trace()],
            candidates=[_candidate()],
        )
        failed = evaluate_answerable_scenario(
            self.protocol,
            scenario_id="fail",
            query_encode_count=1,
            gold=[_trace()],
            candidates=[],
        )
        skipped = evaluate_unanswerable_scenario(
            self.protocol,
            scenario_id="skip",
            query_encode_count=0,
            gold=[],
            candidates=[],
            gold_rank=None,
        )
        success = evaluate_overall_acceptance(
            self.protocol,
            [passed, skipped],
            required_answerable_scenario_ids=["pass"],
        )
        failure = evaluate_overall_acceptance(
            self.protocol,
            [passed, failed, skipped],
            required_answerable_scenario_ids=["pass", "fail"],
        )
        self.assertTrue(success["passed"])
        self.assertEqual(success["answerable_pass_rate"], 1.0)
        self.assertFalse(failure["passed"])
        self.assertEqual(failure["answerable_pass_rate"], 0.5)
        self.assertEqual(failure["skipped_unanswerable_count"], 1)

    def test_overall_rejects_missing_or_duplicate_required_answerable_scenarios(self):
        passed = evaluate_answerable_scenario(
            self.protocol,
            scenario_id="pass",
            query_encode_count=1,
            gold=[_trace()],
            candidates=[_candidate()],
        )
        with self.assertRaisesRegex(LiveDenseProtocolError, "cover exactly"):
            evaluate_overall_acceptance(
                self.protocol,
                [passed],
                required_answerable_scenario_ids=["pass", "missing"],
            )
        with self.assertRaisesRegex(LiveDenseProtocolError, "nonempty and unique"):
            evaluate_overall_acceptance(
                self.protocol,
                [passed],
                required_answerable_scenario_ids=["pass", "pass"],
            )

    def test_unanswerable_cannot_produce_query_gold_candidate_or_rank(self):
        invalid = (
            {"query_encode_count": 1, "gold": [], "candidates": [], "gold_rank": None},
            {"query_encode_count": 0, "gold": [_trace()], "candidates": [], "gold_rank": None},
            {"query_encode_count": 0, "gold": [], "candidates": [_candidate()], "gold_rank": None},
            {"query_encode_count": 0, "gold": [], "candidates": [], "gold_rank": 1},
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(LiveDenseProtocolError):
                    evaluate_unanswerable_scenario(
                        self.protocol,
                        scenario_id="invalid-unanswerable",
                        **values,
                    )

    def test_fake_metrics_cannot_be_threshold_and_hit5_disclosure_is_required(self):
        self.assertFalse(self.protocol.reporting.fake_provider_metrics_can_set_acceptance_threshold)
        self.assertEqual(
            self.protocol.reporting.hit_at_5_disclosure,
            "computed-from-at-most-top-3-not-five-retrieved",
        )
        self.assertEqual(self.protocol.retrieval.top_k, 3)

    def test_missing_required_field_and_unknown_version_fail_closed(self):
        missing = self._mutated_protocol(lambda value: value.pop("retrieval"))
        unknown = self._mutated_protocol(lambda value: value.update(protocol_version=2))
        for path in (missing, unknown):
            with self.subTest(path=path):
                with self.assertRaises(LiveDenseProtocolError):
                    load_live_dense_protocol(path)

    def test_missing_protocol_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(LiveDenseProtocolError, "is missing"):
                load_live_dense_protocol(Path(directory) / "missing.json")

    def test_non_dense_wrong_topk_and_illegal_ordering_fail_closed(self):
        paths = (
            self._mutated_protocol(lambda value: value["retrieval"].update(mode="hybrid")),
            self._mutated_protocol(lambda value: value["retrieval"].update(top_k=5)),
            self._mutated_protocol(
                lambda value: value["stable_ordering"].update(manifest_field="stable_chunk_id")
            ),
        )
        for path in paths:
            with self.subTest(path=path):
                with self.assertRaises(LiveDenseProtocolError):
                    load_live_dense_protocol(path)

    def test_illegal_partition_formula_and_loose_overall_threshold_fail_closed(self):
        paths = (
            self._mutated_protocol(
                lambda value: value["partition"].update(a_count_formula="N // 2")
            ),
            self._mutated_protocol(
                lambda value: value["overall_acceptance"].update(answerable_pass_rate=0.8)
            ),
        )
        for path in paths:
            with self.subTest(path=path):
                with self.assertRaises(LiveDenseProtocolError):
                    load_live_dense_protocol(path)

    def test_repository_without_protocol_association_fails(self):
        validate_protocol_repository_coverage(
            self.protocol, [{"repo_id": "python-repo", "language": "python"}]
        )
        with self.assertRaisesRegex(LiveDenseProtocolError, "no associated"):
            validate_protocol_repository_coverage(
                self.protocol, [{"repo_id": "rust-repo", "language": "rust"}]
            )


if __name__ == "__main__":
    unittest.main()
