from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.retrieval_phase5.contracts import FROZEN_PATHS, read_frozen_text
from app.retrieval_phase5.metrics import strict_gold_match
from app.retrieval_phase5.contracts import MATCHER_VERSION
from app.retrieval_phase6.contracts import (
    PHASE6_FORMAL_TOP_K,
    PHASE6_PATH_E_LABEL,
    Phase6ContractError,
    _adapt_httpx_query,
    load_phase6_benchmark,
)


ROOT = Path(__file__).resolve().parents[2]
PHASE6_SOURCE = ROOT / "benchmarks" / "retrieval_v2_phase6"
PHASE5_SOURCE = ROOT / "benchmarks" / "m5" / "datasets" / "pilot-v1"


def _copy_frozen(root: Path) -> tuple[Path, Path]:
    phase6 = root / "phase6"
    phase5 = root / "phase5"
    shutil.copytree(PHASE6_SOURCE, phase6)
    shutil.copytree(PHASE5_SOURCE, phase5)
    return phase6, phase5


def _rewrite_component(phase6: Path, name: str, value: object) -> None:
    if name.endswith(".jsonl"):
        payload = "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for item in value
        ).encode("utf-8")
    else:
        payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    (phase6 / name).write_bytes(payload)
    manifest_path = phase6 / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["file_hashes_before_manifest"][name] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_bytes((json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))


def _httpx_records(path: Path = PHASE6_SOURCE) -> list[dict]:
    return [json.loads(line) for line in read_frozen_text(path / "httpx_queries.jsonl").splitlines() if line.strip()]


class RetrievalPhase6ContractTests(unittest.TestCase):
    def test_frozen_cross_repo_benchmark_loads_without_rewriting_phase5_gold(self):
        root = Path(__file__).resolve().parents[2]
        snapshot = load_phase6_benchmark(
            root / "benchmarks" / "retrieval_v2_phase6",
            root / "benchmarks" / "m5" / "datasets" / "pilot-v1",
        )
        self.assertEqual(snapshot.repository_ids, ("click", "httpx"))
        self.assertEqual(snapshot.new_repository_ids, ("httpx",))
        self.assertEqual(
            {repo: (len(snapshot.scenarios_by_repo[repo]), len(snapshot.answerable_by_repo[repo])) for repo in snapshot.repository_ids},
            {"click": (12, 11), "httpx": (22, 20)},
        )
        self.assertEqual(snapshot.matcher_version, MATCHER_VERSION)
        self.assertEqual(
            snapshot.matcher_hash,
            "169b0515ffcdd889212f88cc51430ac8b706eb25817e7647e7b064b45a405cb6",
        )
        self.assertEqual(
            snapshot.stratum_counts("httpx"),
            {
                "direct_behavior_location": 6,
                "hierarchy_sensitive": 4,
                "relation_dependent": 6,
                "symbol_focused": 4,
                "unanswerable": 2,
            },
        )
        self.assertEqual(len(snapshot.query_ids), 34)
        self.assertEqual(len(set(snapshot.query_ids)), 34)
        self.assertEqual(snapshot.formal_top_k, PHASE6_FORMAL_TOP_K)
        self.assertEqual(snapshot.protocol["paths"]["E"]["label"], PHASE6_PATH_E_LABEL)
        self.assertEqual(
            next(item.label for item in FROZEN_PATHS if item.path_id == "E"),
            "v2 + hierarchy + relation",
        )
        self.assertEqual(snapshot.click.dataset_hash, "40bbc1f2c39a94e5c9d02e949c6fefdc5fe79c0e6b46ff0666448656c1a2f32f")
        self.assertEqual(snapshot.manifest["dataset_hash"], "f65d3544d520b3d71de151f15eb7963c1e2595f0872185885fb9d003b8d380b1")
        self.assertEqual(snapshot.manifest["frozen_text_hash_policy"], "utf8-lf-v1")
        self.assertEqual(snapshot.manifest["gold_hash"], "29b8ac2613bd33a644da09615e4d7a1f8a69e8e53314f7281137989fe4931181")
        self.assertEqual(snapshot.manifest["query_hash"], "ddf738c7dc5a6a5cb1666a65cb371d37d6bfb81e440ebf539627351f891e4b67")

    def test_relation_and_hierarchy_gold_survive_scenario_adaptation(self):
        root = Path(__file__).resolve().parents[2]
        snapshot = load_phase6_benchmark(
            root / "benchmarks" / "retrieval_v2_phase6",
            root / "benchmarks" / "m5" / "datasets" / "pilot-v1",
        )
        relation = next(
            item for item in snapshot.scenarios_by_repo["httpx"]
            if item.scenario_id == "httpx-phase6-relation-01"
        )
        hierarchy = next(
            item for item in snapshot.scenarios_by_repo["httpx"]
            if item.scenario_id == "httpx-phase6-hierarchy-01"
        )
        self.assertEqual(relation.expected_source_spans[0].qualified_symbol, "request")
        self.assertEqual(relation.expected_relation_edges[0].source_symbol, "get")
        self.assertEqual(relation.expected_relation_edges[0].target_symbol, "request")
        self.assertIn(".", hierarchy.expected_source_spans[0].qualified_symbol)

    def test_loader_reads_each_frozen_file_once(self):
        counts: dict[Path, int] = {}
        original = Path.read_bytes

        def counted(path: Path) -> bytes:
            resolved = path.resolve()
            counts[resolved] = counts.get(resolved, 0) + 1
            return original(path)

        with patch.object(Path, "read_bytes", new=counted):
            load_phase6_benchmark(PHASE6_SOURCE, PHASE5_SOURCE)
        expected = {
            (PHASE6_SOURCE / name).resolve()
            for name in (
                "manifest.json", "click_strata.json", "httpx_queries.jsonl", "matcher.json",
                "protocol.json", "repositories.json", "selection.json", "strata.json",
            )
        }
        expected.update((PHASE5_SOURCE / name).resolve() for name in ("manifest.json", "repositories.json", "scenarios.jsonl"))
        self.assertEqual({path: counts.get(path, 0) for path in expected}, {path: 1 for path in expected})

    def test_formal_top_k_mismatch_is_rejected_in_manifest_or_protocol(self):
        for location in ("manifest", "protocol"):
            with self.subTest(location=location), tempfile.TemporaryDirectory() as directory:
                phase6, phase5 = _copy_frozen(Path(directory))
                if location == "manifest":
                    path = phase6 / "manifest.json"
                    manifest = json.loads(path.read_bytes())
                    manifest["formal_top_k"] = 9
                    path.write_bytes((json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8"))
                else:
                    protocol = json.loads((phase6 / "protocol.json").read_bytes())
                    protocol["formal_top_k"] = 9
                    _rewrite_component(phase6, "protocol.json", protocol)
                with self.assertRaisesRegex(Phase6ContractError, "formal_top_k"):
                    load_phase6_benchmark(phase6, phase5)

    def test_frozen_text_policy_and_identity_cross_bindings_fail_closed(self):
        cases = (
            (
                "frozen_text_hash_policy",
                "sha256-raw-v1",
                "frozen text hash policy",
            ),
            (
                "click.dataset_hash",
                "1e658877c9b0aa9481700fa09868242b596b4a9c818db9e3159e192825155eb3",
                "Phase 5 Click benchmark identity",
            ),
            (
                "dataset_hash",
                "44291240a1874899adfc4978be7fbdda3fd2bfd1ef1bc8fe31cc523777a7643a",
                "Phase 6 dataset identity",
            ),
        )
        for key, value, error in cases:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                phase6, phase5 = _copy_frozen(Path(directory))
                path = phase6 / "manifest.json"
                manifest = json.loads(path.read_bytes())
                if key == "click.dataset_hash":
                    manifest["click"]["dataset_hash"] = value
                else:
                    manifest[key] = value
                path.write_bytes(
                    (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
                )
                with self.assertRaisesRegex(Phase6ContractError, error):
                    load_phase6_benchmark(phase6, phase5)

    def test_multi_gold_projects_complete_atomic_candidates(self):
        records = _httpx_records()
        primary = records[0]
        source = records[1]
        primary["acceptable_multi_gold"] = [{
            key: source[key]
            for key in (
                "gold_revision", "gold_path", "gold_qualified_symbol", "gold_span",
                "gold_content_hash", "gold_chunk_identity",
            )
        }]
        scenario = _adapt_httpx_query(primary)
        self.assertEqual(len(scenario.expected_source_spans), 2)

        def candidate(span):
            return {
                "validation_status": "valid",
                "repository_revision": scenario.repository_revision,
                "path": span.path,
                "qualified_name": span.qualified_symbol,
                "start_line": span.start_line,
                "end_line": span.end_line,
                "content_hash": span.content_hash,
            }

        first, second = scenario.expected_source_spans
        self.assertTrue(strict_gold_match(candidate(first), scenario)[0])
        self.assertTrue(strict_gold_match(candidate(second), scenario)[0])
        mixed = candidate(first)
        mixed["path"] = second.path
        self.assertFalse(strict_gold_match(mixed, scenario)[0])

    def test_repository_relative_paths_and_revision_are_loader_enforced(self):
        for invalid in ("/absolute.py", "C:/absolute.py", r"C:drive.py", r"\\server\share.py", "pkg/../escape.py"):
            with self.subTest(path=invalid), tempfile.TemporaryDirectory() as directory:
                phase6, phase5 = _copy_frozen(Path(directory))
                records = _httpx_records(phase6)
                records[0]["gold_path"] = invalid
                _rewrite_component(phase6, "httpx_queries.jsonl", records)
                with self.assertRaisesRegex(Phase6ContractError, "repository-relative|parent escape"):
                    load_phase6_benchmark(phase6, phase5)
        with tempfile.TemporaryDirectory() as directory:
            phase6, phase5 = _copy_frozen(Path(directory))
            records = _httpx_records(phase6)
            records[0]["gold_revision"] = "f" * 40
            _rewrite_component(phase6, "httpx_queries.jsonl", records)
            with self.assertRaisesRegex(Phase6ContractError, "revision"):
                load_phase6_benchmark(phase6, phase5)


if __name__ == "__main__":
    unittest.main()
