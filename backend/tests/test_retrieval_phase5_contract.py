from __future__ import annotations

import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from app.m5.embedding import FakeEmbeddingBackend
from app.services.embedding_service import EffectiveEmbeddingIdentity
from app.retrieval_phase5.contracts import (
    BGE_M3_SNAPSHOT_REVISION,
    FORMAL_TOP_K,
    FROZEN_TEXT_HASH_POLICY,
    FROZEN_PATHS,
    ManifestError,
    build_frozen_manifest,
    ensure_formal_embedding_identity,
    file_hash,
    frozen_text_hash,
    immutable_write_json,
    manifest_run_identity,
    read_frozen_text,
)
from app.retrieval_phase5.runner import load_click_benchmark


def _real_identity() -> EffectiveEmbeddingIdentity:
    return EffectiveEmbeddingIdentity(
        identity_schema_version="embedding-effective-v1",
        provider="sentence-transformers",
        backend_type="sentence-transformers",
        model_name="BAAI/bge-m3",
        configured_revision="a" * 40,
        resolved_revision="a" * 40,
        local_snapshot_identity="snapshot-sha256:" + "b" * 64,
        backend_model_identity="model-sha256:" + "c" * 64,
        model_identity="embedding-sha256:" + "d" * 64,
        dimension=1024,
        normalized=True,
        text_format_version="code-chunk-v1",
        document_prefix_identity="text-sha256:" + "e" * 64,
        query_prefix_identity="text-sha256:" + "e" * 64,
        max_length=8192,
        batch_size=8,
        pooling_identity="text-sha256:" + "f" * 64,
        embedding_config_hash="1" * 64,
        device="cuda",
        dtype="float32",
        cache_identity="2" * 64,
        is_real=True,
    )


class RetrievalPhase5ContractTests(unittest.TestCase):
    def test_frozen_text_transport_normalizes_newlines_but_raw_hash_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lf = root / "lf.json"
            crlf = root / "crlf.json"
            lone_cr = root / "lone-cr.json"
            lf.write_bytes(b'{"value":"x"}\n')
            crlf.write_bytes(b'{"value":"x"}\r\n')
            lone_cr.write_bytes(b'{"value":"x"}\r')

            expected = hashlib.sha256(b'{"value":"x"}\n').hexdigest()
            self.assertEqual(frozen_text_hash(read_frozen_text(lf)), expected)
            self.assertEqual(frozen_text_hash(read_frozen_text(crlf)), expected)
            self.assertEqual(frozen_text_hash(read_frozen_text(lone_cr)), expected)
            self.assertNotEqual(file_hash(lf), file_hash(crlf))
            self.assertEqual(file_hash(lf), hashlib.sha256(lf.read_bytes()).hexdigest())

    def test_frozen_text_rejects_bom_and_invalid_utf8(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bom = root / "bom.json"
            invalid = root / "invalid.json"
            bom.write_bytes(b"\xef\xbb\xbf{}\n")
            invalid.write_bytes(b"{\xff}\n")
            for path in (bom, invalid):
                with self.subTest(path=path.name), self.assertRaises(ManifestError):
                    read_frozen_text(path)

    def test_click_identity_is_stable_for_lf_and_crlf_transport(self):
        repository_root = Path(__file__).resolve().parents[2]
        source = repository_root / "benchmarks" / "m5" / "datasets" / "pilot-v1"
        expected = load_click_benchmark(source)
        self.assertEqual(
            expected.dataset_hash,
            "40bbc1f2c39a94e5c9d02e949c6fefdc5fe79c0e6b46ff0666448656c1a2f32f",
        )
        self.assertNotEqual(expected.dataset_hash, "1e658877c9b0aa9481700fa09868242b596b4a9c818db9e3159e192825155eb3")
        self.assertEqual(next(item.label for item in FROZEN_PATHS if item.path_id == "E"), "v2 + hierarchy + relation")
        with tempfile.TemporaryDirectory() as directory:
            lf_copy = Path(directory) / "lf"
            crlf_copy = Path(directory) / "crlf"
            lf_copy.mkdir()
            crlf_copy.mkdir()
            for name in ("manifest.json", "repositories.json", "scenarios.jsonl"):
                normalized = read_frozen_text(source / name)
                (lf_copy / name).write_bytes(normalized)
                (crlf_copy / name).write_bytes(normalized.replace(b"\n", b"\r\n"))
            for copy in (lf_copy, crlf_copy):
                with self.subTest(transport=copy.name):
                    transported = load_click_benchmark(copy)
                    self.assertEqual(
                        (transported.dataset_hash, transported.query_hash, transported.gold_hash),
                        (expected.dataset_hash, expected.query_hash, expected.gold_hash),
                    )

    def test_five_paths_are_exactly_frozen(self):
        self.assertEqual(
            [(item.path_id, item.request_parameters) for item in FROZEN_PATHS],
            [
                ("A", {"retrieval_version": "v1", "hierarchy_mode": "off", "relation_mode": "off"}),
                ("B", {"retrieval_version": "v2", "hierarchy_mode": "off", "relation_mode": "off"}),
                ("C", {"retrieval_version": "v2", "hierarchy_mode": "normalize_v1", "relation_mode": "off"}),
                ("D", {"retrieval_version": "v2", "hierarchy_mode": "off", "relation_mode": "expand_v1"}),
                ("E", {"retrieval_version": "v2", "hierarchy_mode": "normalize_v1", "relation_mode": "expand_v1"}),
            ],
        )
        self.assertEqual(FORMAL_TOP_K, 8)

    def test_formal_identity_rejects_fake_injected_and_silent_fallback(self):
        real = _real_identity()
        ensure_formal_embedding_identity(real, backend=object())
        local = replace(
            real,
            model_name=f"local:{BGE_M3_SNAPSHOT_REVISION}",
            configured_revision=BGE_M3_SNAPSHOT_REVISION,
            resolved_revision=BGE_M3_SNAPSHOT_REVISION,
        )
        ensure_formal_embedding_identity(local, backend=object())
        for invalid in (
            replace(real, is_real=False),
            replace(real, provider="fake-bge-m3"),
            replace(real, backend_type="fake"),
            replace(real, dimension=16),
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ManifestError):
                ensure_formal_embedding_identity(invalid, backend=object())
        with self.assertRaises(ManifestError):
            ensure_formal_embedding_identity(real, backend=FakeEmbeddingBackend())

    def test_manifest_has_required_hashes_versions_and_shared_contract(self):
        manifest = build_frozen_manifest(
            repository_commit="1" * 40,
            branch="v3-agent-development",
            corpus_project_id="project-click",
            corpus_repository_revision="2" * 40,
            corpus_database_hash="3" * 64,
            relation_graph_hash="4" * 64,
            dataset_hash="5" * 64,
            query_hash="6" * 64,
            gold_hash="7" * 64,
            matcher_hash="8" * 64,
            query_count=12,
            answerable_query_count=11,
            embedding_identity=_real_identity(),
            model_local_path="<local-bge-m3-snapshot>",
            cache_namespace="retrieval-v2-phase5-test",
            environment={"python_version": "3.12", "torch_version": "2.12"},
            source_files={"query_file": "scenarios.jsonl", "gold_file": "scenarios.jsonl"},
            timestamp="2026-08-01T00:00:00+00:00",
        )
        required = {
            "evaluation_version", "timestamp", "repository_commit", "branch",
            "corpus_project_id", "corpus_repository_revision", "dataset_name",
            "dataset_version", "dataset_hash", "query_file", "query_hash",
            "gold_file", "gold_hash", "matcher_name", "matcher_version",
            "matcher_hash", "query_count", "answerable_query_count",
            "gold_granularity", "retrieval_paths", "request_parameters",
            "top_k_values", "metric_definitions", "embedding_model",
            "embedding_revision", "embedding_local_path", "embedding_dimension",
            "pooling", "normalize", "query_prefix", "document_prefix",
            "max_length", "precision", "device", "batch_size", "cache_namespace",
            "random_seed", "python_version", "torch_version", "cuda_version",
            "gpu_name", "relation_graph_version", "relation_expansion_version",
            "relation_selection_version", "relation_priority_version",
            "relation_whitelist_version", "hierarchy_normalization_version",
            "weighted_rrf_version", "corpus_database_hash", "relation_graph_hash",
        }
        self.assertTrue(required.issubset(manifest))
        self.assertEqual(manifest["query_count"], 12)
        self.assertEqual(manifest["answerable_query_count"], 11)
        self.assertEqual(manifest["frozen_text_hash_policy"], FROZEN_TEXT_HASH_POLICY)

    def test_manifest_change_changes_run_identity_and_existing_output_is_not_overwritten(self):
        base = {"evaluation_version": "retrieval-v2-phase5@1", "dataset_hash": "a" * 64}
        self.assertNotEqual(
            manifest_run_identity(base),
            manifest_run_identity({**base, "dataset_hash": "b" * 64}),
        )
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "manifest.json"
            immutable_write_json(target, base)
            with self.assertRaises(FileExistsError):
                immutable_write_json(target, base)


if __name__ == "__main__":
    unittest.main()
