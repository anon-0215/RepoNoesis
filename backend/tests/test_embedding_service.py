import math
import struct
import tempfile
import unittest
from pathlib import Path

from app.config import EmbeddingSettings
from app.services.embedding_service import (
    CODE_CHUNK_TEXT_FORMAT_VERSION,
    EmbeddingConfigurationError,
    EmbeddingEncodeError,
    EmbeddingService,
    build_code_chunk_embedding_input_hash,
    build_code_chunk_document_text,
    build_embedding_config_hash,
    build_embedding_input_hash,
    build_model_identity,
)


def _settings(**overrides):
    values = {
        "enabled": True,
        "model_name_or_path": "fake-model",
        "device": "auto",
        "batch_size": 2,
        "max_length": 128,
        "normalize": True,
        "cache_dir": Path("embedding-cache"),
        "query_prefix": "",
        "document_prefix": "",
        "model_revision": "fake-revision",
    }
    values.update(overrides)
    return EmbeddingSettings(**values)


class FakeEmbeddingBackend:
    def __init__(self, vector_fn=None, fail=False):
        self.vector_fn = vector_fn or (lambda text: [3.0, 4.0])
        self.fail = fail
        self.load_calls = 0
        self.encode_calls = []
        self.loaded_device = None

    def load_model(self, model_name_or_path, device, cache_dir, max_length, model_revision):
        self.load_calls += 1
        self.loaded_device = device
        self.loaded_model = model_name_or_path
        self.loaded_cache_dir = cache_dir
        self.loaded_max_length = max_length
        self.loaded_model_revision = model_revision

    def encode(self, texts, batch_size, normalize):
        self.encode_calls.append((list(texts), batch_size, normalize))
        if self.fail:
            raise RuntimeError("backend exploded")
        return [self.vector_fn(text) for text in texts]

    def get_embedding_dimension(self):
        return 2

    def get_model_revision(self):
        return self.loaded_model_revision

    def unload_model(self):
        self.loaded_device = None


class EmbeddingServiceTests(unittest.TestCase):
    def test_model_is_loaded_lazily(self):
        backend = FakeEmbeddingBackend()
        service = EmbeddingService(
            _settings(),
            backend_factory=lambda: backend,
            cuda_available=lambda: False,
        )

        self.assertEqual(backend.load_calls, 0)
        self.assertEqual(service.encode_documents([]), [])
        self.assertEqual(backend.load_calls, 0)

        service.encode_documents(["hello"])
        self.assertEqual(backend.load_calls, 1)

    def test_auto_device_prefers_cuda_when_available(self):
        backend = FakeEmbeddingBackend()
        service = EmbeddingService(
            _settings(device="auto"),
            backend_factory=lambda: backend,
            cuda_available=lambda: True,
        )

        service.load_model()

        self.assertEqual(backend.loaded_device, "cuda")

    def test_forced_cpu_uses_cpu_even_when_cuda_is_available(self):
        backend = FakeEmbeddingBackend()
        service = EmbeddingService(
            _settings(device="cpu"),
            backend_factory=lambda: backend,
            cuda_available=lambda: True,
        )

        service.load_model()

        self.assertEqual(backend.loaded_device, "cpu")

    def test_forced_cuda_fails_clearly_when_cuda_is_unavailable(self):
        backend = FakeEmbeddingBackend()
        service = EmbeddingService(
            _settings(device="cuda"),
            backend_factory=lambda: backend,
            cuda_available=lambda: False,
        )

        with self.assertRaises(EmbeddingConfigurationError) as context:
            service.load_model()

        self.assertIn("CUDA is not available", str(context.exception))
        self.assertEqual(backend.load_calls, 0)

    def test_empty_document_list_does_not_call_model(self):
        backend = FakeEmbeddingBackend()
        service = EmbeddingService(
            _settings(),
            backend_factory=lambda: backend,
            cuda_available=lambda: False,
        )

        self.assertEqual(service.encode_documents([]), [])
        self.assertEqual(backend.encode_calls, [])

    def test_batch_encoding_uses_configured_batch_size(self):
        backend = FakeEmbeddingBackend()
        service = EmbeddingService(
            _settings(batch_size=7),
            backend_factory=lambda: backend,
            cuda_available=lambda: False,
        )

        vectors = service.encode_documents(["one", "two"])

        self.assertEqual(len(vectors), 2)
        self.assertEqual(backend.encode_calls[0][1], 7)

    def test_configured_model_revision_is_passed_to_backend_and_identity(self):
        backend = FakeEmbeddingBackend()
        service = EmbeddingService(
            _settings(model_revision="commit-abc"),
            backend_factory=lambda: backend,
            cuda_available=lambda: False,
        )

        service.load_model()

        self.assertEqual(backend.loaded_model_revision, "commit-abc")
        self.assertEqual(service.get_model_identity().model_revision, "commit-abc")

    def test_max_length_is_clamped_before_loading_backend(self):
        low_backend = FakeEmbeddingBackend()
        low_service = EmbeddingService(
            _settings(max_length=1),
            backend_factory=lambda: low_backend,
            cuda_available=lambda: False,
        )
        high_backend = FakeEmbeddingBackend()
        high_service = EmbeddingService(
            _settings(max_length=999999),
            backend_factory=lambda: high_backend,
            cuda_available=lambda: False,
        )

        low_service.load_model()
        high_service.load_model()

        self.assertEqual(low_backend.loaded_max_length, 16)
        self.assertEqual(high_backend.loaded_max_length, 8192)

    def test_outputs_are_float32_values(self):
        backend = FakeEmbeddingBackend(lambda text: [1.0 / 3.0, 2.0 / 3.0])
        service = EmbeddingService(
            _settings(normalize=False),
            backend_factory=lambda: backend,
            cuda_available=lambda: False,
        )

        value = service.encode_documents(["x"])[0][0]

        self.assertEqual(value, struct.unpack("<f", struct.pack("<f", value))[0])

    def test_normalized_vectors_have_unit_norm(self):
        backend = FakeEmbeddingBackend(lambda text: [3.0, 4.0])
        service = EmbeddingService(
            _settings(normalize=True),
            backend_factory=lambda: backend,
            cuda_available=lambda: False,
        )

        vector = service.encode_documents(["x"])[0]

        self.assertAlmostEqual(math.sqrt(sum(value * value for value in vector)), 1.0, places=6)

    def test_model_loads_only_once_across_requests(self):
        backend = FakeEmbeddingBackend()
        service = EmbeddingService(
            _settings(),
            backend_factory=lambda: backend,
            cuda_available=lambda: False,
        )

        service.encode_documents(["one"])
        service.encode_query("question")

        self.assertEqual(backend.load_calls, 1)

    def test_fake_backend_reports_available_without_sentence_transformers(self):
        backend = FakeEmbeddingBackend()
        service = EmbeddingService(
            _settings(),
            backend_factory=lambda: backend,
            cuda_available=lambda: False,
        )

        self.assertTrue(service.is_available())

    def test_encoding_exception_has_context(self):
        backend = FakeEmbeddingBackend(fail=True)
        service = EmbeddingService(
            _settings(),
            backend_factory=lambda: backend,
            cuda_available=lambda: False,
        )

        with self.assertRaises(EmbeddingEncodeError) as context:
            service.encode_documents(["secret source code"])

        message = str(context.exception)
        self.assertIn("failed to encode 1 documents", message)
        self.assertIn("fake-model", message)
        self.assertNotIn("secret source code", message)

    def test_code_chunk_document_text_format_is_fixed(self):
        text = build_code_chunk_document_text(
            {
                "path": "app\\services\\auth.py",
                "chunk_type": "method",
                "qualified_name": "AuthService.authenticate",
                "content": "def authenticate():\n    return True\n",
            }
        )

        self.assertEqual(
            text,
            "\n".join(
                [
                    f"format: {CODE_CHUNK_TEXT_FORMAT_VERSION}",
                    "path: app/services/auth.py",
                    "type: method",
                    "symbol: AuthService.authenticate",
                    "code:",
                    "def authenticate():\n    return True\n",
                ]
            ),
        )

    def test_embedding_input_hash_uses_final_prefixed_text(self):
        chunk = {
            "path": "app/auth.py",
            "chunk_type": "function",
            "qualified_name": "authenticate_user",
            "content": "def authenticate_user():\n    return True\n",
        }
        settings = _settings(document_prefix="passage: ")
        document_text = build_code_chunk_document_text(chunk)

        self.assertEqual(
            build_code_chunk_embedding_input_hash(chunk, settings),
            build_embedding_input_hash("passage: " + document_text),
        )

    def test_embedding_config_hash_changes_with_prefix_max_length_and_normalize(self):
        base = build_embedding_config_hash(_settings())

        self.assertNotEqual(base, build_embedding_config_hash(_settings(document_prefix="doc: ")))
        self.assertNotEqual(base, build_embedding_config_hash(_settings(query_prefix="query: ")))
        self.assertNotEqual(base, build_embedding_config_hash(_settings(max_length=256)))
        self.assertNotEqual(base, build_embedding_config_hash(_settings(normalize=False)))

    def test_remote_model_identity_includes_configured_revision(self):
        identity = build_model_identity("BAAI/bge-m3", "cpu", "abc123")

        self.assertEqual(identity.model_name, "BAAI/bge-m3")
        self.assertEqual(identity.model_revision, "abc123")

    def test_verified_local_snapshot_separates_revision_and_model_identity(self):
        revision = "5617a9f61b028005a4858fdac845db406aefb181"
        with tempfile.TemporaryDirectory() as directory:
            snapshot = self._make_snapshot(Path(directory), revision, refs_revision=revision)

            identity = build_model_identity(str(snapshot), "cuda", revision)

            self.assertEqual(identity.configured_revision, revision)
            self.assertEqual(identity.resolved_revision, revision)
            self.assertRegex(identity.resolved_revision, r"^[0-9a-f]{40}$")
            self.assertTrue(identity.local_snapshot_identity.startswith("path-sha256:"))
            self.assertTrue(identity.model_identity.startswith("model-sha256:"))
            self.assertNotEqual(identity.model_identity, identity.resolved_revision)
            self.assertNotIn(str(snapshot), str(identity.to_dict()))

    def test_configured_revision_conflict_fails_closed(self):
        revision = "1" * 40
        with tempfile.TemporaryDirectory() as directory:
            snapshot = self._make_snapshot(Path(directory), revision, refs_revision=revision)

            with self.assertRaisesRegex(
                EmbeddingConfigurationError, "configured embedding revision"
            ):
                build_model_identity(str(snapshot), "cpu", "2" * 40)

    def test_refs_main_conflict_fails_closed(self):
        revision = "1" * 40
        with tempfile.TemporaryDirectory() as directory:
            snapshot = self._make_snapshot(Path(directory), revision, refs_revision="2" * 40)

            with self.assertRaisesRegex(EmbeddingConfigurationError, "refs/main"):
                build_model_identity(str(snapshot), "cpu", revision)

    def test_sha_named_arbitrary_directory_is_not_a_resolved_snapshot(self):
        revision = "1" * 40
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / revision
            snapshot.mkdir()
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "modules.json").write_text("[]", encoding="utf-8")

            identity = build_model_identity(str(snapshot), "cpu", revision)

            self.assertIsNone(identity.resolved_revision)
            self.assertIsNotNone(identity.local_snapshot_identity)

    def test_missing_or_invalid_configured_revision_remains_unresolved(self):
        revision = "1" * 40
        with tempfile.TemporaryDirectory() as directory:
            snapshot = self._make_snapshot(Path(directory), revision, refs_revision=revision)

            missing = build_model_identity(str(snapshot), "cpu", "")
            invalid = build_model_identity(str(snapshot), "cpu", "not-a-commit")

            self.assertIsNone(missing.resolved_revision)
            self.assertIsNone(invalid.resolved_revision)

    def test_different_local_snapshots_have_different_model_identities(self):
        revision = "1" * 40
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_snapshot = self._make_snapshot(Path(first), revision, refs_revision=revision)
            second_snapshot = self._make_snapshot(Path(second), revision, refs_revision=revision)

            first_identity = build_model_identity(str(first_snapshot), "cpu", revision)
            second_identity = build_model_identity(str(second_snapshot), "cpu", revision)

            self.assertNotEqual(
                first_identity.local_snapshot_identity,
                second_identity.local_snapshot_identity,
            )
            self.assertNotEqual(first_identity.model_identity, second_identity.model_identity)

    @staticmethod
    def _make_snapshot(root: Path, revision: str, refs_revision: str | None) -> Path:
        model_root = root / "models--BAAI--bge-m3"
        snapshot = model_root / "snapshots" / revision
        (snapshot / "1_Pooling").mkdir(parents=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        (snapshot / "modules.json").write_text("[]", encoding="utf-8")
        (snapshot / "1_Pooling" / "config.json").write_text("{}", encoding="utf-8")
        if refs_revision is not None:
            refs = model_root / "refs"
            refs.mkdir()
            (refs / "main").write_text(refs_revision, encoding="utf-8")
        return snapshot


if __name__ == "__main__":
    unittest.main()
