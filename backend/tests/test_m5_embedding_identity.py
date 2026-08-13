import tempfile
import unittest
from pathlib import Path

from app.config import EmbeddingSettings
from app.m5.embedding import FakeEmbeddingBackend, M5EmbeddingProvider


class M5EmbeddingIdentityTests(unittest.TestCase):
    def test_identity_is_composite_stable_and_does_not_expose_local_path(self):
        revision = "5617a9f61b028005a4858fdac845db406aefb181"
        with tempfile.TemporaryDirectory() as directory:
            snapshot = self._make_snapshot(Path(directory), revision)
            provider = M5EmbeddingProvider(
                self._settings(snapshot, revision),
                cache_directory=Path(directory) / "cache",
                allow_model_load=True,
                allow_network=False,
                backend_factory=lambda: FakeEmbeddingBackend(1024),
                cuda_available=lambda: True,
            )

            identity = provider.identity
            serialized = str(identity.to_dict())

            self.assertEqual(identity.configured_revision, revision)
            self.assertEqual(identity.resolved_revision, revision)
            self.assertRegex(identity.resolved_revision, r"^[0-9a-f]{40}$")
            self.assertTrue(identity.model_identity.startswith("embedding-sha256:"))
            self.assertNotEqual(identity.model_identity, revision)
            self.assertNotIn(str(snapshot), serialized)

    def test_output_affecting_configuration_changes_composite_identity(self):
        revision = "1" * 40
        with tempfile.TemporaryDirectory() as directory:
            snapshot = self._make_snapshot(Path(directory), revision)
            first = M5EmbeddingProvider(
                self._settings(snapshot, revision, normalize=True),
                cache_directory=Path(directory) / "cache-a",
                allow_model_load=True,
                allow_network=False,
                backend_factory=lambda: FakeEmbeddingBackend(1024),
                cuda_available=lambda: True,
            ).identity
            second = M5EmbeddingProvider(
                self._settings(snapshot, revision, normalize=False),
                cache_directory=Path(directory) / "cache-b",
                allow_model_load=True,
                allow_network=False,
                backend_factory=lambda: FakeEmbeddingBackend(1024),
                cuda_available=lambda: True,
            ).identity

            self.assertNotEqual(first.model_identity, second.model_identity)
            self.assertNotEqual(first.cache_identity, second.cache_identity)

    @staticmethod
    def _settings(snapshot: Path, revision: str, *, normalize: bool = True) -> EmbeddingSettings:
        return EmbeddingSettings(
            enabled=True,
            model_name_or_path=str(snapshot),
            model_revision=revision,
            device="cuda",
            batch_size=1,
            max_length=512,
            normalize=normalize,
            cache_dir=snapshot.parent,
            query_prefix="query: ",
            document_prefix="passage: ",
        )

    @staticmethod
    def _make_snapshot(root: Path, revision: str) -> Path:
        model_root = root / "models--BAAI--bge-m3"
        snapshot = model_root / "snapshots" / revision
        (snapshot / "1_Pooling").mkdir(parents=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        (snapshot / "modules.json").write_text("[]", encoding="utf-8")
        (snapshot / "1_Pooling" / "config.json").write_text("{}", encoding="utf-8")
        refs = model_root / "refs"
        refs.mkdir()
        (refs / "main").write_text(revision, encoding="utf-8")
        return snapshot


if __name__ == "__main__":
    unittest.main()
