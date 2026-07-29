import hashlib
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from app.config import EmbeddingSettings
from app.database import Database
from app.m5.dense_artifact import (
    DENSE_ARTIFACT_SCHEMA_VERSION,
    DenseArtifactError,
    DenseArtifactLegacyError,
    StandaloneDenseArtifact,
)
from app.m5.embedding import FakeEmbeddingBackend, M5EmbeddingProvider
from app.m5.identity import identity_digest
from app.services.embedding_indexer import EmbeddingIndexer
from app.services.embedding_service import (
    CODE_CHUNK_TEXT_FORMAT_VERSION,
    EmbeddingModelIdentity,
    build_code_chunk_embedding_input_hash,
    build_effective_embedding_identity,
)


class CountingEmbeddingBackend(FakeEmbeddingBackend):
    def __init__(self, dimension: int = 4) -> None:
        super().__init__(dimension)
        self.encoded_text_count = 0
        self.load_count = 0

    def load_model(self, *args, **kwargs):
        self.load_count += 1
        return super().load_model(*args, **kwargs)

    def encode(self, texts, batch_size, normalize):
        self.encoded_text_count += len(texts)
        return super().encode(texts, batch_size, normalize)


class M5DenseArtifactTests(unittest.TestCase):
    def test_wrapper_returns_effective_identity_and_backend_remains_auditable(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = CountingEmbeddingBackend(4)
            provider = self._provider(Path(directory), backend)

            effective = provider.get_model_identity()
            backend_identity = provider.get_backend_identity()
            same = provider.get_model_identity()

            self.assertTrue(effective.model_identity.startswith("embedding-sha256:"))
            self.assertEqual(effective, same)
            self.assertEqual(effective.backend_model_identity, backend_identity.model_identity)
            self.assertEqual(effective.dimension, 4)
            serialized = json.dumps(effective.to_dict(), sort_keys=True)
            self.assertNotIn(str(Path(directory).resolve()), serialized)
            self.assertNotRegex(serialized, r"[A-Za-z]:[\\/]")

            changed = build_effective_embedding_identity(
                backend_identity,
                provider.settings,
                5,
                is_real=True,
            )
            self.assertNotEqual(effective.model_identity, changed.model_identity)

    def test_cache_identity_variations_are_misses_and_records_can_coexist(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "cache.sqlite")
            project_id = self._project_id(db)
            db.save_code_chunks_for_project(project_id, [self._chunk("a")])
            chunk = db.get_code_chunks(project_id)[0]
            settings = self._settings(Path(directory))
            backend_identity = EmbeddingModelIdentity(
                "fake-model", "backend-v1", "cpu", "configured", None, None
            )
            base = build_effective_embedding_identity(
                backend_identity, settings, 4, is_real=False
            )
            self._store(db, chunk, base, [0.5, 0.5, 0.5, 0.5])
            self.assertEqual(self._missing(db, project_id, chunk, base), [])

            variants = [
                replace(base, model_identity="embedding-sha256:" + "1" * 64),
                build_effective_embedding_identity(backend_identity, settings, 5, is_real=False),
                build_effective_embedding_identity(
                    backend_identity,
                    replace(settings, normalize=False),
                    4,
                    is_real=False,
                ),
                build_effective_embedding_identity(
                    backend_identity,
                    replace(settings, document_prefix="passage: "),
                    4,
                    is_real=False,
                ),
                replace(base, embedding_config_hash="2" * 64),
            ]
            for variant in variants:
                missing = self._missing(db, project_id, chunk, variant)
                self.assertEqual(len(missing), 1)

            text_format_missing = db.get_code_chunks_missing_embeddings(
                project_id,
                base.model_name,
                base.backend_model_identity,
                "code-chunk-v2",
                base.embedding_config_hash,
                base.normalized,
                {chunk["id"]: build_code_chunk_embedding_input_hash(chunk, settings)},
                effective_identity=base,
            )
            self.assertEqual(len(text_format_missing), 1)

            dimension_variant = variants[1]
            self._store(db, chunk, dimension_variant, [1.0, 0.0, 0.0, 0.0, 0.0])
            with db.connect() as conn:
                rows = conn.execute(
                    "SELECT wrapper_model_identity, embedding_dimension "
                    "FROM code_chunk_embeddings ORDER BY embedding_dimension"
                ).fetchall()
            self.assertEqual(len(rows), 2)
            self.assertEqual([row["embedding_dimension"] for row in rows], [4, 5])

    def test_create_extend_resume_only_encode_new_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = Database(root / "dense.sqlite")
            project_id = self._project_id(db)
            chunks = [self._chunk(name) for name in ("a", "b", "c", "d")]
            db.save_code_chunks_for_project(project_id, chunks[:2])
            backend = CountingEmbeddingBackend(4)
            provider = self._provider(root, backend)
            artifact = self._artifact(root)

            stage_a = EmbeddingIndexer(db, provider).index_project(
                project_id, artifact=artifact, artifact_mode="create"
            )
            self.assertEqual((stage_a.generated_chunks, stage_a.cached_chunks), (2, 0))
            self.assertEqual(backend.encoded_text_count, 2)

            db.save_code_chunks_for_project(project_id, chunks)
            stage_b = EmbeddingIndexer(db, provider).index_project(
                project_id, artifact=artifact, artifact_mode="extend"
            )
            self.assertEqual((stage_b.generated_chunks, stage_b.cached_chunks), (2, 2))
            self.assertEqual(backend.encoded_text_count, 4)

            stage_c = EmbeddingIndexer(db, provider).index_project(
                project_id, artifact=artifact, artifact_mode="resume"
            )
            self.assertEqual((stage_c.generated_chunks, stage_c.cached_chunks), (0, 4))
            self.assertEqual(backend.encoded_text_count, 4)
            manifest = json.loads((root / "artifact" / "manifest.json").read_text())
            checkpoint = json.loads((root / "artifact" / "checkpoint.json").read_text())
            self.assertEqual(manifest["indexed_chunk_count"], 4)
            self.assertEqual(manifest["checkpoint_status"], "completed")
            self.assertEqual(
                manifest["artifact_identity_digest"],
                checkpoint["artifact_identity_digest"],
            )

    def test_tampered_identity_fields_fail_before_encode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = Database(root / "dense.sqlite")
            project_id = self._project_id(db)
            db.save_code_chunks_for_project(project_id, [self._chunk("a")])
            backend = CountingEmbeddingBackend(4)
            provider = self._provider(root, backend)
            artifact = self._artifact(root)
            EmbeddingIndexer(db, provider).index_project(
                project_id, artifact=artifact, artifact_mode="create"
            )
            baseline_calls = backend.encoded_text_count

            mutations = [
                ("effective_embedding_identity", "model_identity", "embedding-sha256:" + "9" * 64),
                ("effective_embedding_identity", "resolved_revision", "f" * 40),
                ("effective_embedding_identity", "dimension", 5),
                ("effective_embedding_identity", "normalized", False),
                ("effective_embedding_identity", "embedding_config_hash", "8" * 64),
                ("repository", "repository_content_identity", "sha256:" + "7" * 64),
                (None, "chunk_inventory_identity", "inventory-sha256:" + "6" * 64),
            ]
            for index, (section, field, value) in enumerate(mutations):
                copy_root = root / f"copy-{index}"
                shutil.copytree(root / "artifact", copy_root)
                manifest_path = copy_root / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                target = manifest["artifact_identity"]
                if section is None:
                    target[field] = value
                else:
                    target[section][field] = value
                manifest["artifact_identity_digest"] = identity_digest(target)
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                checkpoint_path = copy_root / "checkpoint.json"
                checkpoint = json.loads(checkpoint_path.read_text())
                checkpoint["artifact_identity_digest"] = manifest["artifact_identity_digest"]
                checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

                probe_backend = CountingEmbeddingBackend(4)
                with self.assertRaises(DenseArtifactError):
                    EmbeddingIndexer(db, self._provider(root, probe_backend)).index_project(
                        project_id,
                        artifact=StandaloneDenseArtifact(
                            copy_root,
                            repository_id="itsdangerous",
                            repository_revision="a" * 40,
                            repository_content_identity="sha256:" + "b" * 64,
                        ),
                        artifact_mode="resume",
                    )
                self.assertEqual(probe_backend.load_count, 0)
                self.assertEqual(probe_backend.encoded_text_count, 0)
                self.assertEqual(backend.encoded_text_count, baseline_calls)

            schema_copy = root / "schema-copy"
            shutil.copytree(root / "artifact", schema_copy)
            schema_manifest = json.loads((schema_copy / "manifest.json").read_text())
            schema_manifest["artifact_schema_version"] = DENSE_ARTIFACT_SCHEMA_VERSION + 1
            (schema_copy / "manifest.json").write_text(json.dumps(schema_manifest))
            schema_backend = CountingEmbeddingBackend(4)
            with self.assertRaises(DenseArtifactLegacyError):
                EmbeddingIndexer(db, self._provider(root, schema_backend)).index_project(
                    project_id,
                    artifact=StandaloneDenseArtifact(
                        schema_copy,
                        repository_id="itsdangerous",
                        repository_revision="a" * 40,
                        repository_content_identity="sha256:" + "b" * 64,
                    ),
                    artifact_mode="resume",
                )
            self.assertEqual(schema_backend.load_count, 0)
            self.assertEqual(schema_backend.encoded_text_count, 0)
            self.assertEqual(backend.encoded_text_count, baseline_calls)

    def test_artifact_metadata_contains_no_paths_or_secret_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = Database(root / "dense.sqlite")
            project_id = self._project_id(db)
            db.save_code_chunks_for_project(project_id, [self._chunk("a")])
            provider = self._provider(root, CountingEmbeddingBackend(4))
            EmbeddingIndexer(db, provider).index_project(
                project_id,
                artifact=self._artifact(root),
                artifact_mode="create",
            )
            audit = "\n".join(
                path.read_text(encoding="utf-8")
                for path in (root / "artifact").glob("*.json")
            )
            self.assertNotIn(str(root.resolve()), audit)
            self.assertNotRegex(audit, r"[A-Za-z]:[\\/]")
            self.assertNotRegex(audit.lower(), r"authorization|api[_-]?key|sk-[a-z0-9]")
            database_bytes = (root / "dense.sqlite").read_bytes()
            self.assertNotIn(str(root.resolve()).encode("utf-8"), database_bytes)
            lowered = database_bytes.lower()
            for marker in (b"authorization", b"api_key", b"api-key", b"sk-"):
                self.assertNotIn(marker, lowered)

    @staticmethod
    def _settings(root: Path, **overrides) -> EmbeddingSettings:
        values = {
            "enabled": True,
            "model_name_or_path": "fake-model",
            "model_revision": "configured",
            "device": "cpu",
            "batch_size": 1,
            "max_length": 128,
            "normalize": True,
            "cache_dir": root / "cache",
            "query_prefix": "",
            "document_prefix": "",
        }
        values.update(overrides)
        return EmbeddingSettings(**values)

    @classmethod
    def _provider(cls, root: Path, backend: CountingEmbeddingBackend) -> M5EmbeddingProvider:
        return M5EmbeddingProvider(
            cls._settings(root),
            cache_directory=root / "cache",
            allow_model_load=True,
            allow_network=False,
            backend_factory=lambda: backend,
            cuda_available=lambda: False,
        )

    @staticmethod
    def _artifact(root: Path) -> StandaloneDenseArtifact:
        return StandaloneDenseArtifact(
            root / "artifact",
            repository_id="itsdangerous",
            repository_revision="a" * 40,
            repository_content_identity="sha256:" + "b" * 64,
        )

    @staticmethod
    def _project_id(db: Database) -> str:
        return db.create_project(
            {
                "repo_url": "https://github.com/pallets/itsdangerous",
                "owner": "pallets",
                "repo": "itsdangerous",
                "default_branch": "main",
                "repository_revision": "a" * 40,
            }
        )

    @staticmethod
    def _chunk(name: str) -> dict:
        content = f"def {name}():\n    return '{name}'\n"
        return {
            "repository_revision": "a" * 40,
            "language": "python",
            "path": f"src/itsdangerous/{name}.py",
            "chunk_type": "function",
            "symbol_name": name,
            "qualified_name": name,
            "parent_symbol": "",
            "start_line": 1,
            "end_line": 2,
            "content": content,
            "content_hash": hashlib.sha256(content.encode()).hexdigest(),
        }

    @staticmethod
    def _store(db, chunk, identity, vector):
        settings = M5DenseArtifactTests._settings(Path("cache"))
        db.upsert_code_chunk_embeddings(
            [
                {
                    "code_chunk_id": chunk["id"],
                    "content_hash": chunk["content_hash"],
                    "embedding_input_hash": build_code_chunk_embedding_input_hash(
                        chunk, settings
                    ),
                    "model_name": identity.model_name,
                    "model_revision": identity.backend_model_identity,
                    "identity_schema_version": identity.identity_schema_version,
                    "wrapper_model_identity": identity.model_identity,
                    "resolved_revision": identity.resolved_revision or "",
                    "identity_eligible": True,
                    "text_format_version": CODE_CHUNK_TEXT_FORMAT_VERSION,
                    "embedding_config_hash": identity.embedding_config_hash,
                    "embedding_dimension": len(vector),
                    "embedding_dtype": "float32",
                    "normalized": identity.normalized,
                    "vector": vector,
                }
            ]
        )

    @staticmethod
    def _missing(db, project_id, chunk, identity):
        settings = M5DenseArtifactTests._settings(Path("cache"))
        return db.get_code_chunks_missing_embeddings(
            project_id,
            identity.model_name,
            identity.backend_model_identity,
            CODE_CHUNK_TEXT_FORMAT_VERSION,
            identity.embedding_config_hash,
            identity.normalized,
            {chunk["id"]: build_code_chunk_embedding_input_hash(chunk, settings)},
            effective_identity=identity,
        )


if __name__ == "__main__":
    unittest.main()
