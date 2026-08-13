import hashlib
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import app.m5.dense_artifact as dense_artifact_module
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
            chunks = [self._chunk(f"chunk_{index:03}") for index in range(79)]
            db.save_code_chunks_for_project(project_id, chunks[:40])
            backend = CountingEmbeddingBackend(4)
            provider = self._provider(root, backend)
            artifact = self._artifact(root)

            stage_a = EmbeddingIndexer(db, provider).index_project(
                project_id, artifact=artifact, artifact_mode="create"
            )
            self.assertEqual((stage_a.generated_chunks, stage_a.cached_chunks), (40, 0))
            self.assertEqual(backend.encoded_text_count, 40)
            stage_a_files = self._artifact_state(root / "artifact")

            db.save_code_chunks_for_project(project_id, chunks)
            stage_b = EmbeddingIndexer(db, provider).index_project(
                project_id, artifact=artifact, artifact_mode="extend"
            )
            self.assertEqual((stage_b.generated_chunks, stage_b.cached_chunks), (39, 40))
            self.assertEqual(backend.encoded_text_count, 79)
            stage_b_files = self._artifact_state(root / "artifact")
            self.assertNotEqual(stage_a_files["manifest"]["sha256"], stage_b_files["manifest"]["sha256"])
            self.assertNotEqual(stage_a_files["checkpoint"]["sha256"], stage_b_files["checkpoint"]["sha256"])
            with db.connect() as conn:
                rows_after_b = conn.execute(
                    "SELECT COUNT(*) FROM code_chunk_embeddings WHERE identity_eligible = 1"
                ).fetchone()[0]
            self.assertEqual(rows_after_b, 79)

            stage_c = EmbeddingIndexer(db, provider).index_project(
                project_id, artifact=artifact, artifact_mode="resume"
            )
            self.assertEqual((stage_c.generated_chunks, stage_c.cached_chunks), (0, 79))
            self.assertEqual(backend.encoded_text_count, 79)
            stage_c_files = self._artifact_state(root / "artifact")
            self.assertEqual(stage_c_files, stage_b_files)
            with db.connect() as conn:
                rows_after_c = conn.execute(
                    "SELECT COUNT(*) FROM code_chunk_embeddings WHERE identity_eligible = 1"
                ).fetchone()[0]
            self.assertEqual(rows_after_c, 79)
            manifest = json.loads((root / "artifact" / "manifest.json").read_text())
            checkpoint = json.loads((root / "artifact" / "checkpoint.json").read_text())
            self.assertEqual(manifest["indexed_chunk_count"], 79)
            self.assertEqual(manifest["checkpoint_status"], "completed")
            self.assertEqual(
                manifest["artifact_identity_digest"],
                checkpoint["artifact_identity_digest"],
            )
            self.assertEqual(list((root / "artifact").glob("*.tmp")), [])

    def test_complete_is_byte_hash_timestamp_and_mtime_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = Database(root / "dense.sqlite")
            project_id = self._project_id(db)
            db.save_code_chunks_for_project(project_id, [self._chunk("a")])
            artifact = self._artifact(root)
            EmbeddingIndexer(db, self._provider(root, CountingEmbeddingBackend(4))).index_project(
                project_id, artifact=artifact, artifact_mode="create"
            )
            before = self._artifact_state(root / "artifact")

            self.assertFalse(artifact.complete(1))

            self.assertEqual(self._artifact_state(root / "artifact"), before)
            self.assertEqual(list((root / "artifact").glob("*.tmp")), [])

    def test_zero_encode_resume_updates_incomplete_persistent_state(self):
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
            with patch.object(dense_artifact_module, "_utc_now", return_value="2026-07-29T01:00:00+00:00"):
                self.assertTrue(artifact.update_progress(0, status="indexing"))
            incomplete = self._artifact_state(root / "artifact")
            calls_before = backend.encoded_text_count

            with patch.object(dense_artifact_module, "_utc_now", return_value="2026-07-29T02:00:00+00:00"):
                stats = EmbeddingIndexer(db, provider).index_project(
                    project_id, artifact=artifact, artifact_mode="resume"
                )

            recovered = self._artifact_state(root / "artifact")
            self.assertEqual((stats.generated_chunks, stats.cached_chunks), (0, 1))
            self.assertEqual(backend.encoded_text_count, calls_before)
            self.assertNotEqual(recovered["manifest"]["bytes"], incomplete["manifest"]["bytes"])
            self.assertNotEqual(recovered["checkpoint"]["bytes"], incomplete["checkpoint"]["bytes"])
            self.assertEqual(recovered["manifest"]["json"]["indexed_chunk_count"], 1)
            self.assertEqual(recovered["manifest"]["json"]["checkpoint_status"], "completed")
            self.assertEqual(recovered["manifest"]["json"]["updated_at"], "2026-07-29T02:00:00+00:00")
            self.assertEqual(recovered["checkpoint"]["json"]["status"], "completed")
            self.assertEqual(recovered["checkpoint"]["json"]["updated_at"], "2026-07-29T02:00:00+00:00")

    def test_metadata_write_failure_restores_pair_and_retry_succeeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = Database(root / "dense.sqlite")
            project_id = self._project_id(db)
            db.save_code_chunks_for_project(project_id, [self._chunk("a")])
            artifact = self._artifact(root)
            EmbeddingIndexer(db, self._provider(root, CountingEmbeddingBackend(4))).index_project(
                project_id, artifact=artifact, artifact_mode="create"
            )
            original = self._artifact_state(root / "artifact")
            real_atomic_json = dense_artifact_module._atomic_json

            for failed_name in ("checkpoint.json", "manifest.json"):
                def fail_selected(path, value, *, selected=failed_name):
                    if path.name == selected:
                        raise OSError(f"forced {selected} failure")
                    return real_atomic_json(path, value)

                with patch.object(dense_artifact_module, "_atomic_json", side_effect=fail_selected):
                    with self.assertRaises(DenseArtifactError):
                        artifact.update_progress(0, status="indexing")
                restored = self._artifact_state(root / "artifact")
                for name in ("manifest", "checkpoint"):
                    self.assertEqual(restored[name]["bytes"], original[name]["bytes"])
                    self.assertEqual(restored[name]["sha256"], original[name]["sha256"])
                    self.assertEqual(restored[name]["json"], original[name]["json"])
                self.assertEqual(list((root / "artifact").glob("*.tmp")), [])
                self.assertEqual(list((root / "artifact").glob("*.rollback")), [])
                with db.connect() as conn:
                    self.assertEqual(
                        conn.execute("SELECT COUNT(*) FROM code_chunk_embeddings").fetchone()[0],
                        1,
                    )

            with patch.object(dense_artifact_module, "_utc_now", return_value="2026-07-29T03:00:00+00:00"):
                self.assertTrue(artifact.update_progress(0, status="indexing"))
            self.assertEqual(
                json.loads((root / "artifact" / "manifest.json").read_text())["checkpoint_status"],
                "indexing",
            )
            self.assertTrue(artifact.complete(1))
            artifact.preflight(db.get_code_chunks(project_id), mode="resume")
            final_manifest = json.loads((root / "artifact" / "manifest.json").read_text())
            final_checkpoint = json.loads((root / "artifact" / "checkpoint.json").read_text())
            self.assertEqual(final_manifest["checkpoint_status"], "completed")
            self.assertEqual(final_manifest["indexed_chunk_count"], 1)
            self.assertEqual(final_checkpoint["status"], "completed")
            self.assertEqual(final_checkpoint["indexed_chunk_count"], 1)

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
            original_state = self._artifact_state(root / "artifact")

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
            self.assertEqual(self._artifact_state(root / "artifact"), original_state)

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

    @staticmethod
    def _artifact_state(root: Path) -> dict:
        result = {}
        for name in ("manifest", "checkpoint"):
            path = root / f"{name}.json"
            content = path.read_bytes()
            value = json.loads(content)
            result[name] = {
                "bytes": content,
                "sha256": hashlib.sha256(content).hexdigest(),
                "times": {
                    key: value[key]
                    for key in ("created_at", "updated_at")
                    if key in value
                },
                "mtime_ns": path.stat().st_mtime_ns,
                "json": value,
            }
        return result


if __name__ == "__main__":
    unittest.main()
