import hashlib
import unittest
from pathlib import Path
import tempfile

from app.config import EmbeddingSettings
from app.database import Database
from app.services.embedding_indexer import EmbeddingIndexer
from app.services.embedding_service import CODE_CHUNK_TEXT_FORMAT_VERSION, EmbeddingService


def _settings(**overrides):
    values = {
        "enabled": True,
        "model_name_or_path": "fake-model",
        "device": "cpu",
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
    def __init__(self, fail_on: str | None = None):
        self.fail_on = fail_on
        self.load_calls = 0
        self.encode_calls = 0

    def load_model(self, model_name_or_path, device, cache_dir, max_length, model_revision):
        self.load_calls += 1
        self.loaded_model_revision = model_revision

    def encode(self, texts, batch_size, normalize):
        self.encode_calls += 1
        if self.fail_on and any(self.fail_on in text for text in texts):
            raise RuntimeError("selected chunk failed")
        return [[1.0, 0.0] if "auth" in text else [0.0, 1.0] for text in texts]

    def get_embedding_dimension(self):
        return 2

    def get_model_revision(self):
        return self.loaded_model_revision

    def unload_model(self):
        pass


def _project_id(db: Database) -> str:
    return db.create_project(
        {
            "repo_url": "https://github.com/demo/sample",
            "owner": "demo",
            "repo": "sample",
            "default_branch": "main",
        }
    )


def _chunk(path: str, name: str, content: str | None = None) -> dict:
    source = content or f"def {name}():\n    return '{name}'\n"
    return {
        "repository_revision": "abc123",
        "language": "python",
        "path": path,
        "chunk_type": "function",
        "symbol_name": name,
        "qualified_name": name,
        "parent_symbol": "",
        "start_line": 1,
        "end_line": len(source.splitlines()),
        "content": source,
        "content_hash": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }


class EmbeddingIndexerTests(unittest.TestCase):
    def test_first_index_generates_all_embeddings(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "first.sqlite")
            project_id = _project_id(db)
            db.save_code_chunks_for_project(
                project_id,
                [_chunk("auth.py", "auth"), _chunk("db.py", "init_db")],
            )
            backend = FakeEmbeddingBackend()
            service = EmbeddingService(
                _settings(),
                backend_factory=lambda: backend,
                cuda_available=lambda: False,
            )

            stats = EmbeddingIndexer(db, service).index_project(project_id)

            self.assertEqual(stats.total_chunks, 2)
            self.assertEqual(stats.cached_chunks, 0)
            self.assertEqual(stats.generated_chunks, 2)
            self.assertEqual(stats.failed_chunks, 0)
            self.assertEqual(stats.dimension, 2)

    def test_second_index_reuses_cache_without_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "cached.sqlite")
            project_id = _project_id(db)
            db.save_code_chunks_for_project(
                project_id,
                [_chunk("auth.py", "auth"), _chunk("db.py", "init_db")],
            )
            first_backend = FakeEmbeddingBackend()
            first_service = EmbeddingService(
                _settings(),
                backend_factory=lambda: first_backend,
                cuda_available=lambda: False,
            )
            EmbeddingIndexer(db, first_service).index_project(project_id)

            second_backend = FakeEmbeddingBackend()
            second_service = EmbeddingService(
                _settings(),
                backend_factory=lambda: second_backend,
                cuda_available=lambda: False,
            )
            stats = EmbeddingIndexer(db, second_service).index_project(project_id)

            self.assertEqual(stats.cached_chunks, 2)
            self.assertEqual(stats.generated_chunks, 0)
            self.assertEqual(stats.dimension, 2)
            self.assertEqual(second_backend.load_calls, 1)
            self.assertEqual(second_backend.encode_calls, 0)

    def test_only_changed_chunk_is_reindexed(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "incremental.sqlite")
            project_id = _project_id(db)
            original_auth = _chunk("auth.py", "auth")
            original_db = _chunk("db.py", "init_db")
            db.save_code_chunks_for_project(project_id, [original_auth, original_db])
            service = EmbeddingService(
                _settings(),
                backend_factory=lambda: FakeEmbeddingBackend(),
                cuda_available=lambda: False,
            )
            EmbeddingIndexer(db, service).index_project(project_id)

            changed_db = _chunk("db.py", "init_db", "def init_db():\n    return 'changed'\n")
            db.save_code_chunks_for_project(project_id, [original_auth, changed_db])
            backend = FakeEmbeddingBackend()
            second_service = EmbeddingService(
                _settings(),
                backend_factory=lambda: backend,
                cuda_available=lambda: False,
            )
            stats = EmbeddingIndexer(db, second_service).index_project(project_id)

            self.assertEqual(stats.cached_chunks, 1)
            self.assertEqual(stats.generated_chunks, 1)
            self.assertEqual(stats.failed_chunks, 0)

    def test_document_prefix_change_reindexes_unchanged_content(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "prefix.sqlite")
            project_id = _project_id(db)
            db.save_code_chunks_for_project(project_id, [_chunk("auth.py", "auth")])
            EmbeddingIndexer(
                db,
                EmbeddingService(
                    _settings(document_prefix=""),
                    backend_factory=lambda: FakeEmbeddingBackend(),
                    cuda_available=lambda: False,
                ),
            ).index_project(project_id)

            backend = FakeEmbeddingBackend()
            stats = EmbeddingIndexer(
                db,
                EmbeddingService(
                    _settings(document_prefix="passage: "),
                    backend_factory=lambda: backend,
                    cuda_available=lambda: False,
                ),
            ).index_project(project_id)

            self.assertEqual(stats.cached_chunks, 0)
            self.assertEqual(stats.generated_chunks, 1)
            self.assertEqual(backend.encode_calls, 1)

    def test_query_prefix_change_reindexes_by_config_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "query-prefix.sqlite")
            project_id = _project_id(db)
            db.save_code_chunks_for_project(project_id, [_chunk("auth.py", "auth")])
            EmbeddingIndexer(
                db,
                EmbeddingService(
                    _settings(query_prefix=""),
                    backend_factory=lambda: FakeEmbeddingBackend(),
                    cuda_available=lambda: False,
                ),
            ).index_project(project_id)

            backend = FakeEmbeddingBackend()
            stats = EmbeddingIndexer(
                db,
                EmbeddingService(
                    _settings(query_prefix="query: "),
                    backend_factory=lambda: backend,
                    cuda_available=lambda: False,
                ),
            ).index_project(project_id)

            self.assertEqual(stats.cached_chunks, 0)
            self.assertEqual(stats.generated_chunks, 1)

    def test_model_revision_change_reindexes_unchanged_content(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "revision.sqlite")
            project_id = _project_id(db)
            db.save_code_chunks_for_project(project_id, [_chunk("auth.py", "auth")])
            EmbeddingIndexer(
                db,
                EmbeddingService(
                    _settings(model_revision="rev-a"),
                    backend_factory=lambda: FakeEmbeddingBackend(),
                    cuda_available=lambda: False,
                ),
            ).index_project(project_id)

            backend = FakeEmbeddingBackend()
            stats = EmbeddingIndexer(
                db,
                EmbeddingService(
                    _settings(model_revision="rev-b"),
                    backend_factory=lambda: backend,
                    cuda_available=lambda: False,
                ),
            ).index_project(project_id)

            self.assertEqual(stats.cached_chunks, 0)
            self.assertEqual(stats.generated_chunks, 1)

    def test_normalize_change_reindexes_unchanged_content(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "normalize.sqlite")
            project_id = _project_id(db)
            db.save_code_chunks_for_project(project_id, [_chunk("auth.py", "auth")])
            EmbeddingIndexer(
                db,
                EmbeddingService(
                    _settings(normalize=True),
                    backend_factory=lambda: FakeEmbeddingBackend(),
                    cuda_available=lambda: False,
                ),
            ).index_project(project_id)

            backend = FakeEmbeddingBackend()
            stats = EmbeddingIndexer(
                db,
                EmbeddingService(
                    _settings(normalize=False),
                    backend_factory=lambda: backend,
                    cuda_available=lambda: False,
                ),
            ).index_project(project_id)

            self.assertEqual(stats.cached_chunks, 0)
            self.assertEqual(stats.generated_chunks, 1)

    def test_deleted_chunk_clears_cached_embedding(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "delete.sqlite")
            project_id = _project_id(db)
            auth = _chunk("auth.py", "auth")
            upload = _chunk("upload.py", "upload")
            db.save_code_chunks_for_project(project_id, [auth, upload])
            service = EmbeddingService(
                _settings(),
                backend_factory=lambda: FakeEmbeddingBackend(),
                cuda_available=lambda: False,
            )
            EmbeddingIndexer(db, service).index_project(project_id)

            db.save_code_chunks_for_project(project_id, [auth])

            with db.connect() as conn:
                count = conn.execute("SELECT COUNT(*) FROM code_chunk_embeddings").fetchone()[0]
            self.assertEqual(count, 1)

    def test_partial_encoding_failure_counts_failed_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "partial.sqlite")
            project_id = _project_id(db)
            db.save_code_chunks_for_project(
                project_id,
                [_chunk("auth.py", "auth"), _chunk("bad.py", "bad")],
            )
            backend = FakeEmbeddingBackend(fail_on="bad")
            service = EmbeddingService(
                _settings(batch_size=2),
                backend_factory=lambda: backend,
                cuda_available=lambda: False,
            )

            stats = EmbeddingIndexer(db, service).index_project(project_id)

            self.assertEqual(stats.generated_chunks, 1)
            self.assertEqual(stats.failed_chunks, 1)
            self.assertTrue(stats.warnings)

    def test_disabled_embeddings_do_not_load_model(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "disabled.sqlite")
            project_id = _project_id(db)
            db.save_code_chunks_for_project(project_id, [_chunk("auth.py", "auth")])
            backend = FakeEmbeddingBackend()
            service = EmbeddingService(
                _settings(enabled=False),
                backend_factory=lambda: backend,
                cuda_available=lambda: False,
            )

            stats = EmbeddingIndexer(db, service).index_project(project_id)

            self.assertEqual(stats.total_chunks, 1)
            self.assertEqual(stats.generated_chunks, 0)
            self.assertEqual(backend.load_calls, 0)
            self.assertIn("disabled", stats.warnings[0].lower())

    def test_embedding_failure_does_not_destroy_saved_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "analysis-safe.sqlite")
            project_id = _project_id(db)
            chunk = _chunk("bad.py", "bad")
            db.save_analysis(
                project_id,
                {
                    "primary_language": "Python",
                    "frameworks": [],
                    "files": [],
                    "modules": [],
                    "overview": "saved",
                },
                [
                    {
                        "path": "bad.py",
                        "extension": ".py",
                        "language": "Python",
                        "size": len(chunk["content"]),
                        "content": chunk["content"],
                        "summary": "bad",
                        "importance": 1,
                        "is_core": True,
                        "imports": [],
                        "exports": [],
                        "symbols": [],
                    }
                ],
                [],
                [chunk],
            )
            service = EmbeddingService(
                _settings(),
                backend_factory=lambda: FakeEmbeddingBackend(fail_on="bad"),
                cuda_available=lambda: False,
            )

            stats = EmbeddingIndexer(db, service).index_project(project_id)

            self.assertEqual(stats.failed_chunks, 1)
            self.assertEqual(db.get_project(project_id)["status"], "done")
            self.assertEqual(len(db.get_code_chunks(project_id)), 1)


if __name__ == "__main__":
    unittest.main()
