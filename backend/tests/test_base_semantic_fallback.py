from __future__ import annotations

from dataclasses import replace
from contextlib import nullcontext
import asyncio
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from app.database import Database
from app.services.agent_contracts import AgentLimits
from app.services.agent_core import run_bounded_agent
from app.services.embedding_service import EmbeddingService, EmbeddingModelLoadError
from app.services.ask_progress import ProgressRecorder
from app.services.agent_contracts import CancellationToken, RequestBudget
from app.services.semantic_retriever import SemanticSearchOutcome
from tests.m1_helpers import disabled_embedding_service, make_project
from tests.test_ask_execution_modes import _FinalLlm
from tests.test_hybrid_retriever import semantic_result
from tests.test_semantic_retriever import QueryBackend, _record
from app.services.lexical_retriever import LexicalRetriever


class BaseSemanticFallbackTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.db = Database(Path(self.directory.name) / "isolated.sqlite")
        self.project_id, self.bundle = make_project(self.db, [
            ("src/auth.py", "authenticate_user", "def authenticate_user(password):\n    return verify(password)\n"),
        ])
        settings = replace(disabled_embedding_service().settings, enabled=True)
        self.embedding = EmbeddingService(settings, backend_factory=lambda: None, cuda_available=lambda: False)

    def _run(self, query, semantic, *, timeout_ms=300, recorder=None, cancellation=None, request_budget=None, retrieval_version="v1"):
        llm = _FinalLlm()
        semantic_patch = (patch("app.services.semantic_retriever.SemanticRetriever.search", semantic)
                          if semantic is not None else nullcontext())
        with semantic_patch:
            result = run_bounded_agent(
                query, self.bundle, llm, self.db, self.embedding,
                execution_mode="rag",
                limits=AgentLimits(total_deadline_ms=4000, default_tool_timeout_ms=timeout_ms,
                                   min_final_answer_budget_ms=1000),
                diagnostics_recorder=recorder, cancellation=cancellation,
                request_budget=request_budget,
                retrieval_version=retrieval_version,
            )
        return result, llm

    def test_slow_semantic_does_not_discard_lexical_evidence(self):
        def slow(_retriever, *_args, **_kwargs):
            time.sleep(0.75)
            return SemanticSearchOutcome("ok", [], "fake", 0)

        started = time.monotonic()
        result, llm = self._run("authenticate_user", slow)
        self.assertLess(time.monotonic() - started, 0.55)
        self.assertEqual(result["agent_status"], "completed", result)
        self.assertEqual(len(result["evidence"]), 1)
        self.assertEqual(llm.calls, 1)

    def test_cold_model_load_wait_is_bounded_and_shared_state_later_ready(self):
        release = threading.Event()
        entered = threading.Event()
        class ColdBackend(QueryBackend):
            def load_model(self, *args):
                entered.set()
                release.wait(2)
                super().load_model(*args)
        backend = ColdBackend()
        self.embedding = EmbeddingService(
            self.embedding.settings, backend_factory=lambda: backend, cuda_available=lambda: False
        )
        recorder = ProgressRecorder(lambda *_args: None)
        try:
            result, _ = self._run("authenticate_user", None, recorder=recorder)
            self.assertTrue(entered.is_set(), recorder.snapshot())
            self.assertEqual(result["agent_status"], "completed")
            self.assertEqual(recorder.snapshot()["base_retrieval"]["retrieval_detail"]["semantic_status"], "timed_out")
            self.assertEqual(self.embedding.model_load_state, "loading")
        finally:
            release.set()
        for _ in range(30):
            if self.embedding.model_load_state == "ready":
                break
            time.sleep(0.01)
        self.assertEqual(self.embedding.model_load_state, "ready")
        self.assertEqual(backend.load_calls, 1)

    def test_loaded_model_slow_query_encoding_is_bounded(self):
        release = threading.Event()
        entered = threading.Event()
        class SlowBackend(QueryBackend):
            def encode(self, *args, **kwargs):
                entered.set()
                release.wait(2)
                return super().encode(*args, **kwargs)
        backend = SlowBackend()
        self.embedding = EmbeddingService(
            self.embedding.settings, backend_factory=lambda: backend, cuda_available=lambda: False
        )
        self.embedding.load_model(local_files_only=True)
        chunk = self.db.get_code_chunks(self.project_id)[0]
        self.db.upsert_code_chunk_embeddings([_record(chunk, [1.0, 0.0, 0.0], self.embedding.settings)])
        recorder = ProgressRecorder(lambda *_args: None)
        try:
            result, _ = self._run("authenticate_user", None, recorder=recorder)
            self.assertTrue(entered.is_set(), recorder.snapshot())
            self.assertEqual(result["agent_status"], "completed")
            self.assertEqual(recorder.snapshot()["base_retrieval"]["retrieval_detail"]["semantic_status"], "timed_out")
            self.assertNotIn("query_encode_ms", recorder.snapshot()["base_retrieval"]["retrieval_detail"])
        finally:
            release.set()

    def test_loaded_model_slow_vector_read_is_bounded(self):
        backend = QueryBackend()
        self.embedding = EmbeddingService(
            self.embedding.settings, backend_factory=lambda: backend, cuda_available=lambda: False
        )
        self.embedding.load_model(local_files_only=True)
        entered = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        original = self.db.get_code_chunk_embeddings_for_project
        def slow_read(*args, **kwargs):
            entered.set()
            release.wait(2)
            try:
                return original(*args, **kwargs)
            finally:
                finished.set()
        recorder = ProgressRecorder(lambda *_args: None)
        try:
            with patch.object(self.db, "get_code_chunk_embeddings_for_project", side_effect=slow_read):
                result, _ = self._run("authenticate_user", None, recorder=recorder)
            self.assertTrue(entered.is_set())
            self.assertEqual(result["agent_status"], "completed")
            detail = recorder.snapshot()["base_retrieval"]["retrieval_detail"]
            self.assertEqual(detail["semantic_status"], "timed_out")
            self.assertNotIn("vector_read_ms", detail)
        finally:
            release.set()
            self.assertTrue(finished.wait(2))

    def test_concurrent_cold_requests_share_one_model_load(self):
        release = threading.Event()
        entered = threading.Event()
        class ColdBackend(QueryBackend):
            def load_model(self, *args):
                entered.set()
                release.wait(2)
                super().load_model(*args)
        backend = ColdBackend()
        self.embedding = EmbeddingService(
            self.embedding.settings, backend_factory=lambda: backend, cuda_available=lambda: False
        )
        results = []
        def request():
            results.append(self._run("authenticate_user", None)[0])
        threads = [threading.Thread(target=request) for _ in range(2)]
        try:
            for thread in threads:
                thread.start()
            self.assertTrue(entered.wait(1))
            for thread in threads:
                thread.join(1)
            self.assertEqual(len(results), 2)
            self.assertTrue(all(item["agent_status"] == "completed" for item in results))
        finally:
            release.set()
            for thread in threads:
                thread.join(2)
        for _ in range(30):
            if self.embedding.model_load_state == "ready":
                break
            time.sleep(0.01)
        self.assertEqual(backend.load_calls, 1)

    def test_normal_semantic_result_keeps_hybrid_ranking(self):
        chunk = self.db.get_code_chunks(self.project_id)[0]
        def normal(_retriever, *_args, **_kwargs):
            return SemanticSearchOutcome("ok", [semantic_result(chunk, self.project_id, 0.8)], "fake", 1)

        recorder = ProgressRecorder(lambda *_args: None)
        result, llm = self._run("authenticate_user", normal, recorder=recorder)
        self.assertEqual(result["agent_status"], "completed")
        self.assertEqual(result["retrieval_mode"], "hybrid")
        self.assertEqual(result["evidence"][0]["retrieval_sources"], ["lexical", "semantic"])
        self.assertEqual(recorder.snapshot()["base_retrieval"]["retrieval_detail"]["semantic_status"], "completed")
        self.assertEqual(llm.calls, 1)

    def test_optional_v2_base_also_bounds_slow_dense_stage(self):
        def slow(_retriever, *_args, **_kwargs):
            time.sleep(0.75)
            return SemanticSearchOutcome("ok", [], "fake", 0)
        recorder = ProgressRecorder(lambda *_args: None)
        result, llm = self._run("authenticate_user", slow, recorder=recorder, retrieval_version="v2")
        self.assertEqual(result["agent_status"], "completed", result)
        self.assertGreaterEqual(len(result["evidence"]), 1)
        self.assertEqual(recorder.snapshot()["base_retrieval"]["retrieval_detail"]["semantic_status"], "timed_out")
        self.assertEqual(llm.calls, 1)

    def test_late_semantic_result_never_changes_finished_evidence(self):
        started = threading.Event()
        release = threading.Event()
        chunk = self.db.get_code_chunks(self.project_id)[0]
        def late(_retriever, *_args, **_kwargs):
            started.set()
            release.wait(2)
            return SemanticSearchOutcome("ok", [semantic_result(chunk, self.project_id, 0.9)], "fake", 1)

        recorder = ProgressRecorder(lambda *_args: None)
        try:
            result, _llm = self._run("authenticate_user", late, recorder=recorder)
            self.assertTrue(started.is_set())
            self.assertEqual(result["agent_status"], "completed")
            self.assertEqual(result["retrieval_mode"], "lexical")
            self.assertEqual(recorder.snapshot()["base_retrieval"]["retrieval_detail"]["semantic_status"], "timed_out")
            evidence = list(result["evidence"])
        finally:
            release.set()
        time.sleep(0.03)
        self.assertEqual(result["evidence"], evidence)
        self.assertEqual(recorder.snapshot()["base_retrieval"]["retrieval_detail"]["retrieval_source"], "lexical")

    def test_slow_workers_have_fixed_capacity_and_recover_after_return(self):
        release = threading.Event()
        entered = 0
        lock = threading.Lock()
        def slow(_retriever, *_args, **_kwargs):
            nonlocal entered
            with lock:
                entered += 1
            release.wait(2)
            return SemanticSearchOutcome("ok", [], "fake", 0)

        try:
            first, _ = self._run("authenticate_user", slow)
            second, _ = self._run("authenticate_user", slow)
            recorder = ProgressRecorder(lambda *_args: None)
            third, _ = self._run("authenticate_user", slow, recorder=recorder)
            self.assertEqual(entered, 2)
            self.assertEqual([first["agent_status"], second["agent_status"], third["agent_status"]], ["completed"] * 3)
            self.assertEqual(recorder.snapshot()["base_retrieval"]["retrieval_detail"]["semantic_status"], "capacity")
        finally:
            release.set()

    def test_cancelled_request_cannot_promote_lexical_fallback(self):
        cancellation = CancellationToken()
        cancellation.cancel()
        def slow(_retriever, *_args, **_kwargs):
            time.sleep(0.01)
            return SemanticSearchOutcome("ok", [], "fake", 0)
        result, llm = self._run("authenticate_user", slow, cancellation=cancellation)
        self.assertNotEqual(result["agent_status"], "completed")
        self.assertEqual(result["evidence"], [])
        self.assertEqual(llm.calls, 0)

    def test_expired_request_cannot_promote_lexical_fallback(self):
        now = time.monotonic()
        budget = RequestBudget.from_deadline(
            started_at=now - 2, deadline_at=now - 1, final_answer_reserve_ms=1000
        )
        result, llm = self._run("authenticate_user", None, request_budget=budget)
        self.assertEqual(result["agent_status"], "budget_exhausted")
        self.assertEqual(result["evidence"], [])
        self.assertEqual(llm.calls, 0)

    def test_wrong_project_revision_and_identity_never_promote_on_fallback(self):
        source = LexicalRetriever(self.db).search(self.project_id, "authenticate_user")[0]
        def failed(_retriever, *_args, **_kwargs):
            raise EmbeddingModelLoadError("unavailable")
        for bad in (
            replace(source, project_id="another-project"),
            replace(source, repository_revision="another-revision"),
            replace(source, content_hash="0" * 64),
        ):
            with self.subTest(bad=bad):
                with patch("app.services.lexical_retriever.LexicalRetriever.search", return_value=[bad]):
                    result, llm = self._run("authenticate_user", failed)
                self.assertEqual(result["agent_status"], "insufficient_evidence")
                self.assertEqual(result["evidence"], [])
                self.assertEqual(llm.calls, 0)

    def test_route_and_stream_share_fallback_and_save_only_success(self):
        from app import main
        self.bundle["project"]["source_type"] = "local"
        class Learning:
            def get_learning_context(self, _project_id):
                return None
        semantic_finished = threading.Event()
        semantic_count = 0
        semantic_lock = threading.Lock()
        def slow(_retriever, *_args, **_kwargs):
            nonlocal semantic_count
            time.sleep(0.75)
            with semantic_lock:
                semantic_count += 1
                if semantic_count == 2:
                    semantic_finished.set()
            return SemanticSearchOutcome("ok", [], "fake", 0)
        async def stream_request():
            response = await main.ask_project_stream(
                self.project_id,
                main.AskRequest(question="no_matching_source_terms", execution_mode="rag"),
            )
            return [json.loads(frame) async for frame in response.body_iterator]

        with (patch.object(main, "db", self.db),
              patch.object(main, "embedding_service", self.embedding),
              patch.object(main, "learning_service", Learning()),
              patch.object(main, "_bundle_or_404", return_value=self.bundle),
              patch.object(main, "llm", _FinalLlm()),
              patch.object(main, "agent_limits", AgentLimits(
                  total_deadline_ms=4000, default_tool_timeout_ms=300,
                  min_final_answer_budget_ms=1000)),
              patch.object(main, "_log_ask_success"), patch.object(main, "_log_ask_failure"),
              patch("app.services.semantic_retriever.SemanticRetriever.search", slow)):
            success = main.ask_project(
                self.project_id,
                main.AskRequest(question="authenticate_user", execution_mode="rag"),
            )
            failure_frames = asyncio.run(stream_request())
        self.assertEqual(success["execution_summary"]["base_retrieval"]["retrieval_detail"]["semantic_status"], "timed_out")
        self.assertEqual(success["execution_summary"]["planner_requests_attempted"], 0)
        self.assertEqual(failure_frames[-1]["type"], "failed")
        self.assertEqual(failure_frames[-1]["failure"]["diagnostics"]["base_retrieval"]["status"], "zero_hit")
        self.assertEqual([frame["type"] for frame in failure_frames].count("failed"), 1)
        self.assertNotIn("planner_started", [frame["type"] for frame in failure_frames])
        self.assertTrue(semantic_finished.wait(2))
        with self.db.connect() as connection:
            saved = connection.execute(
                "SELECT COUNT(*) FROM chat_answers WHERE project_id = ?", (self.project_id,)
            ).fetchone()[0]
        self.assertEqual(saved, 1)

    def test_recoverable_model_failure_keeps_lexical_evidence(self):
        def failed(_retriever, *_args, **_kwargs):
            raise EmbeddingModelLoadError("test failure")

        result, llm = self._run("authenticate_user", failed)
        self.assertEqual(result["agent_status"], "completed")
        self.assertEqual(len(result["evidence"]), 1)
        self.assertEqual(llm.calls, 1)

    def test_no_lexical_hit_and_semantic_failure_stays_insufficient(self):
        def failed(_retriever, *_args, **_kwargs):
            raise EmbeddingModelLoadError("test failure")

        result, llm = self._run("no_matching_source_terms", failed)
        self.assertEqual(result["agent_status"], "insufficient_evidence")
        self.assertEqual(result["evidence"], [])
        self.assertEqual(llm.calls, 0)


if __name__ == "__main__":
    unittest.main()
