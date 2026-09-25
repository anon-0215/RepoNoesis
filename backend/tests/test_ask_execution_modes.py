from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from app.database import Database
from app.services.agent_core import run_bounded_agent
from app.services.ask_progress import ProgressRecorder
from app.services.llm_client import ProviderError
from tests.m1_helpers import disabled_embedding_service, make_project
from tests.test_m2_agent import ScriptedPlanner, decision


class _FinalLlm:
    available = True

    def __init__(self):
        self.calls = 0

    def chat(self, _messages, **_kwargs):
        self.calls += 1
        return json.dumps({"parts": [{"text": "Grounded answer", "evidence_aliases": ["A1"]}]})

    def require_available(self):
        return None


class _RouteLlm(_FinalLlm):
    def __init__(self):
        super().__init__()
        self.planner_calls = 0

    def chat(self, messages, **kwargs):
        if kwargs.get("purpose") == "planner":
            self.planner_calls += 1
            return json.dumps(decision("answer"))
        return super().chat(messages, **kwargs)


class ExecutionModeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.db = Database(Path(self.directory.name) / "modes.sqlite")
        self.project_id, self.bundle = make_project(self.db, [
            ("src/auth.py", "authenticate_user", "def authenticate_user(password):\n    return verify(password)\n"),
            ("src/upload.py", "upload_file", "def upload_file(path):\n    return save(path)\n"),
        ])

    def test_rag_skips_planner_but_uses_canonical_evidence_and_validation(self):
        llm = _FinalLlm()
        planner = ScriptedPlanner([decision("answer")])
        events = []
        result = run_bounded_agent(
            "authenticate_user", self.bundle, llm, self.db,
            disabled_embedding_service(), planner=planner, execution_mode="rag",
            diagnostics_recorder=ProgressRecorder(lambda kind, data: events.append((kind, data))),
        )
        self.assertEqual(planner.calls, 0)
        self.assertEqual(result["budget_usage"]["tool_calls_used"], 0)
        self.assertEqual(result["answer_mode"], "llm_grounded")
        self.assertEqual(result["citations"][0]["evidence_id"], "E1")
        self.assertIn("base_completed", [kind for kind, _ in events])
        self.assertIn("citation_checked", [kind for kind, _ in events])
        self.assertEqual(
            [data["checkpoint"] for kind, data in events if kind == "citation_checked"],
            ["initial_evidence", "generation_input", "post_answer_evidence"],
        )
        self.assertTrue(any(kind == "citation_checked" and data["phase"] == "after_answer" for kind, data in events))
        self.assertNotIn("planner_started", [kind for kind, _ in events])
        self.assertNotIn("relation_checked", [kind for kind, _ in events])

    def test_agent_calls_planner_and_supplementary_tool(self):
        planner = ScriptedPlanner([
            decision("continue", "search_code", {"query": "upload_file"}),
            decision("answer"),
        ])
        events = []
        result = run_bounded_agent(
            "authenticate_user", self.bundle, _FinalLlm(), self.db,
            disabled_embedding_service(), planner=planner, execution_mode="agent",
            diagnostics_recorder=ProgressRecorder(lambda kind, data: events.append((kind, data))),
        )
        self.assertGreaterEqual(planner.calls, 2)
        self.assertEqual(result["budget_usage"]["tool_calls_used"], 1)
        kinds = [kind for kind, _ in events]
        self.assertLess(kinds.index("base_completed"), kinds.index("planner_started"))
        self.assertLess(kinds.index("tool_completed"), kinds.index("answer_started"))
        self.assertIn("citation_checked", kinds)
        self.assertEqual(
            [data["checkpoint"] for kind, data in events if kind == "citation_checked"],
            ["initial_evidence", "generation_input", "post_answer_evidence"],
        )

    def test_recoverable_planner_stop_continues_to_validated_answer(self):
        planner = ScriptedPlanner(["invalid", "still invalid"])
        events = []
        result = run_bounded_agent(
            "authenticate_user", self.bundle, _FinalLlm(), self.db,
            disabled_embedding_service(), planner=planner, execution_mode="agent",
            diagnostics_recorder=ProgressRecorder(lambda kind, data: events.append((kind, data))),
        )
        self.assertEqual(result["agent_status"], "completed")
        self.assertIn(("planner_stopped", {"reason": "planner_repair_failed"}), events)
        kinds = [kind for kind, _ in events]
        self.assertLess(kinds.index("planner_stopped"), kinds.index("answer_started"))
        self.assertIn("citation_checked", kinds)
        self.assertEqual(
            [data["checkpoint"] for kind, data in events if kind == "citation_checked"],
            ["initial_evidence", "generation_input", "post_answer_evidence"],
        )

    def test_generation_failure_after_base_is_not_a_completed_answer(self):
        class FailingLlm(_FinalLlm):
            def chat(self, _messages, **_kwargs):
                raise ProviderError("provider_unavailable", "safe unavailable")

        events = []
        with self.assertRaises(ProviderError):
            run_bounded_agent(
                "authenticate_user", self.bundle, FailingLlm(), self.db,
                disabled_embedding_service(), execution_mode="rag",
                diagnostics_recorder=ProgressRecorder(lambda kind, data: events.append((kind, data))),
            )
        kinds = [kind for kind, _ in events]
        self.assertIn("base_completed", kinds)
        self.assertIn("answer_started", kinds)
        self.assertTrue(any(kind == "citation_checked" and data["phase"] == "before_answer" for kind, data in events))
        self.assertFalse(any(kind == "citation_checked" and data["phase"] == "after_answer" for kind, data in events))

    def test_rag_zero_hit_does_not_report_empty_evidence_as_citation_pass(self):
        from app.services.retrieval_v2 import RetrievalExecutionOutcome
        from app.services.ask_diagnostics import build_ask_failure_detail

        events = []
        llm = _FinalLlm()
        empty = RetrievalExecutionOutcome([], "lexical", [], "v1", "test")
        recorder = ProgressRecorder(lambda kind, data: events.append((kind, data)))
        with patch("app.services.agent_tools.retrieve_code", return_value=empty):
            result = run_bounded_agent(
                "unmatched question", self.bundle, llm, self.db,
                disabled_embedding_service(), execution_mode="rag",
                diagnostics_recorder=recorder,
            )
        detail = build_ask_failure_detail(
            result=result, recorder_snapshot=recorder.snapshot(),
            retrieval_version="v1", hierarchy_mode="off", relation_mode="off",
        )
        self.assertEqual(result["agent_status"], "insufficient_evidence")
        self.assertEqual(llm.calls, 0)
        self.assertIn(("base_completed", {"status": "zero_hit", "new_evidence_count": 0}), events)
        self.assertFalse(any(kind == "citation_checked" for kind, _ in events))
        self.assertEqual(detail["diagnostics"]["base_retrieval"]["status"], "zero_hit")
        self.assertEqual(detail["diagnostics"]["base_retrieval"]["new_evidence_count"], 0)

    def test_request_mode_validation_and_legacy_default(self):
        from app.main import AskRequest, app
        from pydantic import ValidationError
        from tests.test_ask_observability import _asgi_post
        self.assertEqual(AskRequest(question="question").execution_mode, "agent")
        with self.assertRaises(ValidationError):
            AskRequest(question="question", execution_mode="other")
        status, _body = asyncio.run(_asgi_post(
            app, f"/api/projects/{self.project_id}/ask",
            {"question": "question", "execution_mode": "other"},
        ))
        self.assertEqual(status, 422)

    def test_route_runs_once_and_persists_once_per_success(self):
        from app import main
        llm = _RouteLlm()
        self.bundle["project"]["source_type"] = "local"

        class Learning:
            def get_learning_context(self, _project_id):
                return None

        with (patch.object(main, "db", self.db), patch.object(main, "llm", llm),
              patch.object(main, "embedding_service", disabled_embedding_service()),
              patch.object(main, "learning_service", Learning()),
              patch.object(main, "_bundle_or_404", return_value=self.bundle)):
            rag = main.ask_project(self.project_id, main.AskRequest(question="authenticate_user", execution_mode="rag"))
            agent = main.ask_project(self.project_id, main.AskRequest(question="authenticate_user"))
            class FailingLlm(_FinalLlm):
                def chat(self, _messages, **_kwargs):
                    raise ProviderError("provider_unavailable", "safe unavailable")

            with patch.object(main, "llm", FailingLlm()):
                with self.assertRaises(main.HTTPException):
                    main.ask_project(self.project_id, main.AskRequest(question="authenticate_user", execution_mode="rag"))
        self.assertEqual(rag["execution_mode"], "rag")
        self.assertEqual(agent["execution_mode"], "agent")
        self.assertIsNone(rag["execution_summary"]["relation_validation_passed"])
        self.assertEqual(llm.planner_calls, 1)
        with self.db.connect() as connection:
            saved = connection.execute("SELECT COUNT(*) FROM chat_answers WHERE project_id = ?", (self.project_id,)).fetchone()[0]
        self.assertEqual(saved, 2)

    def test_stream_sends_intermediate_frame_before_terminal(self):
        from app import main
        release = threading.Event()

        def execution(_project_id, request, *, recorder, request_id):
            recorder.begin_base()
            recorder.record_base_retrieval(
                attempted=True, status="succeeded", retrieval_hit_count=1,
                normalized_candidate_count=1, valid_candidate_count=1,
                new_evidence_count=1, rejected_candidate_count=0,
                rejection_code_counts={},
            )
            release.wait(5)
            return {"request_id": request_id, "execution_mode": request.execution_mode}

        async def inspect():
            payload = {"question": "question", "execution_mode": "rag", "client_request_id": "local-1",
                       "repository_revision": self.bundle["project"]["repository_revision"]}
            body = json.dumps(payload).encode()
            messages = []
            intermediate = asyncio.Event()
            incoming = True

            async def receive():
                nonlocal incoming
                if incoming:
                    incoming = False
                    return {"type": "http.request", "body": body, "more_body": False}
                await asyncio.Event().wait()

            async def send(message):
                messages.append(message)
                if message["type"] == "http.response.body" and b'"type":"base_completed"' in message.get("body", b""):
                    intermediate.set()

            path = f"/api/projects/{self.project_id}/ask/stream"
            task = asyncio.create_task(main.app({
                "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
                "http_version": "1.1", "method": "POST", "scheme": "http", "path": path,
                "raw_path": path.encode(), "query_string": b"", "root_path": "",
                "headers": [(b"content-type", b"application/json")],
                "client": ("127.0.0.1", 12345), "server": ("127.0.0.1", 8000),
            }, receive, send))
            try:
                await asyncio.wait_for(intermediate.wait(), 5)
            except TimeoutError as exc:
                raise AssertionError((messages, task.done(), task.exception() if task.done() else None)) from exc
            self.assertFalse(any(b'"type":"completed"' in m.get("body", b"") for m in messages))
            release.set()
            await asyncio.wait_for(task, 5)
            frames = [json.loads(m["body"]) for m in messages if m["type"] == "http.response.body" and m.get("body")]
            return next(m["status"] for m in messages if m["type"] == "http.response.start"), frames

        with patch.object(main, "db", self.db), patch.object(main, "_execute_ask", side_effect=execution):
            try:
                status, frames = asyncio.run(inspect())
            finally:
                release.set()
        self.assertEqual(status, 200)
        self.assertEqual([frame["type"] for frame in frames], [
            "request_received", "base_started", "base_completed", "completed",
        ])
        self.assertEqual([frame["sequence"] for frame in frames], [1, 2, 3, 4])

    def test_isolated_stream_outcomes_keep_real_checkpoints_and_persistence(self):
        from app import main
        from app.services.retrieval_v2 import RetrievalExecutionOutcome

        self.bundle["project"]["source_type"] = "local"

        class Learning:
            def get_learning_context(self, _project_id):
                return None

        class RecoveringLlm(_FinalLlm):
            def chat(self, messages, **kwargs):
                if kwargs.get("purpose") == "planner":
                    return "invalid planner response"
                return super().chat(messages, **kwargs)

        async def request(mode):
            response = await main.ask_project_stream(
                self.project_id,
                main.AskRequest(question="authenticate_user", execution_mode=mode),
            )
            return [json.loads(frame) async for frame in response.body_iterator]

        with (patch.object(main, "db", self.db),
              patch.object(main, "embedding_service", disabled_embedding_service()),
              patch.object(main, "learning_service", Learning()),
              patch.object(main, "_bundle_or_404", return_value=self.bundle)):
            with patch.object(main, "llm", _FinalLlm()):
                rag = asyncio.run(request("rag"))
            with patch.object(main, "llm", RecoveringLlm()):
                agent = asyncio.run(request("agent"))
            empty = RetrievalExecutionOutcome([], "lexical", [], "v1", "test")
            with (patch.object(main, "llm", _FinalLlm()),
                  patch("app.services.agent_tools.retrieve_code", return_value=empty)):
                insufficient = asyncio.run(request("rag"))

        for frames in (rag, agent, insufficient):
            self.assertEqual([frame["sequence"] for frame in frames], list(range(1, len(frames) + 1)))
            self.assertEqual(len({frame["request_id"] for frame in frames}), 1)
            self.assertEqual(frames[0]["type"], "request_received")
        self.assertEqual(rag[-1]["type"], "completed")
        self.assertEqual(rag[-1]["result"]["execution_mode"], "rag")
        self.assertEqual(
            [frame["checkpoint"] for frame in rag if frame["type"] == "citation_checked"],
            ["initial_evidence", "generation_input", "post_answer_evidence"],
        )
        self.assertEqual(agent[-1]["type"], "completed")
        self.assertEqual(agent[-1]["result"]["execution_mode"], "agent")
        self.assertIn("planner_stopped", [frame["type"] for frame in agent])
        self.assertEqual(insufficient[-1]["type"], "failed")
        self.assertEqual(insufficient[-1]["failure"]["diagnostics"]["base_retrieval"]["status"], "zero_hit")
        self.assertNotIn("citation_checked", [frame["type"] for frame in insufficient])
        with self.db.connect() as connection:
            saved = connection.execute("SELECT COUNT(*) FROM chat_answers WHERE project_id = ?", (self.project_id,)).fetchone()[0]
        self.assertEqual(saved, 2)
