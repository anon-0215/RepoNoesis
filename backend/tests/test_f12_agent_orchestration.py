from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app import main
from app.database import Database
from app.services.agent_contracts import AgentLimits
from app.services.agent_core import run_bounded_agent
from app.services.agent_tools import build_m2_tool_registry
from app.services.ask_diagnostics import build_ask_failure_detail, format_ask_failure_log
from app.services.smoke_diagnostics import SmokeDiagnosticsRecorder
from tests.m1_helpers import disabled_embedding_service, make_project
from tests.test_m2_agent import NoLlm, ScriptedPlanner, decision


async def _post(app, path: str, payload: dict):
    requests = [
        {
            "type": "http.request",
            "body": json.dumps(payload).encode("utf-8"),
            "more_body": False,
        }
    ]
    responses = []

    async def receive():
        if requests:
            return requests.pop(0)
        return {"type": "http.disconnect"}

    async def send(message):
        responses.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": b"",
            "root_path": "",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 8000),
        },
        receive,
        send,
    )
    start = next(item for item in responses if item["type"] == "http.response.start")
    body = b"".join(
        item.get("body", b"")
        for item in responses
        if item["type"] == "http.response.body"
    )
    return start["status"], json.loads(body)


class _LearningService:
    def get_learning_context(self, _project_id):
        return None


class _ScriptedProvider:
    available = True

    def __init__(self, planner_decisions, *, final_answer=None, events=None):
        self.planner_decisions = list(planner_decisions)
        self.final_answer = final_answer or json.dumps(
            {"parts": [{"text": "Validated behavior", "evidence_aliases": ["A1"]}]}
        )
        self.events = events if events is not None else []
        self.planner_calls = 0
        self.final_calls = 0

    def require_available(self):
        return None

    def chat(self, _messages, **kwargs):
        purpose = kwargs.get("purpose")
        self.events.append(f"provider:{purpose or 'unknown'}")
        if purpose == "planner":
            value = self.planner_decisions[
                min(self.planner_calls, len(self.planner_decisions) - 1)
            ]
            self.planner_calls += 1
            return json.dumps(value)
        self.final_calls += 1
        return self.final_answer


class F12AgentOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = Database(Path(self.directory.name) / "f12.sqlite")
        self.project_id, self.bundle = make_project(
            self.database,
            [
                (
                    "src/auth.py",
                    "authenticate_user",
                    "def authenticate_user(password):\n    return verify(password)\n",
                ),
                (
                    "src/upload.py",
                    "upload_file",
                    "def upload_file(path):\n    return save(path)\n",
                ),
            ],
        )
        self.bundle["project"]["source_type"] = "local"

    def _chat_count(self):
        with self.database.connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM chat_answers").fetchone()[0])

    def _route(self, provider, payload, *, limits=None):
        with (
            patch.object(main, "db", self.database),
            patch.object(main, "llm", provider),
            patch.object(main, "embedding_service", disabled_embedding_service()),
            patch.object(main, "learning_service", _LearningService()),
            patch.object(main, "_bundle_or_404", return_value=self.bundle),
            patch.object(main, "agent_limits", limits or AgentLimits()),
        ):
            return asyncio.run(
                _post(
                    main.app,
                    f"/api/projects/{self.project_id}/ask",
                    payload,
                )
            )

    def test_server_constraints_override_nonempty_conflicting_planner_arguments(self):
        planner = ScriptedPlanner(
            [
                decision(
                    "continue",
                    "search_code",
                    {
                        "query": "upload_file",
                        "path": "src/upload.py",
                        "language": "javascript",
                        "symbol": "upload_file",
                        "top_k": 8,
                    },
                ),
                decision("answer"),
            ]
        )
        calls = []
        from app.services import agent_tools

        real_retrieve = agent_tools.retrieve_code

        def capture(*args, **kwargs):
            calls.append(dict(kwargs))
            return real_retrieve(*args, **kwargs)

        with patch.object(agent_tools, "retrieve_code", side_effect=capture):
            result = run_bounded_agent(
                "authenticate_user",
                self.bundle,
                NoLlm(),
                self.database,
                disabled_embedding_service(),
                planner=planner,
                path="src\\auth.py",
                language="PYTHON",
                symbol="authenticate_user",
                evidence_count=1,
            )

        self.assertEqual(len(calls), 2)
        self.assertTrue(all(item["path"] == "src/auth.py" for item in calls))
        self.assertTrue(all(item["language"] == "PYTHON" for item in calls))
        self.assertTrue(all(item["symbol"] == "authenticate_user" for item in calls))
        self.assertTrue(all(item["evidence_count"] == 1 for item in calls))
        self.assertEqual(result["evidence"][0]["path"], "src/auth.py")

    def test_seed_precedes_planner_and_counts_as_tool_not_step(self):
        events = []

        class CapturingPlanner(ScriptedPlanner):
            def decide(inner_self, state, *, repair_hint=None):
                events.append("planner")
                self.assertEqual(state["known_evidence_ids"], ["E1"])
                return super().decide(state, repair_hint=repair_hint)

        planner = CapturingPlanner([decision("answer")])
        recorder = SmokeDiagnosticsRecorder()
        from app.services import agent_tools

        real_retrieve = agent_tools.retrieve_code

        def capture(*args, **kwargs):
            events.append("seed")
            return real_retrieve(*args, **kwargs)

        with patch.object(agent_tools, "retrieve_code", side_effect=capture):
            result = run_bounded_agent(
                "authenticate_user",
                self.bundle,
                NoLlm(),
                self.database,
                disabled_embedding_service(),
                planner=planner,
                path="src/auth.py",
                symbol="authenticate_user",
                evidence_count=1,
                diagnostics_recorder=recorder,
                request_id="seed-before-planner",
            )

        self.assertEqual(events[:2], ["seed", "planner"])
        self.assertEqual(result["budget_usage"]["tool_calls_used"], 1)
        self.assertEqual(result["budget_usage"]["steps_used"], 1)
        execution = recorder.snapshot()["tool_executions"]
        self.assertEqual(
            execution,
            [
                {
                    "phase": "seed",
                    "tool_name": "search_code",
                    "status": "succeeded",
                    "result_count": 1,
                    "evidence_added": 1,
                    "reason_code": None,
                }
            ],
        )

    def test_locator_steps_exhaust_then_use_one_formal_grounded_answer(self):
        planner = ScriptedPlanner(
            [
                decision("continue", "lookup_symbol", {"symbol": "authenticate_user"}),
                decision(
                    "continue",
                    "read_source",
                    {"path": "src/auth.py", "start_line": 1, "end_line": 2},
                ),
                decision("continue", "lookup_symbol", {"symbol": "authenticate_user"}),
            ]
        )
        provider = _ScriptedProvider([])
        recorder = SmokeDiagnosticsRecorder()
        result = run_bounded_agent(
            "authenticate_user",
            self.bundle,
            provider,
            self.database,
            disabled_embedding_service(),
            planner=planner,
            path="src/auth.py",
            symbol="authenticate_user",
            evidence_count=1,
            limits=replace(AgentLimits(), max_agent_steps=3),
            diagnostics_recorder=recorder,
            request_id="locator-recovery",
        )

        self.assertEqual(planner.calls, 3)
        self.assertEqual(provider.final_calls, 1)
        self.assertEqual(result["agent_status"], "completed")
        self.assertEqual(result["answer_mode"], "llm_grounded")
        self.assertEqual(result["budget_usage"]["steps_used"], 3)
        self.assertEqual(result["budget_usage"]["tool_calls_used"], 4)
        self.assertEqual(
            [item["phase"] for item in recorder.snapshot()["tool_executions"]],
            ["seed", "planner", "planner", "planner"],
        )

    def test_zero_seed_and_unconstrained_request_keep_failure_and_old_behavior(self):
        missing_recorder = SmokeDiagnosticsRecorder()
        missing = run_bounded_agent(
            "authenticate_user",
            self.bundle,
            NoLlm(),
            self.database,
            disabled_embedding_service(),
            planner=ScriptedPlanner([decision("answer")]),
            path="src/missing.py",
            symbol="missing_symbol",
            diagnostics_recorder=missing_recorder,
            request_id="missing-seed",
        )
        ordinary_recorder = SmokeDiagnosticsRecorder()
        ordinary = run_bounded_agent(
            "authenticate_user",
            self.bundle,
            NoLlm(),
            self.database,
            disabled_embedding_service(),
            planner=ScriptedPlanner([decision("answer")]),
            diagnostics_recorder=ordinary_recorder,
            request_id="ordinary-request",
        )

        self.assertEqual(missing["agent_status"], "insufficient_evidence")
        self.assertEqual(missing["evidence"], [])
        self.assertEqual(
            missing_recorder.snapshot()["tool_executions"][0]["evidence_added"], 0
        )
        self.assertEqual(ordinary["budget_usage"]["tool_calls_used"], 0)
        self.assertNotIn("tool_executions", ordinary_recorder.snapshot())
        self.assertEqual(self._chat_count(), 0)

    def test_off_modes_do_not_disable_base_seed(self):
        result = run_bounded_agent(
            "authenticate_user",
            self.bundle,
            NoLlm(),
            self.database,
            disabled_embedding_service(),
            planner=ScriptedPlanner([decision("answer")]),
            path="src/auth.py",
            symbol="authenticate_user",
            hierarchy_mode="off",
            relation_mode="off",
        )
        self.assertEqual(len(result["evidence"]), 1)
        self.assertEqual(result["analysis_mode"], "retrieval_only")

    def test_tool_diagnostics_are_allowlisted_content_free_and_bounded(self):
        registry = build_m2_tool_registry(AgentLimits())
        search = registry.get("search_code")

        def unsafe_failure(_context, _parameters):
            raise RuntimeError(
                "PRIVATE-EXCEPTION query=PRIVATE-QUERY path=PRIVATE-PATH "
                "source=def private provider=PRIVATE-PROVIDER"
            )

        registry._tools["search_code"] = replace(search, handler=unsafe_failure)
        recorder = SmokeDiagnosticsRecorder()
        result = run_bounded_agent(
            "PRIVATE-QUERY",
            self.bundle,
            NoLlm(),
            self.database,
            disabled_embedding_service(),
            planner=ScriptedPlanner([decision("answer")]),
            path="PRIVATE-PATH",
            registry=registry,
            diagnostics_recorder=recorder,
            request_id="safe-tool-diagnostics",
        )
        snapshot = recorder.snapshot()
        detail = build_ask_failure_detail(
            result=result,
            recorder_snapshot=snapshot,
            retrieval_version="v1",
            hierarchy_mode="off",
            relation_mode="off",
        )
        logged = json.loads(format_ask_failure_log(detail))
        serialized = json.dumps(detail) + json.dumps(logged)

        self.assertEqual(snapshot["tool_executions"][0]["reason_code"], "tool_failed")
        self.assertIsInstance(snapshot["tool_executions"][0]["result_count"], int)
        self.assertEqual(logged["diagnostics"], detail["diagnostics"])
        self.assertLessEqual(len(json.dumps(detail["diagnostics"]).encode("utf-8")), 4_096)
        for marker in (
            "PRIVATE-EXCEPTION",
            "PRIVATE-QUERY",
            "PRIVATE-PATH",
            "def private",
            "PRIVATE-PROVIDER",
        ):
            self.assertNotIn(marker, serialized)

        many = SmokeDiagnosticsRecorder()
        many.begin_agent(["search_code"], request_id="many-tools")
        for index in range(16):
            many.record_tool_execution(
                phase="planner" if index else "seed",
                tool_name="search_code",
                status="failed",
                result_count=index,
                evidence_added=0,
                reason_code="tool_failed",
            )
        many_detail = build_ask_failure_detail(
            result=result,
            recorder_snapshot=many.snapshot(),
            retrieval_version="v1",
            hierarchy_mode="off",
            relation_mode="off",
        )
        self.assertLessEqual(
            len(json.dumps(many_detail["diagnostics"]).encode("utf-8")), 4_096
        )

    def test_formal_route_locator_recovery_persists_once_and_validator_failures_do_not(self):
        events = []
        provider = _ScriptedProvider(
            [
                decision("continue", "lookup_symbol", {"symbol": "authenticate_user"}),
                decision(
                    "continue",
                    "read_source",
                    {"path": "src/auth.py", "start_line": 1, "end_line": 2},
                ),
                decision("continue", "lookup_symbol", {"symbol": "authenticate_user"}),
            ],
            events=events,
        )
        from app.services import agent_tools

        real_retrieve = agent_tools.retrieve_code

        def capture(*args, **kwargs):
            events.append("tool:seed")
            return real_retrieve(*args, **kwargs)

        captured_constraints = []

        def capture_constraints(*args, **kwargs):
            events.append("tool:seed")
            captured_constraints.append(dict(kwargs))
            return real_retrieve(*args, **kwargs)

        with patch.object(agent_tools, "retrieve_code", side_effect=capture_constraints):
            status, body = self._route(
                provider,
                {
                    "question": "authenticate_user",
                    "path": "src/auth.py",
                    "language": "python",
                    "symbol": "authenticate_user",
                    "evidence_count": 1,
                    "hierarchy_mode": "off",
                    "relation_mode": "off",
                },
                limits=replace(AgentLimits(), max_agent_steps=3),
            )
        self.assertEqual(status, 200)
        self.assertEqual(events[0], "tool:seed")
        self.assertEqual(provider.planner_calls, 3)
        self.assertEqual(provider.final_calls, 1)
        self.assertEqual(captured_constraints[0]["path"], "src/auth.py")
        self.assertEqual(captured_constraints[0]["language"], "python")
        self.assertEqual(captured_constraints[0]["symbol"], "authenticate_user")
        self.assertEqual(body["answer_mode"], "llm_grounded")
        self.assertEqual(body["budget_usage"]["tool_calls_used"], 4)
        self.assertEqual(self._chat_count(), 1)

        before = self._chat_count()
        citation_provider = _ScriptedProvider([decision("answer")])
        with patch(
            "app.services.evidence.CitationValidator.validate_all",
            return_value=([], ["safe invalid citation"]),
        ):
            citation_status, citation_body = self._route(
                citation_provider,
                {
                    "question": "authenticate_user",
                    "path": "src/auth.py",
                    "symbol": "authenticate_user",
                    "evidence_count": 1,
                },
            )
        self.assertEqual(citation_status, 502)
        self.assertEqual(
            citation_body["detail"]["code"],
            "citation_evidence_binding_failed",
        )
        self.assertEqual(
            citation_body["detail"]["diagnostics"]["citation_failure_reason_code"],
            "citation_evidence_binding_failed",
        )
        self.assertEqual(citation_provider.final_calls, 0)
        self.assertEqual(self._chat_count(), before)

        zero_provider = _ScriptedProvider([decision("answer")])
        zero_status, zero_body = self._route(
            zero_provider,
            {
                "question": "authenticate_user",
                "path": "src/missing.py",
                "symbol": "missing_symbol",
                "evidence_count": 1,
            },
        )
        self.assertEqual(zero_status, 422)
        self.assertEqual(zero_body["detail"]["code"], "evidence_insufficient")
        self.assertFalse(zero_body["detail"]["diagnostics"]["final_answer_attempted"])
        self.assertEqual(self._chat_count(), before)

        relation_provider = _ScriptedProvider([decision("answer")])
        with patch(
            "app.services.agent_core.RelationValidator.validate_chains",
            return_value=([], ["safe invalid relation"]),
        ):
            relation_status, relation_body = self._route(
                relation_provider,
                {
                    "question": "authenticate_user",
                    "path": "src/auth.py",
                    "symbol": "authenticate_user",
                    "evidence_count": 1,
                },
            )
        self.assertEqual(relation_status, 502)
        self.assertEqual(
            relation_body["detail"]["code"], "relation_validation_failed"
        )
        self.assertEqual(self._chat_count(), before)

    def test_consecutive_requests_keep_seed_evidence_diagnostics_and_ids_isolated(self):
        first_recorder = SmokeDiagnosticsRecorder()
        second_recorder = SmokeDiagnosticsRecorder()
        first = run_bounded_agent(
            "authenticate_user",
            self.bundle,
            NoLlm(),
            self.database,
            disabled_embedding_service(),
            planner=ScriptedPlanner([decision("answer")]),
            path="src/auth.py",
            symbol="authenticate_user",
            diagnostics_recorder=first_recorder,
            request_id="request-one",
        )
        second = run_bounded_agent(
            "authenticate_user",
            self.bundle,
            NoLlm(),
            self.database,
            disabled_embedding_service(),
            planner=ScriptedPlanner([decision("answer")]),
            path="src/missing.py",
            symbol="missing_symbol",
            diagnostics_recorder=second_recorder,
            request_id="request-two",
        )

        self.assertEqual(first["request_id"], "request-one")
        self.assertEqual(second["request_id"], "request-two")
        self.assertEqual(len(first["evidence"]), 1)
        self.assertEqual(second["evidence"], [])
        self.assertEqual(first_recorder.snapshot()["evidence_count"], 1)
        self.assertEqual(second_recorder.snapshot().get("evidence_count", 0), 0)
        self.assertEqual(
            first_recorder.snapshot()["tool_executions"][0]["evidence_added"], 1
        )
        self.assertEqual(
            second_recorder.snapshot()["tool_executions"][0]["evidence_added"], 0
        )


if __name__ == "__main__":
    unittest.main()
