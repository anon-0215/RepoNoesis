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

    def test_unconstrained_base_retrieval_precedes_planner_without_using_planner_state(self):
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
                evidence_count=1,
                diagnostics_recorder=recorder,
                request_id="seed-before-planner",
            )

        self.assertEqual(events[:2], ["seed", "planner"])
        self.assertEqual(events.count("seed"), 1)
        self.assertEqual(result["budget_usage"]["tool_calls_used"], 0)
        self.assertEqual(result["budget_usage"]["steps_used"], 1)
        snapshot = recorder.snapshot()
        self.assertEqual(snapshot.get("tool_calls_attempted", 0), 0)
        self.assertEqual(snapshot.get("tool_executions", []), [])
        self.assertEqual(
            snapshot["base_retrieval"],
            {
                "attempted": True,
                "status": "succeeded",
                "retrieval_hit_count": 1,
                "normalized_candidate_count": 1,
                "valid_candidate_count": 1,
                "new_evidence_count": 1,
                "rejected_candidate_count": 0,
                "rejection_code_counts": {},
            },
        )

    def test_blank_question_is_rejected_before_route_or_direct_agent_work(self):
        invalid_questions = ("", "   ", "\t\n \r")
        with patch.object(main, "run_bounded_agent") as route_agent:
            for question in invalid_questions:
                with self.subTest(layer="api", question=repr(question)):
                    status, body = asyncio.run(
                        _post(
                            main.app,
                            f"/api/projects/{self.project_id}/ask",
                            {"question": question},
                        )
                    )
                    self.assertEqual(status, 422)
                    self.assertEqual(body["detail"][0]["loc"][-1], "question")
            route_agent.assert_not_called()

        from app.services import agent_tools

        for question in invalid_questions:
            provider = _ScriptedProvider([decision("answer")])
            planner = ScriptedPlanner([decision("answer")])
            with (
                self.subTest(layer="agent", question=repr(question)),
                patch.object(agent_tools, "retrieve_code") as retrieve,
                patch.object(self.database, "save_chat_answer") as persist,
                self.assertRaisesRegex(ValueError, "non-whitespace"),
            ):
                run_bounded_agent(
                    question,
                    self.bundle,
                    provider,
                    self.database,
                    disabled_embedding_service(),
                    planner=planner,
                )
            retrieve.assert_not_called()
            persist.assert_not_called()
            self.assertEqual(planner.calls, 0)
            self.assertEqual(provider.planner_calls, 0)
            self.assertEqual(provider.final_calls, 0)

    def test_padded_nonblank_question_retains_business_text_and_uses_trimmed_base_query(self):
        question = "  authenticate_user  "
        self.assertEqual(main.AskRequest(question=question).question, question)
        queries = []
        from app.services import agent_tools

        real_retrieve = agent_tools.retrieve_code

        def capture(*args, **kwargs):
            queries.append(args[3])
            return real_retrieve(*args, **kwargs)

        with patch.object(agent_tools, "retrieve_code", side_effect=capture):
            result = run_bounded_agent(
                question,
                self.bundle,
                NoLlm(),
                self.database,
                disabled_embedding_service(),
                planner=ScriptedPlanner([decision("answer")]),
            )

        self.assertEqual(queries, ["authenticate_user"])
        self.assertEqual(result["agent_status"], "completed")

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
            limits=replace(
                AgentLimits(), max_agent_steps=3, max_no_progress_steps=3
            ),
            diagnostics_recorder=recorder,
            request_id="locator-recovery",
        )

        self.assertEqual(planner.calls, 3)
        self.assertEqual(provider.final_calls, 1)
        self.assertEqual(result["agent_status"], "completed")
        self.assertEqual(result["answer_mode"], "llm_grounded")
        self.assertEqual(result["budget_usage"]["steps_used"], 3)
        self.assertEqual(result["budget_usage"]["tool_calls_used"], 3)
        self.assertEqual(
            [item["phase"] for item in recorder.snapshot()["tool_executions"]],
            ["planner", "planner", "planner"],
        )

    def test_zero_base_retrieval_keeps_failure_but_unconstrained_request_is_seeded(self):
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
        self.assertEqual(missing_recorder.snapshot().get("tool_executions", []), [])
        self.assertEqual(
            missing_recorder.snapshot()["base_retrieval"]["status"],
            "zero_hit",
        )
        self.assertEqual(ordinary["budget_usage"]["tool_calls_used"], 0)
        self.assertEqual(len(ordinary["evidence"]), 1)
        self.assertEqual(ordinary_recorder.snapshot().get("tool_executions", []), [])
        self.assertEqual(
            ordinary_recorder.snapshot()["base_retrieval"]["status"],
            "succeeded",
        )
        self.assertEqual(self._chat_count(), 0)

    def test_base_retrieval_audit_covers_rejection_failure_and_request_isolation(self):
        from app.services import agent_tools

        initial = agent_tools.retrieve_code(
            self.database,
            disabled_embedding_service(),
            self.project_id,
            "authenticate_user",
            evidence_count=1,
        )
        forged = replace(initial, results=[replace(initial.results[0], path="src/forged.py")])
        empty = replace(initial, results=[])

        records = []
        for request_id, outcome in (("base-rejected", forged), ("base-zero", empty)):
            recorder = SmokeDiagnosticsRecorder()
            with patch.object(agent_tools, "retrieve_code", return_value=outcome):
                run_bounded_agent(
                    "authenticate_user",
                    self.bundle,
                    NoLlm(),
                    self.database,
                    disabled_embedding_service(),
                    planner=ScriptedPlanner([decision("answer")]),
                    diagnostics_recorder=recorder,
                    request_id=request_id,
                )
            records.append(recorder.snapshot())

        rejected = records[0]["base_retrieval"]
        self.assertEqual(rejected["status"], "all_rejected")
        self.assertEqual(rejected["retrieval_hit_count"], 1)
        self.assertEqual(rejected["normalized_candidate_count"], 1)
        self.assertEqual(rejected["valid_candidate_count"], 0)
        self.assertEqual(rejected["new_evidence_count"], 0)
        self.assertEqual(rejected["rejected_candidate_count"], 1)
        self.assertEqual(rejected["rejection_code_counts"], {"path_mismatch": 1})
        self.assertEqual(records[1]["base_retrieval"]["status"], "zero_hit")
        self.assertEqual(records[1]["base_retrieval"]["retrieval_hit_count"], 0)
        self.assertEqual(records[0].get("tool_executions", []), [])
        self.assertEqual(records[1].get("tool_executions", []), [])
        self.assertNotEqual(records[0]["request_id"], records[1]["request_id"])

        failed_recorder = SmokeDiagnosticsRecorder()
        with patch.object(agent_tools, "retrieve_code", side_effect=RuntimeError("sensitive body")):
            run_bounded_agent(
                "authenticate_user",
                self.bundle,
                NoLlm(),
                self.database,
                disabled_embedding_service(),
                planner=ScriptedPlanner([decision("answer")]),
                diagnostics_recorder=failed_recorder,
                request_id="base-safe-failure",
            )
        failed = failed_recorder.snapshot()
        self.assertEqual(failed["base_retrieval"]["status"], "failed")
        self.assertNotIn("sensitive body", json.dumps(failed))

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

    def test_question_only_route_seeds_before_planner_and_persists_once(self):
        events = []
        provider = _ScriptedProvider([decision("answer")], events=events)
        from app.services import agent_tools

        real_retrieve = agent_tools.retrieve_code

        def capture(*args, **kwargs):
            events.append("base-retrieval")
            return real_retrieve(*args, **kwargs)

        with patch.object(agent_tools, "retrieve_code", side_effect=capture):
            status, body = self._route(
                provider,
                {"question": "Where is authenticate_user defined?"},
            )

        self.assertEqual(status, 200)
        self.assertEqual(events[:2], ["base-retrieval", "provider:planner"])
        self.assertEqual(events.count("base-retrieval"), 1)
        self.assertEqual(provider.planner_calls, 1)
        self.assertEqual(provider.final_calls, 1)
        self.assertEqual(body["budget_usage"]["tool_calls_used"], 0)
        self.assertEqual(len(body["evidence"]), 1)
        self.assertEqual(self._chat_count(), 1)

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
            planner=ScriptedPlanner(
                [
                    decision("continue", "search_code", {"query": "PRIVATE-QUERY"}),
                    decision("answer"),
                ]
            ),
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
                limits=replace(
                    AgentLimits(), max_agent_steps=3, max_no_progress_steps=3
                ),
            )
        self.assertEqual(status, 200)
        self.assertEqual(events[0], "tool:seed")
        self.assertEqual(provider.planner_calls, 3)
        self.assertEqual(provider.final_calls, 1)
        self.assertEqual(captured_constraints[0]["path"], "src/auth.py")
        self.assertEqual(captured_constraints[0]["language"], "python")
        self.assertEqual(captured_constraints[0]["symbol"], "authenticate_user")
        self.assertEqual(body["answer_mode"], "llm_grounded")
        self.assertEqual(body["budget_usage"]["tool_calls_used"], 3)
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
        self.assertEqual(first_recorder.snapshot().get("tool_executions", []), [])
        self.assertEqual(second_recorder.snapshot().get("tool_executions", []), [])

    def test_zero_tool_budget_still_runs_base_and_allows_one_terminal_planner_decision(self):
        events = []

        class CapturingPlanner(ScriptedPlanner):
            def decide(inner_self, state, *, repair_hint=None):
                events.append("planner")
                self.assertEqual(state["known_evidence_ids"], ["E1"])
                self.assertEqual(state["remaining_budget"]["tool_calls"], 0)
                return super().decide(state, repair_hint=repair_hint)

        planner = CapturingPlanner([decision("answer")])
        from app.services import agent_tools

        real_retrieve = agent_tools.retrieve_code

        def capture(*args, **kwargs):
            events.append("base")
            return real_retrieve(*args, **kwargs)

        with patch.object(agent_tools, "retrieve_code", side_effect=capture):
            result = run_bounded_agent(
                "  authenticate_user  ",
                self.bundle,
                NoLlm(),
                self.database,
                disabled_embedding_service(),
                planner=planner,
                limits=replace(AgentLimits(), max_tool_calls=0),
            )

        self.assertEqual(events[:2], ["base", "planner"])
        self.assertEqual(events.count("base"), 1)
        self.assertEqual(planner.calls, 1)
        self.assertEqual(result["agent_status"], "completed")
        self.assertEqual(result["budget_usage"]["tool_calls_used"], 0)

    def test_seed_does_not_repeat_block_same_planner_query_or_consume_no_progress(self):
        planner = ScriptedPlanner(
            [
                decision("continue", "search_code", {"query": "authenticate_user"}),
                decision("answer"),
            ]
        )
        recorder = SmokeDiagnosticsRecorder()
        result = run_bounded_agent(
            "authenticate_user",
            self.bundle,
            NoLlm(),
            self.database,
            disabled_embedding_service(),
            planner=planner,
            limits=replace(AgentLimits(), max_tool_calls=1, max_no_progress_steps=2),
            diagnostics_recorder=recorder,
        )

        execution = recorder.snapshot()["tool_executions"]
        self.assertEqual(len(execution), 1)
        self.assertEqual(execution[0]["phase"], "planner")
        self.assertEqual(execution[0]["status"], "succeeded")
        self.assertEqual(execution[0]["result_count"], 0)
        self.assertIsNone(execution[0]["reason_code"])
        self.assertEqual(result["budget_usage"]["tool_calls_used"], 1)


if __name__ == "__main__":
    unittest.main()
