from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError

from app import main
from app.config import LLMSettings
from app.database import Database
from app.services.agent_contracts import (
    AgentLimits,
    CancellationToken,
    RequestBudget,
    normalize_repository_relative_path,
)
from app.services.agent_core import run_bounded_agent, validate_planner_decision
from app.services.agent_tools import build_m2_tool_registry
from app.services.ask_diagnostics import format_ask_failure_log
from app.services.llm_client import LLMClient, ProviderError
from app.services.smoke_diagnostics import SmokeDiagnosticsRecorder
from tests.m1_helpers import disabled_embedding_service, make_project
from tests.test_f12_agent_orchestration import _LearningService, _post
from tests.test_m2_agent import NoLlm, ScriptedPlanner, decision


class _Clock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class _Response:
    status = 200

    def __init__(self, content: str) -> None:
        self.body = json.dumps(
            {"choices": [{"message": {"content": content}}]}
        ).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self.body


class _CutoffProviderHarness:
    def __init__(
        self,
        clock: _Clock,
        *,
        planner_end: float = 7.1,
        planner_error: str = "deadline",
        final_error: str | None = None,
        final_answer: str = '{"parts":[{"text":"Validated behavior","evidence_aliases":["A1"]}]}',
        cancellation: CancellationToken | None = None,
    ) -> None:
        self.clock = clock
        self.planner_end = planner_end
        self.planner_error = planner_error
        self.final_error = final_error
        self.final_answer = final_answer
        self.cancellation = cancellation
        self.planner_calls = 0
        self.final_calls = 0

    def opener(self, request, *, timeout):
        del timeout
        payload = json.loads(request.data.decode("utf-8"))
        system_prompt = payload["messages"][0]["content"]
        if "bounded_repository_planner" in system_prompt:
            self.planner_calls += 1
            if self.cancellation is not None:
                self.cancellation.cancel()
            if self.planner_error == "provider":
                raise urllib.error.URLError("safe fake planner failure")
            self.clock.value = self.planner_end
            raise TimeoutError("safe fake planner cutoff")
        self.final_calls += 1
        if self.final_error == "provider":
            raise urllib.error.URLError("safe fake final failure")
        return _Response(self.final_answer)


def _client(harness: _CutoffProviderHarness) -> LLMClient:
    return LLMClient(
        LLMSettings(
            provider="openai_compatible",
            base_url="https://provider.invalid/v1",
            api_key="test-placeholder",
            model="configured-model",
            timeout_seconds=20.0,
            max_tokens=1600,
            max_retries=0,
        ),
        opener=harness.opener,
        sleep=lambda _seconds: None,
        monotonic=harness.clock,
    )


class F13AgentOrchestrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = Database(Path(self.directory.name) / "f13.sqlite")
        self.project_id, self.bundle = make_project(
            self.database,
            [
                (
                    "src/auth.py",
                    "authenticate_user",
                    "def authenticate_user(password):\n    return verify(password)\n",
                ),
                (
                    "src/admin.py",
                    "authenticate_user",
                    "def authenticate_user(token):\n    return bool(token)\n",
                ),
                (
                    "src/upload.py",
                    "upload_file",
                    "def upload_file(path):\n    return save(path)\n",
                ),
            ],
        )
        self.bundle["project"]["source_type"] = "local"
        self.limits = replace(
            AgentLimits(),
            total_deadline_ms=10_000,
            min_final_answer_budget_ms=3_000,
        )

    def _chat_count(self) -> int:
        with self.database.connect() as connection:
            return int(
                connection.execute("SELECT COUNT(*) FROM chat_answers").fetchone()[0]
            )

    def _budget(self) -> RequestBudget:
        return RequestBudget.from_deadline(
            started_at=0.0,
            deadline_at=10.0,
            final_answer_reserve_ms=3_000,
        )

    def _clock_patches(self, clock: _Clock):
        return (
            patch("app.services.agent_core.time.monotonic", side_effect=clock),
            patch("app.services.agent_tools.time.monotonic", side_effect=clock),
            patch("app.services.qa_agent.time.monotonic", side_effect=clock),
            patch("app.main.time.monotonic", side_effect=clock),
        )

    def _run_cutoff(
        self,
        harness: _CutoffProviderHarness,
        *,
        path: str | None = "src/auth.py",
        symbol: str | None = "authenticate_user",
        cancellation: CancellationToken | None = None,
        limits: AgentLimits | None = None,
        recorder: SmokeDiagnosticsRecorder | None = None,
    ):
        client = _client(harness)
        patches = self._clock_patches(harness.clock)
        with patches[0], patches[1], patches[2], patches[3]:
            return run_bounded_agent(
                "authenticate_user",
                self.bundle,
                client,
                self.database,
                disabled_embedding_service(),
                path=path,
                symbol=symbol,
                evidence_count=1,
                cancellation=cancellation,
                limits=limits or self.limits,
                diagnostics_recorder=recorder,
                request_id="f13-cutoff",
                request_budget=self._budget(),
            )

    def _route(self, harness: _CutoffProviderHarness):
        client = _client(harness)
        patches = self._clock_patches(harness.clock)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patch.object(main, "db", self.database),
            patch.object(main, "llm", client),
            patch.object(main, "embedding_service", disabled_embedding_service()),
            patch.object(main, "learning_service", _LearningService()),
            patch.object(main, "_bundle_or_404", return_value=self.bundle),
            patch.object(main, "agent_limits", self.limits),
        ):
            return asyncio.run(
                _post(
                    main.app,
                    f"/api/projects/{self.project_id}/ask",
                    {
                        "question": "authenticate_user",
                        "path": "src/auth.py",
                        "symbol": "authenticate_user",
                        "evidence_count": 1,
                    },
                )
            )

    def test_production_planner_work_cutoff_recovers_once_and_route_persists_once(self):
        clock = _Clock()
        harness = _CutoffProviderHarness(clock)
        recorder = SmokeDiagnosticsRecorder()
        result = self._run_cutoff(harness, recorder=recorder)

        self.assertEqual(harness.planner_calls, 1)
        self.assertEqual(harness.final_calls, 1)
        self.assertEqual(result["agent_status"], "completed")
        self.assertEqual(result["answer_mode"], "llm_grounded")
        self.assertEqual(len(result["evidence"]), 1)
        diagnostics = recorder.snapshot()
        self.assertEqual(diagnostics["planner_requests_attempted"], 1)
        self.assertTrue(diagnostics["final_answer_attempted"])
        self.assertNotIn("agent_failure_reason_code", diagnostics)
        self.assertEqual(diagnostics["provider_logical_calls"], 2)

        route_clock = _Clock()
        route_harness = _CutoffProviderHarness(route_clock)
        status, body = self._route(route_harness)
        self.assertEqual(status, 200)
        self.assertEqual(body["agent_status"], "completed")
        self.assertEqual(body["answer_mode"], "llm_grounded")
        self.assertEqual(route_harness.planner_calls, 1)
        self.assertEqual(route_harness.final_calls, 1)
        self.assertEqual(self._chat_count(), 1)

    def test_planner_deadline_nonrecoverable_conditions_propagate(self):
        cases = (
            ("request-deadline", {"planner_end": 10.1}, {}),
            ("no-evidence", {}, {"path": None, "symbol": None}),
            ("provider-error", {"planner_error": "provider"}, {}),
        )
        for name, harness_kwargs, run_kwargs in cases:
            with self.subTest(name=name):
                clock = _Clock()
                harness = _CutoffProviderHarness(clock, **harness_kwargs)
                with self.assertRaises(ProviderError):
                    self._run_cutoff(harness, **run_kwargs)
                self.assertEqual(harness.final_calls, 0)

        cancellation = CancellationToken()
        clock = _Clock()
        harness = _CutoffProviderHarness(clock, cancellation=cancellation)
        with self.assertRaises(ProviderError):
            self._run_cutoff(
                harness,
                cancellation=cancellation,
            )
        self.assertEqual(harness.final_calls, 0)

        clock = _Clock()
        harness = _CutoffProviderHarness(clock)
        no_final_budget = replace(self.limits, max_final_answer_tokens=0)
        with self.assertRaises(ProviderError):
            self._run_cutoff(harness, limits=no_final_budget)
        self.assertEqual(harness.final_calls, 0)

    def test_final_provider_and_validation_failures_keep_zero_persistence(self):
        provider_harness = _CutoffProviderHarness(
            _Clock(),
            final_error="provider",
        )
        status, body = self._route(provider_harness)
        self.assertEqual(status, 503)
        self.assertEqual(body["detail"]["code"], "provider_unavailable")
        self.assertEqual(provider_harness.planner_calls, 1)
        self.assertEqual(provider_harness.final_calls, 1)
        self.assertEqual(self._chat_count(), 0)

        validation_harness = _CutoffProviderHarness(
            _Clock(),
                final_answer='{"parts":[{"text":"Unsafe answer","evidence_aliases":["A999"]}]}',
        )
        status, body = self._route(validation_harness)
        self.assertEqual(status, 502)
        self.assertEqual(body["detail"]["code"], "citation_unknown")
        self.assertEqual(validation_harness.planner_calls, 1)
        self.assertEqual(validation_harness.final_calls, 2)
        self.assertTrue(
            body["detail"]["diagnostics"]["final_answer_repair_attempted"]
        )
        self.assertFalse(
            body["detail"]["diagnostics"]["final_answer_repair_succeeded"]
        )
        self.assertEqual(self._chat_count(), 0)

    def test_top_k_uses_formal_defaults_and_request_cap_for_all_supported_tools(self):
        for action, arguments, expected in (
            ("lookup_symbol", {"symbol": "authenticate_user"}, 1),
            (
                "lookup_symbol",
                {"symbol": "authenticate_user", "top_k": 20},
                1,
            ),
            ("search_code", {"query": "authenticate_user", "top_k": 20}, 1),
        ):
            with self.subTest(action=action, arguments=arguments):
                registry = build_m2_tool_registry(AgentLimits())
                spec = registry.get(action)
                captured = []

                def capture(context, parameters, *, _handler=spec.handler):
                    captured.append(parameters.model_dump())
                    return _handler(context, parameters)

                registry._tools[action] = replace(spec, handler=capture)
                result = run_bounded_agent(
                    "authenticate_user",
                    self.bundle,
                    NoLlm(),
                    self.database,
                    disabled_embedding_service(),
                    planner=ScriptedPlanner(
                        [decision("continue", action, arguments), decision("answer")]
                    ),
                    evidence_count=1,
                    registry=registry,
                )
                self.assertEqual(captured[0]["top_k"], expected)
                self.assertEqual(result["budget_usage"]["tool_calls_used"], 1)

        planner_arguments = {
            "path": "src/auth.py",
            "start_line": 1,
            "end_line": 2,
        }
        registry = build_m2_tool_registry(AgentLimits())
        read_spec = registry.get("read_source")
        captured_read = []

        def capture_read(context, parameters):
            captured_read.append(parameters.model_dump())
            return read_spec.handler(context, parameters)

        registry._tools["read_source"] = replace(read_spec, handler=capture_read)
        run_bounded_agent(
            "authenticate_user",
            self.bundle,
            NoLlm(),
            self.database,
            disabled_embedding_service(),
            planner=ScriptedPlanner(
                [
                    decision("continue", "read_source", planner_arguments),
                    decision("answer"),
                ]
            ),
            evidence_count=1,
            registry=registry,
        )
        self.assertNotIn("top_k", captured_read[0])
        self.assertNotIn("top_k", planner_arguments)

    def test_invalid_top_k_reaches_formal_parameter_validation_unchanged(self):
        for invalid in ("1", 0, -1, 21):
            with self.subTest(top_k=invalid):
                arguments = {"symbol": "authenticate_user", "top_k": invalid}
                validation = validate_planner_decision(
                    decision("continue", "lookup_symbol", arguments),
                    build_m2_tool_registry(AgentLimits()),
                )
                self.assertFalse(validation.valid)
                self.assertEqual(validation.failure.stage, "semantic")
                self.assertEqual(
                    validation.failure.stable_code,
                    "semantic_invalid_tool_contract",
                )
                self.assertEqual(
                    validation.failure.field_path, ("arguments", "top_k")
                )
                self.assertEqual(arguments["top_k"], invalid)

    def test_server_bound_windows_path_is_shared_without_mutating_inputs(self):
        search_arguments = {
            "query": "upload_file",
            "path": "src/upload.py",
            "top_k": 2,
        }
        lookup_arguments = {
            "symbol": "authenticate_user",
            "path": "src/admin.py",
        }
        read_arguments = {
            "path": "src/upload.py",
            "start_line": 1,
            "end_line": 2,
        }
        originals = json.loads(
            json.dumps([search_arguments, lookup_arguments, read_arguments])
        )
        request = main.AskRequest(
            question="authenticate_user",
            path="src\\auth.py",
            evidence_count=1,
        )
        registry = build_m2_tool_registry(AgentLimits())
        captured = {"lookup_symbol": [], "read_source": []}
        for action in captured:
            spec = registry.get(action)

            def capture(context, parameters, *, _action=action, _handler=spec.handler):
                captured[_action].append(parameters.model_dump())
                return _handler(context, parameters)

            registry._tools[action] = replace(spec, handler=capture)

        from app.services import agent_tools

        real_retrieve = agent_tools.retrieve_code
        search_paths = []

        def capture_search(*args, **kwargs):
            search_paths.append(kwargs.get("path"))
            return real_retrieve(*args, **kwargs)

        with patch.object(agent_tools, "retrieve_code", side_effect=capture_search):
            result = run_bounded_agent(
                "authenticate_user",
                self.bundle,
                NoLlm(),
                self.database,
                disabled_embedding_service(),
                planner=ScriptedPlanner(
                    [
                        decision("continue", "search_code", search_arguments),
                        decision("continue", "lookup_symbol", lookup_arguments),
                        decision("continue", "read_source", read_arguments),
                        decision("answer"),
                    ]
                ),
                path=request.path,
                evidence_count=1,
                limits=replace(AgentLimits(), max_agent_steps=4),
                registry=registry,
            )

        self.assertTrue(search_paths)
        self.assertTrue(all(path == "src/auth.py" for path in search_paths))
        self.assertEqual(captured["lookup_symbol"][0]["path"], "src/auth.py")
        self.assertEqual(captured["read_source"][0]["path"], "src/auth.py")
        self.assertEqual(request.path, "src\\auth.py")
        self.assertEqual(
            [search_arguments, lookup_arguments, read_arguments],
            originals,
        )
        self.assertEqual(result["evidence"][0]["path"], "src/auth.py")

    def test_path_normalizer_rejects_unsafe_inputs_and_preserves_posix_behavior(self):
        self.assertEqual(
            normalize_repository_relative_path("src\\auth.py"), "src/auth.py"
        )
        self.assertEqual(
            normalize_repository_relative_path("src/auth.py"), "src/auth.py"
        )
        for unsafe in (
            "../secret.py",
            "/secret.py",
            "C:\\secret.py",
            "C:/secret.py",
            "\\\\server\\share\\secret.py",
            "",
            "   ",
            "src/\0secret.py",
        ):
            with self.subTest(path=repr(unsafe)):
                with self.assertRaises(ValueError):
                    normalize_repository_relative_path(unsafe)
                with self.assertRaises(ValidationError):
                    main.AskRequest(question="q", path=unsafe)

    def test_unknown_tool_has_one_fixed_safe_diagnostic_and_shared_failure_payload(self):
        raw_unknown = "private_unregistered_tool"
        recorder = SmokeDiagnosticsRecorder()
        result = run_bounded_agent(
            "authenticate_user",
            self.bundle,
            NoLlm(),
            self.database,
            disabled_embedding_service(),
            planner=ScriptedPlanner(
                [decision("continue", raw_unknown, {}), decision("answer")]
            ),
            diagnostics_recorder=recorder,
            request_id="unknown-tool-request",
        )
        diagnostics = recorder.snapshot()
        self.assertEqual(diagnostics.get("tool_executions", []), [])
        self.assertEqual(diagnostics.get("tool_calls_attempted", 0), 0)
        self.assertEqual(diagnostics.get("tool_calls_failed", 0), 0)
        self.assertEqual(result["budget_usage"]["tool_calls_used"], 0)
        self.assertEqual(result["evidence"], [])
        attempts = diagnostics["planner_attempts"]
        self.assertEqual(attempts[0]["stage"], "semantic")
        self.assertEqual(
            attempts[0]["stable_code"], "semantic_invalid_tool_contract"
        )
        self.assertEqual(attempts[0]["field_path"], ["action"])
        self.assertTrue(attempts[1]["repair_attempt"])
        serialized = json.dumps(diagnostics)
        self.assertNotIn(raw_unknown, serialized)
        self.assertLessEqual(len(serialized.encode("utf-8")), 4_096)

        provider = type(
            "UnknownToolProvider",
            (),
            {
                "available": True,
                "require_available": lambda self: None,
                "chat": lambda self, _messages, **_kwargs: json.dumps(
                    decision("continue", raw_unknown, {})
                ),
            },
        )()
        captured = []
        with (
            patch.object(main, "db", self.database),
            patch.object(main, "llm", provider),
            patch.object(main, "embedding_service", disabled_embedding_service()),
            patch.object(main, "learning_service", _LearningService()),
            patch.object(main, "_bundle_or_404", return_value=self.bundle),
            patch.object(main, "_log_ask_failure", side_effect=captured.append),
        ):
            with self.assertRaises(HTTPException) as raised:
                main.ask_project(
                    self.project_id,
                    main.AskRequest(question="authenticate_user"),
                )
        detail = raised.exception.detail
        self.assertIs(captured[0], detail)
        self.assertEqual(captured[0]["diagnostics"], detail["diagnostics"])
        self.assertEqual(
            json.loads(format_ask_failure_log(detail))["diagnostics"],
            detail["diagnostics"],
        )
        self.assertEqual(detail["code"], "planner_repair_failed")
        self.assertEqual(detail["diagnostics"]["tool_executions"], [])
        self.assertEqual(detail["diagnostics"]["tool_calls_used"], 0)
        self.assertEqual(
            [item["stable_code"] for item in detail["diagnostics"]["planner_attempts"]],
            ["semantic_invalid_tool_contract", "semantic_invalid_tool_contract"],
        )
        self.assertNotIn(raw_unknown, json.dumps(detail))


if __name__ == "__main__":
    unittest.main()
