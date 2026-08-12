from __future__ import annotations

import asyncio
from contextlib import ExitStack
from dataclasses import replace
import json
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app import main
from app.database import Database
from app.services import agent_core
from app.services.agent_contracts import AgentLimits, RequestBudget
from app.services.agent_core import run_bounded_agent
from app.services.agent_tools import EvidenceStore, build_m2_tool_registry
from app.services.ask_diagnostics import format_ask_failure_log
from app.services.llm_client import ProviderError
from app.services.smoke_diagnostics import SmokeDiagnosticsRecorder
from tests.m1_helpers import disabled_embedding_service, make_project
from tests.test_f12_agent_orchestration import _LearningService, _post
from tests.test_f13_agent_orchestration import (
    _Clock,
    _CutoffProviderHarness,
    _client,
)
from tests.test_m2_agent import decision


class _LogCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(dict(record.__dict__))


class _ProductionProvider:
    available = True

    def __init__(self, planner_events, *, final_answer=None) -> None:
        self.planner_events = list(planner_events)
        self.final_answer = final_answer or json.dumps(
            {"parts": [{"text": "Grounded", "evidence_aliases": ["A1"]}]}
        )
        self.planner_calls = 0
        self.final_calls = 0

    def require_available(self) -> None:
        return None

    def chat(self, messages, **kwargs):
        purpose = kwargs.get("purpose")
        is_planner = purpose == "planner" or "bounded_repository_planner" in str(
            messages[0].get("content", "")
        )
        if is_planner:
            event = self.planner_events[
                min(self.planner_calls, len(self.planner_events) - 1)
            ]
            self.planner_calls += 1
            if isinstance(event, BaseException):
                raise event
            return json.dumps(event)
        self.final_calls += 1
        return self.final_answer


class F15AgentOrchestrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = Database(Path(self.directory.name) / "f15.sqlite")
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
        self.limits = replace(
            AgentLimits(),
            max_agent_steps=5,
            total_deadline_ms=10_000,
            min_final_answer_budget_ms=3_000,
        )

    def _chat_count(self) -> int:
        with self.database.connect() as connection:
            return int(
                connection.execute("SELECT COUNT(*) FROM chat_answers").fetchone()[0]
            )

    @staticmethod
    def _clock_patches(clock: _Clock):
        return (
            patch("app.services.agent_core.time.monotonic", side_effect=clock),
            patch("app.services.agent_tools.time.monotonic", side_effect=clock),
            patch("app.services.qa_agent.time.monotonic", side_effect=clock),
            patch("app.main.time.monotonic", side_effect=clock),
        )

    def _route(
        self,
        provider,
        payload: dict,
        *,
        limits: AgentLimits | None = None,
        registry=None,
        clock: _Clock | None = None,
        captured_results: list[dict] | None = None,
    ):
        original_run = agent_core.run_bounded_agent

        def capture_run(*args, **kwargs):
            result = original_run(*args, **kwargs)
            if captured_results is not None:
                captured_results.append(result)
            return result

        with ExitStack() as stack:
            stack.enter_context(patch.object(main, "db", self.database))
            stack.enter_context(patch.object(main, "llm", provider))
            stack.enter_context(
                patch.object(main, "embedding_service", disabled_embedding_service())
            )
            stack.enter_context(patch.object(main, "learning_service", _LearningService()))
            stack.enter_context(patch.object(main, "_bundle_or_404", return_value=self.bundle))
            stack.enter_context(
                patch.object(main, "agent_limits", limits or self.limits)
            )
            stack.enter_context(
                patch.object(main, "run_bounded_agent", side_effect=capture_run)
            )
            if registry is not None:
                stack.enter_context(
                    patch.object(
                        agent_core, "build_m2_tool_registry", return_value=registry
                    )
                )
            if clock is not None:
                for clock_patch in self._clock_patches(clock):
                    stack.enter_context(clock_patch)
            return asyncio.run(
                _post(
                    main.app,
                    f"/api/projects/{self.project_id}/ask",
                    payload,
                )
            )

    def test_tool_evidence_then_work_cutoff_cannot_recover(self):
        clock = _Clock()
        provider = _ProductionProvider(
            [decision("continue", "search_code", {"query": "authenticate_user"})]
        )
        captured_results: list[dict] = []
        captured_failures: list[dict] = []
        evidence_added: list[int] = []
        original_add = EvidenceStore.add

        def add_then_cross_cutoff(store, owner_id, evidence):
            added = original_add(store, owner_id, evidence)
            evidence_added.append(len(added))
            clock.value = 7.1
            return added

        with (
            patch.object(
                EvidenceStore, "add", autospec=True, side_effect=add_then_cross_cutoff
            ),
            patch.object(
                agent_core, "answer_from_evidence", wraps=agent_core.answer_from_evidence
            ) as finalization,
            patch.object(
                self.database,
                "save_chat_answer",
                wraps=self.database.save_chat_answer,
            ) as save,
            patch.object(main, "_log_ask_failure", side_effect=captured_failures.append),
        ):
            status, body = self._route(
                provider,
                {"question": "authenticate_user", "evidence_count": 1},
                clock=clock,
                captured_results=captured_results,
            )

        self.assertEqual(evidence_added, [1])
        self.assertEqual(finalization.call_count, 0)
        self.assertEqual(provider.final_calls, 0)
        self.assertEqual(save.call_count, 0)
        self.assertEqual(self._chat_count(), 0)
        self.assertNotEqual(status, 200)
        self.assertEqual(body["detail"]["code"], "final_answer_not_attempted")
        self.assertEqual(captured_failures[0]["code"], body["detail"]["code"])
        self.assertNotEqual(captured_results[0]["agent_status"], "completed")
        self.assertNotEqual(captured_results[0]["answer_mode"], "llm_grounded")

    def test_evidence_then_planner_token_budget_exhaustion_cannot_recover(self):
        first = decision(
            "continue", "search_code", {"query": "authenticate_user", "top_k": 1}
        )
        encoded = json.dumps(first)
        first_tokens = max(1, (len(encoded) + 3) // 4)
        limits = replace(
            self.limits,
            max_total_planner_output_tokens=first_tokens,
            max_planner_output_tokens_per_step=max(512, first_tokens),
        )
        provider = _ProductionProvider([first])
        captured_results: list[dict] = []
        with (
            patch.object(
                agent_core, "answer_from_evidence", wraps=agent_core.answer_from_evidence
            ) as finalization,
            patch.object(
                self.database,
                "save_chat_answer",
                wraps=self.database.save_chat_answer,
            ) as save,
        ):
            status, body = self._route(
                provider,
                {"question": "authenticate_user", "evidence_count": 1},
                limits=limits,
                captured_results=captured_results,
            )

        self.assertEqual(provider.planner_calls, 1)
        self.assertEqual(provider.final_calls, 0)
        self.assertEqual(finalization.call_count, 0)
        self.assertEqual(save.call_count, 0)
        self.assertEqual(self._chat_count(), 0)
        self.assertEqual(status, 503)
        self.assertEqual(body["detail"]["code"], "planner_budget_exhausted")
        self.assertEqual(body["detail"]["diagnostics"]["evidence_count"], 1)
        self.assertNotEqual(captured_results[0]["agent_status"], "completed")
        self.assertNotEqual(captured_results[0]["answer_mode"], "llm_grounded")

    def test_f13_planner_deadline_authorization_recovers_once_and_persists_once(self):
        clock = _Clock()
        harness = _CutoffProviderHarness(clock)
        client = _client(harness)
        with (
            patch.object(
                agent_core, "answer_from_evidence", wraps=agent_core.answer_from_evidence
            ) as finalization,
            patch.object(
                self.database,
                "save_chat_answer",
                wraps=self.database.save_chat_answer,
            ) as save,
        ):
            status, body = self._route(
                client,
                {
                    "question": "authenticate_user",
                    "path": "src/auth.py",
                    "symbol": "authenticate_user",
                    "evidence_count": 1,
                },
                clock=clock,
            )

        self.assertEqual(status, 200)
        self.assertEqual(body["agent_status"], "completed")
        self.assertEqual(body["answer_mode"], "llm_grounded")
        self.assertEqual(harness.planner_calls, 1)
        self.assertEqual(harness.final_calls, 1)
        self.assertEqual(finalization.call_count, 1)
        self.assertEqual(save.call_count, 1)
        self.assertEqual(self._chat_count(), 1)

    def test_non_deadline_planner_provider_error_with_evidence_never_recovers(self):
        clock = _Clock()
        harness = _CutoffProviderHarness(clock, planner_error="provider")
        client = _client(harness)
        with (
            patch.object(
                agent_core, "answer_from_evidence", wraps=agent_core.answer_from_evidence
            ) as finalization,
            patch.object(
                self.database,
                "save_chat_answer",
                wraps=self.database.save_chat_answer,
            ) as save,
        ):
            status, body = self._route(
                client,
                {
                    "question": "authenticate_user",
                    "path": "src/auth.py",
                    "symbol": "authenticate_user",
                    "evidence_count": 1,
                },
                clock=clock,
            )

        self.assertEqual(status, 503)
        self.assertEqual(body["detail"]["code"], "provider_unavailable")
        self.assertEqual(body["detail"]["diagnostics"]["evidence_count"], 1)
        self.assertEqual(harness.planner_calls, 1)
        self.assertEqual(harness.final_calls, 0)
        self.assertEqual(finalization.call_count, 0)
        self.assertEqual(save.call_count, 0)
        self.assertEqual(self._chat_count(), 0)

    def test_unknown_raw_name_is_resolved_once_and_never_leaks_or_executes(self):
        raw_unknown = "Authorization:Bearer secret_token=abc\r\n" + "X" * 40
        provider = _ProductionProvider(
            [
                decision("continue", raw_unknown, {}),
                decision("insufficient_evidence"),
            ]
        )
        registry = build_m2_tool_registry(self.limits)
        captured_results: list[dict] = []
        captured_failures: list[dict] = []
        collector = _LogCollector()
        log = logging.getLogger("app.services.agent_core")
        previous_level = log.level
        log.setLevel(logging.INFO)
        log.addHandler(collector)
        try:
            with (
                patch.object(registry, "get", wraps=registry.get) as get_tool,
                patch.object(registry, "execute", wraps=registry.execute) as execute,
                patch.object(
                    registry,
                    "execute_resolved",
                    wraps=registry.execute_resolved,
                ) as execute_resolved,
                patch.object(
                    self.database,
                    "save_chat_answer",
                    wraps=self.database.save_chat_answer,
                ) as save,
                patch.object(main, "_log_ask_failure", side_effect=captured_failures.append),
            ):
                status, body = self._route(
                    provider,
                    {"question": "authenticate_user", "evidence_count": 1},
                    registry=registry,
                    captured_results=captured_results,
                )
        finally:
            log.removeHandler(collector)
            log.setLevel(previous_level)

        raw_gets = [call for call in get_tool.call_args_list if call.args == (raw_unknown,)]
        self.assertEqual(len(raw_gets), 1)
        self.assertEqual(execute.call_count, 0)
        self.assertEqual(execute_resolved.call_count, 0)
        self.assertEqual(save.call_count, 0)
        self.assertEqual(self._chat_count(), 0)
        self.assertEqual(status, 422)
        self.assertEqual(body["detail"]["diagnostics"]["evidence_count"], 0)
        self.assertEqual(body["detail"]["diagnostics"]["tool_calls_used"], 0)
        self.assertEqual(
            captured_results[0]["agent_trace"][0]["action"],
            "insufficient_evidence",
        )
        attempts = body["detail"]["diagnostics"]["planner_attempts"]
        self.assertEqual(attempts[0]["stage"], "semantic")
        self.assertEqual(
            attempts[0]["stable_code"], "semantic_invalid_tool_contract"
        )
        self.assertEqual(attempts[0]["field_path"], ["action"])
        self.assertTrue(attempts[1]["repair_attempt"])
        serialized = json.dumps(
            {
                "body": body,
                "result": captured_results,
                "failure": captured_failures,
                "failure_log": format_ask_failure_log(captured_failures[0]),
                "records": collector.records,
            },
            default=str,
        )
        self.assertNotIn(raw_unknown, serialized)
        self.assertNotIn("Authorization:Bearer", serialized)
        self.assertNotIn('"tool_version": "unknown"', serialized)

    def test_unknown_then_valid_tool_keeps_one_lookup_and_successful_finalization(self):
        raw_unknown = "Authorization:Bearer secret_token=abc\0" + "Y" * 40
        provider = _ProductionProvider(
            [
                decision("continue", raw_unknown, {}),
                decision(
                    "continue",
                    "search_code",
                    {"query": "authenticate_user", "path": "src/auth.py"},
                ),
                decision("answer"),
            ]
        )
        registry = build_m2_tool_registry(self.limits)
        captured_results: list[dict] = []
        with (
            patch.object(registry, "get", wraps=registry.get) as get_tool,
            patch.object(
                registry,
                "execute_resolved",
                wraps=registry.execute_resolved,
            ) as execute_resolved,
            patch.object(
                agent_core, "answer_from_evidence", wraps=agent_core.answer_from_evidence
            ) as finalization,
            patch.object(
                self.database,
                "save_chat_answer",
                wraps=self.database.save_chat_answer,
            ) as save,
        ):
            status, body = self._route(
                provider,
                {"question": "authenticate_user", "evidence_count": 1},
                registry=registry,
                captured_results=captured_results,
            )

        raw_gets = [call for call in get_tool.call_args_list if call.args == (raw_unknown,)]
        self.assertEqual(len(raw_gets), 1)
        self.assertEqual(execute_resolved.call_count, 1)
        resolved_spec = execute_resolved.call_args.args[2]
        self.assertEqual(resolved_spec.name, "search_code")
        self.assertEqual(resolved_spec.input_model.__name__, "SearchCodeInput")
        self.assertEqual(status, 200)
        self.assertEqual(provider.final_calls, 1)
        self.assertEqual(finalization.call_count, 1)
        self.assertEqual(save.call_count, 1)
        self.assertEqual(self._chat_count(), 1)
        self.assertEqual(body["budget_usage"]["tool_calls_used"], 1)
        self.assertEqual(body["agent_trace"][0]["action"], "search_code")
        self.assertTrue(body["evidence"])
        self.assertNotIn(raw_unknown, json.dumps({"body": body, "result": captured_results}))

    def test_request_local_recovery_store_registry_recorder_and_request_id_are_isolated(self):
        registry = build_m2_tool_registry(self.limits)
        recorder_a = SmokeDiagnosticsRecorder()
        recorder_b = SmokeDiagnosticsRecorder()
        clock = _Clock()
        authorized_harness = _CutoffProviderHarness(clock)
        authorized_client = _client(authorized_harness)
        raw_unknown = "Authorization:secret\n" + "Z" * 40
        non_deadline = ProviderError(
            code="provider_error",
            message="safe fake provider failure",
            status_code=502,
        )
        second_provider = _ProductionProvider(
            [decision("continue", raw_unknown, {}), non_deadline]
        )
        budget = RequestBudget.from_deadline(
            started_at=0.0,
            deadline_at=10.0,
            final_answer_reserve_ms=3_000,
        )
        with (
            patch.object(registry, "get", wraps=registry.get) as get_tool,
            patch.object(
                agent_core, "answer_from_evidence", wraps=agent_core.answer_from_evidence
            ) as finalization,
        ):
            with ExitStack() as stack:
                for clock_patch in self._clock_patches(clock)[:3]:
                    stack.enter_context(clock_patch)
                result_a = run_bounded_agent(
                    "authenticate_user",
                    self.bundle,
                    authorized_client,
                    self.database,
                    disabled_embedding_service(),
                    path="src/auth.py",
                    symbol="authenticate_user",
                    evidence_count=1,
                    limits=self.limits,
                    registry=registry,
                    diagnostics_recorder=recorder_a,
                    request_id="f15-request-a",
                    request_budget=budget,
                )
            with self.assertRaises(ProviderError):
                run_bounded_agent(
                    "authenticate_user",
                    self.bundle,
                    second_provider,
                    self.database,
                    disabled_embedding_service(),
                    evidence_count=1,
                    limits=self.limits,
                    registry=registry,
                    diagnostics_recorder=recorder_b,
                    request_id="f15-request-b",
                )

        snapshot_a = recorder_a.snapshot()
        snapshot_b = recorder_b.snapshot()
        self.assertEqual(result_a["agent_status"], "completed")
        self.assertEqual(result_a["answer_mode"], "llm_grounded")
        self.assertEqual(finalization.call_count, 1)
        self.assertEqual(authorized_harness.final_calls, 1)
        self.assertEqual(second_provider.final_calls, 0)
        self.assertEqual(snapshot_a["request_id"], "f15-request-a")
        self.assertEqual(snapshot_b["request_id"], "f15-request-b")
        self.assertEqual(snapshot_a["evidence_count"], 1)
        self.assertEqual(snapshot_b["evidence_count"], 0)
        self.assertIsNot(recorder_a, recorder_b)
        raw_gets = [call for call in get_tool.call_args_list if call.args == (raw_unknown,)]
        self.assertEqual(len(raw_gets), 1)


if __name__ == "__main__":
    unittest.main()
