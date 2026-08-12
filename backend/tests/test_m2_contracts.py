from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from app.config import get_agent_limits
from app.database import Database
from app.services.agent_contracts import (
    AgentLimits,
    AgentStep,
    PlannerDecision,
    RequestBudget,
    ToolCall,
    ToolObservation,
)
from app.services.agent_core import (
    build_planner_json_schema,
    canonical_planner_json_schema,
    run_bounded_agent,
    validate_planner_decision,
)
from app.services.agent_tools import build_m2_tool_registry
from tests.m1_helpers import disabled_embedding_service, make_project
from tests.test_m2_agent import ScriptedPlanner, decision


class RecordingFinalLlm:
    available = True

    def __init__(self):
        self.kwargs = None

    def chat(self, _messages, **kwargs):
        self.kwargs = kwargs
        return json.dumps(
            {"parts": [{"text": "证据位置", "evidence_aliases": ["A1"]}]},
            ensure_ascii=False,
        )


class M2ContractTests(unittest.TestCase):
    def setUp(self):
        self.registry = build_m2_tool_registry(AgentLimits())

    def test_frozen_default_limits(self):
        self.assertEqual(
            asdict(AgentLimits()),
            {
                "max_agent_steps": 5,
                "max_tool_calls": 8,
                "max_calls_per_step": 1,
                "max_same_tool_calls": 3,
                "max_no_progress_steps": 2,
                "total_deadline_ms": 60000,
                "default_tool_timeout_ms": 40000,
                "min_final_answer_budget_ms": 5000,
                "max_search_results": 20,
                "max_observation_bytes": 65536,
                "max_source_read_lines": 200,
                "max_source_read_bytes": 32768,
                "max_accumulated_evidence_context_bytes": 49152,
                "max_planner_output_tokens_per_step": 512,
                "max_total_planner_output_tokens": 2048,
                "max_final_answer_tokens": 1600,
                "default_relation_depth": 1,
                "max_relation_depth": 2,
                "max_relation_seed_nodes": 8,
                "max_relation_neighbors_per_node": 20,
                "max_relation_nodes": 64,
                "max_relation_edges": 128,
                "max_relation_paths": 24,
                "max_relation_observation_bytes": 65536,
                "max_relation_evidence_items": 16,
                "max_learning_state_items": 16,
                "max_recent_learning_events": 8,
                "max_plan_steps_in_learning_context": 12,
                "max_learning_context_bytes": 16384,
            },
        )

    def test_environment_can_reduce_but_not_raise_server_limits(self):
        with patch.dict(
            os.environ,
            {
                "AGENT_MAX_STEPS": "3",
                "AGENT_MAX_TOOL_CALLS": "999",
                "AGENT_TOTAL_DEADLINE_MS": "500",
                "AGENT_TOOL_TIMEOUT_MS": "15000",
                "AGENT_MAX_SOURCE_READ_LINES": "50",
                "AGENT_MAX_FINAL_ANSWER_TOKENS": "not-an-int",
                "AGENT_FINAL_ANSWER_RESERVE_MS": "999999",
            },
            clear=False,
        ):
            limits = get_agent_limits()
        self.assertEqual(limits.max_agent_steps, 3)
        self.assertEqual(limits.max_tool_calls, 8)
        self.assertEqual(limits.total_deadline_ms, 60_000)
        self.assertEqual(limits.default_tool_timeout_ms, 15_000)
        self.assertEqual(limits.max_source_read_lines, 50)
        self.assertEqual(limits.max_final_answer_tokens, 1_600)
        self.assertEqual(limits.min_final_answer_budget_ms, 5_000)

    def test_default_tool_budget_covers_observed_cold_start_without_double_reserve(self):
        limits = AgentLimits()
        budget = RequestBudget.create(started_at=0.0, limits=limits)
        tool_started_at = 1.812

        stage = budget.tool_deadline(
            started_at=tool_started_at,
            timeout_ms=limits.default_tool_timeout_ms,
        )

        self.assertEqual(stage.reason, "tool_timeout")
        self.assertAlmostEqual(stage.deadline_monotonic, 41.812)
        self.assertGreater(stage.deadline_monotonic - tool_started_at, 30.516)
        self.assertAlmostEqual(
            budget.request_deadline_at - stage.deadline_monotonic,
            18.188,
        )

    def test_planner_contract_forbids_extra_fields_and_long_private_trace(self):
        with self.assertRaises(ValidationError):
            PlannerDecision.model_validate(
                {
                    "status": "answer",
                    "decision_summary": "done",
                    "private_chain_of_thought": "secret",
                }
            )
        with self.assertRaises(ValidationError):
            PlannerDecision.model_validate(
                {
                    "status": "answer",
                    "decision_summary": "x" * 241,
                }
            )

    def test_planner_parser_accepts_only_raw_or_one_exact_outer_fence(self):
        body = (
            '{"status":"answer","action":null,"arguments":{},'
            '"decision_summary":"done"}'
        )
        accepted = (
            body,
            "\ufeff  " + body + "\r\n",
            "```json\n" + body + "\n```",
            "```\r\n" + body + "\r\n```",
        )
        for raw in accepted:
            with self.subTest(raw=raw[:12]):
                result = validate_planner_decision(raw, self.registry)
                self.assertTrue(result.valid)
                self.assertEqual(result.decision.status, "answer")

        rejected = (
            "Here is JSON:\n" + body,
            "```json\n" + body + "\n```\n```\n" + body + "\n```",
            body + " trailing",
            body + "\n" + body,
            body[:-1],
        )
        for raw in rejected:
            with self.subTest(raw=raw[:20]):
                failure = validate_planner_decision(raw, self.registry).failure
                self.assertEqual(failure.stage, "parser")
                self.assertEqual(failure.stable_code, "invalid_json")
                self.assertEqual(failure.field_path, ())
                serialized = repr(failure.to_safe_dict())
                self.assertNotIn(raw, serialized)

        top_level = validate_planner_decision("[]", self.registry).failure
        self.assertEqual(
            (top_level.stage, top_level.stable_code, top_level.field_path),
            ("parser", "wrong_top_level_type", ()),
        )

    def test_planner_schema_failures_have_stable_code_and_safe_path(self):
        cases = (
            (
                {"status": "answer", "action": None, "arguments": {}},
                "schema_missing_field",
                ("decision_summary",),
            ),
            (
                {
                    "status": "answer",
                    "action": None,
                    "arguments": {},
                    "decision_summary": "done",
                    "extra": "not allowed",
                },
                "schema_extra_field",
                ("extra",),
            ),
            (
                {
                    "status": "answer",
                    "action": None,
                    "arguments": {},
                    "decision_summary": 1,
                },
                "schema_invalid_type",
                ("decision_summary",),
            ),
            (
                {
                    "status": "finished",
                    "action": None,
                    "arguments": {},
                    "decision_summary": "done",
                },
                "schema_invalid_literal",
                ("status",),
            ),
        )
        for raw, code, path in cases:
            with self.subTest(code=code):
                failure = validate_planner_decision(raw, self.registry).failure
                self.assertEqual((failure.stage, failure.stable_code), ("schema", code))
                self.assertEqual(failure.field_path, path)

    def test_planner_semantics_reject_unknown_tool_bad_input_and_terminal_action(self):
        cases = (
            (
                decision("continue", "shell", {"command": "whoami"}),
                "semantic_invalid_tool_contract",
                ("action",),
            ),
            (
                decision("continue", "search_code", {"query": 1}),
                "semantic_invalid_tool_contract",
                ("arguments", "query"),
            ),
            (
                decision("answer", "search_code", {"query": "x"}),
                "semantic_invalid_decision",
                ("action",),
            ),
        )
        for raw, code, path in cases:
            with self.subTest(code=code, path=path):
                failure = validate_planner_decision(raw, self.registry).failure
                self.assertEqual((failure.stage, failure.stable_code), ("semantic", code))
                self.assertEqual(failure.field_path, path)

    def test_planner_schema_is_canonical_and_includes_every_production_tool(self):
        schema = build_planner_json_schema(self.registry)
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), {"status", "decision_summary"})
        self.assertEqual(
            set(schema["x-tool-input-schemas"]),
            {item["name"] for item in self.registry.list_tools()},
        )
        self.assertEqual(
            canonical_planner_json_schema(self.registry),
            canonical_planner_json_schema(self.registry),
        )
        for item in self.registry.list_tools():
            self.assertEqual(
                schema["x-tool-input-schemas"][item["name"]],
                item["input_schema"],
            )
    def test_runtime_contracts_and_public_trace_are_complete_but_source_free(self):
        call = ToolCall("C1", "S1", "search_code", "1", {"query": "secret"}, 100, {})
        observation = ToolObservation(
            "C1",
            "succeeded",
            structured_results={"content": "sensitive source"},
            metrics={"duration_ms": 1, "result_count": 1, "output_bytes": 10},
        )
        step = AgentStep(
            "S1",
            "goal",
            "search_code",
            [call],
            [observation],
            "search relevant code",
            "running",
            {"steps": 4},
        )
        public = step.to_public_dict()
        self.assertNotIn("parameters", str(public))
        self.assertNotIn("sensitive source", str(public))
        self.assertEqual(public["tool_calls"][0]["result_count"], 1)

    def test_agent_enforces_final_generation_token_limit(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        database = Database(Path(directory.name) / "contract.sqlite")
        _project_id, bundle = make_project(
            database,
            [
                (
                    "src/auth.py",
                    "authenticate_user",
                    "def authenticate_user(password):\n    return verify(password)\n",
                )
            ],
        )
        llm = RecordingFinalLlm()
        planner = ScriptedPlanner(
            [
                decision("continue", "search_code", {"query": "authenticate_user"}),
                decision("answer"),
            ]
        )
        result = run_bounded_agent(
            "authenticate_user",
            bundle,
            llm,
            database,
            disabled_embedding_service(),
            planner=planner,
        )
        self.assertEqual(result["agent_status"], "completed")
        self.assertEqual(llm.kwargs["max_tokens"], 1600)
        self.assertLessEqual(llm.kwargs["timeout_seconds"], 60)


if __name__ == "__main__":
    unittest.main()
