from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.config import LLMSettings, get_agent_limits, get_llm_settings
from app.database import Database
from app.services import agent_core
from app.services.agent_contracts import AgentLimits
from app.services.agent_core import LLMPlanner, run_bounded_agent
from app.services.agent_tools import build_m2_tool_registry
from app.services.llm_client import LLMClient
from app.services.qa_agent import _chat_final_answer
from tests.m1_helpers import disabled_embedding_service, make_project
from tests.test_m2_agent import NoLlm, ScriptedPlanner, decision


class QualityBudgetConfigurationTests(unittest.TestCase):
    def test_legacy_defaults_and_invalid_value_fallback_are_preserved(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_agent_limits(), AgentLimits())
        with patch.dict(os.environ, {
            "AGENT_TOTAL_DEADLINE_MS": "180000",
            "AGENT_MAX_PLANNER_TOKENS_PER_STEP": "1024",
            "AGENT_MAX_TOTAL_PLANNER_TOKENS": "8192",
            "AGENT_MAX_FINAL_ANSWER_TOKENS": "4096",
            "AGENT_BASE_EVIDENCE_COUNT": "3",
        }, clear=True):
            self.assertEqual(get_agent_limits(), AgentLimits())

    def test_quality_defaults_and_valid_overrides_are_resolved(self):
        with patch.dict(os.environ, {"AGENT_BUDGET_PROFILE": "quality"}, clear=True):
            limits = get_agent_limits()
        self.assertEqual((limits.budget_profile, limits.base_evidence_count), ("quality", 5))
        self.assertEqual((limits.total_deadline_ms, limits.min_final_answer_budget_ms), (180000, 60000))
        self.assertEqual((limits.max_agent_steps, limits.max_tool_calls), (6, 10))
        self.assertEqual((limits.max_planner_output_tokens_per_step, limits.max_total_planner_output_tokens, limits.max_final_answer_tokens), (1024, 8192, 4096))
        with patch.dict(os.environ, {
            "AGENT_BUDGET_PROFILE": "quality",
            "AGENT_TOTAL_DEADLINE_MS": "300000",
            "AGENT_FINAL_ANSWER_RESERVE_MS": "120000",
            "AGENT_BASE_EVIDENCE_COUNT": "8",
            "AGENT_MAX_FINAL_ANSWER_TOKENS": "16384",
        }, clear=True):
            overridden = get_agent_limits()
        self.assertEqual((overridden.total_deadline_ms, overridden.min_final_answer_budget_ms), (300000, 120000))
        self.assertEqual((overridden.base_evidence_count, overridden.max_final_answer_tokens), (8, 16384))

    def test_quality_rejects_invalid_values_without_echoing_them(self):
        invalid = {
            "AGENT_TOTAL_DEADLINE_MS": "300001",
            "AGENT_FINAL_ANSWER_RESERVE_MS": "120001",
            "AGENT_MAX_STEPS": "9",
            "AGENT_MAX_TOOL_CALLS": "13",
            "AGENT_MAX_PLANNER_TOKENS_PER_STEP": "4097",
            "AGENT_MAX_TOTAL_PLANNER_TOKENS": "32769",
            "AGENT_MAX_FINAL_ANSWER_TOKENS": "16385",
            "AGENT_BASE_EVIDENCE_COUNT": "9",
            "AGENT_MAX_OBSERVATION_BYTES": "private-invalid-value",
        }
        for key, value in invalid.items():
            with self.subTest(key=key), patch.dict(os.environ, {
                "AGENT_BUDGET_PROFILE": "quality", key: value,
            }, clear=True):
                with self.assertRaises(ValueError) as error:
                    get_agent_limits()
                self.assertIn(key, str(error.exception))
                self.assertNotIn("private-invalid-value", str(error.exception))
        for values in (
            {"AGENT_TOTAL_DEADLINE_MS": "60000", "AGENT_FINAL_ANSWER_RESERVE_MS": "60000"},
            {"AGENT_MAX_TOTAL_PLANNER_TOKENS": "512"},
        ):
            with self.subTest(values=values), patch.dict(os.environ, {
                "AGENT_BUDGET_PROFILE": "quality", **values,
            }, clear=True), self.assertRaises(ValueError):
                get_agent_limits()
        with patch.dict(os.environ, {"AGENT_BUDGET_PROFILE": "private-invalid-value"}, clear=True):
            with self.assertRaises(ValueError) as error:
                get_agent_limits()
            self.assertNotIn("private-invalid-value", str(error.exception))

    def test_generation_capacity_effort_and_retry_policy_are_optional_and_separate(self):
        with patch.dict(os.environ, {}, clear=True):
            defaults = get_llm_settings()
        self.assertIsNone(defaults.planner_max_generation_tokens)
        self.assertIsNone(defaults.answer_max_generation_tokens)
        self.assertIsNone(defaults.planner_reasoning_effort)
        self.assertIsNone(defaults.answer_reasoning_effort)
        self.assertFalse(defaults.retry_transient_only)
        with patch.dict(os.environ, {
            "LLM_PLANNER_MAX_GENERATION_TOKENS": "32768",
            "LLM_ANSWER_MAX_GENERATION_TOKENS": "65536",
            "LLM_PLANNER_REASONING_EFFORT": "high",
            "LLM_ANSWER_REASONING_EFFORT": "max",
            "LLM_RETRY_TRANSIENT_ONLY": "true",
        }, clear=True):
            settings = get_llm_settings()
        self.assertEqual((settings.planner_max_generation_tokens, settings.answer_max_generation_tokens), (32768, 65536))
        self.assertEqual((settings.planner_reasoning_effort, settings.answer_reasoning_effort), ("high", "max"))
        self.assertTrue(settings.retry_transient_only)
        self.assertEqual(settings.max_tokens, 1600)

    def test_explicit_invalid_generation_configuration_fails_closed(self):
        invalid = {
            "LLM_PLANNER_MAX_GENERATION_TOKENS": "0",
            "LLM_ANSWER_MAX_GENERATION_TOKENS": "65537",
            "LLM_PLANNER_REASONING_EFFORT": "medium",
            "LLM_ANSWER_REASONING_EFFORT": "private-invalid-value",
            "LLM_RETRY_TRANSIENT_ONLY": "maybe",
        }
        for key, value in invalid.items():
            with self.subTest(key=key), patch.dict(os.environ, {key: value}, clear=True):
                with self.assertRaises(ValueError) as error:
                    get_llm_settings()
                self.assertIn(key, str(error.exception))
                self.assertNotIn("private-invalid-value", str(error.exception))


class QualityBaseEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = Database(Path(self.directory.name) / "quality-fixture.sqlite")
        _, self.bundle = make_project(self.database, [
            (f"src/sample_{index}.py", f"sample_{index}", f"def sample_{index}():\n    return 'sharedvalue'\n")
            for index in range(10)
        ])

    def test_quality_base_five_preserves_capacity_eight_for_later_search(self):
        with patch.dict(os.environ, {"AGENT_BUDGET_PROFILE": "quality"}, clear=True):
            limits = get_agent_limits()
        planner = ScriptedPlanner([
            decision("continue", "search_code", {"query": "sharedvalue", "top_k": 8}),
            decision("answer"),
        ])
        with patch.object(agent_core, "_execute_deterministic_base_retrieval", wraps=agent_core._execute_deterministic_base_retrieval) as base_call, patch.object(agent_core, "_execute_agent_tool", wraps=agent_core._execute_agent_tool) as tool_call:
            result = run_bounded_agent(
                "sharedvalue", self.bundle, NoLlm(), self.database,
                disabled_embedding_service(), planner=planner, limits=limits,
                evidence_count=8,
            )
        self.assertEqual(base_call.call_args.kwargs["arguments"]["top_k"], 5)
        self.assertEqual(tool_call.call_args.kwargs["arguments"]["top_k"], 8)
        self.assertEqual(len(result["evidence"]), 6)
        self.assertEqual(tool_call.call_args.kwargs["state"].context.evidence_admission_policy(), {
            "capacity": 8, "remaining_capacity": 2, "reserved_for_directed": 2,
            "broad_search_remaining": 0, "max_new_per_supplement": 1,
            "max_new_per_read_source": 2,
        })
        self.assertEqual(result["budget_usage"]["budget_profile"], "quality")
        self.assertEqual(result["budget_usage"]["base_evidence_count"], 5)
        self.assertEqual(result["budget_usage"]["base_evidence_count_effective"], 5)

    def test_base_configuration_is_bounded_by_request_capacity_and_legacy_is_unchanged(self):
        with patch.dict(os.environ, {"AGENT_BUDGET_PROFILE": "quality"}, clear=True):
            quality = get_agent_limits()
        for limits, capacity, expected in ((quality, 3, 3), (AgentLimits(), 8, 8)):
            with self.subTest(profile=limits.budget_profile), patch.object(agent_core, "_execute_deterministic_base_retrieval", wraps=agent_core._execute_deterministic_base_retrieval) as base_call:
                result = run_bounded_agent(
                    "sharedvalue", self.bundle, NoLlm(), self.database,
                    disabled_embedding_service(), execution_mode="rag",
                    limits=limits, evidence_count=capacity,
                )
            self.assertEqual(base_call.call_args.kwargs["arguments"]["top_k"], expected)
            self.assertEqual(len(result["evidence"]), expected)
            self.assertEqual(result["budget_usage"]["base_evidence_count_effective"], expected)


class QualityProviderPurposeTests(unittest.TestCase):
    @staticmethod
    def state():
        return {
            "remaining_budget": {"tool_calls": 3, "time_ms": 10000},
            "user_goal": "fixture", "observations": [], "known_evidence_ids": [],
            "known_symbols": [], "known_symbol_ids": [],
        }

    def test_real_client_gets_purpose_without_a_recorder(self):
        client = LLMClient(LLMSettings("openai_compatible", "https://provider.invalid/v1", "fixture", "fixture"))
        limits = AgentLimits()
        planner = LLMPlanner(client, build_m2_tool_registry(limits), limits)
        with patch.object(client, "chat", return_value=json.dumps(decision("answer"))) as chat:
            planner.decide(self.state())
            self.assertEqual(chat.call_args.kwargs["purpose"], "planner")
            self.assertNotIn("diagnostics_recorder", chat.call_args.kwargs)
            _chat_final_answer(client, [], max_tokens=1600, timeout_seconds=5, diagnostics_recorder=None, request_deadline_at=None)
            self.assertEqual(chat.call_args.kwargs["purpose"], "final_answer")

    def test_simple_fake_remains_compatible_without_a_recorder(self):
        class StrictFake:
            def chat(self, messages, *, temperature, max_tokens, timeout_seconds):
                return json.dumps(decision("answer"))

        fake = StrictFake()
        limits = AgentLimits()
        planner = LLMPlanner(fake, build_m2_tool_registry(limits), limits)
        planner.decide(self.state())
        _chat_final_answer(fake, [], max_tokens=1600, timeout_seconds=5, diagnostics_recorder=None, request_deadline_at=None)


if __name__ == "__main__":
    unittest.main()
