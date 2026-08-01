from __future__ import annotations

import unittest

from app.config import LLMSettings
from app.services.agent_contracts import AgentLimits
from app.services.agent_core import LLMPlanner
from app.services.agent_tools import ToolRegistry
from app.services.qa_agent import _answer_with_grounded_llm


class _RecordingLlm:
    available = True

    def __init__(self) -> None:
        self.settings = LLMSettings(
            provider="openai_compatible",
            base_url="https://provider.invalid/v1",
            api_key="placeholder",
            model="configured-model",
            planner_thinking="disabled",
            answer_thinking="enabled",
        )
        self.calls: list[dict] = []

    def chat(self, _messages, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return '{"status":"answer","decision_summary":"done"}'
        return "Grounded answer [E1] src/example.py:1-2."


class _Evidence:
    evidence_id = "E1"
    path = "src/example.py"
    start_line = 1
    end_line = 2
    qualified_name = "example"
    symbol_name = "example"
    excerpt = "def example():\n    return 1"


class ProductProviderIntegrationTests(unittest.TestCase):
    def test_planner_and_final_answer_use_separate_thinking_and_existing_budgets(self):
        llm = _RecordingLlm()
        limits = AgentLimits()
        planner = LLMPlanner(llm, ToolRegistry(), limits)
        planner.decide(
            {
                "remaining_budget": {"time_ms": 60_000},
                "user_goal": "explain example",
                "observations": [],
                "known_evidence_ids": [],
                "known_symbols": [],
                "known_symbol_ids": [],
            }
        )
        _answer_with_grounded_llm(
            "explain example",
            [_Evidence()],
            llm,
            max_tokens=limits.max_final_answer_tokens,
            timeout_seconds=30,
        )

        planner_call, answer_call = llm.calls
        self.assertEqual(planner_call["thinking"], "disabled")
        self.assertEqual(answer_call["thinking"], "enabled")
        self.assertEqual(
            planner_call["max_tokens"], limits.max_planner_output_tokens_per_step
        )
        self.assertEqual(answer_call["max_tokens"], limits.max_final_answer_tokens)
        for call in llm.calls:
            self.assertNotIn("reasoning_effort", call)
            self.assertNotIn("response_format", call)
            self.assertNotIn("stream", call)


if __name__ == "__main__":
    unittest.main()
