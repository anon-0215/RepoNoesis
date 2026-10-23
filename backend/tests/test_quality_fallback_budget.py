from __future__ import annotations

from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.config import LLMSettings
from app.database import Database
from app.services.agent_contracts import AgentLimits
from app.services.agent_core import run_bounded_agent
from app.services.llm_client import LLMClient
from app.services.smoke_diagnostics import SmokeDiagnosticsRecorder
from tests.m1_helpers import disabled_embedding_service, make_project


class QualityFallbackBudgetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = Database(Path(self.directory.name) / "fallback-fixture.sqlite")
        _, self.bundle = make_project(self.database, [
            ("src/example.py", "example", "def example():\n    return 1\n"),
        ])

    def run_fallback(self, *, limits, generation_limit, long_answer):
        text = "Detailed explanation " * 85 if long_answer else "Returns one."
        content = json.dumps({"parts": [
            {"text": text.strip(), "evidence_aliases": ["A1"]}
            for _ in range(4 if long_answer else 1)
        ]})
        requests = []

        def fake_open(request, **_kwargs):
            requests.append(json.loads(request.data))
            response = io.BytesIO(json.dumps({
                "choices": [{"finish_reason": "stop", "message": {"content": content}}],
            }).encode())
            response.status = 200
            return response

        llm = LLMClient(LLMSettings(
            provider="openai_compatible", base_url="https://provider.invalid/v1",
            api_key="fixture-key", model="fixture-model",
            answer_max_generation_tokens=generation_limit,
        ), opener=fake_open)
        recorder = SmokeDiagnosticsRecorder()
        with patch("app.services.agent_core.build_tool_context", side_effect=ValueError("fixture")):
            result = run_bounded_agent(
                "example", self.bundle, llm, self.database,
                disabled_embedding_service(), limits=limits,
                diagnostics_recorder=recorder,
            )
        return result, recorder.snapshot(), requests

    def test_quality_rejects_visible_overflow_with_large_generation_capacity(self):
        result, snapshot, requests = self.run_fallback(
            limits=replace(AgentLimits(), budget_profile="quality", max_final_answer_tokens=128),
            generation_limit=32768, long_answer=True,
        )
        self.assertEqual(requests[0]["max_tokens"], 32768)
        self.assertEqual(result["agent_mode"], "deterministic_fallback")
        self.assertEqual(result["answer_mode"], "deterministic")
        self.assertNotIn("Detailed explanation", result["answer"])
        self.assertEqual(snapshot["final_answer_failure_reason_code"], "answer_token_budget_exceeded")

    def test_quality_accepts_grounded_answer_within_visible_budget(self):
        result, snapshot, _ = self.run_fallback(
            limits=replace(AgentLimits(), budget_profile="quality", max_final_answer_tokens=128),
            generation_limit=32768, long_answer=False,
        )
        self.assertEqual(result["answer_mode"], "llm_grounded")
        self.assertIn("Returns one.", result["answer"])
        self.assertNotIn("final_answer_failure_reason_code", snapshot)

    def test_legacy_explicit_generation_override_still_has_visible_guard(self):
        result, snapshot, requests = self.run_fallback(
            limits=AgentLimits(), generation_limit=32768, long_answer=True,
        )
        self.assertEqual(requests[0]["max_tokens"], 32768)
        self.assertEqual(result["answer_mode"], "deterministic")
        self.assertEqual(snapshot["final_answer_failure_reason_code"], "answer_token_budget_exceeded")

    def test_legacy_without_generation_override_keeps_previous_behavior(self):
        result, snapshot, requests = self.run_fallback(
            limits=AgentLimits(), generation_limit=None, long_answer=True,
        )
        self.assertEqual(requests[0]["max_tokens"], 1600)
        self.assertEqual(result["answer_mode"], "llm_grounded")
        self.assertIn("Detailed explanation", result["answer"])
        self.assertNotIn("final_answer_failure_reason_code", snapshot)


if __name__ == "__main__":
    unittest.main()
