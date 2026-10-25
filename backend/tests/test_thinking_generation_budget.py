from __future__ import annotations

import json
import unittest
import urllib.error

from app.config import LLMSettings
from app.services.chat_history import ExecutionSummaryResponse, prepare_recovery_metadata, RecoveryMetadata
from app.services.llm_client import LLMClient, ProviderError
from app.services.smoke_diagnostics import SmokeDiagnosticsRecorder


class _Response:
    status = 200

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def _settings(**overrides):
    values = dict(provider="openai_compatible", base_url="https://provider.invalid/v1",
                  api_key="test-only", model="configured-model", max_tokens=1600,
                  max_retries=1)
    return LLMSettings(**(values | overrides))


def _payload(finish_reason="stop", **usage):
    return {
        "model": "returned-model-0813",
        "choices": [{"finish_reason": finish_reason, "message": {
            "content": "final answer", "reasoning_content": "PRIVATE_REASONING"}}],
        "usage": usage,
    }


class ThinkingGenerationBudgetTests(unittest.TestCase):
    def test_stages_use_generation_capacity_without_recorder(self):
        sent = []

        def opener(request, **_kwargs):
            sent.append(json.loads(request.data))
            return _Response(_payload())

        client = LLMClient(_settings(
            planner_max_generation_tokens=16384, answer_max_generation_tokens=32768,
            planner_reasoning_effort="low", answer_reasoning_effort="high"), opener=opener)
        for purpose, visible_limit in (("planner", 1024), ("final_answer", 4096)):
            self.assertEqual(client.chat([], purpose=purpose, thinking="enabled",
                                         max_tokens=visible_limit), "final answer")
        self.assertEqual([p["max_tokens"] for p in sent], [16384, 32768])
        self.assertEqual([p["reasoning_effort"] for p in sent], ["low", "high"])
        self.assertTrue(all(p["thinking"] == {"type": "enabled"} for p in sent))

    def test_unset_extensions_preserve_legacy_payload(self):
        sent = []

        def opener(request, **_kwargs):
            sent.append(json.loads(request.data))
            return _Response(_payload())

        client = LLMClient(_settings(), opener=opener)
        client.chat([], purpose="planner", max_tokens=512)
        client.chat([], purpose="final_answer", max_tokens=1600)
        self.assertEqual([p["max_tokens"] for p in sent], [512, 1600])
        self.assertTrue(all("reasoning_effort" not in p and "thinking" not in p for p in sent))

    def test_invalid_stage_settings_fail_before_any_http(self):
        for overrides, thinking in (
            ({"answer_max_generation_tokens": True}, "enabled"),
            ({"answer_max_generation_tokens": 65537}, "enabled"),
            ({"answer_reasoning_effort": "invalid"}, "enabled"),
            ({"answer_reasoning_effort": "high"}, "disabled"),
        ):
            with self.subTest(overrides=overrides):
                client = LLMClient(_settings(**overrides), opener=lambda *_a, **_k: self.fail("HTTP attempted"))
                with self.assertRaises(ValueError):
                    client.chat([], purpose="final_answer", thinking=thinking)

    def test_truncated_response_keeps_usage_through_repeated_sanitization(self):
        recorder = SmokeDiagnosticsRecorder()
        client = LLMClient(_settings(), diagnostics_recorder=recorder, opener=lambda *_a, **_k: _Response(
            _payload("length", prompt_tokens=100, completion_tokens=200, total_tokens=300,
                     completion_tokens_details={"reasoning_tokens": 190},
                     prompt_cache_hit_tokens=20, prompt_cache_miss_tokens=80)))
        with self.assertRaises(ProviderError) as error:
            client.chat([], purpose="final_answer")
        self.assertEqual(error.exception.code, "provider_output_truncated")
        safe = error.exception.to_safe_dict()
        self.assertEqual(safe["diagnostics"]["usage"]["reasoning_tokens"], 190)
        self.assertEqual(safe["diagnostics"]["response_model"], "returned-model-0813")
        summary = recorder.snapshot()["provider_usage_summary"]
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["fields"]["reasoning_tokens"], {"known_subtotal": 190, "covered_attempts": 1})
        self.assertEqual(summary["fields"]["prompt_cache_hit_tokens"]["known_subtotal"], 20)
        self.assertNotIn("PRIVATE_REASONING", json.dumps(recorder.snapshot()) + json.dumps(safe))

    def test_invalid_usage_is_unknown_and_response_model_is_sanitized(self):
        payload = _payload(prompt_tokens=True, completion_tokens=-2, total_tokens=3.5,
                           reasoning_tokens="secret", completion_tokens_details={"reasoning_tokens": False})
        payload["model"] = "unsafe model\nprivate"
        recorder = SmokeDiagnosticsRecorder()
        client = LLMClient(_settings(), opener=lambda *_a, **_k: _Response(payload))
        client.chat([], purpose="final_answer", diagnostics_recorder=recorder)
        summary = recorder.snapshot()["provider_usage_summary"]
        self.assertEqual(summary["status"], "unknown")
        self.assertEqual(summary["fields"], {})
        self.assertNotIn("unsafe model", json.dumps(recorder.snapshot()))

    def test_transient_only_does_not_retry_ambiguous_network_failure(self):
        for exception in (TimeoutError(), urllib.error.URLError("test-only")):
            attempts = []

            def opener(*_args, **_kwargs):
                attempts.append(1)
                raise exception

            client = LLMClient(_settings(retry_transient_only=True), opener=opener)
            with self.assertRaises(ProviderError):
                client.chat([], purpose="planner")
            self.assertEqual(len(attempts), 1)

    def test_explicit_transient_error_retries_once_and_keeps_partial_usage(self):
        for status in (429, 500, 503):
            with self.subTest(status=status):
                attempts = []
                recorder = SmokeDiagnosticsRecorder()

                def opener(*_args, **_kwargs):
                    attempts.append(1)
                    if len(attempts) == 1:
                        raise urllib.error.HTTPError("https://provider.invalid", status, "test-only", {}, None)
                    return _Response(_payload(prompt_tokens=10, completion_tokens=20, total_tokens=30))

                client = LLMClient(_settings(retry_transient_only=True), opener=opener,
                                   sleep=lambda _: None, diagnostics_recorder=recorder)
                client.chat([], purpose="planner")
                summary = recorder.snapshot()["provider_usage_summary"]
                self.assertEqual((len(attempts), summary["http_attempts"]), (2, 2))
                self.assertEqual(summary["status"], "partial")
                self.assertEqual(summary["fields"]["total_tokens"]["known_subtotal"], 30)

    def test_usage_crosses_api_projection_and_saved_history_without_fabricating_legacy(self):
        from app.main import _ask_execution_summary

        recorder = SmokeDiagnosticsRecorder()
        client = LLMClient(_settings(), opener=lambda *_a, **_k: _Response(
            _payload(prompt_tokens=1, completion_tokens=2, total_tokens=3)))
        client.chat([], purpose="final_answer", diagnostics_recorder=recorder)
        result = dict(agent_status="completed", answer_mode="llm_grounded", evidence=[], citations=[])
        summary = _ask_execution_summary(result, recorder.snapshot())
        summary = ExecutionSummaryResponse.model_validate(summary).model_dump()
        self.assertEqual(summary["provider_usage_summary"]["status"], "complete")
        saved = prepare_recovery_metadata(result | {"execution_summary": summary}, "p", "r")
        recovered = RecoveryMetadata.model_validate_json(json.dumps(saved))
        self.assertEqual(recovered.execution_summary.provider_usage_summary["fields"]["total_tokens"]["known_subtotal"], 3)
        legacy = _ask_execution_summary(result, {})
        self.assertNotIn("provider_usage_summary", legacy)
        self.assertIsNone(ExecutionSummaryResponse.model_validate(legacy).provider_usage_summary)


if __name__ == "__main__":
    unittest.main()
