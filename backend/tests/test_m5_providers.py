from __future__ import annotations

import json
import math
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from app.m5.providers import (
    BudgetedProvider,
    FakeDeterministicProvider,
    OpenAICompatibleProvider,
    OpenAICompatibleSettings,
    ProviderConfigurationError,
    ProviderLLMClient,
    StructuredLearningEvaluator,
    redact_secrets,
    validate_vector,
)
from app.services.agent_contracts import CancellationToken


class M5ProviderTests(unittest.TestCase):
    def test_real_provider_is_disabled_without_explicit_gates(self):
        with self.assertRaises(ProviderConfigurationError):
            OpenAICompatibleProvider(OpenAICompatibleSettings("https://example.com", "key", "model"))

    def test_real_evaluator_requires_paid_gate(self):
        settings = OpenAICompatibleSettings(
            "https://example.com", "key", "model", allow_network=True, allow_real_llm=True
        )
        with self.assertRaises(ProviderConfigurationError):
            OpenAICompatibleProvider(settings, evaluator=True)

    def test_secret_redaction_is_recursive(self):
        value = redact_secrets({"api_key": "secret", "nested": {"Authorization": "Bearer x"}, "ok": 1})
        self.assertEqual(value, {"api_key": "[REDACTED]", "nested": {"Authorization": "[REDACTED]"}, "ok": 1})

    def test_fake_provider_records_usage_and_identity(self):
        provider = FakeDeterministicProvider()
        result = provider.invoke(
            [{"role": "user", "content": "Evidence E1"}], temperature=0, max_output_tokens=100,
            timeout_seconds=1, maximum_attempts=1, seed=7,
        )
        self.assertEqual(result.status, "succeeded")
        self.assertFalse(result.identity.is_real)
        self.assertGreater(result.usage.total_tokens or 0, 0)
        self.assertEqual(result.usage.estimated_cost_usd, 0.0)

    def test_cancellation_is_honored(self):
        token = CancellationToken(); token.cancel()
        result = FakeDeterministicProvider().invoke(
            [], temperature=0, max_output_tokens=10, timeout_seconds=1,
            maximum_attempts=1, cancellation=token,
        )
        self.assertEqual(result.status, "cancelled")

    def test_provider_budget_prevents_duplicate_paid_call(self):
        provider = BudgetedProvider(
            FakeDeterministicProvider(), maximum_calls=1,
            maximum_input_tokens=1000, maximum_output_tokens=1000,
        )
        kwargs = dict(temperature=0, max_output_tokens=10, timeout_seconds=1, maximum_attempts=1)
        self.assertEqual(provider.invoke([], **kwargs).status, "succeeded")
        exhausted = provider.invoke([], **kwargs)
        self.assertEqual(exhausted.error_type, "provider_budget_exhausted")
        self.assertEqual(provider.call_count, 1)

    def test_llm_adapter_keeps_rich_results(self):
        client = ProviderLLMClient(FakeDeterministicProvider())
        self.assertIn("[E1]", client.chat([{"role": "user", "content": "E1"}]) or "")
        self.assertEqual(len(client.results), 1)

    def test_invalid_structured_output_is_ungradable(self):
        class BadProvider(FakeDeterministicProvider):
            def invoke(self, *args, **kwargs):
                result = super().invoke(*args, **kwargs)
                return type(result)(**{**result.__dict__, "content": '{"mastery":"mastered"}'})
        evaluator = StructuredLearningEvaluator(BadProvider(capability="structured_evaluator"))
        output = evaluator.evaluate({"rubric": [], "evidence": []}, "answer")
        self.assertEqual(output["verdict"], "ungradable")

    def test_vector_validation_rejects_nan_infinity_and_dimension(self):
        with self.assertRaises(ValueError): validate_vector([1.0, math.nan])
        with self.assertRaises(ValueError): validate_vector([1.0, math.inf])
        with self.assertRaises(ValueError): validate_vector([1.0], expected_dimension=2)

    def test_invalid_json_response_is_classified(self):
        settings = OpenAICompatibleSettings(
            "https://example.com", "key", "model", allow_network=True, allow_real_llm=True
        )
        provider = OpenAICompatibleProvider(settings)
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, size): return b"not-json"
        with patch("urllib.request.urlopen", return_value=Response()):
            result = provider.smoke_check()
        self.assertEqual(result.error_type, "invalid_json")

    def test_429_is_bounded_and_classified(self):
        settings = OpenAICompatibleSettings(
            "https://example.com", "key", "model", allow_network=True, allow_real_llm=True
        )
        provider = OpenAICompatibleProvider(settings)
        error = urllib.error.HTTPError("url", 429, "rate", {}, None)
        with patch("urllib.request.urlopen", side_effect=error) as call:
            result = provider.invoke([], temperature=0, max_output_tokens=8, timeout_seconds=1, maximum_attempts=2)
        self.assertEqual(result.error_type, "rate_limited")
        self.assertEqual(call.call_count, 2)

    def test_usage_absent_is_unknown_not_zero(self):
        settings = OpenAICompatibleSettings(
            "https://example.com", "key", "model", allow_network=True, allow_real_llm=True
        )
        provider = OpenAICompatibleProvider(settings)
        payload = json.dumps({"model": "actual", "choices": [{"message": {"content": "ok"}}]}).encode()
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, size): return payload
        with patch("urllib.request.urlopen", return_value=Response()):
            result = provider.smoke_check()
        self.assertIsNone(result.usage.total_tokens)
        self.assertEqual(result.actual_model, "actual")


if __name__ == "__main__":
    unittest.main()
