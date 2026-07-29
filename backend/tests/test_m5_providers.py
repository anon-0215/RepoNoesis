from __future__ import annotations

import json
import math
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.m5.providers import (
    BudgetedProvider,
    FakeDeterministicProvider,
    OpenAICompatibleProvider,
    OpenAICompatibleSettings,
    ProviderConfigurationError,
    PricingConfig,
    ProviderLLMClient,
    StructuredLearningEvaluator,
    redact_secrets,
    validate_vector,
    normalized_endpoint_identity,
)
from app.m5.contracts import ProviderResult, ProviderUsage
from app.m5.embedding import M5EmbeddingProvider, fake_embedding_service
from app.services.agent_contracts import CancellationToken


class M5ProviderTests(unittest.TestCase):
    def test_real_provider_is_disabled_without_explicit_gates(self):
        with self.assertRaises(ProviderConfigurationError):
            OpenAICompatibleProvider(OpenAICompatibleSettings("https://example.com", "key", "model"))

    def test_real_evaluator_requires_paid_gate(self):
        settings = OpenAICompatibleSettings(
            "https://example.com", "key", "model", "revision-1", allow_network=True, allow_real_llm=True
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
            "https://example.com", "key", "model", "revision-1", allow_network=True, allow_real_llm=True
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
            "https://example.com", "key", "model", "revision-1", allow_network=True, allow_real_llm=True
        )
        provider = OpenAICompatibleProvider(settings)
        error = urllib.error.HTTPError("url", 429, "rate", {}, None)
        with patch("urllib.request.urlopen", side_effect=error) as call:
            result = provider.invoke([], temperature=0, max_output_tokens=8, timeout_seconds=1, maximum_attempts=2)
        self.assertEqual(result.error_type, "rate_limited")
        self.assertEqual(call.call_count, 2)

    def test_usage_absent_is_unknown_not_zero(self):
        settings = OpenAICompatibleSettings(
            "https://example.com", "key", "model", "revision-1", allow_network=True, allow_real_llm=True
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
        self.assertEqual(result.usage.cost_status, "unknown")
        self.assertEqual(result.usage.cost_unknown_reason, "token_usage_missing")
        self.assertEqual(result.actual_model, "actual")

    def test_unknown_price_with_usage_remains_unknown(self):
        settings = OpenAICompatibleSettings(
            "https://example.com", "key", "model", "revision-1",
            allow_network=True, allow_real_llm=True,
        )
        payload = json.dumps({"choices": [{"message": {"content": "ok"}}],
                              "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}}).encode()
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, size): return payload
        with patch("urllib.request.urlopen", return_value=Response()):
            usage = OpenAICompatibleProvider(settings).smoke_check().usage
        self.assertEqual(usage.cost_status, "unknown")
        self.assertIsNone(usage.estimated_cost_usd)
        self.assertEqual(usage.cost_unknown_reason, "pricing_not_configured")

    def test_explicit_pricing_calculates_auditable_cost(self):
        pricing = PricingConfig("model", "USD", 2.0, 4.0, 1_000, "operator-config-v1")
        settings = OpenAICompatibleSettings(
            "https://example.com", "key", "model", "revision-1",
            allow_network=True, allow_real_llm=True, pricing=pricing,
        )
        payload = json.dumps({"choices": [{"message": {"content": "ok"}}],
                              "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}}).encode()
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, size): return payload
        with patch("urllib.request.urlopen", return_value=Response()):
            usage = OpenAICompatibleProvider(settings).smoke_check().usage
        self.assertEqual(usage.cost_status, "calculated")
        self.assertAlmostEqual(usage.estimated_cost_usd or 0, 0.4)

    def test_retries_count_against_request_budget(self):
        settings = OpenAICompatibleSettings(
            "https://example.com", "key", "model", "revision-1",
            allow_network=True, allow_real_llm=True,
        )
        budgeted = BudgetedProvider(
            OpenAICompatibleProvider(settings), maximum_requests=2,
            maximum_input_tokens=1000, maximum_output_tokens=1000,
        )
        error = urllib.error.HTTPError("url", 429, "rate", {}, None)
        kwargs = dict(temperature=0, max_output_tokens=8, timeout_seconds=1, maximum_attempts=3)
        with patch("urllib.request.urlopen", side_effect=error):
            result = budgeted.invoke([], **kwargs)
        self.assertEqual(result.attempt_count, 2)
        self.assertEqual(budgeted.request_count, 2)
        self.assertEqual(budgeted.retry_count, 1)
        self.assertEqual(budgeted.invoke([], **kwargs).error_type, "provider_budget_exhausted")

    def test_token_and_cost_limits_stop_subsequent_request(self):
        class CostProvider(FakeDeterministicProvider):
            def invoke(self, *args, **kwargs):
                result = super().invoke(*args, **kwargs)
                usage = ProviderUsage(10, 5, 15, 0.5, "calculated", "USD", None)
                return ProviderResult(**{**result.__dict__, "usage": usage})
        budgeted = BudgetedProvider(
            CostProvider(), maximum_requests=5, maximum_input_tokens=10,
            maximum_output_tokens=100, maximum_cost_usd=0.5,
        )
        kwargs = dict(temperature=0, max_output_tokens=8, timeout_seconds=1, maximum_attempts=1)
        self.assertEqual(budgeted.invoke([], **kwargs).status, "succeeded")
        self.assertIn(budgeted.stop_reason, {"input_token_budget_exhausted", "cost_budget_exhausted"})
        self.assertEqual(budgeted.invoke([], **kwargs).error_type, "provider_budget_exhausted")
        cost_budgeted = BudgetedProvider(
            CostProvider(), maximum_requests=5, maximum_input_tokens=100,
            maximum_output_tokens=100, maximum_cost_usd=0.5,
        )
        self.assertEqual(cost_budgeted.invoke([], **kwargs).status, "succeeded")
        self.assertEqual(cost_budgeted.stop_reason, "cost_budget_exhausted")
        self.assertEqual(cost_budgeted.invoke([], **kwargs).error_type, "provider_budget_exhausted")

    def test_endpoint_identity_drops_credentials_query_and_secret(self):
        identity = normalized_endpoint_identity("https://user:api-secret@example.com/api/?token=api-secret")
        self.assertTrue(identity.startswith("https://example.com|path-sha256:"))
        self.assertNotIn("/api", identity)
        self.assertNotIn("api-secret", identity)
        settings = OpenAICompatibleSettings(
            "https://example.com", "api-secret", "model", "revision-1",
            allow_network=True, allow_real_llm=True,
        )
        provider = OpenAICompatibleProvider(settings)
        self.assertNotIn("api-secret", json.dumps(provider.identity.to_dict()))
        self.assertNotIn("api-secret", json.dumps(settings.safe_dict()))

    def test_embedding_wrapper_enforces_local_files_only_without_network_gate(self):
        wrapper = object.__new__(M5EmbeddingProvider)
        wrapper.allow_network = False
        wrapper._dimension = None
        wrapper.service = MagicMock()
        wrapper.service.encode_query.return_value = [1.0, 0.0]
        wrapper.service.encode_documents.return_value = [[1.0, 0.0]]
        wrapper.service.ensure_effective_embedding_identity.return_value = (
            fake_embedding_service(Path("unused")).ensure_effective_embedding_identity()
        )
        wrapper.encode_query("query")
        wrapper.encode_documents(["document"])
        wrapper.ensure_model_identity()
        wrapper.service.encode_query.assert_called_once_with("query", local_files_only=True)
        wrapper.service.encode_documents.assert_called_once_with(["document"], local_files_only=True)
        wrapper.service.ensure_effective_embedding_identity.assert_called_once_with(
            local_files_only=True
        )


if __name__ == "__main__":
    unittest.main()
