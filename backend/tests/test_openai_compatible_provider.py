from __future__ import annotations

import io
import json
import unittest
import urllib.error

from app.config import LLMSettings
from app.services.llm_client import LLMClient, ProviderError


class _Response:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self._body


def _settings(**overrides) -> LLMSettings:
    values = {
        "provider": "openai_compatible",
        "base_url": "https://provider.invalid/v1",
        "api_key": "super-secret",
        "model": "configured-model",
        "timeout_seconds": 2.0,
        "max_tokens": 128,
        "temperature": 0.1,
        "max_retries": 2,
    }
    values.update(overrides)
    return LLMSettings(**values)


class OpenAICompatibleProviderTests(unittest.TestCase):
    def test_chat_completions_uses_configured_values(self):
        captured = {}

        def opener(request, timeout):
            captured["url"] = request.full_url
            captured["authorization"] = request.headers["Authorization"]
            captured["payload"] = json.loads(request.data)
            captured["timeout"] = timeout
            return _Response({"choices": [{"message": {"content": "ok"}}]})

        client = LLMClient(_settings(), opener=opener, sleep=lambda _value: None)
        result = client.chat([{"role": "user", "content": "hello"}])
        self.assertEqual(result, "ok")
        self.assertEqual(captured["url"], "https://provider.invalid/v1/chat/completions")
        self.assertEqual(captured["payload"]["model"], "configured-model")
        self.assertEqual(captured["authorization"], "Bearer super-secret")

    def test_retryable_status_has_bounded_retries(self):
        calls = []

        def opener(_request, _timeout):
            calls.append(1)
            if len(calls) < 3:
                raise urllib.error.HTTPError(
                    "https://provider.invalid/v1/chat/completions",
                    503,
                    "unavailable",
                    {},
                    io.BytesIO(b"upstream diagnostic"),
                )
            return _Response({"choices": [{"message": {"content": "recovered"}}]})

        client = LLMClient(_settings(), opener=opener, sleep=lambda _value: None)
        self.assertEqual(client.chat([{"role": "user", "content": "hello"}]), "recovered")
        self.assertEqual(len(calls), 3)

    def test_auth_error_is_typed_non_retryable_and_redacted(self):
        def opener(_request, _timeout):
            raise urllib.error.HTTPError(
                "https://provider.invalid/v1/chat/completions",
                401,
                "super-secret was rejected",
                {},
                io.BytesIO(b"super-secret"),
            )

        client = LLMClient(_settings(), opener=opener, sleep=lambda _value: None)
        with self.assertRaises(ProviderError) as raised:
            client.chat([{"role": "user", "content": "hello"}])
        self.assertEqual(raised.exception.code, "provider_authentication_failed")
        self.assertFalse(raised.exception.retryable)
        self.assertNotIn("super-secret", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
