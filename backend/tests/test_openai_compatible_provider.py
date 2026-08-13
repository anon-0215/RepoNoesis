from __future__ import annotations

import io
import json
import hashlib
import time
import unittest
import urllib.error
from threading import Event

from app.config import LLMSettings
from app.services.llm_client import LLMClient, ProviderError
from app.services.smoke_diagnostics import SmokeDiagnosticsRecorder


class _Clock:
    def __init__(self, value=100.0):
        self.value = float(value)

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += float(seconds)


class _Response:
    status = 200

    def __init__(self, payload):
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
    def test_small_real_wall_clock_budget_does_not_accumulate_fixed_retry_timeouts(self):
        blocker = Event()
        calls = []

        def opener(_request, *, timeout):
            calls.append(timeout)
            blocker.wait(0.06)
            raise urllib.error.URLError("safe fake timeout")

        client = LLMClient(
            _settings(timeout_seconds=45.0, max_retries=2),
            opener=opener,
        )
        started = time.monotonic()
        with self.assertRaises(ProviderError) as raised:
            client.chat(
                [{"role": "user", "content": "hello"}],
                deadline_monotonic=started + 0.05,
            )
        elapsed = time.monotonic() - started

        self.assertEqual(raised.exception.code, "deadline_exceeded")
        self.assertEqual(len(calls), 1)
        self.assertLess(elapsed, 0.5)

    def test_each_http_attempt_recomputes_timeout_from_absolute_deadline(self):
        clock = _Clock()
        timeouts = []

        def opener(_request, *, timeout):
            timeouts.append(timeout)
            clock.advance(0.75)
            if len(timeouts) == 1:
                raise urllib.error.URLError("safe fake failure")
            return _Response({"choices": [{"message": {"content": "ok"}}]})

        client = LLMClient(
            _settings(timeout_seconds=2.0, max_retries=1),
            opener=opener,
            sleep=lambda seconds: clock.advance(seconds),
            monotonic=clock,
        )
        result = client.chat(
            [{"role": "user", "content": "hello"}],
            deadline_monotonic=clock() + 2.0,
        )

        self.assertEqual(result, "ok")
        self.assertEqual(len(timeouts), 2)
        self.assertLess(timeouts[1], timeouts[0])
        self.assertLessEqual(timeouts[1], 1.0)

    def test_consumed_budget_prevents_a_second_http_attempt(self):
        clock = _Clock()
        calls = []

        def opener(_request, *, timeout):
            calls.append(timeout)
            clock.advance(timeout)
            raise urllib.error.URLError("safe fake timeout")

        client = LLMClient(
            _settings(timeout_seconds=45.0, max_retries=2),
            opener=opener,
            sleep=lambda seconds: clock.advance(seconds),
            monotonic=clock,
        )
        with self.assertRaises(ProviderError) as raised:
            client.chat(
                [{"role": "user", "content": "hello"}],
                deadline_monotonic=clock() + 0.2,
            )

        self.assertEqual(raised.exception.code, "deadline_exceeded")
        self.assertEqual(len(calls), 1)
        self.assertLessEqual(calls[0], 0.200_001)

    def test_backoff_is_checked_against_remaining_budget(self):
        clock = _Clock()
        calls = []
        sleeps = []

        def opener(_request, *, timeout):
            calls.append(timeout)
            clock.advance(0.1)
            raise urllib.error.URLError("safe fake failure")

        client = LLMClient(
            _settings(timeout_seconds=2.0, max_retries=2),
            opener=opener,
            sleep=lambda seconds: (sleeps.append(seconds), clock.advance(seconds)),
            monotonic=clock,
        )
        with self.assertRaises(ProviderError) as raised:
            client.chat(
                [{"role": "user", "content": "hello"}],
                deadline_monotonic=clock() + 0.2,
            )

        self.assertEqual(raised.exception.code, "deadline_exceeded")
        self.assertEqual(len(calls), 1)
        self.assertEqual(sleeps, [])

    def test_http_attempt_diagnostics_are_distinct_from_logical_calls(self):
        clock = _Clock()
        recorder = SmokeDiagnosticsRecorder()
        calls = []

        def opener(_request, *, timeout):
            calls.append(timeout)
            clock.advance(0.05)
            if len(calls) == 1:
                raise urllib.error.URLError("safe fake failure")
            return _Response({"choices": [{"message": {"content": "ok"}}]})

        client = LLMClient(
            _settings(max_retries=1),
            opener=opener,
            sleep=lambda seconds: clock.advance(seconds),
            monotonic=clock,
        )
        client.chat(
            [{"role": "user", "content": "hello"}],
            purpose="planner",
            diagnostics_recorder=recorder,
            deadline_monotonic=clock() + 2.0,
        )
        snapshot = recorder.snapshot()

        self.assertEqual(snapshot["provider_logical_calls"], 1)
        self.assertEqual(snapshot["provider_http_attempt_count"], 2)
        self.assertEqual(snapshot["provider_attempt_outcomes"], ["network_error", "success"])
        self.assertEqual(len(snapshot["provider_attempt_durations_ms"]), 2)
        self.assertEqual(len(snapshot["provider_attempt_timeouts_ms"]), 2)
        self.assertGreater(snapshot["backoff_total_ms"], 0)

    def test_chat_completions_uses_configured_values(self):
        captured = {}

        def opener(request, *, timeout):
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

    def test_content_is_returned_without_reasoning_content(self):
        payload = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": " final answer ",
                        "reasoning_content": "private reasoning must not be returned",
                    },
                }
            ]
        }
        client = LLMClient(
            _settings(),
            opener=lambda *_args, **_kwargs: _Response(payload),
            sleep=lambda _value: None,
        )
        self.assertEqual(
            client.chat([{"role": "user", "content": "hello"}]), "final answer"
        )

    def test_length_with_empty_content_is_typed_truncation_without_retry(self):
        calls = []
        payload = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "content": None,
                        "reasoning_content": "sensitive-reasoning-body",
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 128,
                "total_tokens": 138,
            },
        }

        def opener(*_args, **_kwargs):
            calls.append(1)
            return _Response(payload)

        client = LLMClient(_settings(), opener=opener, sleep=lambda _value: None)
        with self.assertRaises(ProviderError) as raised:
            client.chat([{"role": "user", "content": "hello"}])
        self.assertEqual(raised.exception.code, "provider_output_truncated")
        self.assertFalse(raised.exception.retryable)
        self.assertEqual(len(calls), 1)
        safe = raised.exception.to_safe_dict()
        self.assertEqual(safe["diagnostics"]["finish_reason"], "length")
        self.assertEqual(safe["diagnostics"]["usage"]["total_tokens"], 138)
        self.assertNotIn("sensitive-reasoning-body", repr(safe))

    def test_length_with_nonempty_content_is_rejected_before_content_return(self):
        body = '{"status":"answer","decision_summary":"truncated"}'
        client = LLMClient(
            _settings(),
            opener=lambda *_args, **_kwargs: _Response(
                {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {
                                "content": body,
                                "reasoning_content": "private reasoning",
                            },
                        }
                    ]
                }
            ),
            sleep=lambda _value: None,
        )

        with self.assertRaises(ProviderError) as raised:
            client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual(raised.exception.code, "provider_output_truncated")
        diagnostics = raised.exception.to_safe_dict()["diagnostics"]
        self.assertEqual(diagnostics["output_chars"], len(body))
        self.assertEqual(
            diagnostics["output_sha256"],
            hashlib.sha256(body.encode("utf-8")).hexdigest(),
        )
        self.assertNotIn(body, repr(raised.exception.to_safe_dict()))
        self.assertNotIn("private reasoning", repr(raised.exception.to_safe_dict()))

    def test_missing_finish_reason_with_nonempty_content_remains_compatible(self):
        client = LLMClient(
            _settings(),
            opener=lambda *_args, **_kwargs: _Response(
                {"choices": [{"message": {"content": "ok"}}]}
            ),
            sleep=lambda _value: None,
        )
        self.assertEqual(client.chat([{"role": "user", "content": "hello"}]), "ok")

    def test_reasoning_content_is_never_used_when_content_is_empty(self):
        client = LLMClient(
            _settings(),
            opener=lambda *_args, **_kwargs: _Response(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": "",
                                "reasoning_content": "not final content",
                            },
                        }
                    ]
                }
            ),
            sleep=lambda _value: None,
        )
        with self.assertRaises(ProviderError) as raised:
            client.chat([{"role": "user", "content": "hello"}])
        self.assertEqual(raised.exception.code, "provider_empty_content")
        self.assertNotIn("not final content", repr(raised.exception.to_safe_dict()))

    def test_stop_with_empty_content_is_typed_empty_content(self):
        payload = {
            "choices": [
                {"finish_reason": "stop", "message": {"content": ""}}
            ]
        }
        client = LLMClient(
            _settings(),
            opener=lambda *_args, **_kwargs: _Response(payload),
            sleep=lambda _value: None,
        )
        with self.assertRaises(ProviderError) as raised:
            client.chat([{"role": "user", "content": "hello"}])
        self.assertEqual(raised.exception.code, "provider_empty_content")

    def test_invalid_response_shapes_are_typed_and_safe(self):
        cases = (
            ("missing choices", {}),
            ("empty choices", {"choices": []}),
            ("invalid first choice", {"choices": ["not-an-object"]}),
            ("invalid message", {"choices": [{"message": []}]}),
            ("top-level array", []),
            (
                "invalid content type",
                {"choices": [{"message": {"content": {"secret": "body"}}}]},
            ),
            (
                "invalid reasoning type",
                {
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "reasoning_content": {"secret": "reasoning"},
                            }
                        }
                    ]
                },
            ),
        )
        for label, payload in cases:
            with self.subTest(label=label):
                client = LLMClient(
                    _settings(),
                    opener=lambda *_args, payload=payload, **_kwargs: _Response(payload),
                    sleep=lambda _value: None,
                )
                with self.assertRaises(ProviderError) as raised:
                    client.chat([{"role": "user", "content": "hello"}])
                self.assertEqual(raised.exception.code, "provider_invalid_response")
                serialized = repr(raised.exception.to_safe_dict())
                self.assertNotIn("body", serialized)
                self.assertNotIn("reasoning", serialized.lower().replace("reasoning_content", ""))
                self.assertNotIn("super-secret", serialized)
                self.assertNotIn("Authorization", serialized)

    def test_thinking_is_omitted_unless_explicitly_selected(self):
        captured = []

        def opener(request, *, timeout):
            captured.append(json.loads(request.data))
            return _Response(
                {"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]}
            )

        client = LLMClient(_settings(), opener=opener, sleep=lambda _value: None)
        client.chat([{"role": "user", "content": "hello"}])
        client.chat([{"role": "user", "content": "hello"}], thinking="disabled")
        client.chat([{"role": "user", "content": "hello"}], thinking="enabled")
        self.assertNotIn("thinking", captured[0])
        self.assertEqual(captured[1]["thinking"], {"type": "disabled"})
        self.assertEqual(captured[2]["thinking"], {"type": "enabled"})

    def test_default_payload_omits_optional_provider_capabilities(self):
        captured = {}

        def opener(request, *, timeout):
            captured.update(json.loads(request.data))
            return _Response(
                {"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]}
            )

        client = LLMClient(_settings(), opener=opener, sleep=lambda _value: None)
        client.chat([{"role": "user", "content": "hello"}])
        self.assertEqual(captured["max_tokens"], 128)
        self.assertNotIn("thinking", captured)
        self.assertNotIn("reasoning_effort", captured)
        self.assertNotIn("response_format", captured)
        self.assertNotIn("stream", captured)

    def test_successful_response_records_only_safe_provider_metadata(self):
        recorder = SmokeDiagnosticsRecorder()
        payload = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "sensitive-content-body",
                        "reasoning_content": "sensitive-reasoning-body",
                    },
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9},
        }
        client = LLMClient(
            _settings(),
            opener=lambda *_args, **_kwargs: _Response(payload),
            sleep=lambda _value: None,
            diagnostics_recorder=recorder,
        )
        self.assertEqual(
            client.chat([{"role": "user", "content": "secret prompt"}], purpose="planner"),
            "sensitive-content-body",
        )

        diagnostics = recorder.snapshot()["provider_calls"][0]
        self.assertEqual(diagnostics["purpose"], "planner")
        self.assertTrue(diagnostics["request_started"])
        self.assertTrue(diagnostics["response_received"])
        self.assertEqual(diagnostics["http_status"], 200)
        self.assertTrue(diagnostics["response_json_valid"])
        self.assertTrue(diagnostics["choices_present"])
        self.assertEqual(diagnostics["choices_count"], 1)
        self.assertEqual(diagnostics["finish_reason"], "stop")
        self.assertEqual(diagnostics["usage"]["total_tokens"], 9)
        self.assertTrue(diagnostics["content_present"])
        self.assertFalse(diagnostics["content_empty"])
        self.assertEqual(diagnostics["content_type"], "string")
        self.assertTrue(diagnostics["reasoning_content_present"])
        self.assertEqual(diagnostics["reasoning_content_type"], "string")
        serialized = json.dumps(diagnostics)
        self.assertNotIn("sensitive-content-body", serialized)
        self.assertNotIn("sensitive-reasoning-body", serialized)
        self.assertNotIn("secret prompt", serialized)

    def test_call_scoped_recorder_supports_ordinary_request_isolation(self):
        recorder = SmokeDiagnosticsRecorder()
        client = LLMClient(
            _settings(),
            opener=lambda *_args, **_kwargs: _Response(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": "grounded answer",
                                "reasoning_content": "private reasoning",
                            },
                        }
                    ]
                }
            ),
            sleep=lambda _value: None,
        )

        result = client.chat(
            [{"role": "user", "content": "private prompt"}],
            purpose="final_answer",
            diagnostics_recorder=recorder,
        )

        self.assertEqual(result, "grounded answer")
        snapshot = recorder.snapshot()
        self.assertEqual(len(snapshot["provider_calls"]), 1)
        self.assertEqual(snapshot["provider_calls"][0]["purpose"], "final_answer")
        serialized = json.dumps(snapshot)
        self.assertNotIn("grounded answer", serialized)
        self.assertNotIn("private reasoning", serialized)
        self.assertNotIn("private prompt", serialized)

    def test_failure_before_response_does_not_invent_status_or_usage(self):
        recorder = SmokeDiagnosticsRecorder()

        def opener(_request, *, timeout):
            raise urllib.error.URLError("offline-test-failure")

        client = LLMClient(
            _settings(max_retries=0),
            opener=opener,
            sleep=lambda _value: None,
            diagnostics_recorder=recorder,
        )
        with self.assertRaises(ProviderError):
            client.chat([{"role": "user", "content": "hello"}], purpose="planner")
        diagnostics = recorder.snapshot()["provider_calls"][0]
        self.assertTrue(diagnostics["request_started"])
        self.assertFalse(diagnostics["response_received"])
        self.assertNotIn("http_status", diagnostics)
        self.assertNotIn("usage", diagnostics)

    def test_retryable_status_has_bounded_retries(self):
        calls = []

        def opener(_request, *, timeout):
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
        def opener(_request, *, timeout):
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
