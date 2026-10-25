from __future__ import annotations

import json
import unittest

from app.services.ask_diagnostics import (
    _bounded_payload,
    build_ask_failure_detail,
    build_ask_success_diagnostics,
)
from app.services.smoke_diagnostics import (
    MAX_PROVIDER_CALLS,
    MAX_SMOKE_DIAGNOSTICS_BYTES,
    SmokeDiagnosticsRecorder,
    sanitize_provider_usage_summary,
)


class ProviderUsageSummaryTests(unittest.TestCase):
    def finish(self, recorder, call_id, outcome="success"):
        recorder.record_provider_attempt(
            call_id, outcome=outcome, duration_ms=3, timeout_ms=100
        )

    def summary(self, recorder):
        return recorder.snapshot()["provider_usage_summary"]

    def test_logical_call_without_http_attempt_is_not_usage(self):
        recorder = SmokeDiagnosticsRecorder()
        recorder.start_provider_call("planner")
        self.assertEqual(self.summary(recorder), {
            "status": "not_attempted", "http_attempts": 0,
            "usage_available_attempts": 0, "fields": {},
        })

    def test_duplicate_callbacks_count_once_and_missing_fields_stay_missing(self):
        recorder = SmokeDiagnosticsRecorder()
        call_id = recorder.start_provider_call("final_answer")
        metadata = {"usage": {"prompt_tokens": 12, "completion_tokens": 5,
                              "total_tokens": 17, "reasoning_tokens": 0}}
        recorder.record_provider_response(call_id, metadata)
        recorder.record_provider_response(call_id, metadata)
        self.finish(recorder, call_id)
        summary = self.summary(recorder)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["http_attempts"], 1)
        self.assertEqual(summary["usage_available_attempts"], 1)
        self.assertEqual(summary["fields"]["total_tokens"], {
            "known_subtotal": 17, "covered_attempts": 1,
        })
        self.assertEqual(summary["fields"]["reasoning_tokens"]["known_subtotal"], 0)
        self.assertNotIn("prompt_cache_hit_tokens", summary["fields"])

    def test_retry_counts_http_and_does_not_reuse_previous_usage(self):
        recorder = SmokeDiagnosticsRecorder()
        call_id = recorder.start_provider_call("planner")
        recorder.record_provider_response(call_id, {"usage": {
            "prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6,
        }})
        self.finish(recorder, call_id, "invalid_response")
        recorder.record_provider_response(call_id, {"response_received": True})
        self.finish(recorder, call_id, "http_error")
        summary = self.summary(recorder)
        self.assertEqual(recorder.snapshot()["provider_logical_calls"], 1)
        self.assertEqual(summary["http_attempts"], 2)
        self.assertEqual(summary["usage_available_attempts"], 1)
        self.assertEqual(summary["status"], "partial")
        self.assertEqual(summary["fields"]["total_tokens"]["known_subtotal"], 6)

    def test_network_failure_stays_unknown_not_zero(self):
        recorder = SmokeDiagnosticsRecorder()
        call_id = recorder.start_provider_call("final_answer")
        self.finish(recorder, call_id, "network_error")
        self.assertEqual(self.summary(recorder), {
            "status": "unknown", "http_attempts": 1,
            "usage_available_attempts": 0, "fields": {},
        })

    def test_partial_usage_and_optional_counters_have_individual_coverage(self):
        recorder = SmokeDiagnosticsRecorder()
        first = recorder.start_provider_call("planner")
        recorder.record_provider_response(first, {"usage": {
            "prompt_tokens": 10, "prompt_cache_hit_tokens": 7,
            "prompt_cache_miss_tokens": 3,
        }})
        self.finish(recorder, first)
        second = recorder.start_provider_call("final_answer")
        recorder.record_provider_response(second, {"usage": {
            "prompt_tokens": 8, "completion_tokens": 5, "total_tokens": 13,
        }})
        self.finish(recorder, second)
        summary = self.summary(recorder)
        self.assertEqual(summary["status"], "partial")
        self.assertEqual(summary["usage_available_attempts"], 2)
        self.assertEqual(summary["fields"]["prompt_tokens"], {
            "known_subtotal": 18, "covered_attempts": 2,
        })
        self.assertEqual(summary["fields"]["completion_tokens"]["covered_attempts"], 1)
        self.assertEqual(summary["fields"]["prompt_cache_hit_tokens"]["known_subtotal"], 7)

    def test_detail_capacity_and_size_truncation_do_not_erase_totals(self):
        recorder = SmokeDiagnosticsRecorder()
        attempts = MAX_PROVIDER_CALLS + 5
        for _ in range(attempts):
            call_id = recorder.start_provider_call("planner")
            recorder.record_provider_response(call_id, {
                "response_received": True, "response_json_valid": True,
                "choices_present": True, "choices_count": 1,
                "content_present": True, "content_type": "string",
                "finish_reason_present": True, "finish_reason": "stop",
                "output_sha256": "a" * 64,
                "usage": {"prompt_tokens": 10, "completion_tokens": 2,
                          "total_tokens": 12, "reasoning_tokens": 1,
                          "prompt_cache_hit_tokens": 8, "prompt_cache_miss_tokens": 2},
            })
            self.finish(recorder, call_id)
        snapshot = recorder.snapshot()
        summary = snapshot["provider_usage_summary"]
        self.assertTrue(snapshot["diagnostics_truncated"])
        self.assertLessEqual(len(json.dumps(snapshot).encode()), MAX_SMOKE_DIAGNOSTICS_BYTES)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["http_attempts"], attempts)
        self.assertEqual(summary["fields"]["total_tokens"], {
            "known_subtotal": 12 * attempts, "covered_attempts": attempts,
        })

    def test_metadata_accepts_only_allowlisted_nonnegative_integer_usage(self):
        recorder = SmokeDiagnosticsRecorder()
        call_id = recorder.start_provider_call("planner")
        recorder.record_provider_response(call_id, {"usage": {
            "prompt_tokens": True, "completion_tokens": 1.5,
            "total_tokens": -1, "reasoning_tokens": "secret",
            "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 2,
            "reasoning_content": "private reasoning", "api_key": "secret",
        }})
        self.finish(recorder, call_id)
        snapshot = recorder.snapshot()
        self.assertEqual(set(snapshot["provider_usage_summary"]["fields"]), {
            "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
        })
        self.assertNotIn("secret", json.dumps(snapshot))
        self.assertNotIn("private reasoning", json.dumps(snapshot))

    def test_summary_sanitizer_is_idempotent_and_ignores_claimed_status(self):
        summary = sanitize_provider_usage_summary({
            "status": "complete", "http_attempts": 2, "usage_available_attempts": 1,
            "fields": {"prompt_tokens": {"known_subtotal": 3, "covered_attempts": 1},
                       "secret": {"known_subtotal": 7, "covered_attempts": 1}},
            "api_key": "secret",
        })
        self.assertEqual(summary["status"], "partial")
        self.assertEqual(sanitize_provider_usage_summary(summary), summary)
        self.assertNotIn("secret", json.dumps(summary))
        for value in (None, {}, {"http_attempts": True},
                      {"http_attempts": 1, "usage_available_attempts": 2, "fields": {}}):
            self.assertIsNone(sanitize_provider_usage_summary(value))

    def test_actual_model_metadata_is_bounded_and_content_free(self):
        recorder = SmokeDiagnosticsRecorder()
        first = recorder.start_provider_call("planner")
        recorder.record_provider_response(first, {"response_model": "vendor/model-2026.10"})
        second = recorder.start_provider_call("final_answer")
        recorder.record_provider_response(second, {"response_model": "secret\nbody"})
        third = recorder.start_provider_call("planner")
        recorder.record_provider_response(third, {"response_model": "a" * 129})
        calls = recorder.snapshot()["provider_calls"]
        self.assertEqual(calls[0]["response_model"], "vendor/model-2026.10")
        self.assertNotIn("response_model", calls[1])
        self.assertNotIn("response_model", calls[2])
        self.assertNotIn("secret", json.dumps(calls))

    def test_public_success_failure_and_clipping_keep_summary(self):
        recorder = SmokeDiagnosticsRecorder()
        call_id = recorder.start_provider_call("final_answer")
        recorder.record_provider_response(call_id, {"usage": {
            "prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9,
        }})
        self.finish(recorder, call_id, "invalid_response")
        snapshot = recorder.snapshot()
        common = {"result": {}, "recorder_snapshot": snapshot,
                  "retrieval_version": "v1", "hierarchy_mode": "off", "relation_mode": "off"}
        success = build_ask_success_diagnostics(**common)
        failure = build_ask_failure_detail(**common, terminal_reason="provider_invalid_response")
        for diagnostic in (success, failure["diagnostics"]):
            self.assertEqual(diagnostic["provider_usage_summary"], self.summary(recorder))
            clipped = _bounded_payload({**diagnostic, "tool_executions": [
                {"detail": "x" * 400} for _ in range(100)
            ]})
            self.assertEqual(clipped["provider_usage_summary"], self.summary(recorder))
            self.assertLessEqual(len(json.dumps(clipped).encode()), 4096)
        common["recorder_snapshot"] = {}
        self.assertNotIn("provider_usage_summary", build_ask_success_diagnostics(**common))
        self.assertNotIn("provider_usage_summary", build_ask_failure_detail(
            **common, terminal_reason="provider_unavailable"
        )["diagnostics"])


if __name__ == "__main__":
    unittest.main()
