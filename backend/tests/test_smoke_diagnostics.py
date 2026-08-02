from __future__ import annotations

import json
import unittest

from app.services.smoke_diagnostics import (
    MAX_SMOKE_DIAGNOSTICS_BYTES,
    SmokeDiagnosticsRecorder,
    SmokeGateError,
)


class SmokeDiagnosticsTests(unittest.TestCase):
    def test_stage_and_error_fields_are_fixed_and_redacted(self):
        recorder = SmokeDiagnosticsRecorder()
        recorder.enter_stage("agent_planner")
        recorder.record_agent_result(
            {
                "agent_mode": "bounded",
                "answer_mode": "deterministic",
                "agent_status": "completed",
                "evidence": [{}],
                "citations": [{}],
                "answer": "sensitive-answer-body",
            }
        )
        error = SmokeGateError(
            code="smoke_provider_grounding_failed",
            gate="C",
            stage="gate_assertion",
            exception_type="RuntimeError",
            diagnostics=recorder.snapshot(),
        )

        payload = error.to_safe_dict()
        self.assertEqual(payload["code"], "smoke_provider_grounding_failed")
        self.assertEqual(payload["gate"], "C")
        self.assertEqual(payload["stage"], "gate_assertion")
        self.assertEqual(payload["exception_type"], "RuntimeError")
        self.assertEqual(payload["diagnostics"]["evidence_count"], 1)
        self.assertNotIn("sensitive-answer-body", json.dumps(payload))
        with self.assertRaises(ValueError):
            recorder.enter_stage("dynamic-untrusted-stage")

    def test_diagnostics_are_bounded_and_provider_fields_are_allowlisted(self):
        recorder = SmokeDiagnosticsRecorder()
        for _index in range(100):
            call_id = recorder.start_provider_call("planner")
            recorder.record_provider_response(
                call_id,
                {
                    "response_received": True,
                    "http_status": 200,
                    "response_json_valid": True,
                    "choices_present": True,
                    "choices_count": 1,
                    "finish_reason": "stop",
                    "content_present": True,
                    "content_empty": False,
                    "content_type": "string",
                    "reasoning_content_present": True,
                    "reasoning_content_type": "string",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                    "response_body": "forbidden-response-body",
                    "authorization": "forbidden-credential",
                },
            )

        payload = recorder.snapshot()
        serialized = json.dumps(payload, sort_keys=True)
        self.assertLessEqual(len(serialized.encode("utf-8")), MAX_SMOKE_DIAGNOSTICS_BYTES)
        self.assertTrue(payload["diagnostics_truncated"])
        self.assertNotIn("forbidden-response-body", serialized)
        self.assertNotIn("forbidden-credential", serialized)

    def test_final_answer_failure_reason_is_a_fixed_enum(self):
        recorder = SmokeDiagnosticsRecorder()
        recorder.record_final_answer_failure("citation_unknown")
        self.assertEqual(
            recorder.snapshot()["final_answer_failure_reason_code"],
            "citation_unknown",
        )
        with self.assertRaises(ValueError):
            recorder.record_final_answer_failure("dynamic-sensitive-reason")


if __name__ == "__main__":
    unittest.main()
