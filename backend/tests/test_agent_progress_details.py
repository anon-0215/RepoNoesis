from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from app.database import Database
from app.services.agent_core import run_bounded_agent
from app.services.ask_progress import ProgressRecorder
from app.services.smoke_diagnostics import SmokeDiagnosticsRecorder
from tests.m1_helpers import disabled_embedding_service, make_project
from tests.test_m2_agent import ScriptedPlanner, decision


class FinalLlm:
    available = True

    def chat(self, _messages, **_kwargs):
        return json.dumps({"parts": [{"text": "Grounded answer", "evidence_aliases": ["A1"]}]})


class AgentProgressDetailsTests(unittest.TestCase):
    def test_real_registry_execution_emits_targets_without_changing_answer_or_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(Path(directory) / "isolated.sqlite")
            _, bundle = make_project(db, [
                ("src/auth.py", "authenticate_user", "def authenticate_user(password):\n    return verify(password)\n"),
                ("src/upload.py", "upload_file", "def upload_file(path):\n    return save(path)\n"),
            ])
            decisions = [
                decision("continue", "lookup_symbol", {"symbol": "upload_file", "match_mode": "exact"}),
                decision("continue", "read_source", {"path": "src/upload.py", "start_line": 1, "end_line": 2}),
                decision("answer"),
            ]
            events = []
            progress = ProgressRecorder(lambda kind, data: events.append((kind, data)))
            results = []
            for recorder in (SmokeDiagnosticsRecorder(), progress):
                results.append(run_bounded_agent(
                    "authenticate_user", bundle, FinalLlm(), db, disabled_embedding_service(),
                    planner=ScriptedPlanner(decisions), execution_mode="agent", diagnostics_recorder=recorder,
                ))
            for key in ("answer", "citations", "agent_status", "answer_mode"):
                self.assertEqual(results[0][key], results[1][key], key)
            self.assertEqual(results[0]["budget_usage"]["tool_calls_used"], 2)
            self.assertEqual(results[1]["budget_usage"]["tool_calls_used"], 2)
            tools = [data for kind, data in events if kind == "tool_completed"]
            self.assertEqual([entry["tool_target"] for entry in tools], [
                {"symbol": "upload_file"}, {"path": "src/upload.py", "start_line": 1, "end_line": 2},
            ])
            self.assertTrue(all(entry["status"] == "succeeded" for entry in tools))
            # A successful repeat read may admit zero NEW evidence: still display
            # the actual read target, without claiming a newly found implementation.
            self.assertEqual(tools[-1]["evidence_added"], 0)
            self.assertNotIn("tool_target", json.dumps(progress.snapshot()))
            self.assertNotIn("upload_file", json.dumps(progress.snapshot()))

    def emit(self, tool, parameters=None, status="succeeded", phase="planner"):
        events = []
        recorder = ProgressRecorder(lambda kind, data: events.append((kind, data)))
        recorder.begin_agent([tool])
        recorder.record_tool_execution(
            phase=phase, tool_name=tool, status=status, result_count=0,
            evidence_added=0, reason_code=None, effective_parameters=parameters,
        )
        return events

    def test_missing_unsafe_or_unrelated_parameters_never_leak_into_progress(self):
        cases = [
            ("lookup_symbol", None), ("lookup_symbol", {"symbol": "a\nsecret"}),
            ("lookup_symbol", {"symbol": "x" * 161}),
            ("lookup_symbol", {"symbol": "a\u202eb"}),
            ("search_code", {"query": "private prompt"}),
        ]
        for path in ("C:/Users/private.py", "/tmp/private.py", "../secret.py", "a//b.py", "a\u202eb.py", "a\x00b.py", "x" * 501):
            cases.append(("read_source", {"path": path, "start_line": 1, "end_line": 2}))
        for start, end in ((True, 2), (1, False), (3, 2), (0, 2), (1, 10_000_001), ("1", 2)):
            cases.append(("read_source", {"path": "src/a.py", "start_line": start, "end_line": end}))
        for tool, parameters in cases:
            with self.subTest(tool=tool, parameters=parameters):
                data = self.emit(tool, parameters)[0][1]
                self.assertEqual(set(data), {"tool", "status", "evidence_added"})

    def test_only_safe_fields_are_projected_and_failures_do_not_claim_targets(self):
        params = {"symbol": "Context.close", "query": "private", "output": "secret"}
        self.assertEqual(self.emit("lookup_symbol", params)[0][1]["tool_target"], {"symbol": "Context.close"})
        for status in ("failed", "rejected", "timed_out"):
            with self.subTest(status=status):
                self.assertNotIn("tool_target", self.emit("lookup_symbol", params, status)[0][1])
        self.assertEqual(self.emit("read_source", {"path": "src\\中文.py", "start_line": 1, "end_line": 2})[0][1]["tool_target"]["path"], "src/中文.py")


if __name__ == "__main__":
    unittest.main()
