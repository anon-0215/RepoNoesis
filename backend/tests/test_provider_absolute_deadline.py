"""Short, credential-free loopback transport deadline regressions."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import threading
import time
import unittest
import urllib.error
from unittest.mock import patch

from app.config import LLMSettings
from app.services.llm_client import LLMClient, ProviderError, provider_cancellation_scope
from app.services.agent_contracts import AgentLimits, CancellationToken
from app.services.provider_transport import active_worker_count
from app.services.smoke_diagnostics import SmokeDiagnosticsRecorder

MEASUREMENTS = []


@contextmanager
def loopback(mode="trickle"):
    stop = threading.Event()
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(payload)
            content = json.dumps({"parts": [{"text": "Returns one.", "evidence_aliases": ["A1"]}]})
            if mode == "repair" and len(seen) == 1:
                content = "malformed initial final answer"
            body = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": content}}],
                "model": "loopback-fixture", "usage": {"prompt_tokens": 11,
                    "completion_tokens": 7, "total_tokens": 18}}).encode()
            if mode == "headers":
                stop.wait(1.2)
            if mode == "retry":
                self.send_response(503)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                if mode == "delayed" or mode == "repair" and len(seen) > 1:
                    stop.wait(1.2)
                    self.wfile.write(body)
                elif mode in {"fast", "repair"} or payload.get("temperature") == 0.1:
                    self.wfile.write(body)
                else:
                    for offset in range(0, len(body), 12):
                        if stop.wait(0.06):
                            break
                        self.wfile.write(body[offset:offset + 12])
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = False
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02})
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", seen
    finally:
        stop.set()
        server.shutdown()
        server.server_close()
        thread.join(2)
        assert not thread.is_alive()


def client(url, **kwargs):
    return LLMClient(LLMSettings(provider="openai_compatible", base_url=url,
        api_key="loopback-placeholder", model="fixture", timeout_seconds=2,
        max_retries=0), **kwargs)


class ProviderAbsoluteDeadlineTests(unittest.TestCase):
    def tearDown(self):
        until = time.monotonic() + 0.5
        while active_worker_count() and time.monotonic() < until:
            time.sleep(0.01)
        self.assertEqual(active_worker_count(), 0, "Local transport workers were not reaped")

    def test_trickling_body_is_stopped_at_absolute_deadline(self):
        with loopback() as (url, seen):
            started = time.monotonic()
            with self.assertRaises(ProviderError) as caught:
                client(url).chat([], deadline_monotonic=started + 0.45)
            elapsed = time.monotonic() - started
            self.assertEqual(caught.exception.code, "deadline_exceeded")
            self.assertLess(elapsed, 0.8)
            self.assertEqual(len(seen), 1)
            MEASUREMENTS.append({"case": "trickle", "deadline_seconds": 0.45, "elapsed_seconds": elapsed})

    def test_delayed_headers_and_delayed_body_are_interruptible(self):
        for mode in ("headers", "delayed"):
            with self.subTest(mode=mode), loopback(mode) as (url, seen):
                recorder = SmokeDiagnosticsRecorder()
                started = time.monotonic()
                with self.assertRaises(ProviderError) as caught:
                    client(url).chat([], purpose="planner", diagnostics_recorder=recorder,
                        deadline_monotonic=started + 0.35)
                elapsed = time.monotonic() - started
                self.assertEqual(caught.exception.code, "deadline_exceeded")
                self.assertLess(elapsed, 0.7)
                snapshot = recorder.snapshot()
                self.assertEqual(snapshot["provider_attempt_outcomes"], ["timeout"])
                self.assertEqual(snapshot["provider_usage_summary"]["status"], "unknown")
                if mode == "delayed":
                    self.assertEqual(snapshot["provider_calls"][0]["http_status"], 200)
                MEASUREMENTS.append({"case": mode, "deadline_seconds": 0.35, "elapsed_seconds": elapsed})

    def test_retry_and_backoff_share_the_same_deadline(self):
        # Worker cold start does not promise exactly two wire arrivals. Use the
        # client's existing injected clock/sleep/opener boundary to prove retry
        # and backoff arithmetic at the unchanged 650ms request cutoff.
        clock=[100.0];attempts=[]
        def fail(request,**kwargs):
            attempts.append(clock[0]);clock[0]+=.1
            raise urllib.error.HTTPError(request.full_url,503,"",{},None)
        def sleep(seconds):clock[0]+=seconds
        llm=client("http://127.0.0.1:1",opener=fail,monotonic=lambda:clock[0],sleep=sleep)
        llm.settings=replace(llm.settings,max_retries=5)
        recorder=SmokeDiagnosticsRecorder()
        with self.assertRaises(ProviderError) as caught:
            llm.chat([],purpose="planner",diagnostics_recorder=recorder,deadline_monotonic=100.65)
        self.assertEqual(caught.exception.code,"deadline_exceeded")
        self.assertEqual(len(attempts),2)
        self.assertTrue(all(at<100.65 for at in attempts))
        self.assertLess(clock[0],100.65)
        self.assertGreater(recorder.snapshot()["backoff_total_ms"],0)
        self.assertEqual(active_worker_count(),0)

    def test_cancel_body_and_next_request_remains_isolated(self):
        token = CancellationToken()
        with loopback("delayed") as (url, seen):
            timer = threading.Timer(0.2, token.cancel)
            timer.start()
            self.addCleanup(timer.join)
            started = time.monotonic()
            with provider_cancellation_scope(lambda: token.cancelled):
                with self.assertRaises(ProviderError) as caught:
                    client(url).chat([], deadline_monotonic=started + 2)
            self.assertEqual(caught.exception.code, "cancelled")
            self.assertLess(time.monotonic() - started, 0.6)
        with loopback("fast") as (url, seen):
            recorder = SmokeDiagnosticsRecorder()
            self.assertIn("Returns one", client(url).chat([], purpose="final_answer", diagnostics_recorder=recorder))
            self.assertEqual(len(seen), 1)
            self.assertEqual(recorder.snapshot()["provider_usage_summary"]["fields"]["total_tokens"],
                {"known_subtotal": 18, "covered_attempts": 1})

    def test_cancel_backoff_does_not_retry(self):
        token = CancellationToken()
        attempts = []
        def fail(request, **kwargs):
            attempts.append(1)
            raise urllib.error.HTTPError(request.full_url, 503, "", {}, None)
        timer = threading.Timer(0.04, token.cancel)
        timer.start()
        self.addCleanup(timer.join)
        llm = client("http://127.0.0.1:1", opener=fail)
        llm.settings = replace(llm.settings, max_retries=2)
        with provider_cancellation_scope(lambda: token.cancelled):
            with self.assertRaises(ProviderError) as caught:
                llm.chat([], deadline_monotonic=time.monotonic() + 1)
        self.assertEqual(caught.exception.code, "cancelled")
        self.assertEqual(len(attempts), 1)

    def test_saturated_worker_pool_cannot_spawn_or_wait_without_bound(self):
        from app.services.provider_transport import _SLOTS
        for _ in range(4):
            self.assertTrue(_SLOTS.acquire(blocking=False))
        try:
            with loopback("fast") as (url, seen):
                started = time.monotonic()
                with self.assertRaises(ProviderError) as caught:
                    client(url).chat([], deadline_monotonic=started + 0.08)
                self.assertEqual(caught.exception.code, "deadline_exceeded")
                self.assertLess(time.monotonic() - started, 0.3)
                self.assertEqual(seen, [])
                self.assertEqual(active_worker_count(), 0)
        finally:
            for _ in range(4):
                _SLOTS.release()

    def test_large_stdin_write_cannot_block_owner_watchdog(self):
        # A deliberately stalled local helper never reads stdin. This covers
        # Windows' synchronous pipe-write trap, independently of HTTP reads.
        from app.services import provider_transport
        with tempfile.TemporaryDirectory() as temporary:
            worker = Path(temporary)/"provider_transport_worker.py"
            worker.write_text("import time; time.sleep(3)\n", encoding="utf-8")
            with patch.object(provider_transport, "__file__", str(Path(temporary)/"provider_transport.py")):
                started = time.monotonic()
                with self.assertRaises(ProviderError) as caught:
                    client("http://127.0.0.1:1").chat([
                        {"role": "user", "content": "large-input-" * 100000}],
                        deadline_monotonic=started + 0.2)
                self.assertEqual(caught.exception.code, "deadline_exceeded")
                self.assertLess(time.monotonic() - started, 0.55)
                self.assertEqual(active_worker_count(), 0)

    def test_spawn_failure_releases_capacity(self):
        from app.services.provider_transport import _SLOTS
        with patch("app.services.provider_transport.subprocess.Popen", side_effect=OSError("fixture spawn failure")):
            with self.assertRaises(ProviderError) as caught:
                client("http://127.0.0.1:1").chat([])
            self.assertEqual(caught.exception.code, "provider_unavailable")
        for _ in range(4):
            self.assertTrue(_SLOTS.acquire(blocking=False))
        for _ in range(4):
            _SLOTS.release()

    def test_cancellation_scope_propagates_original_frozen_provider_error(self):
        error = ProviderError("provider_empty_content", "fixture")
        with self.assertRaises(ProviderError) as caught:
            with provider_cancellation_scope(lambda: False):
                raise error
        self.assertIs(caught.exception, error)

    def test_repeated_timeouts_reap_children_and_preserve_unknown_usage(self):
        with loopback("delayed") as (url, seen):
            for _ in range(3):
                recorder = SmokeDiagnosticsRecorder()
                with self.assertRaises(ProviderError):
                    client(url).chat([], purpose="planner", diagnostics_recorder=recorder,
                        deadline_monotonic=time.monotonic() + 0.18)
                snapshot = recorder.snapshot()
                self.assertEqual(snapshot["provider_usage_summary"]["fields"], {})
                self.assertEqual(active_worker_count(), 0)
                self.assertEqual(snapshot["provider_attempt_outcomes"],["timeout"])
                self.assertEqual(len(snapshot["provider_calls"]),1)
            # Three logical attempts need not reach HTTP within 180ms cold start.
            self.assertLessEqual(len(seen),3)
            count=len(seen);time.sleep(.05)
            self.assertEqual(len(seen),count,"late request after worker cleanup")

    def _run_agent(self, llm, **kwargs):
        from app.database import Database
        from app.services.agent_core import run_bounded_agent
        from tests.m1_helpers import disabled_embedding_service, make_project
        with tempfile.TemporaryDirectory() as temporary:
            database = Database(Path(temporary)/"deadline-fixture.sqlite")
            _, bundle = make_project(database, [("src/example.py", "example", "def example():\n    return 1\n")])
            return run_bounded_agent("example", bundle, llm, database,
                disabled_embedding_service(), **kwargs)

    def test_planner_cutoff_really_leaves_final_reserve(self):
        recorder = SmokeDiagnosticsRecorder()
        limits = replace(AgentLimits(), total_deadline_ms=2000, min_final_answer_budget_ms=1400)
        from app.services import agent_finalization_phase
        original = agent_finalization_phase.run_finalization_phase
        entries = []
        def enter(**kwargs):
            entries.append(kwargs["state"].budget.request_remaining_ms(time.monotonic()))
            return original(**kwargs)
        with loopback() as (url, seen), patch("app.services.agent_core.run_finalization_phase", side_effect=enter):
            started = time.monotonic()
            result = self._run_agent(client(url), limits=limits, diagnostics_recorder=recorder)
            elapsed = time.monotonic() - started
            self.assertEqual(result["agent_status"], "completed")
            self.assertEqual(result["answer_mode"], "llm_grounded")
            self.assertGreaterEqual(entries[0], 1200)
            self.assertEqual(len(seen), 2)
            self.assertEqual([p["temperature"] for p in seen], [0.0, 0.1])
            self.assertEqual(result["budget_usage"]["tool_calls_used"], 0)
            self.assertEqual(len(result["citations"]), 1)
            MEASUREMENTS.append({"case": "planner_to_final", "total_ms": 2000,
                "reserve_ms": 1400, "final_entry_remaining_ms": entries[0], "elapsed_seconds": elapsed})

    def _controlled_final(self, *, expire_at):
        # Explicit transport double with one shared virtual request clock. This
        # proves exact gate/call behavior without relying on worker startup speed.
        import io
        from app.services import qa_agent
        clock = [100.0]
        dispatches, deadlines, repair_entries = [], [], []
        def opener(request, **kwargs):
            dispatches.append(clock[0])
            if len(dispatches) == 1:
                clock[0] += .1
                content = "malformed initial final answer"
                if expire_at == "initial":
                    clock[0] = deadlines[0] + .01
            else:
                clock[0] = deadlines[0] + .01
                content = json.dumps({"parts": [{"text": "Returns one.", "evidence_aliases": ["A1"]}]})
            response = io.BytesIO(json.dumps({"choices": [{"message": {"content": content}}]}).encode())
            response.status = 200
            return response
        recorder = SmokeDiagnosticsRecorder()
        llm = client("http://127.0.0.1:1", opener=opener, monotonic=lambda: clock[0])
        original_chat = llm.chat
        def chat(*args, **kwargs):
            deadlines.append(kwargs["deadline_monotonic"])
            return original_chat(*args, **kwargs)
        original_builder = qa_agent._build_final_answer_repair_messages
        def repair_messages(**kwargs):
            repair_entries.append({"deadline": kwargs["request_deadline_at"],
                                   "remaining_ms": (kwargs["request_deadline_at"]-clock[0])*1000})
            if expire_at == "repair_messages":
                clock[0] = kwargs["request_deadline_at"] + .01
            return original_builder(**kwargs)
        with patch("time.monotonic", side_effect=lambda: clock[0]), patch.object(llm,"chat",side_effect=chat), patch.object(qa_agent,"_build_final_answer_repair_messages",side_effect=repair_messages):
            result = self._run_agent(llm, execution_mode="rag",
                limits=replace(AgentLimits(), total_deadline_ms=650, min_final_answer_budget_ms=200),
                diagnostics_recorder=recorder)
        self.assertEqual(result["agent_status"], "budget_exhausted")
        self.assertEqual(result["answer"], "")
        self.assertTrue(all(deadline == 100.65 for deadline in deadlines))
        self.assertTrue(all(at < 100.65 for at in dispatches), "request dispatched after total cutoff")
        self.assertEqual(active_worker_count(),0)
        MEASUREMENTS.append({"case": "controlled_"+expire_at, "total_ms":650,
            "dispatches":dispatches,"deadlines":deadlines,"repair_entries":repair_entries,
            "status":result["agent_status"],"workers":active_worker_count()})
        return dispatches, deadlines, repair_entries, recorder.snapshot()

    def test_final_protocol_repair_has_no_new_total_budget(self):
        calls, deadlines, repair, snapshot = self._controlled_final(expire_at="repair_response")
        self.assertEqual(len(calls),2)
        self.assertEqual(deadlines,[100.65,100.65])
        self.assertAlmostEqual(repair[0]["remaining_ms"],550)
        self.assertTrue(snapshot["final_answer_repair_attempted"])

    def test_initial_final_exhaustion_does_not_dispatch_repair(self):
        calls, deadlines, repair, snapshot = self._controlled_final(expire_at="initial")
        self.assertEqual(len(calls),1)
        self.assertEqual(len(deadlines),1)
        self.assertEqual(repair,[])
        self.assertFalse(snapshot["final_answer_repair_attempted"])

    def test_repair_message_building_cannot_create_a_new_budget(self):
        calls, deadlines, repair, snapshot = self._controlled_final(expire_at="repair_messages")
        self.assertEqual(len(calls),1)
        self.assertEqual(len(deadlines),1)
        self.assertEqual(len(repair),1)
        self.assertEqual(repair[0]["deadline"],deadlines[0])
        self.assertFalse(snapshot["final_answer_repair_attempted"])

    def test_real_loopback_repair_cutoff_reaps_local_transport(self):
        # Cold process startup may consume the remaining budget before a second
        # wire request exists. Exact two-call gating is proved above; this checks
        # the real coordinator/worker boundary at the unchanged 650ms cutoff.
        recorder=SmokeDiagnosticsRecorder();calls=[]
        with loopback("repair") as (url,seen):
            llm=client(url);original=llm.chat
            def chat(*args,**kwargs):
                calls.append({"at":time.monotonic(),"deadline":kwargs["deadline_monotonic"]})
                return original(*args,**kwargs)
            with patch.object(llm,"chat",side_effect=chat):
                result=self._run_agent(llm,execution_mode="rag",
                    limits=replace(AgentLimits(),total_deadline_ms=650,min_final_answer_budget_ms=200),
                    diagnostics_recorder=recorder)
            self.assertEqual(result["agent_status"],"budget_exhausted")
            self.assertEqual(result["answer"],"")
            self.assertGreaterEqual(len(calls),1)
            self.assertLessEqual(len(calls),2)
            self.assertEqual(len({c["deadline"] for c in calls}),1)
            self.assertTrue(all(c["at"]<c["deadline"] for c in calls))
            self.assertLess(time.monotonic()-calls[0]["deadline"],.3)
            self.assertEqual(active_worker_count(),0)
            count=len(seen);time.sleep(.1)
            self.assertEqual(len(seen),count,"request appeared after owner deadline/cleanup")
            MEASUREMENTS.append({"case":"real_repair_cutoff","total_ms":650,"calls":calls,
                "wire_requests":count,"status":result["agent_status"],"workers":active_worker_count()})

    def test_cancelled_initial_final_cannot_start_repair_or_fallback(self):
        token = CancellationToken()
        import io
        requests = []
        def opener(request, **kwargs):
            requests.append(request)
            token.cancel()
            response = io.BytesIO(json.dumps({"choices": [{"message": {"content": "invalid"}}]}).encode())
            response.status = 200
            return response
        result = self._run_agent(client("http://127.0.0.1:1", opener=opener),
            cancellation=token, execution_mode="rag")
        self.assertEqual(result["agent_status"], "cancelled")
        self.assertEqual(result["answer"], "")
        self.assertEqual(len(requests), 1)

    def test_validation_deadlines_have_an_explicit_empty_terminal_result(self):
        from app.services import agent_core
        original = agent_core._validated_relation_context
        for checkpoint in ("initial_evidence", "post_answer_evidence"):
            with self.subTest(checkpoint=checkpoint), loopback("fast") as (url, seen):
                def validate(state, *args, **kwargs):
                    if kwargs["checkpoint"] == checkpoint:
                        time.sleep(max(0, state.context.deadline_monotonic - time.monotonic()) + 0.005)
                    return original(state, *args, **kwargs)
                recorder = SmokeDiagnosticsRecorder()
                with patch.object(agent_core, "_validated_relation_context", side_effect=validate):
                    result = self._run_agent(client(url), execution_mode="rag",
                        limits=replace(AgentLimits(), total_deadline_ms=400, min_final_answer_budget_ms=50),
                        diagnostics_recorder=recorder)
                self.assertEqual(result["agent_status"], "budget_exhausted")
                self.assertEqual(result["answer"], "")
                self.assertEqual(result["citations"], [])
                self.assertEqual(len(seen), 0 if checkpoint == "initial_evidence" else 1)
                self.assertTrue(recorder.snapshot()["request_deadline_reached"])

    def test_late_injected_response_does_not_become_success_or_known_usage(self):
        import io
        clock = [100.0]
        def opener(request, **kwargs):
            clock[0] += 1
            response = io.BytesIO(json.dumps({"choices": [{"message": {"content": "late"}}],
                "usage": {"total_tokens": 42}}).encode())
            response.status = 200
            return response
        recorder = SmokeDiagnosticsRecorder()
        llm = client("http://127.0.0.1:1", opener=opener, monotonic=lambda: clock[0])
        with self.assertRaises(ProviderError) as caught:
            llm.chat([], purpose="planner", diagnostics_recorder=recorder, deadline_monotonic=100.1)
        self.assertEqual(caught.exception.code, "deadline_exceeded")
        self.assertEqual(recorder.snapshot()["provider_attempt_outcomes"], ["timeout"])
        self.assertEqual(recorder.snapshot()["provider_usage_summary"]["fields"], {})


if __name__ == "__main__":
    unittest.main()
