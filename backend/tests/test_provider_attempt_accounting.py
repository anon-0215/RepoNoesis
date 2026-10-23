"""Offline accounting regressions using the default isolated transport."""
import time
import json
import hashlib
import tempfile
import threading
from pathlib import Path
from dataclasses import replace
import unittest
from unittest.mock import patch

from app.services.llm_client import ProviderError, provider_cancellation_scope
from app.services.smoke_diagnostics import SmokeDiagnosticsRecorder
from app.services.provider_transport import (_SLOTS, active_worker_count,
    transport_audit_scope, TransportCollectionError)
from tests.test_provider_absolute_deadline import client, loopback


class ProviderAttemptAccountingTests(unittest.TestCase):
    def tearDown(self):
        self.assertEqual(active_worker_count(), 0)

    def policy(self, url):
        from urllib.parse import urlsplit
        u = urlsplit(url)
        return {'allowed_origins': [[u.scheme, u.hostname, u.port]], 'loopback_only': True}

    def assert_not_started(self, recorder):
        value = recorder.snapshot()
        self.assertEqual(value['provider_logical_calls'], 1)
        self.assertEqual(value['provider_usage_summary']['http_attempts'], 0)

    def test_slot_deadline_preserves_logical_call_without_http(self):
        for _ in range(4):
            self.assertTrue(_SLOTS.acquire(blocking=False))
        try:
            with loopback('fast') as (url, seen):
                recorder = SmokeDiagnosticsRecorder()
                with self.assertRaises(ProviderError):
                    client(url).chat([], purpose='planner', diagnostics_recorder=recorder,
                                     deadline_monotonic=time.monotonic() + .08)
                self.assertEqual(seen, [])
                self.assert_not_started(recorder)
        finally:
            for _ in range(4):
                _SLOTS.release()

    def test_spawn_failure_is_zero_http_and_releases_slot(self):
        recorder = SmokeDiagnosticsRecorder()
        with patch('app.services.provider_transport.subprocess.Popen', side_effect=OSError('fixture spawn failure')):
            with self.assertRaises(ProviderError):
                client('http://127.0.0.1:1').chat([], purpose='planner', diagnostics_recorder=recorder)
        self.assertEqual(active_worker_count(), 0)
        for _ in range(4):
            self.assertTrue(_SLOTS.acquire(blocking=False))
        for _ in range(4):
            _SLOTS.release()
        self.assert_not_started(recorder)

    def test_launched_without_confirmed_state_remains_unknown(self):
        from app.services import provider_transport
        recorder = SmokeDiagnosticsRecorder()
        events = []
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'provider_transport_worker.py').write_text('import time; time.sleep(3)\n', encoding='utf-8')
            with patch.object(provider_transport, '__file__', str(Path(directory, 'provider_transport.py'))):
                with transport_audit_scope(events.append, policy=self.policy('http://127.0.0.1:1')):
                    with self.assertRaises(ProviderError):
                        client('http://127.0.0.1:1').chat([{'role':'user','content':'large'*100000}],
                            purpose='planner', diagnostics_recorder=recorder,
                            deadline_monotonic=time.monotonic()+.2)
        value = recorder.snapshot()
        self.assertEqual(value['provider_http_attempt_count'], 0)  # confirmed lower bound
        self.assertEqual(value['provider_http_attempts_unknown'], 1)
        self.assertFalse(value['provider_http_attempt_count_complete'])
        self.assertEqual(value['provider_usage_summary']['status'], 'unknown')
        self.assertEqual(events[-1]['network_state'], 'unknown')
        from app.services.ask_diagnostics import _absolute_minimal_ask_payload
        self.assertEqual(_absolute_minimal_ask_payload(value)['provider_usage_summary']['http_attempts_unknown'], 1)

    def test_real_worker_parameters_and_start_count_match(self):
        messages = [{'role':'user','content':'中性消息\r\nlast line'}]
        events = []
        with loopback('fast') as (url, seen):
            recorder = SmokeDiagnosticsRecorder()
            model = client(url)
            model.settings = replace(model.settings, planner_max_generation_tokens=123,
                                     planner_reasoning_effort='high')
            with transport_audit_scope(events.append, policy=self.policy(url)):
                model.chat(messages, purpose='planner', thinking='enabled', temperature=.4,
                           diagnostics_recorder=recorder)
            self.assertEqual(len(seen), 1)
        starts = [e for e in events if e['event']=='network_started']
        self.assertEqual(len(starts), 1)
        self.assertEqual(len({e['attempt_id'] for e in events}), 1)
        actual = starts[0]['parameters']
        self.assertEqual(actual, [e['parameters'] for e in events if e['event']=='parent_prepared'][0])
        self.assertEqual(actual['max_tokens'], 123)
        self.assertEqual(actual['thinking'], {'type':'enabled'})
        self.assertEqual(actual['reasoning_effort'], 'high')
        self.assertEqual(actual['temperature'], .4)
        self.assertEqual(actual['model'], seen[0]['model'])
        self.assertEqual(actual['messages_sha256'], hashlib.sha256(json.dumps(seen[0]['messages'],
            ensure_ascii=False, sort_keys=True, separators=(',',':')).encode()).hexdigest())
        value = recorder.snapshot()
        self.assertEqual(value['provider_http_attempt_count'], 1)
        self.assertTrue(value['provider_http_attempt_count_complete'])
        self.assertEqual(value['provider_usage_summary']['fields']['total_tokens']['known_subtotal'], 18)

    def test_timeout_and_local_cancel_keep_confirmed_attempt_and_unknown_usage(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel), loopback('delayed') as (url, seen):
                recorder = SmokeDiagnosticsRecorder()
                token = threading.Event()
                # Cancel after the server has received the request, independent
                # of worker startup speed. The bounded helper is always joined.
                def cancel_after_receipt():
                    until = time.monotonic()+1
                    while not seen and time.monotonic()<until:
                        token.wait(.01)
                    if cancel:
                        token.set()
                thread = threading.Thread(target=cancel_after_receipt)
                thread.start()
                with provider_cancellation_scope(token.is_set):
                    with self.assertRaises(ProviderError) as caught:
                        client(url).chat([], purpose='planner', diagnostics_recorder=recorder,
                            deadline_monotonic=time.monotonic()+.4)
                thread.join(1.2)
                self.assertFalse(thread.is_alive())
                self.assertEqual(caught.exception.code, 'cancelled' if cancel else 'deadline_exceeded')
                value = recorder.snapshot()
                self.assertEqual(value['provider_http_attempt_count'], 1)
                self.assertEqual(value['provider_attempt_outcomes'], ['cancelled' if cancel else 'timeout'])
                self.assertEqual(value['provider_usage_summary']['status'], 'unknown')
                self.assertEqual(value['provider_usage_summary']['fields'], {})

    def test_finite_retry_events_have_separate_attempts(self):
        events = []
        with loopback('retry') as (url, seen):
            recorder = SmokeDiagnosticsRecorder()
            model = client(url)
            model.settings = replace(model.settings, max_retries=1)
            with transport_audit_scope(events.append, policy=self.policy(url)):
                with self.assertRaises(ProviderError):
                    model.chat([], purpose='planner', diagnostics_recorder=recorder)
            self.assertEqual(len(seen), 2)
        starts = [e for e in events if e['event']=='network_started']
        ends = [e for e in events if e['event']=='transport_finished']
        self.assertEqual(len(starts), 2)
        self.assertEqual([e['attempt_index'] for e in starts], [1,2])
        self.assertEqual(len({e['attempt_id'] for e in starts}), 2)
        self.assertEqual(len({e['logical_call_id'] for e in starts}), 1)
        self.assertEqual([e['attempt_id'] for e in starts], [e['attempt_id'] for e in ends])
        self.assertEqual(recorder.snapshot()['provider_http_attempt_count'], 2)
        self.assertEqual(recorder.snapshot()['provider_usage_summary']['usage_available_attempts'], 0)

    def test_collection_failure_is_explicit_and_does_not_leak_worker(self):
        def broken(event):
            if event['event']=='worker_received':
                raise OSError('fixture collector write failure')
        with loopback('fast') as (url, _):
            with transport_audit_scope(broken, policy=self.policy(url)):
                with self.assertRaises(TransportCollectionError):
                    client(url).chat([], purpose='planner', diagnostics_recorder=SmokeDiagnosticsRecorder())
