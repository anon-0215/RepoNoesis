"""Answer-only model routing with fake transport and explicit temporary storage."""
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
import urllib.error
from unittest.mock import patch

from app.config import LLMSettings, get_llm_settings, get_product_config_status
from app.database import Database
from app.services.agent_core import run_bounded_agent
from app.services.evidence import Evidence, EvidenceBuilder
from app.services.llm_client import LLMClient, ProviderError, provider_cancellation_scope
from app.services.qa_agent import answer_from_evidence, answer_question
from app.services.smoke_diagnostics import SmokeDiagnosticsRecorder, _safe_smoke_diagnostics
from tests.m1_helpers import disabled_embedding_service, make_project


ENV = dict(LLM_PROVIDER="openai_compatible", LLM_BASE_URL="https://provider.invalid/v1",
           LLM_API_KEY="FAKE_KEY_DO_NOT_LOG", LLM_MODEL="deepseek-v4-pro")
FLASH_ENV = dict(LLM_ANSWER_MODEL="deepseek-flash", LLM_ANSWER_THINKING="enabled",
                 LLM_ANSWER_REASONING_EFFORT="low", LLM_ANSWER_MAX_GENERATION_TOKENS="16384")


def settings(**overrides):
    with patch.dict(os.environ, ENV | FLASH_ENV, clear=True):
        return replace(get_llm_settings(), max_retries=0, **overrides)


def structured(text="The stored function returns True.", aliases=None):
    return json.dumps({"parts": [{"text": text, "evidence_aliases": aliases or ["A1"]}]})


class Response:
    status = 200

    def __init__(self, content, *, model="provider-returned-snapshot", finish="stop"):
        self.body = json.dumps({"model": model, "choices": [{"finish_reason": finish,
            "message": {"content": content, "reasoning_content": "PRIVATE_THOUGHT"}}]}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


class AnswerModelOptionTests(unittest.TestCase):
    def setUp(self):
        # Each test remains safe even when run without the guarded receipt runner.
        for target in ("urllib.request.urlopen", "app.services.llm_client.LLMClient._isolated_response",
                       "app.services.embedding_service.SentenceTransformerEmbeddingBackend.load_model"):
            guard = patch(target, side_effect=AssertionError("Real network/model loading forbidden"))
            guard.start()
            self.addCleanup(guard.stop)
        original_init = Database.__init__

        def explicit_database(instance, path=None):
            if path is None:
                raise AssertionError("Implicit/default database forbidden")
            return original_init(instance, path)

        guard = patch.object(Database, "__init__", explicit_database)
        guard.start()
        self.addCleanup(guard.stop)

    def fixture(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        db = Database(Path(directory.name) / "answer-option.sqlite")
        project_id, bundle = make_project(db, [
            ("src/auth.py", "authenticate_user", "def authenticate_user():\n    return True\n")])
        evidence = EvidenceBuilder().build_from_code_chunks(db.get_code_chunks(project_id), bundle["project"])
        return db, bundle, evidence

    def client(self, config=None, outputs=None, recorder=None):
        sent = []
        outputs = iter(outputs or [structured()])

        def opener(request, *, timeout):
            body = json.loads(request.data)
            sent.append((body, timeout, request.full_url))
            if body["temperature"] == 0:
                return Response('{"status":"answer","decision_summary":"enough evidence"}')
            return Response(next(outputs))

        return LLMClient(config or settings(), opener=opener, diagnostics_recorder=recorder), sent

    def assert_flash(self, payload):
        self.assertEqual(payload["model"], "deepseek-flash")
        self.assertEqual(payload["thinking"], {"type": "enabled"})
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertEqual(payload["max_tokens"], 16384)

    def test_unset_and_blank_preserve_exact_legacy_requests(self):
        expected = dict(model="deepseek-v4-pro", messages=[], temperature=0.1,
                        max_tokens=1600, thinking={"type": "disabled"})
        for override in (None, "", "   \t"):
            with self.subTest(override=override):
                env = ENV | ({"LLM_ANSWER_MODEL": override} if override is not None else {})
                with patch.dict(os.environ, env, clear=True):
                    config = get_llm_settings()
                self.assertIsNone(config.answer_model)
                client, sent = self.client(config)
                client.chat([], purpose="final_answer", temperature=0.1, thinking="disabled")
                self.assertEqual(sent[0][0], expected)
        # Programmatic whitespace also inherits, without mutating settings.
        client, sent = self.client(settings(answer_model="  ", answer_reasoning_effort=None,
            answer_max_generation_tokens=None))
        client.chat([], purpose="final_answer", temperature=0.1, thinking="disabled")
        self.assertEqual(sent[0][0], expected)

    def test_alternating_calls_are_local_and_other_purposes_are_unchanged(self):
        client, sent = self.client(settings(planner_thinking="disabled"), outputs=[structured()] * 6)
        for purpose in ("final_answer", "planner", None, "final_answer", "planner", "other"):
            client.chat([], purpose=purpose, max_tokens=512,
                        thinking="enabled" if purpose == "final_answer" else None)
        for index in (0, 3):
            self.assert_flash(sent[index][0])
        for index in (1, 2, 4, 5):
            self.assertEqual(sent[index][0], dict(model="deepseek-v4-pro", messages=[],
                                                temperature=0.2, max_tokens=512))
        self.assertEqual(client.model, "deepseek-v4-pro")
        self.assertEqual(client.settings.model, "deepseek-v4-pro")

    def test_rag_and_agent_initial_and_single_protocol_repair(self):
        db, bundle, evidence = self.fixture()
        for mode in ("rag", "agent"):
            for repair in (False, True):
                with self.subTest(mode=mode, repair=repair):
                    recorder = SmokeDiagnosticsRecorder()
                    client, sent = self.client(outputs=(["bad JSON"] if repair else []) + [structured()], recorder=recorder)
                    if mode == "rag":
                        result = answer_question("authenticate_user", bundle, client, db,
                            disabled_embedding_service(), max_answer_tokens=1600, diagnostics_recorder=recorder)
                    else:
                        result = run_bounded_agent("authenticate_user", bundle, client, db,
                            disabled_embedding_service(), diagnostics_recorder=recorder)
                        self.assertEqual(result["agent_status"], "completed")
                    self.assertEqual(result["answer_mode"], "llm_grounded")
                    self.assertTrue(result["citations"])
                    answers = [p for p, _t, _u in sent if p["temperature"] == 0.1]
                    self.assertEqual(len(answers), 2 if repair else 1)
                    for payload in answers:
                        self.assert_flash(payload)
                        self.assertIn("return True", payload["messages"][1]["content"])
                    planners = [p for p, _t, _u in sent if p["temperature"] == 0]
                    self.assertEqual(len(planners), int(mode == "agent"))
                    for payload in planners:
                        self.assertEqual(payload["model"], "deepseek-v4-pro")
                        self.assertEqual(payload["max_tokens"], 512)
                        self.assertNotIn("reasoning_effort", payload)
                    self.assertTrue(all(url == ENV["LLM_BASE_URL"] + "/chat/completions" for _p, _t, url in sent))

    def test_repair_is_still_bounded_to_one(self):
        db, _bundle, evidence = self.fixture()
        client, sent = self.client(outputs=["bad JSON", "still bad JSON"])
        result = answer_from_evidence("explain", evidence, client, db, retrieval_mode="hybrid")
        self.assertEqual(result["answer_mode"], "deterministic")
        self.assertEqual(len(sent), 2)
        for payload, _t, _u in sent:
            self.assert_flash(payload)

    def test_http_retry_reuses_request_and_absolute_deadline(self):
        sent, now = [], [100.0]
        recorder = SmokeDiagnosticsRecorder()

        def opener(request, *, timeout):
            sent.append((request.data, timeout))
            now[0] += 0.5
            if len(sent) == 1:
                raise urllib.error.HTTPError(request.full_url, 503, "FAKE_KEY_DO_NOT_LOG", {}, None)
            return Response(structured())

        config = replace(settings(), max_retries=1)
        client = LLMClient(config, opener=opener, monotonic=lambda: now[0],
            sleep=lambda duration: now.__setitem__(0, now[0] + duration), diagnostics_recorder=recorder)
        client.chat([], purpose="final_answer", thinking="enabled", deadline_monotonic=102.0)
        self.assertEqual(len(sent), 2)
        self.assertEqual(sent[0][0], sent[1][0])
        self.assertEqual([t for _p, t in sent], [2.0, 1.25])
        self.assert_flash(json.loads(sent[0][0]))
        self.assertEqual(recorder.snapshot()["provider_http_attempt_count"], 2)

    def test_errors_and_metadata_distinguish_request_from_response_without_secrets(self):
        for body in (b"not JSON", Response("", finish="length").body, None):
            recorder = SmokeDiagnosticsRecorder()

            def opener(request, **_kwargs):
                if body is None:
                    raise urllib.error.HTTPError(request.full_url, 401, "FAKE_KEY_DO_NOT_LOG", {}, None)
                response = Response("")
                response.body = body
                return response

            client = LLMClient(settings(), opener=opener, diagnostics_recorder=recorder)
            with self.assertRaises(ProviderError) as raised:
                client.chat([], purpose="final_answer", thinking="enabled")
            safe = raised.exception.to_safe_dict()
            self.assertEqual(safe["diagnostics"]["model"], "deepseek-flash")
            snapshot = _safe_smoke_diagnostics(recorder.snapshot())
            call = snapshot["provider_calls"][0]
            self.assertEqual(call["request_model"], "deepseek-flash")
            if body == Response("", finish="length").body:
                self.assertEqual(call["response_model"], "provider-returned-snapshot")
            serialized = json.dumps([safe, snapshot])
            for secret in ("FAKE_KEY_DO_NOT_LOG", "PRIVATE_THOUGHT", "Authorization", "Bearer"):
                self.assertNotIn(secret, serialized)
        recorder = SmokeDiagnosticsRecorder()
        client, _sent = self.client(recorder=recorder)
        client.chat([], purpose="final_answer", thinking="enabled")
        call = recorder.snapshot()["provider_calls"][0]
        self.assertEqual((call["request_model"], call["response_model"]),
                         ("deepseek-flash", "provider-returned-snapshot"))

    def test_configuration_summary_preserves_global_model_and_shows_override(self):
        with patch.dict(os.environ, ENV | FLASH_ENV, clear=True):
            summary = get_product_config_status()["llm"]
        self.assertEqual(summary["model"], "deepseek-v4-pro")
        self.assertEqual(summary["answer_model"], "deepseek-flash")
        self.assertEqual(summary["effective_answer_model"], "deepseek-flash")
        self.assertEqual(summary["answer_max_generation_tokens"], 16384)
        self.assertEqual(summary["answer_reasoning_effort"], "low")
        self.assertNotIn(ENV["LLM_API_KEY"], repr(summary))

    def test_illegal_combinations_fail_before_transport(self):
        for overrides in (dict(answer_thinking="disabled"), dict(answer_reasoning_effort="invalid"),
                          dict(answer_max_generation_tokens=65537), dict(answer_max_generation_tokens=True)):
            db, _bundle, evidence = self.fixture()
            client, sent = self.client(settings(**overrides))
            with self.assertRaises(ValueError):
                answer_from_evidence("explain", evidence, client, db, retrieval_mode="hybrid")
            self.assertEqual(sent, [])

    def test_visible_budget_and_citation_identity_and_range_remain_enforced(self):
        db, _bundle, evidence = self.fixture()
        client, sent = self.client(outputs=[structured("A" * 1000)])
        result = answer_from_evidence("explain", evidence, client, db, retrieval_mode="hybrid", max_answer_tokens=10)
        self.assertEqual(result["answer_mode"], "deterministic")
        self.assert_flash(sent[0][0])
        for overrides in (dict(path="forged.py"), dict(end_line=999), dict(content_hash="0" * 64)):
            client, sent = self.client()
            bad = Evidence(**(evidence[0].to_dict() | overrides))
            result = answer_from_evidence("explain", [bad], client, db, retrieval_mode="hybrid")
            self.assertEqual(result["grounding_status"], "insufficient_evidence")
            self.assertEqual(sent, [])

    def test_expired_deadline_and_cancellation_never_send(self):
        client, sent = self.client()
        with self.assertRaises(ProviderError) as raised:
            client.chat([], purpose="final_answer", thinking="enabled", deadline_monotonic=time.monotonic() - 1)
        self.assertEqual(raised.exception.code, "deadline_exceeded")
        with provider_cancellation_scope(lambda: True):
            with self.assertRaises(ProviderError) as raised:
                client.chat([], purpose="final_answer", thinking="enabled")
        self.assertEqual(raised.exception.code, "cancelled")
        self.assertEqual(sent, [])

    def test_repair_shares_absolute_deadline_and_stops_when_initial_call_exhausts_it(self):
        for elapsed, expected_calls in ((0.4, 2), (2.1, 1)):
            with self.subTest(elapsed=elapsed):
                db, _bundle, evidence = self.fixture()
                now, sent = [100.0], []

                def opener(request, *, timeout):
                    sent.append((json.loads(request.data), timeout))
                    now[0] += elapsed
                    return Response("bad JSON" if len(sent) == 1 else structured())

                client = LLMClient(settings(), opener=opener, monotonic=lambda: now[0])
                with patch("app.services.qa_agent.time.monotonic", side_effect=lambda: now[0]):
                    if expected_calls == 1:
                        with self.assertRaises(ProviderError) as raised:
                            answer_from_evidence("explain", evidence, client, db,
                                retrieval_mode="hybrid", request_deadline_at=102.0)
                        self.assertEqual(raised.exception.code, "deadline_exceeded")
                    else:
                        result = answer_from_evidence("explain", evidence, client, db,
                            retrieval_mode="hybrid", request_deadline_at=102.0)
                        self.assertEqual(result["answer_mode"], "llm_grounded")
                self.assertEqual(len(sent), expected_calls)
                self.assertEqual(sent[0][1], 2.0)
                if expected_calls == 2:
                    self.assertAlmostEqual(sent[1][1], 1.6)
                for payload, _timeout in sent:
                    self.assert_flash(payload)

    def test_retry_exhaustion_and_missing_response_identity_are_honest(self):
        sent, recorder = [], SmokeDiagnosticsRecorder()

        def opener(request, **_kwargs):
            sent.append(request.data)
            raise urllib.error.HTTPError(request.full_url, 503, "fake", {}, None)

        client = LLMClient(replace(settings(), max_retries=2), opener=opener,
                           sleep=lambda _seconds: None, diagnostics_recorder=recorder)
        with self.assertRaises(ProviderError) as raised:
            client.chat([], purpose="final_answer", thinking="enabled")
        self.assertEqual(len(sent), 3)
        self.assertEqual(len(set(sent)), 1)
        self.assertEqual(raised.exception.diagnostics["model"], "deepseek-flash")
        self.assertNotIn("response_model", recorder.snapshot()["provider_calls"][0])
        recorder = SmokeDiagnosticsRecorder()
        client = LLMClient(settings(), opener=lambda *_a, **_k: Response(structured(), model=None),
                           diagnostics_recorder=recorder)
        client.chat([], purpose="final_answer", thinking="enabled")
        self.assertEqual(recorder.snapshot()["provider_calls"][0]["request_model"], "deepseek-flash")
        self.assertNotIn("response_model", recorder.snapshot()["provider_calls"][0])


if __name__ == "__main__":
    unittest.main()
