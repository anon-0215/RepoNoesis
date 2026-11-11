import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "backend")]
from acceptance import AcceptanceBudget, AcceptanceMiddleware, install
from app.services import llm_client, provider_transport
from app.config import LLMSettings


def client():
    return llm_client.LLMClient(LLMSettings(provider="openai_compatible", base_url="https://provider.invalid/v1",
        api_key="test-only-placeholder", model="test-model", max_retries=0))


class AcceptanceTests(unittest.TestCase):
    def test_one_question_per_mode_and_two_total_survive_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ledger.json"
            first = AcceptanceBudget(path)
            self.assertTrue(first.question("rag"))
            self.assertFalse(first.question("rag"))
            resumed = AcceptanceBudget(path)
            self.assertTrue(resumed.question("agent"))
            self.assertFalse(resumed.question("agent"))
            self.assertEqual(len(resumed.value["questions"]), 2)

    def test_seventeenth_attempt_is_blocked_before_transport_and_usage_is_safe(self):
        with tempfile.TemporaryDirectory() as folder:
            calls = []
            def transport(*a, **kw):
                calls.append(1)
                return [{"data": {"usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                                  "choices": [{"message": {"content": "private answer"}}]}}]
            class App:
                def add_middleware(self, *a, **kw): pass
            with patch.object(llm_client.LLMClient, "_isolated_response", new=staticmethod(transport)), patch.dict(os.environ, {"LLM_BASE_URL": "https://provider.invalid/v1"}):
                budget = install(App(), Path(folder) / "ledger.json")
                real_client = client()
                for _ in range(16):
                    self.assertEqual(real_client.chat([{"role": "user", "content": "offline fixture"}]), "private answer")
                with self.assertRaises(llm_client.ProviderError):
                    real_client.chat([{"role": "user", "content": "offline fixture"}])
                self.assertEqual(len(calls), 16)
                self.assertEqual(budget.value["attempts_reserved"], 16)
                text = budget.path.read_text()
                self.assertNotIn("private answer", text)
                self.assertEqual(budget.value["attempts"][0]["usage"]["total_tokens"], 3)

    def test_network_failure_halts_all_further_attempts(self):
        with tempfile.TemporaryDirectory() as folder:
            calls = []
            def transport(*a, **kw):
                calls.append(1)
                return [{"error": "network"}]
            class App:
                def add_middleware(self, *a, **kw): pass
            with patch.object(llm_client.LLMClient, "_isolated_response", new=staticmethod(transport)), patch.dict(os.environ, {"LLM_BASE_URL": "https://provider.invalid/v1"}):
                budget = install(App(), Path(folder) / "ledger.json")
                real_client = client()
                with self.assertRaises(llm_client.ProviderError):
                    real_client.chat([{"role": "user", "content": "offline fixture"}])
                with self.assertRaises(llm_client.ProviderError):
                    real_client.chat([{"role": "user", "content": "offline fixture"}])
                self.assertEqual(len(calls), 1)
                self.assertTrue(budget.value["halted"])

    def test_asgi_guard_replays_body_once_and_rejects_duplicate_question(self):
        with tempfile.TemporaryDirectory() as folder:
            budget = AcceptanceBudget(Path(folder) / "ledger.json")
            received = []
            async def app(scope, receive, send):
                received.append((await receive())["body"])
                await send({"type": "http.response.start", "status": 200, "headers": []})
                await send({"type": "http.response.body", "body": b"ok"})
            middleware = AcceptanceMiddleware(app, budget)
            async def invoke(mode):
                sent = []
                async def receive(): return {"type": "http.request", "body": json.dumps({"execution_mode": mode}).encode(), "more_body": False}
                async def send(event): sent.append(event)
                await middleware({"type": "http", "method": "POST", "path": "/api/projects/project/ask/stream"}, receive, send)
                return sent[0]["status"]
            self.assertEqual(asyncio.run(invoke("rag")), 200)
            self.assertEqual(asyncio.run(invoke("rag")), 429)
            self.assertEqual(asyncio.run(invoke("agent")), 200)
            self.assertEqual(len(received), 2)


if __name__ == "__main__":
    unittest.main()
