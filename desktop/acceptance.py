"""Opt-in two-question/16-attempt acceptance guard; absent from normal startup."""
import json
from pathlib import Path
import threading
import time
import uuid
from urllib.parse import urlsplit
from gui_settings import atomic_json


class AcceptanceBudget:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.value = json.loads(self.path.read_text()) if self.path.exists() else {
            "questions": [], "attempts_reserved": 0, "attempts": [], "halted": False}
        self._write()

    def _write(self):
        atomic_json(self.path, self.value)

    def question(self, mode):
        with self.lock:
            if self.value["halted"] or len(self.value["questions"]) >= 2 or any(q["mode"] == mode for q in self.value["questions"]):
                return False
            self.value["questions"].append({"mode": mode, "at": time.time()})
            self._write()
            return True

    def reserve(self):
        from app.services.llm_client import ProviderError
        with self.lock:
            if self.value["halted"] or self.value["attempts_reserved"] >= 16:
                raise ProviderError("desktop_acceptance_paused", "本轮真实调用已暂停或达到 16 次尝试上限。", status_code=503)
            self.value["attempts_reserved"] += 1
            attempt = {"number": self.value["attempts_reserved"], "at": time.time(),
                       "network_started": False, "usage": None}
            self.value["attempts"].append(attempt)
            self._write()
            return attempt

    def update(self, attempt, **values):
        with self.lock:
            attempt.update(values)
            self._write()

    def halt(self, reason):
        with self.lock:
            self.value.update(halted=True, halt_reason=reason)
            self._write()


class AcceptanceMiddleware:
    def __init__(self, app, budget):
        self.app, self.budget = app, budget

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        if scope["type"] != "http" or scope.get("method") != "POST" or not (path.endswith("/ask") or path.endswith("/ask/stream")):
            return await self.app(scope, receive, send)
        messages = []
        body = b""
        while True:
            event = await receive()
            messages.append(event)
            body += event.get("body", b"")
            if not event.get("more_body"):
                break
        try:
            mode = json.loads(body).get("execution_mode", "agent")
        except ValueError:
            mode = None
        if mode in {"rag", "agent"} and not self.budget.question(mode):
            from starlette.responses import JSONResponse
            return await JSONResponse({"detail": {"code": "desktop_acceptance_paused",
                "message": "本轮仅允许 RAG、Agent 各一次；后续真实调用已暂停。", "retryable": False}}, status_code=429)(scope, receive, send)
        async def replay():
            return messages.pop(0) if messages else await receive()
        await self.app(scope, replay, send)


def install(app, path):
    from app.config import get_llm_settings
    from app.services import llm_client, provider_transport
    budget = AcceptanceBudget(path)
    original = llm_client.LLMClient._isolated_response
    base = urlsplit(get_llm_settings().base_url)
    origin = [base.scheme, base.hostname, base.port or (443 if base.scheme == "https" else 80)]
    policy = {"allowed_origins": [origin]}

    def bounded(*args, **kwargs):
        attempt = budget.reserve()
        def observe(event):
            name = event.get("event")
            if name == "network_started":
                budget.update(attempt, network_started=True)
            elif name == "response_headers":
                status = event.get("http_status")
                budget.update(attempt, http_status=status)
                if status in {401, 402, 403, 429}:
                    budget.halt("authentication_or_quota")
        try:
            with provider_transport.transport_audit_scope(observe, policy=policy):
                records = original(*args, **kwargs)
            for record in records:
                if record.get("error") in {"network", "timeout", "audit"}:
                    budget.halt("network_or_timeout")
                data = record.get("data")
                if isinstance(data, dict):
                    usage = llm_client._safe_usage(data.get("usage"))
                    if usage:
                        budget.update(attempt, usage=usage)
            return records
        except (OSError, TimeoutError):
            budget.halt("network_or_timeout")
            raise
        finally:
            budget.update(attempt, finished_at=time.time())
    llm_client.LLMClient._isolated_response = staticmethod(bounded)
    app.add_middleware(AcceptanceMiddleware, budget=budget)
    return budget
