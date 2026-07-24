from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


class LLMClient:
    def __init__(self) -> None:
        self.base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com").rstrip("/")
        self.api_key = os.getenv("LLM_API_KEY", "")
        self.model = os.getenv("LLM_MODEL", "deepseek-chat")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        *,
        max_tokens: int | None = None,
        timeout_seconds: float = 45,
    ) -> str | None:
        if not self.available:
            return None
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max(1, int(max_tokens))
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=max(0.1, min(45.0, float(timeout_seconds))),
            ) as response:
                data: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return None
        choices = data.get("choices") or []
        if not choices:
            return None
        return choices[0].get("message", {}).get("content")

