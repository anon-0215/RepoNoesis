from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from app.config import LLMSettings, get_llm_settings


@dataclass(frozen=True)
class ProviderError(RuntimeError):
    code: str
    message: str
    retryable: bool = False
    status_code: int = 502

    def __str__(self) -> str:
        return self.message

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }


class LLMClient:
    """Product OpenAI-compatible Chat Completions client.

    Provider credentials remain private attributes and are never included in
    status output, exceptions, or response payloads.
    """

    provider_name = "openai_compatible"

    def __init__(
        self,
        settings: LLMSettings | None = None,
        *,
        opener: Callable[..., Any] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.settings = settings or get_llm_settings()
        self.base_url = self.settings.base_url
        self.api_key = self.settings.api_key
        self.model = self.settings.model
        self._opener = opener or urllib.request.urlopen
        self._sleep = sleep or time.sleep

    @property
    def available(self) -> bool:
        return self.settings.configured

    def require_available(self) -> None:
        if self.settings.provider and self.settings.provider != self.provider_name:
            raise ProviderError(
                "unsupported_provider",
                "LLM_PROVIDER must be openai_compatible for the local product path.",
                status_code=503,
            )
        if not self.available:
            raise ProviderError(
                "provider_not_configured",
                "The generation provider is not configured. Complete the backend .env settings and restart the backend.",
                status_code=503,
            )

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        *,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> str | None:
        self.require_available()
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.settings.temperature if temperature is None else temperature,
            "max_tokens": max(1, int(max_tokens or self.settings.max_tokens)),
        }
        request = urllib.request.Request(
            _chat_completions_url(self.base_url),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        timeout = max(
            0.1,
            min(
                self.settings.timeout_seconds,
                float(timeout_seconds or self.settings.timeout_seconds),
            ),
        )
        attempts = self.settings.max_retries + 1
        for attempt in range(attempts):
            try:
                with self._opener(request, timeout) as response:
                    data: dict[str, Any] = json.loads(response.read().decode("utf-8"))
                return _parse_chat_content(data)
            except urllib.error.HTTPError as exc:
                error = _http_error(exc.code)
            except (TimeoutError, urllib.error.URLError):
                error = ProviderError(
                    "provider_unavailable",
                    "The generation provider could not be reached before the timeout.",
                    retryable=True,
                    status_code=503,
                )
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
                raise ProviderError(
                    "provider_invalid_response",
                    "The generation provider returned an invalid Chat Completions response.",
                ) from exc
            if not error.retryable or attempt == attempts - 1:
                raise error
            self._sleep(min(0.25 * (2**attempt), 1.0))
        raise ProviderError("provider_unavailable", "The generation provider is unavailable.")


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _parse_chat_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("missing choices")
    first = choices[0]
    if not isinstance(first, dict) or not isinstance(first.get("message"), dict):
        raise ValueError("missing message")
    content = first["message"].get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("empty content")
    return content.strip()


def _http_error(status: int) -> ProviderError:
    if status in {401, 403}:
        return ProviderError(
            "provider_authentication_failed",
            "The generation provider rejected the configured credentials.",
            status_code=502,
        )
    if status == 429:
        return ProviderError(
            "provider_rate_limited",
            "The generation provider rate limit was reached.",
            retryable=True,
            status_code=503,
        )
    if status >= 500:
        return ProviderError(
            "provider_unavailable",
            "The generation provider is temporarily unavailable.",
            retryable=True,
            status_code=503,
        )
    return ProviderError(
        "provider_request_rejected",
        "The generation provider rejected the Chat Completions request.",
        status_code=502,
    )
