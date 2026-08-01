from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from app.config import LLMSettings, ThinkingMode, get_llm_settings


@dataclass(frozen=True)
class ProviderError(RuntimeError):
    code: str
    message: str
    retryable: bool = False
    status_code: int = 502
    diagnostics: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "diagnostics",
            _compact_diagnostics(self.diagnostics or {}) or None,
        )

    def __str__(self) -> str:
        return self.message

    def to_safe_dict(self) -> dict[str, Any]:
        result = {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.diagnostics:
            result["diagnostics"] = self.diagnostics
        return result


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
        thinking: ThinkingMode | None = None,
    ) -> str | None:
        self.require_available()
        if thinking not in {None, "enabled", "disabled"}:
            raise ValueError("thinking must be enabled, disabled, or omitted")
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.settings.temperature if temperature is None else temperature,
            "max_tokens": max(1, int(max_tokens or self.settings.max_tokens)),
        }
        if thinking is not None:
            payload["thinking"] = {"type": thinking}
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
        http_status: int | None = None
        for attempt in range(attempts):
            try:
                # urllib.request.urlopen's second positional argument is POST data,
                # not the timeout. Keep this keyword-only so request.data remains JSON.
                with self._opener(request, timeout=timeout) as response:
                    http_status = _response_status(response)
                    data: Any = json.loads(response.read().decode("utf-8"))
                return _parse_chat_content(
                    data,
                    http_status=http_status,
                    provider=self.provider_name,
                    model=self.model,
                )
            except urllib.error.HTTPError as exc:
                error = _http_error(exc.code)
            except (TimeoutError, urllib.error.URLError):
                error = ProviderError(
                    "provider_unavailable",
                    "The generation provider could not be reached before the timeout.",
                    retryable=True,
                    status_code=503,
                )
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
                raise _invalid_response_error(
                    {
                        "provider": self.provider_name,
                        "model": self.model,
                        "http_status": http_status,
                        "response_json_valid": False,
                    }
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


def _parse_chat_content(
    data: Any,
    *,
    http_status: int | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> str:
    diagnostics: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "http_status": http_status,
        "top_level_type": _json_type(data),
    }
    if not isinstance(data, dict):
        raise _invalid_response_error(diagnostics)

    choices_present = "choices" in data
    choices = data.get("choices")
    diagnostics.update(
        {
            "choices_present": choices_present,
            "choices_type": _json_type(choices),
            "choices_count": len(choices) if isinstance(choices, list) else None,
        }
    )
    diagnostics["usage"] = _safe_usage(data.get("usage"))
    if not isinstance(choices, list) or not choices:
        raise _invalid_response_error(diagnostics)

    first = choices[0]
    diagnostics["choice_0_type"] = _json_type(first)
    if not isinstance(first, dict):
        raise _invalid_response_error(diagnostics)

    finish_reason_present = "finish_reason" in first
    finish_reason = first.get("finish_reason")
    diagnostics.update(
        {
            "finish_reason_present": finish_reason_present,
            "finish_reason_type": _json_type(finish_reason),
        }
    )
    if isinstance(finish_reason, str) or finish_reason is None:
        diagnostics["finish_reason"] = (
            finish_reason
            if finish_reason
            in {"stop", "length", "tool_calls", "content_filter", "function_call"}
            else "other"
            if finish_reason is not None
            else None
        )
    else:
        raise _invalid_response_error(diagnostics)

    message_present = "message" in first
    message = first.get("message")
    diagnostics.update(
        {
            "message_present": message_present,
            "message_type": _json_type(message),
        }
    )
    if not isinstance(message, dict):
        raise _invalid_response_error(diagnostics)

    content_present = "content" in message
    content = message.get("content")
    content_valid_type = isinstance(content, str) or content is None
    content_empty = content is None or (isinstance(content, str) and not content.strip())
    reasoning_present = "reasoning_content" in message
    reasoning = message.get("reasoning_content")
    reasoning_valid_type = isinstance(reasoning, str) or reasoning is None
    diagnostics.update(
        {
            "content_present": content_present,
            "content_type": _json_type(content),
            "content_empty": content_empty,
            "reasoning_content_present": reasoning_present,
            "reasoning_content_type": _json_type(reasoning),
        }
    )
    if not content_valid_type or (reasoning_present and not reasoning_valid_type):
        raise _invalid_response_error(diagnostics)
    if isinstance(content, str) and content.strip():
        return content.strip()
    if finish_reason == "length":
        raise ProviderError(
            "provider_output_truncated",
            "The generation provider reached the configured output limit before returning final content.",
            retryable=False,
            diagnostics=_compact_diagnostics(diagnostics),
        )
    if finish_reason == "stop":
        raise ProviderError(
            "provider_empty_content",
            "The generation provider completed without final answer content.",
            retryable=False,
            diagnostics=_compact_diagnostics(diagnostics),
        )
    raise _invalid_response_error(diagnostics)


def _invalid_response_error(diagnostics: dict[str, Any]) -> ProviderError:
    return ProviderError(
        "provider_invalid_response",
        "The generation provider returned an invalid Chat Completions response.",
        diagnostics=_compact_diagnostics(diagnostics),
    )


def _compact_diagnostics(value: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "provider",
        "model",
        "http_status",
        "response_json_valid",
        "top_level_type",
        "choices_present",
        "choices_type",
        "choices_count",
        "choice_0_type",
        "finish_reason_present",
        "finish_reason_type",
        "finish_reason",
        "message_present",
        "message_type",
        "content_present",
        "content_type",
        "content_empty",
        "reasoning_content_present",
        "reasoning_content_type",
    }
    result = {
        key: item
        for key, item in value.items()
        if key in allowed and item is not None and isinstance(item, (str, int, float, bool))
    }
    usage = _safe_usage(value.get("usage"))
    if usage:
        result["usage"] = usage
    return result


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "other"


def _safe_usage(value: Any) -> dict[str, int | float]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int | float] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        item = value.get(key)
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            result[key] = item
    details = value.get("completion_tokens_details")
    if isinstance(details, dict):
        reasoning_tokens = details.get("reasoning_tokens")
        if isinstance(reasoning_tokens, (int, float)) and not isinstance(
            reasoning_tokens, bool
        ):
            result["reasoning_tokens"] = reasoning_tokens
    return result


def _response_status(response: Any) -> int | None:
    status = getattr(response, "status", None)
    return status if isinstance(status, int) and not isinstance(status, bool) else None


def _http_error(status: int) -> ProviderError:
    if status in {401, 403}:
        return ProviderError(
            "provider_authentication_failed",
            "The generation provider rejected the configured credentials.",
            status_code=502,
            diagnostics={"http_status": status},
        )
    if status == 429:
        return ProviderError(
            "provider_rate_limited",
            "The generation provider rate limit was reached.",
            retryable=True,
            status_code=503,
            diagnostics={"http_status": status},
        )
    if status >= 500:
        return ProviderError(
            "provider_unavailable",
            "The generation provider is temporarily unavailable.",
            retryable=True,
            status_code=503,
            diagnostics={"http_status": status},
        )
    return ProviderError(
        "provider_request_rejected",
        "The generation provider rejected the Chat Completions request.",
        status_code=502,
        diagnostics={"http_status": status},
    )
