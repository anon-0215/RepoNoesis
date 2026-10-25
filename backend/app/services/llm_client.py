from __future__ import annotations

import json
import hashlib
import re
import time
import uuid
from contextvars import ContextVar
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from app.config import LLMSettings, ThinkingMode, get_llm_settings
from app.services.smoke_diagnostics import ProviderPurpose, SmokeDiagnosticsRecorder
from app.services.provider_transport import isolated_response, TransportCollectionError


_CANCEL_CHECK: ContextVar[Callable[[], bool] | None] = ContextVar("provider_cancel_check", default=None)


class provider_cancellation_scope:
    """Request-local scope without generator exception traceback mutation.

    ProviderError is frozen. contextlib's generator context manager tries to
    assign __traceback__ when propagating it, which would mask the real error.
    """
    def __init__(self, cancelled: Callable[[], bool]):
        self.cancelled = cancelled

    def __enter__(self):
        self.token = _CANCEL_CHECK.set(self.cancelled)
        return self

    def __exit__(self, *_args):
        _CANCEL_CHECK.reset(self.token)
        return False


def raise_if_provider_cancelled():
    check = _CANCEL_CHECK.get()
    if check is not None and check():
        raise ProviderError("cancelled", "The request was cancelled locally; remote generation state is unknown.", status_code=499)


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
        monotonic: Callable[[], float] | None = None,
        diagnostics_recorder: SmokeDiagnosticsRecorder | None = None,
    ) -> None:
        self.settings = settings or get_llm_settings()
        self.base_url = self.settings.base_url
        self.api_key = self.settings.api_key
        self.model = self.settings.model
        self._opener = opener or urllib.request.urlopen
        self._isolated_transport = opener is None
        self._sleep = sleep or time.sleep
        self._monotonic = monotonic or time.monotonic
        self._diagnostics_recorder = diagnostics_recorder

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
        purpose: ProviderPurpose | None = None,
        diagnostics_recorder: SmokeDiagnosticsRecorder | None = None,
        deadline_monotonic: float | None = None,
    ) -> str | None:
        self.require_available()
        if thinking not in {None, "enabled", "disabled"}:
            raise ValueError("thinking must be enabled, disabled, or omitted")
        generation_limit = None
        reasoning_effort = None
        # Resolve per call; never mutate the shared client's default model.
        request_model = self.model
        if purpose == "planner":
            generation_limit = self.settings.planner_max_generation_tokens
            reasoning_effort = self.settings.planner_reasoning_effort
        elif purpose == "final_answer":
            request_model = (self.settings.answer_model or "").strip() or self.model
            generation_limit = self.settings.answer_max_generation_tokens
            reasoning_effort = self.settings.answer_reasoning_effort
        if generation_limit is not None and (
            type(generation_limit) is not int or not 1 <= generation_limit <= 65536
        ):
            raise ValueError("stage generation limit must be an integer from 1 to 65536")
        if reasoning_effort not in {None, "low", "high", "max"}:
            raise ValueError("reasoning_effort must be low, high, max, or omitted")
        if reasoning_effort is not None and thinking == "disabled":
            raise ValueError("reasoning_effort requires thinking enabled or omitted")
        payload = {
            "model": request_model,
            "messages": messages,
            "temperature": self.settings.temperature if temperature is None else temperature,
            # Provider generation tokens include hidden reasoning. Caller limits
            # still bound visible Planner/answer text in their own validators.
            "max_tokens": (
                generation_limit if generation_limit is not None
                else max(1, int(max_tokens or self.settings.max_tokens))
            ),
        }
        if thinking is not None:
            payload["thinking"] = {"type": thinking}
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        request = urllib.request.Request(
            _chat_completions_url(self.base_url),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        recorder = diagnostics_recorder or self._diagnostics_recorder
        provider_call_id = (
            recorder.start_provider_call(purpose)
            if recorder is not None and purpose is not None
            else None
        )
        self._record_provider_metadata(
            provider_call_id, {"request_model": request_model}, recorder=recorder
        )
        configured_timeout = min(
            self.settings.timeout_seconds,
            float(timeout_seconds or self.settings.timeout_seconds),
        )
        attempts = self.settings.max_retries + 1
        logical_call_id = uuid.uuid4().hex
        http_status: int | None = None
        for attempt in range(attempts):
            self._check_active(deadline_monotonic)
            remaining = self._remaining_seconds(deadline_monotonic)
            if remaining is not None and remaining <= 0:
                raise _deadline_error()
            timeout = configured_timeout
            if remaining is not None:
                timeout = min(timeout, remaining)
            if timeout <= 0:
                raise _deadline_error()
            attempt_started = self._monotonic()
            http_status = None  # A retry must not inherit the preceding status.
            attempt_audit = {"network_state": "not_started" if self._isolated_transport else "started",
                             "local_reason": "local_admission", "logical_call_id": logical_call_id,
                             "attempt_index": attempt + 1, "provider_call_id": provider_call_id,
                             "request_id": getattr(recorder, "_agent", {}).get("request_id", "UNKNOWN")}
            try:
                # urllib.request.urlopen's second positional argument is POST data,
                # not the timeout. Keep this keyword-only so request.data remains JSON.
                if self._isolated_transport:
                    records = self._isolated_response(request, timeout=timeout,
                        audit=attempt_audit,
                        check_active=lambda: self._check_active(deadline_monotonic),
                        on_headers=lambda status: self._record_provider_metadata(provider_call_id,
                            {"response_received": True, "http_status": status}, recorder=recorder))
                    data = None
                    for record in records:
                        if "http_status" in record:
                            http_status = record["http_status"]
                            self._record_provider_metadata(provider_call_id,
                                {"response_received": True, "http_status": http_status}, recorder=recorder)
                        if record.get("error") == "http":
                            raise urllib.error.HTTPError(request.full_url, record["status"], "", {}, None)
                        if record.get("error") == "timeout":
                            raise TimeoutError()
                        if record.get("error") == "network":
                            raise urllib.error.URLError("Provider transport failed")
                        if record.get("error") == "audit":
                            raise TransportCollectionError("Worker transport audit denied a destination")
                        if record.get("error") == "invalid_json":
                            raise ValueError("Invalid provider JSON")
                        if "data" in record:
                            data = record["data"]
                else:
                    # Explicit injected openers are a synchronous test seam;
                    # they must cooperate with their timeout. Production uses
                    # isolated_response, including during blocking body reads.
                    with self._opener(request, timeout=timeout) as response:
                        http_status = _response_status(response)
                        self._record_provider_metadata(provider_call_id,
                            {"response_received": True, "http_status": http_status}, recorder=recorder)
                        data = json.loads(response.read().decode("utf-8"))
                self._check_active(deadline_monotonic)
                content = _parse_chat_content(
                    data,
                    http_status=http_status,
                    provider=self.provider_name,
                    model=request_model,
                    diagnostics_callback=lambda metadata: self._record_provider_metadata(
                        provider_call_id, metadata, recorder=recorder
                    ),
                )
                self._check_active(deadline_monotonic)
                self._record_provider_attempt(
                    provider_call_id,
                    outcome="success",
                    started=attempt_started,
                    timeout=timeout,
                    recorder=recorder,
                    transport=attempt_audit,
                )
                return content
            except TransportCollectionError:
                self._record_provider_attempt(provider_call_id, outcome="collection_error",
                    started=attempt_started, timeout=timeout, recorder=recorder, transport=attempt_audit)
                raise  # Collection failure is explicit and never retried.
            except urllib.error.HTTPError as exc:
                self._record_provider_metadata(
                    provider_call_id,
                    {"response_received": True, "http_status": exc.code},
                    recorder=recorder,
                )
                error = _http_error(exc.code)
                exc.close()
                outcome = "http_error"
            except TimeoutError:
                error = ProviderError(
                    "provider_unavailable",
                    "The generation provider could not be reached before the timeout.",
                    retryable=True,
                    status_code=503,
                )
                outcome = "timeout"
            except (urllib.error.URLError, OSError):
                error = ProviderError(
                    "provider_unavailable",
                    "The generation provider could not be reached before the timeout.",
                    retryable=True,
                    status_code=503,
                )
                outcome = "network_error"
            except (UnicodeDecodeError, ValueError, TypeError) as exc:
                self._record_provider_metadata(
                    provider_call_id,
                    {"response_json_valid": False},
                    recorder=recorder,
                )
                self._record_provider_attempt(
                    provider_call_id,
                    outcome="invalid_response",
                    started=attempt_started,
                    timeout=timeout,
                    recorder=recorder,
                    transport=attempt_audit,
                )
                raise _invalid_response_error(
                    {
                        "provider": self.provider_name,
                        "model": request_model,
                        "http_status": http_status,
                        "response_json_valid": False,
                    }
                ) from exc
            except ProviderError as exc:
                self._record_provider_attempt(
                    provider_call_id,
                    outcome="timeout" if exc.code == "deadline_exceeded" else "cancelled" if exc.code == "cancelled" else "invalid_response",
                    started=attempt_started,
                    timeout=timeout,
                    recorder=recorder,
                    transport=attempt_audit,
                )
                raise
            error = ProviderError(
                error.code, error.message, retryable=error.retryable,
                status_code=error.status_code,
                diagnostics={**(error.diagnostics or {}),
                             "provider": self.provider_name, "model": request_model},
            )
            self._record_provider_attempt(
                provider_call_id,
                outcome=outcome,
                started=attempt_started,
                timeout=timeout,
                recorder=recorder,
                transport=attempt_audit,
            )
            self._check_active(deadline_monotonic)
            remaining = self._remaining_seconds(deadline_monotonic)
            if remaining is not None and remaining <= 0:
                raise _deadline_error()
            retry_allowed = error.retryable
            if self.settings.retry_transient_only:
                status = (error.diagnostics or {}).get("http_status")
                retry_allowed = retry_allowed and outcome == "http_error" and (
                    status == 429 or isinstance(status, int) and 500 <= status <= 599
                )
            if not retry_allowed or attempt == attempts - 1:
                raise error
            backoff = min(0.25 * (2**attempt), 1.0)
            remaining = self._remaining_seconds(deadline_monotonic)
            if remaining is not None:
                if remaining <= 0 or remaining < backoff:
                    raise _deadline_error()
            backoff_started = self._monotonic()
            if self._sleep is time.sleep:
                backoff_end = self._monotonic() + backoff
                while self._monotonic() < backoff_end:
                    self._check_active(deadline_monotonic)
                    self._sleep(min(0.02, max(0, backoff_end - self._monotonic())))
            else:
                self._sleep(backoff)
            if recorder is not None:
                recorder.record_backoff(
                    max(0, int((self._monotonic() - backoff_started) * 1000))
                )
            remaining = self._remaining_seconds(deadline_monotonic)
            if remaining is not None and remaining <= 0:
                raise _deadline_error()
        raise ProviderError("provider_unavailable", "The generation provider is unavailable.")

    _isolated_response = staticmethod(isolated_response)

    def _check_active(self, deadline_monotonic):
        raise_if_provider_cancelled()
        remaining = self._remaining_seconds(deadline_monotonic)
        if remaining is not None and remaining <= 0:
            raise _deadline_error()

    def _remaining_seconds(self, deadline_monotonic: float | None) -> float | None:
        if deadline_monotonic is None:
            return None
        return deadline_monotonic - self._monotonic()

    def _record_provider_attempt(
        self,
        call_id: int | None,
        *,
        outcome: str,
        started: float,
        timeout: float,
        recorder: SmokeDiagnosticsRecorder | None,
        transport: dict | None = None,
    ) -> None:
        if recorder is not None:
            recorder.record_provider_attempt(
                call_id,
                outcome=outcome,
                duration_ms=max(0, int((self._monotonic() - started) * 1000)),
                timeout_ms=max(0, int(timeout * 1000)),
                network_state=(transport or {}).get("network_state", "started"),
                local_reason=(transport or {}).get("local_reason"),
            )

    def _record_provider_metadata(
        self,
        call_id: int | None,
        metadata: dict[str, Any],
        *,
        recorder: SmokeDiagnosticsRecorder | None = None,
    ) -> None:
        target = recorder or self._diagnostics_recorder
        if target is not None:
            target.record_provider_response(call_id, metadata)


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _deadline_error() -> ProviderError:
    return ProviderError(
        "deadline_exceeded",
        "The local provider stage deadline was exhausted; remote generation state and unreceived usage are unknown.",
        retryable=False,
        status_code=504,
    )


def _parse_chat_content(
    data: Any,
    *,
    http_status: int | None = None,
    provider: str | None = None,
    model: str | None = None,
    diagnostics_callback: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    diagnostics: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "http_status": http_status,
        "response_json_valid": True,
        "top_level_type": _json_type(data),
    }
    try:
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
        diagnostics["response_model"] = _safe_response_model(data.get("model"))
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
        content_empty = content is None or (
            isinstance(content, str) and not content.strip()
        )
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
        if isinstance(content, str):
            diagnostics.update(
                {
                    "output_chars": len(content),
                    "output_sha256": hashlib.sha256(
                        content.encode("utf-8")
                    ).hexdigest(),
                    "markdown_fence_detected": "```" in content,
                }
            )
        if not content_valid_type or (reasoning_present and not reasoning_valid_type):
            raise _invalid_response_error(diagnostics)
        if finish_reason == "length":
            raise ProviderError(
                "provider_output_truncated",
                "The generation provider reached the configured output limit before returning final content.",
                retryable=False,
                diagnostics=_compact_diagnostics(diagnostics),
            )
        if finish_reason not in {None, "stop"}:
            raise _invalid_response_error(diagnostics)
        if isinstance(content, str) and content.strip():
            if diagnostics_callback is not None:
                diagnostics_callback(_compact_diagnostics(diagnostics))
            return content.strip()
        if finish_reason == "stop":
            raise ProviderError(
                "provider_empty_content",
                "The generation provider completed without final answer content.",
                retryable=False,
                diagnostics=_compact_diagnostics(diagnostics),
            )
        raise _invalid_response_error(diagnostics)
    except ProviderError:
        if diagnostics_callback is not None:
            diagnostics_callback(_compact_diagnostics(diagnostics))
        raise


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
        "output_chars",
        "output_sha256",
        "markdown_fence_detected",
    }
    result = {
        key: item
        for key, item in value.items()
        if key in allowed and item is not None and isinstance(item, (str, int, float, bool))
    }
    usage = _safe_usage(value.get("usage"))
    if usage:
        result["usage"] = usage
    response_model = _safe_response_model(value.get("response_model"))
    if response_model is not None:
        result["response_model"] = response_model
    output_sha256 = result.get("output_sha256")
    if not isinstance(output_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", output_sha256
    ) is None:
        result.pop("output_sha256", None)
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


def _safe_response_model(value: Any) -> str | None:
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9._:/-]{1,128}", value):
        return value
    return None


def _safe_usage(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for key in (
        "prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens",
        "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
    ):
        item = value.get(key)
        if type(item) is int and 0 <= item <= 1_000_000_000:
            result[key] = item
    details = value.get("completion_tokens_details")
    if isinstance(details, dict):
        reasoning_tokens = details.get("reasoning_tokens")
        if type(reasoning_tokens) is int and 0 <= reasoning_tokens <= 1_000_000_000:
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
