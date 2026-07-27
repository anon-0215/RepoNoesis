from __future__ import annotations

import json
import hashlib
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import ValidationError

from app.services.agent_contracts import CancellationToken
from app.services.learning_contracts import EvaluationOutput
from app.m5.contracts import (
    ProviderIdentity,
    ProviderResult,
    ProviderUsage,
)


MAX_PROVIDER_RESPONSE_BYTES = 1_000_000
RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}
SECRET_KEYS = ("api_key", "authorization", "password", "secret")


class Provider(Protocol):
    @property
    def identity(self) -> ProviderIdentity:
        ...

    def invoke(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_output_tokens: int,
        timeout_seconds: float,
        maximum_attempts: int,
        cancellation: CancellationToken | None = None,
        seed: int | None = None,
    ) -> ProviderResult:
        ...

    def smoke_check(self) -> ProviderResult:
        ...


class ProviderConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class PricingConfig:
    model: str
    currency: str
    input_price_per_unit: float
    output_price_per_unit: float
    unit_tokens: int
    source: str

    def validate(self) -> None:
        if not self.model.strip():
            raise ProviderConfigurationError("pricing model registration must be explicit")
        if self.currency != "USD":
            raise ProviderConfigurationError("M5 currently records provider cost in USD")
        if self.input_price_per_unit < 0 or self.output_price_per_unit < 0:
            raise ProviderConfigurationError("pricing values must be non-negative")
        if self.unit_tokens <= 0 or not self.source.strip():
            raise ProviderConfigurationError("pricing unit and source must be explicit")


@dataclass(frozen=True)
class OpenAICompatibleSettings:
    base_url: str
    api_key: str
    model: str
    model_revision: str = "provider-managed"
    allow_network: bool = False
    allow_real_llm: bool = False
    allow_paid_eval: bool = False
    maximum_response_bytes: int = MAX_PROVIDER_RESPONSE_BYTES
    pricing: PricingConfig | None = None

    def validate(self, *, evaluator: bool = False) -> None:
        if not self.allow_network or not self.allow_real_llm:
            raise ProviderConfigurationError("real LLM calls require explicit M5 opt-in")
        if evaluator and not self.allow_paid_eval:
            raise ProviderConfigurationError("real evaluator calls require explicit paid-eval opt-in")
        if not self.api_key:
            raise ProviderConfigurationError("LLM API key is not configured")
        if not self.base_url.startswith(("https://", "http://127.0.0.1", "http://localhost")):
            raise ProviderConfigurationError("provider endpoint must use HTTPS or loopback HTTP")
        if not self.model.strip():
            raise ProviderConfigurationError("provider model is not configured")
        if not self.model_revision.strip() or self.model_revision == "provider-managed":
            raise ProviderConfigurationError("provider model revision/identity must be explicit")
        if self.pricing is not None:
            self.pricing.validate()
            if self.pricing.model != self.model:
                raise ProviderConfigurationError("pricing model does not match provider model")

    def safe_dict(self) -> dict[str, Any]:
        return {
            "endpoint_identity": normalized_endpoint_identity(self.base_url),
            "api_key": "[REDACTED]" if self.api_key else "",
            "model": self.model,
            "model_revision": self.model_revision,
            "allow_network": self.allow_network,
            "allow_real_llm": self.allow_real_llm,
            "allow_paid_eval": self.allow_paid_eval,
            "maximum_response_bytes": self.maximum_response_bytes,
            "pricing": None if self.pricing is None else {
                "currency": self.pricing.currency,
                "model": self.pricing.model,
                "input_price_per_unit": self.pricing.input_price_per_unit,
                "output_price_per_unit": self.pricing.output_price_per_unit,
                "unit_tokens": self.pricing.unit_tokens,
                "source": self.pricing.source,
            },
        }


class OpenAICompatibleProvider:
    """Bounded OpenAI-compatible provider used only after explicit M5 gates."""

    def __init__(self, settings: OpenAICompatibleSettings, *, evaluator: bool = False) -> None:
        settings.validate(evaluator=evaluator)
        self.settings = settings
        self.evaluator = evaluator
        self._identity = ProviderIdentity(
            provider="openai-compatible",
            model=settings.model,
            model_revision=settings.model_revision,
            capability="structured_evaluator" if evaluator else "answer_generation",
            is_real=True,
            endpoint_identity=normalized_endpoint_identity(settings.base_url),
            pricing_identity=_pricing_identity(settings.pricing),
        )

    @property
    def identity(self) -> ProviderIdentity:
        return self._identity

    def invoke(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_output_tokens: int,
        timeout_seconds: float,
        maximum_attempts: int,
        cancellation: CancellationToken | None = None,
        seed: int | None = None,
    ) -> ProviderResult:
        started = time.monotonic()
        attempts = max(1, min(3, int(maximum_attempts)))
        timeout = max(0.1, min(60.0, float(timeout_seconds)))
        output_limit = max(1, min(4_096, int(max_output_tokens)))
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": max(0.0, min(1.0, float(temperature))),
            "max_tokens": output_limit,
        }
        if seed is not None:
            payload["seed"] = int(seed)
        last_error = "provider_error"
        for attempt in range(1, attempts + 1):
            if cancellation and cancellation.cancelled:
                return self._failure("cancelled", "cancelled", started, attempt)
            request = urllib.request.Request(
                f"{self.settings.base_url.rstrip('/')}/v1/chat/completions",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.settings.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    raw = response.read(self.settings.maximum_response_bytes + 1)
                if len(raw) > self.settings.maximum_response_bytes:
                    return self._failure("failed", "oversized_response", started, attempt)
                decoded = json.loads(raw.decode("utf-8"))
                return self._parse_response(decoded, started, attempt, seed is not None)
            except urllib.error.HTTPError as exc:
                last_error = "rate_limited" if exc.code == 429 else (
                    "server_error" if 500 <= exc.code <= 599 else "http_error"
                )
                if exc.code not in RETRYABLE_HTTP_CODES or attempt == attempts:
                    break
            except (TimeoutError, urllib.error.URLError) as exc:
                last_error = "timeout" if isinstance(exc, TimeoutError) else "connection_error"
                if attempt == attempts:
                    break
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
                last_error = "invalid_json"
                break
        status = "timed_out" if last_error == "timeout" else "failed"
        return self._failure(status, last_error, started, attempts)

    def smoke_check(self) -> ProviderResult:
        return self.invoke(
            [{"role": "user", "content": "Return exactly: ok"}],
            temperature=0.0,
            max_output_tokens=8,
            timeout_seconds=10.0,
            maximum_attempts=1,
            seed=20260726,
        )

    def _parse_response(
        self,
        data: Any,
        started: float,
        attempt: int,
        seed_requested: bool,
    ) -> ProviderResult:
        if not isinstance(data, dict):
            return self._failure("failed", "invalid_json", started, attempt)
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return self._failure("failed", "invalid_json", started, attempt)
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            return self._failure("failed", "invalid_json", started, attempt)
        usage_data = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        input_tokens = _optional_nonnegative_int(usage_data.get("prompt_tokens"))
        output_tokens = _optional_nonnegative_int(usage_data.get("completion_tokens"))
        total_tokens = _optional_nonnegative_int(usage_data.get("total_tokens"))
        usage = _priced_usage(input_tokens, output_tokens, total_tokens, self.settings.pricing)
        actual_model = str(data.get("model") or self.settings.model)
        return ProviderResult(
            status="succeeded",
            content=content,
            identity=self.identity,
            usage=usage,
            latency_ms=int((time.monotonic() - started) * 1000),
            actual_model=actual_model,
            attempt_count=attempt,
            seed_supported=None if seed_requested else False,
            raw_metadata={"request_id": str(data.get("id", ""))[:120]},
        )

    def _failure(
        self,
        status: str,
        error_type: str,
        started: float,
        attempt: int,
    ) -> ProviderResult:
        return ProviderResult(
            status=status,  # type: ignore[arg-type]
            content=None,
            identity=self.identity,
            usage=ProviderUsage(cost_unknown_reason="usage_unavailable_for_failed_request"),
            latency_ms=int((time.monotonic() - started) * 1000),
            actual_model=None,
            error_type=error_type,
            attempt_count=attempt,
        )


class FakeDeterministicProvider:
    def __init__(self, capability: str = "answer_generation", model: str = "fake-m5-v1") -> None:
        self._identity = ProviderIdentity(
            provider="fake-deterministic",
            model=model,
            model_revision="fixture-v1",
            capability=capability,
            is_real=False,
            endpoint_identity="local:fake",
            pricing_identity={"status": "known_zero", "source": "deterministic fixture"},
        )
        self.call_count = 0

    @property
    def identity(self) -> ProviderIdentity:
        return self._identity

    def invoke(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_output_tokens: int,
        timeout_seconds: float,
        maximum_attempts: int,
        cancellation: CancellationToken | None = None,
        seed: int | None = None,
    ) -> ProviderResult:
        started = time.monotonic()
        self.call_count += 1
        if cancellation and cancellation.cancelled:
            return ProviderResult(
                status="cancelled",
                content=None,
                identity=self.identity,
                usage=ProviderUsage(cost_unknown_reason="cancelled_before_usage"),
                latency_ms=0,
                error_type="cancelled",
            )
        joined = "\n".join(str(item.get("content", "")) for item in messages)
        if self.identity.capability == "structured_evaluator":
            content = _fake_evaluation(joined)
        else:
            evidence_ids = list(dict.fromkeys(re.findall(r"\bE\d+\b", joined)))
            reference = evidence_ids[0] if evidence_ids else "E1"
            content = f"基于已校验源码证据 [{reference}]，回答仅陈述可由该证据支持的事实。"
        content = content[: max(1, max_output_tokens * 4)]
        input_tokens = max(1, (len(joined) + 3) // 4)
        output_tokens = max(1, (len(content) + 3) // 4)
        return ProviderResult(
            status="succeeded",
            content=content,
            identity=self.identity,
            usage=ProviderUsage(
                input_tokens, output_tokens, input_tokens + output_tokens, 0.0,
                "known_zero", "USD", None,
            ),
            latency_ms=int((time.monotonic() - started) * 1000),
            actual_model=self.identity.model,
            attempt_count=1,
            seed_supported=True,
        )

    def smoke_check(self) -> ProviderResult:
        return self.invoke(
            [{"role": "user", "content": "smoke"}],
            temperature=0.0,
            max_output_tokens=8,
            timeout_seconds=1.0,
            maximum_attempts=1,
            seed=1,
        )


class BudgetedProvider:
    """Hard provider-call and token ledger shared by planner, answer, and evaluator."""

    def __init__(
        self,
        provider: Provider,
        *,
        maximum_requests: int | None = None,
        maximum_calls: int | None = None,
        maximum_input_tokens: int,
        maximum_output_tokens: int,
        maximum_cost_usd: float | None = None,
        maximum_wall_clock_seconds: float = 3_600.0,
    ) -> None:
        self.provider = provider
        self.maximum_requests = maximum_requests if maximum_requests is not None else maximum_calls
        if self.maximum_requests is None:
            raise ValueError("maximum_requests is required")
        self.maximum_input_tokens = maximum_input_tokens
        self.maximum_output_tokens = maximum_output_tokens
        self.maximum_cost_usd = maximum_cost_usd
        self.maximum_wall_clock_seconds = maximum_wall_clock_seconds
        self.request_count = 0
        self.logical_call_count = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.retry_count = 0
        self.usage_complete = True
        self.cost_complete = True
        self.cost_unknown_reasons: set[str] = set()
        self.stop_reason: str | None = None
        self.started_monotonic = time.monotonic()

    @property
    def call_count(self) -> int:
        return self.request_count

    @property
    def identity(self) -> ProviderIdentity:
        return self.provider.identity

    def invoke(self, messages: list[dict[str, str]], **kwargs: Any) -> ProviderResult:
        reason = self._limit_reason()
        if reason is not None:
            self.stop_reason = reason
            return self._exhausted(reason)
        remaining = self.maximum_requests - self.request_count
        kwargs["maximum_attempts"] = min(int(kwargs.get("maximum_attempts", 1)), remaining)
        self.logical_call_count += 1
        result = self.provider.invoke(messages, **kwargs)
        attempts = max(1, result.attempt_count)
        self.request_count += attempts
        self.retry_count += max(0, attempts - 1)
        if result.usage.input_tokens is None or result.usage.output_tokens is None:
            self.usage_complete = False
        else:
            self.input_tokens += result.usage.input_tokens
            self.output_tokens += result.usage.output_tokens
        if result.usage.estimated_cost_usd is None:
            self.cost_complete = False
            self.cost_unknown_reasons.add(result.usage.cost_unknown_reason or "unspecified")
        else:
            self.cost_usd += result.usage.estimated_cost_usd
        self.stop_reason = self._limit_reason()
        return result

    def _limit_reason(self) -> str | None:
        if self.request_count >= self.maximum_requests:
            return "request_budget_exhausted"
        if self.input_tokens >= self.maximum_input_tokens:
            return "input_token_budget_exhausted"
        if self.output_tokens >= self.maximum_output_tokens:
            return "output_token_budget_exhausted"
        if self.maximum_cost_usd is not None and self.cost_usd >= self.maximum_cost_usd:
            return "cost_budget_exhausted"
        if time.monotonic() - self.started_monotonic >= self.maximum_wall_clock_seconds:
            return "wall_clock_budget_exhausted"
        if not self.usage_complete:
            return "token_usage_unavailable"
        if self.maximum_cost_usd is not None and not self.cost_complete:
            return "cost_usage_unavailable"
        return None

    def _exhausted(self, reason: str) -> ProviderResult:
        return ProviderResult(
            status="failed", content=None, identity=self.identity,
            usage=ProviderUsage(cost_unknown_reason="request_not_sent"), latency_ms=0,
            error_type="provider_budget_exhausted", attempt_count=0,
            raw_metadata={"stop_reason": reason},
        )

    def ledger(self) -> dict[str, Any]:
        return {
            "request_count": self.request_count, "logical_call_count": self.logical_call_count,
            "retry_count": self.retry_count, "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens, "usage_complete": self.usage_complete,
            "cost_usd": self.cost_usd if self.cost_complete else "unknown",
            "cost_complete": self.cost_complete, "stop_reason": self.stop_reason,
            "cost_unknown_reasons": sorted(self.cost_unknown_reasons),
            "elapsed_seconds": time.monotonic() - self.started_monotonic,
            "limits": {
                "maximum_requests": self.maximum_requests,
                "maximum_input_tokens": self.maximum_input_tokens,
                "maximum_output_tokens": self.maximum_output_tokens,
                "maximum_cost_usd": self.maximum_cost_usd,
                "maximum_wall_clock_seconds": self.maximum_wall_clock_seconds,
            },
        }

    def restore(self, ledger: dict[str, Any]) -> None:
        limits = ledger.get("limits")
        if limits != self.ledger()["limits"]:
            raise ProviderConfigurationError("checkpoint provider budget identity mismatch")
        self.request_count = int(ledger.get("request_count", 0))
        self.logical_call_count = int(ledger.get("logical_call_count", 0))
        self.retry_count = int(ledger.get("retry_count", 0))
        self.input_tokens = int(ledger.get("input_tokens", 0))
        self.output_tokens = int(ledger.get("output_tokens", 0))
        self.usage_complete = bool(ledger.get("usage_complete", True))
        self.cost_complete = bool(ledger.get("cost_complete", True))
        self.cost_unknown_reasons = {str(item) for item in ledger.get("cost_unknown_reasons", [])}
        stored_cost = ledger.get("cost_usd", 0.0)
        self.cost_usd = float(stored_cost) if isinstance(stored_cost, (int, float)) else 0.0
        self.stop_reason = ledger.get("stop_reason")
        self.started_monotonic = time.monotonic() - max(0.0, float(ledger.get("elapsed_seconds", 0.0)))

    def smoke_check(self) -> ProviderResult:
        return self.invoke(
            [{"role": "user", "content": "Return exactly: ok"}],
            temperature=0.0,
            max_output_tokens=8,
            timeout_seconds=10.0,
            maximum_attempts=1,
            seed=20260726,
        )


class ProviderLLMClient:
    """Compatibility adapter so M1-M4 answer generation uses the M5 contract."""

    def __init__(
        self,
        provider: Provider,
        *,
        maximum_attempts: int = 2,
        seed: int = 20260726,
        cancellation: CancellationToken | None = None,
        request_timeout_seconds: float = 60.0,
        maximum_output_tokens: int = 1_600,
    ) -> None:
        self.provider = provider
        self.maximum_attempts = max(1, min(3, maximum_attempts))
        self.seed = seed
        self.cancellation = cancellation
        self.request_timeout_seconds = max(0.1, min(60.0, request_timeout_seconds))
        self.maximum_output_tokens = max(1, min(4_096, maximum_output_tokens))
        self.results: list[ProviderResult] = []

    @property
    def available(self) -> bool:
        return True

    @property
    def model(self) -> str:
        return self.provider.identity.model

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        *,
        max_tokens: int | None = None,
        timeout_seconds: float = 45,
    ) -> str | None:
        result = self.provider.invoke(
            messages,
            temperature=temperature,
            max_output_tokens=min(max_tokens or self.maximum_output_tokens, self.maximum_output_tokens),
            timeout_seconds=min(timeout_seconds, self.request_timeout_seconds),
            maximum_attempts=self.maximum_attempts,
            cancellation=self.cancellation,
            seed=self.seed,
        )
        self.results.append(result)
        return result.content if result.status == "succeeded" else None


class StructuredLearningEvaluator:
    """M5 evaluator adapter; invalid provider output is explicitly ungradable."""

    def __init__(
        self, provider: Provider, *, seed: int = 20260726,
        maximum_attempts: int = 2, request_timeout_seconds: float = 60.0,
        maximum_output_tokens: int = 1_600,
    ) -> None:
        if provider.identity.capability != "structured_evaluator":
            raise ProviderConfigurationError("provider lacks structured evaluator capability")
        self.client = ProviderLLMClient(
            provider, maximum_attempts=maximum_attempts, seed=seed,
            request_timeout_seconds=request_timeout_seconds,
            maximum_output_tokens=maximum_output_tokens,
        )

    @property
    def results(self) -> list[ProviderResult]:
        return self.client.results

    def evaluate(self, task: dict[str, Any], answer_text: str) -> dict[str, Any]:
        response = self.client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Task: bounded_learning_evaluator. Prompt version: m5-evaluator-v1. "
                        "Return one JSON object with schema version 1. Use only supplied "
                        "criterion IDs and Evidence IDs. Never output mastery, events, plans, "
                        "tools, code execution, shell commands, or unknown fields."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"server_bound_task": task, "untrusted_user_answer": answer_text},
                        ensure_ascii=False,
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=1_200,
            timeout_seconds=20.0,
        )
        if response is None:
            return _ungradable("Evaluator provider failed or timed out.")
        try:
            raw = json.loads(_strip_fence(response))
            validated = EvaluationOutput.model_validate(raw)
            _validate_evaluator_membership(validated, task)
            return validated.model_dump()
        except (json.JSONDecodeError, ValidationError, ValueError, TypeError):
            return _ungradable("Evaluator output failed strict validation.")


def redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): (
                "[REDACTED]"
                if _is_secret_key(str(key))
                else redact_secrets(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    return value


def _validate_evaluator_membership(evaluation: EvaluationOutput, task: dict[str, Any]) -> None:
    criteria = {str(item["criterion_id"]): item for item in task.get("rubric", [])}
    allowed_evidence = {str(item["evidence_id"]) for item in task.get("evidence", [])}
    if evaluation.verdict == "ungradable":
        if evaluation.criterion_results:
            raise ValueError("ungradable output contains criterion results")
        return
    result_ids = [item.criterion_id for item in evaluation.criterion_results]
    if len(result_ids) != len(set(result_ids)) or set(result_ids) != set(criteria):
        raise ValueError("evaluator criterion IDs do not match rubric")
    passed_weight = 0.0
    critical_failed = False
    used: set[str] = set()
    for result in evaluation.criterion_results:
        criterion = criteria[result.criterion_id]
        allowed = set(criterion.get("supporting_evidence_ids", []))
        if not set(result.used_evidence_ids).issubset(allowed):
            raise ValueError("evaluator invented Evidence")
        if result.passed and not result.used_evidence_ids:
            raise ValueError("passed criterion lacks Evidence")
        if result.passed:
            passed_weight += float(criterion["weight"])
            used.update(result.used_evidence_ids)
        elif criterion.get("critical"):
            critical_failed = True
    if not set(evaluation.used_evidence_ids).issubset(allowed_evidence):
        raise ValueError("evaluator invented Evidence summary")
    if set(evaluation.used_evidence_ids) != used:
        raise ValueError("evaluator Evidence summary mismatch")
    total = sum(float(item["weight"]) for item in criteria.values())
    ratio = passed_weight / total if total else 0.0
    verdict = "pass" if not critical_failed and ratio >= 0.8 else ("partial" if ratio > 0 else "fail")
    if evaluation.verdict != verdict:
        raise ValueError("evaluator verdict conflicts with rubric")


def _fake_evaluation(joined: str) -> str:
    try:
        user_json = json.loads(joined.split("\n")[-1])
        task = user_json.get("server_bound_task", {})
        answer = str(user_json.get("untrusted_user_answer", "")).casefold()
    except (json.JSONDecodeError, AttributeError):
        return json.dumps(_ungradable("Fake evaluator could not parse task."))
    forced_fail = "fail" in answer or "错误" in answer
    forced_partial = "partial" in answer or "部分" in answer
    results = []
    used: list[str] = []
    rubric = list(task.get("rubric", []))
    for index, item in enumerate(rubric):
        passed = not forced_fail and (not forced_partial or index == 0)
        ids = list(item.get("supporting_evidence_ids", []))[:1] if passed else []
        used.extend(ids)
        results.append(
            {
                "criterion_id": item.get("criterion_id"),
                "passed": passed,
                "used_evidence_ids": ids,
                "feedback": "deterministic fixture",
            }
        )
    return json.dumps(
        {
            "evaluator_schema_version": 1,
            "verdict": "fail" if forced_fail else ("partial" if forced_partial else "pass"),
            "criterion_results": results,
            "supported_feedback": [] if forced_fail else ["deterministic fixture"],
            "missing_concepts": [] if not (forced_fail or forced_partial) else ["fixture concept"],
            "misconceptions": [],
            "used_evidence_ids": list(dict.fromkeys(used)),
            "warnings": [],
        },
        ensure_ascii=False,
    )


def _ungradable(warning: str) -> dict[str, Any]:
    return {
        "evaluator_schema_version": 1,
        "verdict": "ungradable",
        "criterion_results": [],
        "supported_feedback": [],
        "missing_concepts": [],
        "misconceptions": [],
        "used_evidence_ids": [],
        "warnings": [warning[:500]],
    }


def _strip_fence(value: str) -> str:
    cleaned = value.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return cleaned


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed >= 0 else None


def normalized_endpoint_identity(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url.strip())
    host = (parsed.hostname or "").casefold()
    port = f":{parsed.port}" if parsed.port is not None else ""
    path = "/" + "/".join(part for part in parsed.path.split("/") if part)
    path_digest = hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]
    return f"{parsed.scheme.casefold()}://{host}{port}|path-sha256:{path_digest}"


def _pricing_identity(pricing: PricingConfig | None) -> dict[str, Any]:
    if pricing is None:
        return {"status": "unknown", "reason": "pricing_not_configured"}
    return {
        "status": "configured", "currency": pricing.currency,
        "model": pricing.model,
        "input_price_per_unit": pricing.input_price_per_unit,
        "output_price_per_unit": pricing.output_price_per_unit,
        "unit_tokens": pricing.unit_tokens, "source": pricing.source,
    }


def _priced_usage(
    input_tokens: int | None,
    output_tokens: int | None,
    total_tokens: int | None,
    pricing: PricingConfig | None,
) -> ProviderUsage:
    if input_tokens is None or output_tokens is None:
        return ProviderUsage(
            input_tokens, output_tokens, total_tokens, None,
            "unknown", pricing.currency if pricing else None, "token_usage_missing",
        )
    if pricing is None:
        return ProviderUsage(
            input_tokens, output_tokens, total_tokens, None,
            "unknown", None, "pricing_not_configured",
        )
    cost = (
        input_tokens * pricing.input_price_per_unit
        + output_tokens * pricing.output_price_per_unit
    ) / pricing.unit_tokens
    status = "known_zero" if cost == 0 else "calculated"
    return ProviderUsage(
        input_tokens, output_tokens, total_tokens, cost,
        status, pricing.currency, None,
    )


def _is_secret_key(key: str) -> bool:
    lowered = key.casefold()
    return (
        any(marker in lowered for marker in SECRET_KEYS)
        or lowered == "token"
        or lowered.endswith("_token")
    )


def validate_vector(vector: list[float], *, expected_dimension: int | None = None) -> None:
    if not vector:
        raise ValueError("embedding vector is empty")
    if expected_dimension is not None and len(vector) != expected_dimension:
        raise ValueError("embedding dimension mismatch")
    if any(not math.isfinite(float(value)) for value in vector):
        raise ValueError("embedding vector contains NaN or Infinity")
