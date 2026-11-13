"""Beijing DashScope text-rerank adapter. One bounded HTTP attempt, no generation."""
from __future__ import annotations

import json
import math
import time
import urllib.request
from urllib.parse import urlsplit

from app.config import BailianRerankSettings, get_bailian_rerank_settings
from app.services.llm_client import ProviderError, raise_if_provider_cancelled
from app.services.provider_transport import isolated_response, TransportCollectionError


def validate_endpoint(endpoint: str) -> bool:
    try:
        p = urlsplit(endpoint)
        host = p.hostname or ""
        return (p.scheme == "https" and p.port in {None, 443} and not (
            p.username or p.password or p.query or p.fragment)
            and (host == "dashscope.aliyuncs.com" or (
                host.endswith(".cn-beijing.maas.aliyuncs.com")
                and host.split('.')[0].replace('-', '').isalnum()))
            and p.path == "/api/v1/services/rerank/text-rerank/text-rerank")
    except ValueError:
        return False


def validate_input(query: str, documents: list[str]) -> None:
    # UTF-8 byte count is a conservative upper bound for text token count.
    # Reject rather than silently clip any question or prebuilt source document.
    sizes = [len(x.encode('utf8')) for x in documents]
    q = len(query.encode('utf8'))
    if not query.strip() or not documents or len(documents) > 500 or any(not d for d in documents):
        raise ProviderError('rerank_invalid_input', 'Invalid rerank input.')
    if q > 4000 or any(s > 4000 for s in sizes) or q * len(sizes) + sum(sizes) > 120000:
        raise ProviderError('rerank_input_limit', 'Rerank conservative input limit exceeded; no request sent.')


def validate_results(data: object, count: int) -> list[tuple[int, float]]:
    if not isinstance(data, dict) or data.get('code'):
        raise ProviderError('rerank_service_error', 'Rerank service error.')
    output = data.get('output')
    results = output.get('results') if isinstance(output, dict) else None
    if not isinstance(results, list) or len(results) != count:
        raise ProviderError('rerank_missing_items', 'Rerank response must cover every candidate.')
    rows = []
    for item in results:
        if not isinstance(item, dict):
            raise ProviderError('rerank_invalid_result', 'Invalid rerank result.')
        index, score = item.get('index'), item.get('relevance_score')
        if type(index) is not int or not 0 <= index < count:
            raise ProviderError('rerank_invalid_index', 'Invalid rerank index.')
        if type(score) not in {int, float} or not math.isfinite(score) or not 0 <= score <= 1:
            raise ProviderError('rerank_invalid_score', 'Invalid rerank score.')
        rows.append((index, float(score)))
    if len({index for index, _ in rows}) != count:
        raise ProviderError('rerank_duplicate_index', 'Duplicate rerank index.')
    # Stable tie order follows service order; validate and order by the real score.
    return sorted(rows, key=lambda row: -row[1])


def rerank(query: str, documents: list[str], *, deadline: float,
           audit: dict, settings: BailianRerankSettings | None = None,
           recorder=None) -> list[tuple[int, float]]:
    config = settings or get_bailian_rerank_settings()
    if not config.api_key:
        raise ProviderError('rerank_auth_missing', 'Bailian API key is missing.')
    if not validate_endpoint(config.endpoint) or config.model != 'qwen3-rerank':
        raise ProviderError('rerank_config_invalid', 'Invalid Beijing rerank configuration.')
    validate_input(query, documents)
    end = min(deadline, time.monotonic() + config.timeout_seconds)
    def active():
        raise_if_provider_cancelled()
        if time.monotonic() >= end:
            raise TimeoutError()
    body = {'model': config.model, 'input': {'query': query, 'documents': documents},
            'parameters': {'top_n': len(documents), 'return_documents': False}}
    request = urllib.request.Request(config.endpoint, data=json.dumps(body, ensure_ascii=False).encode('utf8'),
        headers={'Authorization': 'Bearer ' + config.api_key, 'Content-Type': 'application/json'}, method='POST')
    transport = {}
    started = time.monotonic()
    call_id = recorder.start_provider_call('rerank') if recorder is not None else None
    outcome = 'invalid_response'
    audit.update(provider='bailian', model=config.model, region='cn-beijing', http_attempts=0)
    try:
        active()
        records = isolated_response(request, timeout=max(.001, end-time.monotonic()),
                                    check_active=active, audit=transport)
        active()
        data = None
        for row in records:
            if 'http_status' in row:
                audit['http_status'] = row['http_status']
            if row.get('error') == 'http':
                outcome = 'http_error'
                code = 'rerank_auth_failed' if row['status'] in {401, 403} else 'rerank_http_' + str(row['status'])
                raise ProviderError(code, 'Bailian rerank HTTP failure.')
            if row.get('error'):
                outcome = 'timeout' if row['error'] == 'timeout' else 'network_error'
                raise ProviderError('rerank_' + row['error'], 'Bailian rerank transport failure.')
            if 'data' in row:
                data = row['data']
        if isinstance(data, dict):
            usage = data.get('usage')
            audit['usage'] = {k: v for k, v in (usage if isinstance(usage, dict) else {}).items()
                              if k in {'total_tokens', 'prompt_tokens'} and type(v) is int and v >= 0}
            rid = data.get('request_id')
            if isinstance(rid, str) and len(rid) <= 128:
                audit['request_id'] = rid
        result = validate_results(data, len(documents))
        outcome = 'success'
        return result
    except TransportCollectionError:
        outcome = 'collection_error'
        raise
    except TimeoutError:
        outcome = 'timeout'
        raise ProviderError('rerank_timeout', 'Bailian rerank timed out.') from None
    except OSError:
        outcome = 'network_error'
        raise ProviderError('rerank_network', 'Bailian rerank network failure.') from None
    finally:
        audit['http_attempts'] = int(transport.get('network_state') in {'started', 'unknown'})
        audit['network_state'] = transport.get('network_state', 'not_started')
        if recorder is not None:
            recorder.record_provider_response(call_id, {'response_received': 'http_status' in audit,
                'http_status': audit.get('http_status'), 'request_model': config.model,
                'usage': audit.get('usage', {})})
            recorder.record_provider_attempt(call_id, outcome=outcome,
                duration_ms=int((time.monotonic()-started)*1000), timeout_ms=int((end-started)*1000),
                network_state=audit['network_state'], local_reason=transport.get('local_reason'))
