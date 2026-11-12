"""Small durable tool failures; never include model prose or source bodies."""
from __future__ import annotations

import re
from typing import Any

from app.services.agent_contracts import normalize_repository_relative_path

REASONS = {
    'document_relation_unsupported': 'Documents have no AST relations; choose a code or symbol seed.',
    'source_line_out_of_range': 'Range exceeds saved file; use its reported line count.',
    'source_range_exceeded': 'Read exceeds per-call range or byte limit; narrow the range.',
    'source_file_not_found': 'Path absent from saved snapshot; use discovered exact paths.',
    'repository_revision_mismatch': 'Bound snapshot revision changed.',
    'invalid_parameters': 'Tool parameters do not match the schema.',
    'tool_error': 'Tool rejected the operation.',
    'tool_failed': 'Tool execution failed.',
    'tool_timeout': 'Tool timed out.',
    'cancelled': 'Request cancelled.',
    'deadline_exceeded': 'Request deadline reached.',
    'final_answer_not_attempted': 'Work budget exhausted; preserve final answer time.',
}


def receipt(request_id: str, tool: str, call_id: str, order: int,
            parameters: Any, code: str) -> dict[str, Any]:
    values = parameters if isinstance(parameters, dict) else {}
    target: dict[str, Any] = {'status': 'omitted'}
    if tool == 'read_source':
        path = values.get('path')
        try:
            normalized = normalize_repository_relative_path(path) if isinstance(path, str) else ''
        except ValueError:
            normalized = ''
        target = {'status': 'known' if normalized and len(normalized) <= 500 else 'omitted'}
        if target['status'] == 'known':
            target['path'] = normalized
        for field in ('start_line', 'end_line'):
            value = values.get(field)
            if type(value) is int and 1 <= value <= 1_000_000_000:
                target[field] = value
    stable = code if code in REASONS else 'tool_failed'
    return {'request_id': request_id[:64], 'tool': tool[:48], 'call_id': call_id[:40],
            'order': order, 'target': target, 'code': stable, 'reason': REASONS[stable]}


def sanitize_receipts(value: Any) -> list[dict[str, Any]]:
    rows = []
    if not isinstance(value, list):
        return rows
    for item in value[:4]:
        if (not isinstance(item, dict) or not isinstance(item.get('request_id'), str)
                or not re.fullmatch(r'[A-Za-z0-9-]{1,64}', item['request_id'])
                or item.get('tool') not in {'read_source', 'search_code', 'lookup_symbol', 'expand_relations', 'validate_evidence', 'get_learning_context'}
                or type(item.get('order')) is not int or not 1 <= item['order'] <= 1_000_000):
            continue
        target = item.get('target') if isinstance(item.get('target'), dict) else {}
        safe = receipt(item['request_id'], item['tool'], str(item.get('call_id', '')),
                       item['order'], target, item.get('code'))
        if (target.get('status') == 'truncated'
                and re.fullmatch(r'[a-f0-9]{64}', str(target.get('path_sha256', '')))):
            safe['target'].update(status='truncated', path_sha256=target['path_sha256'])
        rows.append(safe)
    return rows
