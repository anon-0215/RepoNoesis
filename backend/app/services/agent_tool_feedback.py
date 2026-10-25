"""Bounded untrusted call feedback. No source content or Evidence promotion."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

PARAMETER_FIELDS = {
    "lookup_symbol": ("symbol", "path", "language", "match_mode", "top_k"),
    "search_code": ("query", "path", "language", "symbol", "top_k"),
    "read_source": ("path", "start_line", "end_line"),
    "validate_evidence": ("evidence_ids",),
    "expand_relations": ("seed_evidence_ids", "seed_symbol_ids", "relation_types", "direction", "max_depth", "per_node_limit"),
    "get_learning_context": (),
}


def json_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def bounded_parameters(tool: str, parameters: dict[str, Any] | None) -> dict[str, Any]:
    if parameters is None:
        return {"status": "UNKNOWN", "values": {}}
    values, truncated, omitted = {}, {}, 0
    for key, value in parameters.items():
        if key not in PARAMETER_FIELDS.get(tool, ()):
            omitted += 1
            continue
        if isinstance(value, str):
            raw = value.encode("utf-8")
            values[key] = raw[:256].decode("utf-8", errors="ignore")
            if len(raw) > 256:
                truncated[key] = {"original_utf8_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        elif type(value) is float and not math.isfinite(value):
            omitted += 1
        elif value is None or type(value) in (bool, int, float):
            values[key] = value if value is None or len(str(value)) <= 32 else "[OMITTED]"
            if values[key] == "[OMITTED]": omitted += 1
        elif isinstance(value, list):
            values[key] = [v for v in value[:8] if isinstance(v, str) and len(v.encode("utf-8")) <= 64]
            if len(values[key]) != len(value): truncated[key] = {"original_items": len(value)}
        else:
            omitted += 1
    return {"status": "bounded_untrusted_parameters", "values": values,
            "truncated_fields": truncated, "omitted_fields_count": omitted}


def call_feedback(call_id: str, calls: dict[str, list[Any]]) -> dict[str, Any]:
    matches = calls.get(call_id, [])
    if len(matches) != 1:
        return {"association": "missing_call_id" if not matches else "ambiguous_call_id",
                "execution_started": "UNKNOWN", "effective_parameters": {"status": "UNKNOWN", "values": {}}}
    call = matches[0]
    proposal = getattr(call, "proposed_parameters", None)
    effective = getattr(call, "effective_parameters", None)
    bound = getattr(call, "bound_constraints", None)
    sources = {}
    for key in PARAMETER_FIELDS.get(call.tool_name, ()):
        if bound is not None and key in bound:
            sources[key] = "server_bound"
        elif proposal is not None and key in proposal:
            sources[key] = "model_selected"
        elif proposal is not None and effective is not None and key in effective:
            sources[key] = "server_default_query" if key == "query" else "schema_default"
        else:
            sources[key] = "UNKNOWN"
    result = {"association": "unique_call_id", "tool": call.tool_name,
        "proposed_parameters": bounded_parameters(call.tool_name, proposal),
        "constrained_parameters": bounded_parameters(call.tool_name, getattr(call, "constrained_parameters", None)),
        "effective_parameters": bounded_parameters(call.tool_name, effective),
        "server_bound_constraints": bounded_parameters(call.tool_name, bound),
        "parameter_sources": sources,
        "adjusted_parameter_fields": [key for key in PARAMETER_FIELDS.get(call.tool_name, ())
            if effective is not None and proposal is not None and key in proposal
            and proposal[key] != effective.get(key)],
        "execution_started": getattr(call, "execution_started", None),
        "execution_stage": getattr(call, "execution_stage", "unknown"),
        "execution_limits": getattr(call, "execution_limits", None),
        "omitted_sections": [],
        "untrusted_data": True}
    if result["execution_started"] is None: result["execution_started"] = "UNKNOWN"
    # Four representations share this bound; retain exact omissions explicitly.
    for field in ("constrained_parameters", "proposed_parameters", "server_bound_constraints", "effective_parameters"):
        if json_bytes(result) <= 4096: break
        result[field] = {"status": "omitted_feedback_byte_budget", "values": {}}
        result["omitted_sections"].append(field)
    return result
