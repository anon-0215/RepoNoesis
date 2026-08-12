from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError, model_validator

from app.services.evidence import Evidence


MAX_FINAL_ANSWER_PARTS = 12
MAX_PART_TEXT_CHARS = 2_000
MAX_ALIASES_PER_PART = 5
MAX_FINAL_ANSWER_OUTPUT_CHARS = 16_000
_ALIAS = re.compile(r"A[1-9][0-9]{0,2}")
_EVIDENCE_MARKER = re.compile(r"\[E[1-9][0-9]*\]")


@dataclass(frozen=True)
class CanonicalCitationDescriptor:
    """One request-local alias bound to server-validated Evidence identity."""

    alias: str
    evidence_id: str
    project_id: str
    repository_revision: str
    path: str
    start_line: int
    end_line: int
    content_hash: str
    chunk_identity: str
    symbol: str
    language: str
    excerpt: str

    @classmethod
    def from_evidence(cls, alias: str, evidence: Evidence) -> "CanonicalCitationDescriptor":
        if _ALIAS.fullmatch(alias) is None:
            raise ValueError("invalid citation alias")
        return cls(
            alias=alias,
            evidence_id=evidence.evidence_id,
            project_id=evidence.project_id,
            repository_revision=evidence.repository_revision,
            path=evidence.path,
            start_line=evidence.start_line,
            end_line=evidence.end_line,
            content_hash=evidence.content_hash,
            chunk_identity=evidence.chunk_identity,
            symbol=evidence.qualified_name or evidence.symbol_name,
            language=evidence.language,
            excerpt=evidence.excerpt,
        )

    def canonical_token(self) -> str:
        if (
            not self.evidence_id
            or not self.path
            or self.start_line < 1
            or self.end_line < self.start_line
        ):
            raise ValueError("invalid canonical citation descriptor")
        return (
            f"[{self.evidence_id}] "
            f"{self.path}:{self.start_line}-{self.end_line}"
        )

    def prompt_evidence(self) -> dict[str, Any]:
        """Expose source content and alias, but not model-writable location fields."""

        return {
            "alias": self.alias,
            "symbol": self.symbol,
            "language": self.language,
            "untrusted_source_excerpt": self.excerpt,
        }


class FinalAnswerPart(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    text: StrictStr = Field(min_length=1, max_length=MAX_PART_TEXT_CHARS)
    evidence_aliases: list[StrictStr] = Field(
        min_length=1,
        max_length=MAX_ALIASES_PER_PART,
    )

    @model_validator(mode="after")
    def validate_part(self) -> "FinalAnswerPart":
        if self.text != self.text.strip():
            raise ValueError("part text must not have surrounding whitespace")
        if len(set(self.evidence_aliases)) != len(self.evidence_aliases):
            raise ValueError("part aliases must be unique")
        return self


class FinalAnswerEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    parts: list[FinalAnswerPart] = Field(
        min_length=1,
        max_length=MAX_FINAL_ANSWER_PARTS,
    )


@dataclass(frozen=True)
class FinalAnswerProtocolFailure:
    stable_code: str
    field_path: tuple[str | int, ...] = ()
    output_chars: int = 0
    output_sha256: str | None = None
    markdown_fence_detected: bool = False
    part_count: int = 0
    alias_count: int = 0

    def to_safe_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "stage": "final_answer",
            "stable_code": self.stable_code,
            "field_path": list(self.field_path),
            "output_chars": max(0, self.output_chars),
            "markdown_fence_detected": self.markdown_fence_detected,
            "part_count": max(0, self.part_count),
            "alias_count": max(0, self.alias_count),
        }
        if self.output_sha256 is not None:
            value["output_sha256"] = self.output_sha256
        return value


@dataclass(frozen=True)
class FinalAnswerProtocolResult:
    answer: str | None = None
    selected_evidence_ids: tuple[str, ...] = ()
    failure: FinalAnswerProtocolFailure | None = None

    @property
    def valid(self) -> bool:
        return self.answer is not None and self.failure is None


def build_canonical_citation_descriptors(
    evidence: list[Evidence],
) -> list[CanonicalCitationDescriptor]:
    descriptors = [
        CanonicalCitationDescriptor.from_evidence(f"A{index}", item)
        for index, item in enumerate(evidence, start=1)
    ]
    if len({item.alias for item in descriptors}) != len(descriptors):
        raise ValueError("citation aliases must be unique")
    if len({item.evidence_id for item in descriptors}) != len(descriptors):
        raise ValueError("Evidence IDs must be unique")
    return descriptors


def build_final_answer_json_schema(
    descriptors: list[CanonicalCitationDescriptor],
) -> dict[str, Any]:
    schema = FinalAnswerEnvelope.model_json_schema()
    aliases = [item.alias for item in descriptors]
    part_schema = schema.get("$defs", {}).get("FinalAnswerPart", {})
    alias_items = (
        part_schema.get("properties", {})
        .get("evidence_aliases", {})
        .get("items")
    )
    if not isinstance(alias_items, dict):
        raise ValueError("final-answer schema alias items are unavailable")
    alias_items["enum"] = aliases
    schema["x-citation-contract"] = {
        "aliases": aliases,
        "canonical_rendering": "server_only",
        "model_location_fields_allowed": False,
    }
    return schema


def render_structured_final_answer(
    raw: Any,
    descriptors: list[CanonicalCitationDescriptor],
) -> FinalAnswerProtocolResult:
    metadata = _output_metadata(raw)
    if not isinstance(raw, str) or not raw.strip():
        return _failure("final_answer_invalid_json", metadata)
    if len(raw) > MAX_FINAL_ANSWER_OUTPUT_CHARS:
        return _failure("final_answer_schema_invalid", metadata)
    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError:
        return _failure("final_answer_invalid_json", metadata)
    try:
        envelope = FinalAnswerEnvelope.model_validate(parsed, strict=True)
    except ValidationError as exc:
        error = exc.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        )[0]
        path = _safe_field_path(error.get("loc"))
        error_type = str(error.get("type", ""))
        if "evidence_aliases" in path:
            if error_type in {"missing", "too_short"}:
                code = "citation_alias_missing"
            elif error_type == "too_long":
                code = "citation_alias_limit_exceeded"
            elif error_type in {"list_type", "string_type"}:
                code = "citation_alias_invalid_type"
            else:
                code = "final_answer_schema_invalid"
        else:
            code = "final_answer_schema_invalid"
        return _failure(code, metadata, field_path=path)

    descriptor_by_alias = {item.alias: item for item in descriptors}
    selected_ids: list[str] = []
    rendered_parts: list[str] = []
    alias_count = 0
    for index, part in enumerate(envelope.parts):
        alias_count += len(part.evidence_aliases)
        for alias_index, alias in enumerate(part.evidence_aliases):
            if alias not in descriptor_by_alias:
                return _failure(
                    "citation_alias_unknown",
                    metadata,
                    field_path=("parts", index, "evidence_aliases", alias_index),
                    part_count=len(envelope.parts),
                    alias_count=alias_count,
                )
        if _contains_model_supplied_location(part.text, descriptors):
            return _failure(
                "final_answer_schema_invalid",
                metadata,
                field_path=("parts", index, "text"),
                part_count=len(envelope.parts),
                alias_count=alias_count,
            )
        try:
            tokens = [
                descriptor_by_alias[alias].canonical_token()
                for alias in part.evidence_aliases
            ]
        except (KeyError, ValueError):
            return _failure(
                "canonical_render_failed",
                metadata,
                field_path=("parts", index, "evidence_aliases"),
                part_count=len(envelope.parts),
                alias_count=alias_count,
            )
        selected_ids.extend(
            descriptor_by_alias[alias].evidence_id
            for alias in part.evidence_aliases
        )
        rendered_parts.append(f"{part.text} {' '.join(tokens)}")
    return FinalAnswerProtocolResult(
        answer="\n\n".join(rendered_parts),
        selected_evidence_ids=tuple(dict.fromkeys(selected_ids)),
    )


def _contains_model_supplied_location(
    text: str,
    descriptors: list[CanonicalCitationDescriptor],
) -> bool:
    if _EVIDENCE_MARKER.search(text):
        return True
    for item in descriptors:
        if (
            item.path in text
            or (item.repository_revision and item.repository_revision in text)
            or (item.content_hash and item.content_hash in text)
            or (item.chunk_identity and item.chunk_identity in text)
        ):
            return True
    return False


def _output_metadata(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str):
        return {
            "output_chars": 0,
            "output_sha256": None,
            "markdown_fence_detected": False,
        }
    return {
        "output_chars": len(raw),
        "output_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "markdown_fence_detected": "```" in raw,
    }


def _failure(
    stable_code: str,
    metadata: dict[str, Any],
    *,
    field_path: tuple[str | int, ...] = (),
    part_count: int = 0,
    alias_count: int = 0,
) -> FinalAnswerProtocolResult:
    return FinalAnswerProtocolResult(
        failure=FinalAnswerProtocolFailure(
            stable_code=stable_code,
            field_path=field_path,
            part_count=part_count,
            alias_count=alias_count,
            **metadata,
        )
    )


def _safe_field_path(value: Any) -> tuple[str | int, ...]:
    if not isinstance(value, (tuple, list)):
        return ()
    safe: list[str | int] = []
    for item in value[:16]:
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            safe.append(item)
        elif isinstance(item, str) and re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]{0,63}", item
        ):
            safe.append(item)
    return tuple(safe)
