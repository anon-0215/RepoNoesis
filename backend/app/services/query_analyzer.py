from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Literal


QueryIntent = Literal["locate", "explain", "impact", "relation", "mixed", "unknown"]
RelationDirection = Literal["inbound", "outbound", "both"]

_IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"
_DOTTED_IDENTIFIER = rf"{_IDENTIFIER}(?:\.{_IDENTIFIER})+"
_CODE_IDENTIFIER = re.compile(rf"(?:{_DOTTED_IDENTIFIER}|{_IDENTIFIER})")
_BACKTICK = re.compile(r"`([^`\r\n]{1,500})`")
_PATH_SYMBOL = re.compile(
    rf"(?P<path>[A-Za-z0-9_./\\-]+\.py)\s*[:#]\s*"
    rf"(?P<symbol>{_DOTTED_IDENTIFIER}|{_IDENTIFIER})"
)
_DOTTED = re.compile(rf"(?<![A-Za-z0-9_.])({_DOTTED_IDENTIFIER})(?![A-Za-z0-9_.])")
_SNAKE = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+)(?![A-Za-z0-9_])")
_CAMEL = re.compile(
    r"(?<![A-Za-z0-9_])((?=[A-Za-z0-9]*[A-Z])(?=[A-Za-z0-9]*[a-z])"
    r"[A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*)(?![A-Za-z0-9_])"
)
_CALL = re.compile(
    rf"(?<![A-Za-z0-9_.])({_DOTTED_IDENTIFIER}|{_IDENTIFIER})\s*\("
)
_EXPLICIT_CONTEXT = re.compile(
    rf"(?:class|method|function|symbol|identifier|类|方法|函数|符号|标识符)"
    rf"\s*(?:called|named|名为|叫)?\s*[`\"']?({_DOTTED_IDENTIFIER}|{_IDENTIFIER})",
    re.IGNORECASE,
)

_INTENT_PATTERNS: tuple[tuple[str, str, tuple[re.Pattern[str], ...]], ...] = (
    (
        "locate",
        "locate_definition_phrase",
        (
            re.compile(r"在哪里定义|定义在哪|哪个文件|哪一个文件|定位|查找.{0,12}定义"),
            re.compile(
                r"\bwhere\s+(?:is|are)\b.{0,80}\bdefined\b|\bwhich\s+file\b|"
                r"\blocate\b|\bfind\s+(?:the\s+)?definition\b|\bdefinition\s+of\b",
                re.IGNORECASE,
            ),
        ),
    ),
    (
        "explain",
        "explain_mechanism_phrase",
        (
            re.compile(r"如何工作|怎么工作|为什么|解释|实现机制|如何实现"),
            re.compile(
                r"\bhow\s+(?:does|do|is|are)\b|\bhow\b.{0,60}\bwork\b|"
                r"\bwhy\b|\bexplain\b|\bimplementation\b",
                re.IGNORECASE,
            ),
        ),
    ),
    (
        "impact",
        "impact_change_phrase",
        (
            re.compile(r"影响什么|谁会受影响|影响范围|(?:修改|变更|改动).{0,30}影响"),
            re.compile(
                r"\bwhat\b.{0,60}\b(?:impact|affect)\b|\bwho\b.{0,40}\baffected\b|"
                r"\bimpact\s+of\b|\bwhat\s+changes?\s+if\b|"
                r"\b(?:change|modified|modify)\b.{0,60}\baffect\b",
                re.IGNORECASE,
            ),
        ),
    ),
    (
        "relation",
        "relation_caller_phrase",
        (
            re.compile(r"谁调用|被谁调用|调用了谁|调用它|和谁有关|装饰器.{0,20}包装"),
            re.compile(
                r"\bwho\s+calls?\b|\bwhat\s+does\b.{0,50}\bcall\b|\bcalled\s+by\b|"
                r"\brelated\s+to\b|\bcallers?\b|\bcallees?\b|\bdecorator\b.{0,30}\bwrap",
                re.IGNORECASE,
            ),
        ),
    ),
)


@dataclass(frozen=True)
class SymbolHint:
    value: str
    path: str | None = None
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalRoutingHint:
    dense_weight: float
    lexical_weight: float
    symbol_weight: float
    relation_direction: RelationDirection
    relation_budget: int
    candidate_pool: int


@dataclass(frozen=True)
class QueryAnalysis:
    primary_intent: QueryIntent
    secondary_intents: tuple[str, ...]
    confidence: float
    reason_codes: tuple[str, ...]
    symbol_hints: tuple[str, ...]
    neutral_fallback: bool
    routing_hint: RetrievalRoutingHint

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["secondary_intents"] = list(self.secondary_intents)
        value["reason_codes"] = list(self.reason_codes)
        value["symbol_hints"] = list(self.symbol_hints)
        return value


class QueryAnalyzer:
    """Deterministic, offline query analysis for Retrieval v2 soft routing."""

    def analyze(self, query: str) -> QueryAnalysis:
        text = str(query).strip()
        matched_intents: list[str] = []
        reason_codes: list[str] = []
        for intent, reason, patterns in _INTENT_PATTERNS:
            if any(pattern.search(text) for pattern in patterns):
                matched_intents.append(intent)
                reason_codes.append(reason)

        hints = extract_symbol_query_hints(text)
        for hint in hints:
            reason_codes.extend(hint.reason_codes)

        if len(matched_intents) > 1:
            primary: QueryIntent = "mixed"
            secondary = tuple(matched_intents)
            confidence = 0.65
            reason_codes.extend(("multiple_intents_detected", "conflicting_intent_rules"))
        elif matched_intents:
            primary = matched_intents[0]  # type: ignore[assignment]
            secondary = ()
            confidence = 0.9
        else:
            primary = "unknown"
            secondary = ()
            confidence = 0.2
            reason_codes.append("unknown_intent_fallback")

        neutral_fallback = primary in {"mixed", "unknown"} or confidence < 0.5
        if neutral_fallback:
            reason_codes.append("neutral_strategy_fallback")
        if confidence < 0.5:
            reason_codes.append("low_confidence_fallback")
        return QueryAnalysis(
            primary_intent=primary,
            secondary_intents=secondary,
            confidence=confidence,
            reason_codes=tuple(_deduplicate(reason_codes)),
            symbol_hints=tuple(hint.value for hint in hints),
            neutral_fallback=neutral_fallback,
            routing_hint=_routing_hint(primary, text, neutral_fallback),
        )


def extract_symbol_query_hints(query: str) -> tuple[SymbolHint, ...]:
    """Extract only code-like identifiers, retaining stable provenance and path context."""

    text = str(query)
    collected: dict[tuple[str, str | None], tuple[int, list[str]]] = {}

    def add(value: str, position: int, reason: str, path: str | None = None) -> None:
        cleaned = value.strip().removesuffix("()").strip()
        if _CODE_IDENTIFIER.fullmatch(cleaned) is None:
            return
        normalized_path = path.replace("\\", "/").lstrip("/") if path else None
        key = (cleaned.casefold(), normalized_path.casefold() if normalized_path else None)
        if key not in collected:
            collected[key] = (position, [cleaned, normalized_path or "", reason])
            return
        current_position, values = collected[key]
        reasons = values[2:]
        if reason not in reasons:
            reasons.append(reason)
        collected[key] = (min(current_position, position), [values[0], values[1], *reasons])

    masked = list(text)
    for match in _BACKTICK.finditer(text):
        raw = match.group(1).strip()
        path_match = _PATH_SYMBOL.fullmatch(raw)
        if path_match:
            add(
                path_match.group("symbol"),
                match.start(),
                "backticked_identifier",
                path_match.group("path"),
            )
        else:
            add(raw, match.start(), "backticked_identifier")
        for index in range(match.start(), match.end()):
            masked[index] = " "
    visible = "".join(masked)

    for match in _PATH_SYMBOL.finditer(visible):
        add(match.group("symbol"), match.start(), "path_symbol_detected", match.group("path"))
    for match in _DOTTED.finditer(visible):
        add(match.group(1), match.start(), "qualified_symbol_detected")
    for match in _SNAKE.finditer(visible):
        add(match.group(1), match.start(), "snake_case_identifier")
    for match in _CAMEL.finditer(visible):
        add(match.group(1), match.start(), "camel_case_identifier")
    for match in _CALL.finditer(visible):
        add(match.group(1), match.start(), "call_syntax_identifier")
    for match in _EXPLICIT_CONTEXT.finditer(visible):
        add(match.group(1), match.start(1), "explicit_symbol_context")

    ordered: list[SymbolHint] = []
    for _key, (position, values) in sorted(
        collected.items(), key=lambda item: (item[1][0], item[0][0], item[0][1] or "")
    ):
        del position
        value, path, *reasons = values
        if "." in value and "qualified_symbol_detected" not in reasons:
            reasons.append("qualified_symbol_detected")
        ordered.append(SymbolHint(value=value, path=path or None, reason_codes=tuple(reasons)))
    return tuple(ordered)


def _routing_hint(
    intent: QueryIntent, query: str, neutral_fallback: bool
) -> RetrievalRoutingHint:
    if neutral_fallback:
        return RetrievalRoutingHint(1.0, 1.0, 1.0, "both", 8, 24)
    direction: RelationDirection = "both"
    caller = re.search(r"谁调用|被谁调用|\bwho\s+calls?\b|\bcalled\s+by\b", query, re.IGNORECASE)
    callee = re.search(r"调用了谁|\bwhat\s+does\b.{0,50}\bcall\b|\bcallees?\b", query, re.IGNORECASE)
    if caller and not callee:
        direction = "inbound"
    elif callee and not caller:
        direction = "outbound"
    if intent == "locate":
        return RetrievalRoutingHint(1.1, 1.05, 1.2, direction, 0, 24)
    if intent == "explain":
        return RetrievalRoutingHint(1.1, 1.15, 1.15, direction, 0, 24)
    if intent == "impact":
        return RetrievalRoutingHint(1.0, 1.05, 1.05, "both", 16, 24)
    return RetrievalRoutingHint(1.0, 1.05, 1.1, direction, 16, 24)


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
