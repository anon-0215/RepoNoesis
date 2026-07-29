from __future__ import annotations

import hashlib
import json
from typing import Any, TypeVar


IdentityError = TypeVar("IdentityError", bound=Exception)


def canonical_identity(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def identity_digest(value: Any) -> str:
    return hashlib.sha256(canonical_identity(value).encode("utf-8")).hexdigest()


def require_identity_digest(
    record: dict[str, Any],
    expected_digest: str,
    *,
    field: str,
    label: str,
    error_type: type[IdentityError],
) -> None:
    if record.get(field) != expected_digest:
        raise error_type(f"{label} identity mismatch")
