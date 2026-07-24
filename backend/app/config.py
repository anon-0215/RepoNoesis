from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from app.services.agent_contracts import AgentLimits


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_MAX_LENGTH_MIN = 16
EMBEDDING_MAX_LENGTH_MAX = 8192


def load_environment() -> None:
    """Load simple KEY=VALUE pairs from project .env files.

    Existing process environment values win over .env values, so users can still
    override settings from the command line when they want to.
    """
    for env_path in (PROJECT_ROOT / ".env", BACKEND_ROOT / ".env"):
        if env_path.exists():
            _load_env_file(env_path)


def get_env_value(key: str, default: str = "") -> str:
    if key in os.environ:
        return os.environ[key]
    for env_path in (BACKEND_ROOT / ".env", PROJECT_ROOT / ".env"):
        if not env_path.exists():
            continue
        values = _read_env_file(env_path)
        if key in values:
            return values[key]
    return os.getenv(key, default)


@dataclass(frozen=True)
class EmbeddingSettings:
    enabled: bool
    model_name_or_path: str
    device: str
    batch_size: int
    max_length: int
    normalize: bool
    cache_dir: Path
    query_prefix: str
    document_prefix: str
    model_revision: str = ""


def get_embedding_settings() -> EmbeddingSettings:
    cache_dir = _env_path("EMBEDDING_CACHE_DIR", PROJECT_ROOT / "embedding_cache")
    return EmbeddingSettings(
        enabled=_env_bool("EMBEDDING_ENABLED", False),
        model_name_or_path=get_env_value("EMBEDDING_MODEL_NAME_OR_PATH", "BAAI/bge-m3").strip()
        or "BAAI/bge-m3",
        device=(get_env_value("EMBEDDING_DEVICE", "auto").strip() or "auto").lower(),
        batch_size=max(1, _env_int("EMBEDDING_BATCH_SIZE", 8)),
        max_length=clamp_embedding_max_length(_env_int("EMBEDDING_MAX_LENGTH", 8192)),
        normalize=_env_bool("EMBEDDING_NORMALIZE", True),
        cache_dir=cache_dir,
        query_prefix=get_env_value("EMBEDDING_QUERY_PREFIX", ""),
        document_prefix=get_env_value("EMBEDDING_DOCUMENT_PREFIX", ""),
        model_revision=get_env_value("EMBEDDING_MODEL_REVISION", "").strip(),
    )


def get_agent_limits() -> AgentLimits:
    """Load bounded M2 limits; environment values may only stay in safe ranges."""
    return AgentLimits(
        max_agent_steps=_bounded_env_int("AGENT_MAX_STEPS", 5, 1, 8),
        max_tool_calls=_bounded_env_int("AGENT_MAX_TOOL_CALLS", 8, 1, 12),
        max_calls_per_step=1,
        max_same_tool_calls=_bounded_env_int("AGENT_MAX_SAME_TOOL_CALLS", 3, 1, 3),
        max_no_progress_steps=_bounded_env_int(
            "AGENT_MAX_NO_PROGRESS_STEPS", 2, 1, 2
        ),
        total_deadline_ms=_bounded_env_int(
            "AGENT_TOTAL_DEADLINE_MS", 60_000, 1_000, 60_000
        ),
        default_tool_timeout_ms=_bounded_env_int(
            "AGENT_TOOL_TIMEOUT_MS", 15_000, 100, 15_000
        ),
        max_search_results=_bounded_env_int(
            "AGENT_MAX_SEARCH_RESULTS", 20, 1, 20
        ),
        max_observation_bytes=_bounded_env_int(
            "AGENT_MAX_OBSERVATION_BYTES", 65_536, 1_024, 65_536
        ),
        max_source_read_lines=_bounded_env_int(
            "AGENT_MAX_SOURCE_READ_LINES", 200, 1, 200
        ),
        max_source_read_bytes=_bounded_env_int(
            "AGENT_MAX_SOURCE_READ_BYTES", 32_768, 1_024, 32_768
        ),
        max_accumulated_evidence_context_bytes=_bounded_env_int(
            "AGENT_MAX_EVIDENCE_CONTEXT_BYTES", 49_152, 1_024, 49_152
        ),
        max_planner_output_tokens_per_step=_bounded_env_int(
            "AGENT_MAX_PLANNER_TOKENS_PER_STEP", 512, 64, 512
        ),
        max_total_planner_output_tokens=_bounded_env_int(
            "AGENT_MAX_TOTAL_PLANNER_TOKENS", 2_048, 64, 2_048
        ),
        max_final_answer_tokens=_bounded_env_int(
            "AGENT_MAX_FINAL_ANSWER_TOKENS", 1_600, 128, 1_600
        ),
    )


def clamp_embedding_max_length(value: int) -> int:
    return min(EMBEDDING_MAX_LENGTH_MAX, max(EMBEDDING_MAX_LENGTH_MIN, int(value)))


def _load_env_file(path: Path) -> None:
    for key, value in _read_env_file(path).items():
        os.environ.setdefault(key, value)


def _env_bool(key: str, default: bool) -> bool:
    value = get_env_value(key, str(default)).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    return default


def _env_int(key: str, default: int) -> int:
    value = get_env_value(key, str(default)).strip()
    try:
        return int(value)
    except ValueError:
        return default


def _bounded_env_int(key: str, default: int, minimum: int, maximum: int) -> int:
    value = _env_int(key, default)
    if value < minimum or value > maximum:
        return default
    return value


def _env_path(key: str, default: Path) -> Path:
    value = get_env_value(key, str(default)).strip()
    path = Path(value) if value else default
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if " #" in value:
            value = value.split(" #", 1)[0].strip()
        if key:
            values[key] = value
    return values
