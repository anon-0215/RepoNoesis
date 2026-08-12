from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlsplit

from app.services.agent_contracts import AgentLimits


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_MAX_LENGTH_MIN = 16
EMBEDDING_MAX_LENGTH_MAX = 8192
ThinkingMode = Literal["enabled", "disabled"]


class EnvironmentLoadError(RuntimeError):
    """Raised when the explicit bootstrap cannot safely load `.env`."""


class RepositoryConfigurationError(RuntimeError):
    """Raised with a fixed message when repository configuration is unsafe."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def load_environment() -> None:
    """Load simple KEY=VALUE pairs from project .env files.

    Existing process environment values win over .env values, so users can still
    override settings from the command line when they want to.
    """
    env_path = PROJECT_ROOT / ".env"
    try:
        if env_path.exists():
            _load_env_file(env_path)
    except (OSError, UnicodeError):
        raise EnvironmentLoadError(
            "Failed to load backend environment configuration."
        ) from None


def get_env_value(key: str, default: str = "") -> str:
    """Read configuration already present in the process environment.

    Disk-backed ``.env`` loading is intentionally restricted to the explicit
    production bootstrap. Ordinary imports and configuration access therefore
    never search for or read an environment file.
    """
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
    provider: str = "local_bge_m3"
    offline: bool = True


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = 45.0
    max_tokens: int = 1600
    temperature: float = 0.2
    max_retries: int = 2
    planner_thinking: ThinkingMode | None = None
    answer_thinking: ThinkingMode | None = None

    @property
    def missing(self) -> tuple[str, ...]:
        missing: list[str] = []
        if self.provider != "openai_compatible":
            missing.append("LLM_PROVIDER")
        if not _valid_provider_base_url(self.base_url):
            missing.append("LLM_BASE_URL")
        if not self.api_key:
            missing.append("LLM_API_KEY")
        if not self.model:
            missing.append("LLM_MODEL")
        return tuple(missing)

    @property
    def configured(self) -> bool:
        return not self.missing


@dataclass(frozen=True)
class RepositorySettings:
    runtime_dir: Path
    clone_timeout_seconds: float = 120.0
    max_files: int = 2000
    max_file_bytes: int = 1_000_000
    max_total_bytes: int = 50_000_000
    git_proxy: str | None = field(default=None, repr=False)


_MAX_GIT_PROXY_LENGTH = 2048


def validate_git_proxy(value: str | None) -> str | None:
    """Validate the explicit clone-only proxy without retaining it in errors."""
    if value is None or value == "":
        return None
    invalid = RepositoryConfigurationError(
        "git_proxy_invalid", "RepoNoesis Git proxy configuration is invalid."
    )
    if (
        not isinstance(value, str)
        or len(value) > _MAX_GIT_PROXY_LENGTH
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        or any(character.isspace() for character in value)
        or "\\" in value
    ):
        raise invalid
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise invalid from None
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.netloc
        or not host
        or parsed.fragment
        or port is not None and not 1 <= port <= 65535
    ):
        raise invalid
    return value


def get_embedding_settings() -> EmbeddingSettings:
    cache_dir = _env_path("EMBEDDING_CACHE_DIR", PROJECT_ROOT / "embedding_cache")
    model = get_env_value("EMBEDDING_MODEL", "").strip()
    if not model:
        model = get_env_value("EMBEDDING_MODEL_NAME_OR_PATH", "BAAI/bge-m3").strip()
    return EmbeddingSettings(
        enabled=_env_bool("EMBEDDING_ENABLED", False),
        model_name_or_path=model or "BAAI/bge-m3",
        device=(get_env_value("EMBEDDING_DEVICE", "auto").strip() or "auto").lower(),
        batch_size=max(1, _env_int("EMBEDDING_BATCH_SIZE", 8)),
        max_length=clamp_embedding_max_length(_env_int("EMBEDDING_MAX_LENGTH", 8192)),
        normalize=_env_bool("EMBEDDING_NORMALIZE", True),
        cache_dir=cache_dir,
        query_prefix=get_env_value("EMBEDDING_QUERY_PREFIX", ""),
        document_prefix=get_env_value("EMBEDDING_DOCUMENT_PREFIX", ""),
        model_revision=get_env_value("EMBEDDING_MODEL_REVISION", "").strip(),
        provider=(get_env_value("EMBEDDING_PROVIDER", "local_bge_m3").strip() or "local_bge_m3"),
        offline=_env_bool("EMBEDDING_OFFLINE", True),
    )


def get_llm_settings() -> LLMSettings:
    return LLMSettings(
        provider=get_env_value("LLM_PROVIDER", "").strip(),
        base_url=get_env_value("LLM_BASE_URL", "").strip().rstrip("/"),
        api_key=get_env_value("LLM_API_KEY", ""),
        model=get_env_value("LLM_MODEL", "").strip(),
        timeout_seconds=_bounded_env_float("LLM_TIMEOUT_SECONDS", 45.0, 1.0, 300.0),
        max_tokens=_bounded_env_int("LLM_MAX_TOKENS", 1600, 1, 32768),
        temperature=_bounded_env_float("LLM_TEMPERATURE", 0.2, 0.0, 2.0),
        max_retries=_bounded_env_int("LLM_MAX_RETRIES", 2, 0, 4),
        planner_thinking=_env_optional_thinking("LLM_PLANNER_THINKING"),
        answer_thinking=_env_optional_thinking("LLM_ANSWER_THINKING"),
    )


def get_repository_settings() -> RepositorySettings:
    return RepositorySettings(
        runtime_dir=_env_path("RUNTIME_DATA_DIR", BACKEND_ROOT / "data" / "runtime"),
        clone_timeout_seconds=_bounded_env_float(
            "GIT_CLONE_TIMEOUT_SECONDS", 120.0, 5.0, 600.0
        ),
        max_files=_bounded_env_int("REPOSITORY_MAX_FILES", 2000, 1, 10000),
        max_file_bytes=_bounded_env_int(
            "REPOSITORY_MAX_FILE_BYTES", 1_000_000, 1024, 10_000_000
        ),
        max_total_bytes=_bounded_env_int(
            "REPOSITORY_MAX_TOTAL_BYTES", 50_000_000, 1024, 500_000_000
        ),
        git_proxy=validate_git_proxy(get_env_value("REPONOESIS_GIT_PROXY", "")),
    )


def get_database_path() -> Path:
    """Return the normalized product database path for environment configuration."""
    return _env_path(
        "GITLEARN_DB", BACKEND_ROOT / "data" / "gitlearn.sqlite"
    ).resolve(strict=False)


def get_product_config_status() -> dict[str, Any]:
    """Return diagnostics that intentionally contain no credential value or metadata."""
    llm = get_llm_settings()
    embedding = get_embedding_settings()
    repository = get_repository_settings()
    embedding_missing: list[str] = []
    configured_embedding_provider = get_env_value("EMBEDDING_PROVIDER", "").strip()
    configured_embedding_model = (
        get_env_value("EMBEDDING_MODEL", "").strip()
        or get_env_value("EMBEDDING_MODEL_NAME_OR_PATH", "").strip()
    )
    configured_embedding_offline = get_env_value("EMBEDDING_OFFLINE", "").strip().lower()
    if not embedding.enabled:
        embedding_missing.append("EMBEDDING_ENABLED")
    if configured_embedding_provider != "local_bge_m3":
        embedding_missing.append("EMBEDDING_PROVIDER")
    if not configured_embedding_model:
        embedding_missing.append("EMBEDDING_MODEL")
    if configured_embedding_offline not in {"1", "true", "yes", "on"}:
        embedding_missing.append("EMBEDDING_OFFLINE")
    return {
        "git_proxy_configured": repository.git_proxy is not None,
        "configuration_file": str(PROJECT_ROOT / ".env"),
        "llm": {
            "provider": llm.provider or None,
            "model": llm.model or None,
            "base_url_configured": bool(llm.base_url),
            "api_key_configured": bool(llm.api_key),
            "planner_thinking": llm.planner_thinking or "omitted",
            "answer_thinking": llm.answer_thinking or "omitted",
            "ready": llm.configured,
            "missing": list(llm.missing),
        },
        "embedding": {
            "provider": embedding.provider,
            "model": _safe_model_label(embedding.model_name_or_path),
            "device": embedding.device,
            "offline": embedding.offline,
            "enabled": embedding.enabled,
            "ready": not embedding_missing,
            "missing": embedding_missing,
        },
        "runtime": {
            "database": str(get_database_path()),
            "data_dir": str(repository.runtime_dir),
            "clone_dir": str(repository.runtime_dir / "repositories"),
        },
    }


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
            "AGENT_TOOL_TIMEOUT_MS", 40_000, 100, 40_000
        ),
        min_final_answer_budget_ms=_bounded_env_int(
            "AGENT_FINAL_ANSWER_RESERVE_MS", 5_000, 100, 30_000
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
        default_relation_depth=_bounded_env_int(
            "AGENT_DEFAULT_RELATION_DEPTH", 1, 1, 2
        ),
        max_relation_depth=_bounded_env_int(
            "AGENT_MAX_RELATION_DEPTH", 2, 1, 2
        ),
        max_relation_seed_nodes=_bounded_env_int(
            "AGENT_MAX_RELATION_SEEDS", 8, 1, 8
        ),
        max_relation_neighbors_per_node=_bounded_env_int(
            "AGENT_MAX_RELATION_NEIGHBORS", 20, 1, 20
        ),
        max_relation_nodes=_bounded_env_int(
            "AGENT_MAX_RELATION_NODES", 64, 2, 64
        ),
        max_relation_edges=_bounded_env_int(
            "AGENT_MAX_RELATION_EDGES", 128, 1, 128
        ),
        max_relation_paths=_bounded_env_int(
            "AGENT_MAX_RELATION_PATHS", 24, 1, 24
        ),
        max_relation_observation_bytes=_bounded_env_int(
            "AGENT_MAX_RELATION_OBSERVATION_BYTES", 65_536, 1_024, 65_536
        ),
        max_relation_evidence_items=_bounded_env_int(
            "AGENT_MAX_RELATION_EVIDENCE", 16, 1, 16
        ),
        max_learning_state_items=_bounded_env_int(
            "AGENT_MAX_LEARNING_STATE_ITEMS", 16, 1, 16
        ),
        max_recent_learning_events=_bounded_env_int(
            "AGENT_MAX_RECENT_LEARNING_EVENTS", 8, 1, 8
        ),
        max_plan_steps_in_learning_context=_bounded_env_int(
            "AGENT_MAX_LEARNING_PLAN_STEPS", 12, 1, 12
        ),
        max_learning_context_bytes=_bounded_env_int(
            "AGENT_MAX_LEARNING_CONTEXT_BYTES", 16_384, 1_024, 16_384
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


def _env_float(key: str, default: float) -> float:
    value = get_env_value(key, str(default)).strip()
    try:
        return float(value)
    except ValueError:
        return default


def _env_optional_thinking(key: str) -> ThinkingMode | None:
    value = get_env_value(key, "").strip().lower()
    if not value:
        return None
    if value in {"enabled", "disabled"}:
        return cast(ThinkingMode, value)
    raise ValueError(f"{key} must be enabled, disabled, or empty.")


def _bounded_env_int(key: str, default: int, minimum: int, maximum: int) -> int:
    value = _env_int(key, default)
    if value < minimum or value > maximum:
        return default
    return value


def _bounded_env_float(
    key: str, default: float, minimum: float, maximum: float
) -> float:
    value = _env_float(key, default)
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


def _valid_provider_base_url(value: str) -> bool:
    if not value:
        return False
    try:
        parsed = urlsplit(value)
        return bool(
            parsed.scheme in {"http", "https"}
            and parsed.hostname
            and not parsed.username
            and not parsed.password
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        return False


def _safe_model_label(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or "\\" in value or value.startswith("."):
        return f"local:{path.name or 'embedding-model'}"
    return value
