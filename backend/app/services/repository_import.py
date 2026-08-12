from __future__ import annotations

import hashlib
import ipaddress
import logging
import math
import os
import re
import shutil
import socket
import stat
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit, urlunsplit

from app.config import RepositorySettings, validate_git_proxy
from app.models import RepoFile, RepositorySnapshot
from app.services.github_client import is_interesting_text_file


logger = logging.getLogger(__name__)
_MAX_SAFE_ELAPSED_MS = 86_400_000
_MAX_SAFE_EXIT_CODE = 2_147_483_647
_IMPORT_DIRECTORY_PATTERN = re.compile(r"^\.import-[0-9a-f]{32}$")
_PROXY_ENVIRONMENT_NAMES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


@dataclass(frozen=True)
class GitCloneFailure:
    stable_code: str
    retryable: bool
    safe_stage: str
    exit_code: int | None
    elapsed_ms: int

    def to_safe_dict(self) -> dict[str, object]:
        return {
            "stable_code": self.stable_code,
            "retryable": self.retryable,
            "safe_stage": self.safe_stage,
            "exit_code": self.exit_code,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass(frozen=True)
class RepositoryImportError(RuntimeError):
    code: str
    message: str
    status_code: int = 400
    retryable: bool = False
    safe_stage: str | None = None
    exit_code: int | None = None
    elapsed_ms: int | None = None
    cleanup_pending: bool = False

    def __str__(self) -> str:
        return self.message

    def to_safe_dict(self, *, request_id: str | None = None) -> dict[str, object]:
        result: dict[str, object] = {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "cleanup_pending": self.cleanup_pending,
        }
        if request_id is not None:
            result["request_id"] = request_id
        if self.safe_stage is not None:
            result.update(
                {
                    "safe_stage": self.safe_stage,
                    "exit_code": self.exit_code,
                    "elapsed_ms": self.elapsed_ms,
                }
            )
        return result


@dataclass(frozen=True)
class ImportedRepository:
    snapshot: RepositorySnapshot
    source_identity: str
    checkout_path: Path


@dataclass(frozen=True)
class CheckoutCleanupResult:
    cleanup_pending: bool
    status: str


def import_repository(
    source_type: str,
    source: str,
    settings: RepositorySettings,
    *,
    request_id: str | None = None,
) -> ImportedRepository:
    if source_type == "local":
        return import_local_repository(source, settings)
    if source_type == "git_url":
        return import_public_git_repository(source, settings, request_id=request_id)
    raise RepositoryImportError(
        "unsupported_source_type", "source_type must be local or git_url."
    )


def import_local_repository(
    source: str, settings: RepositorySettings
) -> ImportedRepository:
    raw = source.strip()
    if not raw:
        raise RepositoryImportError("local_path_missing", "A local repository path is required.")
    try:
        root = Path(raw).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RepositoryImportError(
            "local_path_not_found", "The selected local repository directory does not exist."
        ) from exc
    if not root.is_dir():
        raise RepositoryImportError(
            "local_path_not_directory", "The selected local repository path is not a directory."
        )
    top = _git(root, "rev-parse", "--show-toplevel").strip()
    if not top or Path(top).resolve() != root:
        raise RepositoryImportError(
            "local_repository_root_required",
            "Select the root directory of a Git repository.",
        )
    if _git(root, "status", "--porcelain=v1", "--untracked-files=normal"):
        raise RepositoryImportError(
            "local_repository_dirty",
            "The local repository has uncommitted or untracked changes. Commit or remove them before import so citations match the recorded revision.",
        )
    return _snapshot_checkout(root, "local", str(root), settings)


def validate_public_https_git_url(url: str) -> str:
    raw = url.strip()
    try:
        parsed = urlsplit(raw)
    except ValueError as exc:
        raise RepositoryImportError("git_url_invalid", "The Git URL is invalid.") from exc
    if parsed.scheme.lower() != "https":
        raise RepositoryImportError(
            "git_url_scheme_rejected", "Only public HTTPS Git URLs are supported."
        )
    if not parsed.hostname or parsed.username or parsed.password:
        raise RepositoryImportError(
            "git_url_credentials_rejected",
            "Git URLs must not contain credentials and must include a public host.",
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise RepositoryImportError("git_url_invalid", "The Git URL port is invalid.") from exc
    if parsed.query or parsed.fragment or port not in {None, 443}:
        raise RepositoryImportError(
            "git_url_components_rejected",
            "Git URLs must not contain a query, fragment, or non-HTTPS port.",
        )
    host = parsed.hostname.rstrip(".").lower()
    if host == "localhost" or host.endswith(".localhost"):
        raise RepositoryImportError("git_url_private_host", "Private Git hosts are not supported.")
    addresses: set[str] = set()
    try:
        addresses.add(str(ipaddress.ip_address(host)))
    except ValueError:
        try:
            addresses.update(
                item[4][0] for item in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
            )
        except OSError as exc:
            raise RepositoryImportError(
                "git_url_host_unresolved", "The public Git host could not be resolved."
            ) from exc
    if not addresses or any(not ipaddress.ip_address(value).is_global for value in addresses):
        raise RepositoryImportError("git_url_private_host", "Private Git hosts are not supported.")
    path = parsed.path.rstrip("/")
    if not path or path == "/":
        raise RepositoryImportError("git_url_path_missing", "The Git URL has no repository path.")
    return urlunsplit(("https", host, path, "", ""))


def import_public_git_repository(
    source: str,
    settings: RepositorySettings,
    *,
    request_id: str | None = None,
) -> ImportedRepository:
    url = validate_public_https_git_url(source)
    clone_root = settings.runtime_dir / "repositories"
    clone_root.mkdir(parents=True, exist_ok=True)
    temporary = clone_root / f".import-{uuid.uuid4().hex}"
    env = _git_environment(isolated=True, git_proxy=settings.git_proxy)
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    command = [
        "git",
        "-c",
        "protocol.file.allow=never",
        "-c",
        "http.followRedirects=false",
        "-c",
        "core.hooksPath=NUL" if os.name == "nt" else "core.hooksPath=/dev/null",
        "clone",
        "--depth",
        "1",
        "--single-branch",
        "--no-tags",
        "--no-recurse-submodules",
        "--",
        url,
        str(temporary),
    ]
    started = time.monotonic()
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            timeout=settings.clone_timeout_seconds,
            env=env,
        )
        revision = _git(temporary, "rev-parse", "HEAD").strip()
        stable = clone_root / f"{hashlib.sha256(url.encode()).hexdigest()[:16]}-{revision[:12]}"
        if stable.exists():
            cleanup = _remove_new_checkout(temporary, clone_root)
            if cleanup.cleanup_pending:
                raise RepositoryImportError(
                    "git_cleanup_failed",
                    "The temporary Git checkout could not be safely removed.",
                    500,
                    True,
                    "cleanup",
                    None,
                    _elapsed_ms(started),
                    True,
                )
        else:
            temporary.replace(stable)
        return _snapshot_checkout(stable, "git_url", url, settings)
    except subprocess.TimeoutExpired as exc:
        failure = classify_git_clone_failure(
            b"",
            exit_code=None,
            elapsed_ms=_elapsed_ms(started),
            timed_out=True,
        )
        cleanup = _remove_new_checkout(temporary, clone_root)
        _log_git_clone_failure(failure, request_id, cleanup.cleanup_pending)
        raise _repository_error_for_git_failure(
            failure, cleanup_pending=cleanup.cleanup_pending
        ) from exc
    except subprocess.CalledProcessError as exc:
        failure = classify_git_clone_failure(
            exc.stderr,
            exit_code=exc.returncode,
            elapsed_ms=_elapsed_ms(started),
        )
        cleanup = _remove_new_checkout(temporary, clone_root)
        _log_git_clone_failure(failure, request_id, cleanup.cleanup_pending)
        raise _repository_error_for_git_failure(
            failure, cleanup_pending=cleanup.cleanup_pending
        ) from exc
    except OSError as exc:
        failure = classify_git_clone_failure(
            b"",
            exit_code=None,
            elapsed_ms=_elapsed_ms(started),
            executable_unavailable=isinstance(exc, FileNotFoundError),
        )
        cleanup = _remove_new_checkout(temporary, clone_root)
        _log_git_clone_failure(failure, request_id, cleanup.cleanup_pending)
        raise _repository_error_for_git_failure(
            failure, cleanup_pending=cleanup.cleanup_pending
        ) from exc


def classify_git_clone_failure(
    stderr: bytes | str | None,
    *,
    exit_code: object,
    elapsed_ms: object,
    timed_out: bool = False,
    executable_unavailable: bool = False,
) -> GitCloneFailure:
    """Project one clone failure into a bounded contract; raw output is never retained."""
    safe_exit = (
        exit_code
        if isinstance(exit_code, int)
        and not isinstance(exit_code, bool)
        and -_MAX_SAFE_EXIT_CODE <= exit_code <= _MAX_SAFE_EXIT_CODE
        else None
    )
    safe_elapsed = _bounded_elapsed_ms(elapsed_ms)
    if executable_unavailable:
        return GitCloneFailure(
            "git_executable_unavailable", False, "clone", safe_exit, safe_elapsed
        )
    if timed_out:
        return GitCloneFailure("git_clone_timeout", True, "clone", safe_exit, safe_elapsed)

    if isinstance(stderr, bytes):
        text = stderr.decode("utf-8", errors="replace")
    elif isinstance(stderr, str):
        text = stderr
    else:
        text = ""
    lowered = text.casefold()
    patterns: tuple[tuple[str, tuple[str, ...], bool], ...] = (
        (
            "git_dns_failed",
            (
                "could not resolve host",
                "unable to resolve host",
                "name or service not known",
                "temporary failure in name resolution",
                "no such host is known",
            ),
            True,
        ),
        (
            "git_tls_failed",
            (
                "ssl certificate problem",
                "certificate verify failed",
                "tls handshake",
                "unable to get local issuer certificate",
                "schannel: next initializesecuritycontext failed",
            ),
            False,
        ),
        (
            "git_connection_failed",
            (
                "connection was reset",
                "connection reset by peer",
                "failed to connect",
                "could not connect",
                "connection refused",
                "connection timed out",
                "remote end hung up unexpectedly",
                "unexpected disconnect",
                "early eof",
                "http/2 stream 0 was not closed cleanly",
                "rpc failed; curl 56",
                "rpc failed; curl 92",
            ),
            True,
        ),
        (
            "git_remote_not_found",
            (
                "repository not found",
                "remote: not found",
                "does not appear to be a git repository",
            ),
            False,
        ),
        (
            "git_authentication_required",
            (
                "authentication failed",
                "could not read username",
                "terminal prompts disabled",
                "authentication required",
                "http basic: access denied",
            ),
            False,
        ),
    )
    for stable_code, needles, retryable in patterns:
        if any(needle in lowered for needle in needles):
            return GitCloneFailure(stable_code, retryable, "clone", safe_exit, safe_elapsed)
    if "http/2 stream " in lowered and " was not closed cleanly" in lowered:
        return GitCloneFailure(
            "git_connection_failed", True, "clone", safe_exit, safe_elapsed
        )
    return GitCloneFailure("git_clone_failed", True, "clone", safe_exit, safe_elapsed)


def _bounded_elapsed_ms(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    if not math.isfinite(value):
        return 0
    return max(0, min(int(value), _MAX_SAFE_ELAPSED_MS))


def _elapsed_ms(started: float) -> int:
    return _bounded_elapsed_ms((time.monotonic() - started) * 1000)


def _repository_error_for_git_failure(
    failure: GitCloneFailure, *, cleanup_pending: bool = False
) -> RepositoryImportError:
    status_and_message = {
        "git_executable_unavailable": (
            503,
            "The Git executable is unavailable on the backend host.",
        ),
        "git_dns_failed": (502, "The public Git host could not be resolved."),
        "git_tls_failed": (502, "The public Git TLS connection could not be verified."),
        "git_connection_failed": (502, "The public Git connection was interrupted."),
        "git_remote_not_found": (404, "The public Git repository was not found."),
        "git_authentication_required": (
            401,
            "The Git repository requires authentication and cannot be imported as public.",
        ),
        "git_clone_timeout": (504, "The public Git repository clone timed out."),
        "git_clone_failed": (502, "The public Git repository could not be cloned."),
    }
    status_code, message = status_and_message[failure.stable_code]
    return RepositoryImportError(
        failure.stable_code,
        message,
        status_code,
        failure.retryable,
        failure.safe_stage,
        failure.exit_code,
        failure.elapsed_ms,
        cleanup_pending,
    )


def _log_git_clone_failure(
    failure: GitCloneFailure,
    request_id: str | None,
    cleanup_pending: bool = False,
) -> None:
    safe = failure.to_safe_dict()
    safe["cleanup_pending"] = cleanup_pending
    if request_id is not None:
        safe["request_id"] = request_id
    logger.warning("Public Git clone failed.", extra={"repository_import": safe})


def _snapshot_checkout(
    root: Path, source_type: str, source_location: str, settings: RepositorySettings
) -> ImportedRepository:
    revision = _git(root, "rev-parse", "HEAD").strip()
    if len(revision) != 40:
        raise RepositoryImportError("repository_revision_missing", "The Git revision is unavailable.")
    branch = _git(root, "symbolic-ref", "--short", "HEAD", allow_failure=True).strip() or "detached"
    files = _read_tracked_files(root, settings)
    if not any(file.path.lower().endswith(".py") for file in files):
        raise RepositoryImportError(
            "python_source_missing", "No supported tracked Python source files were found."
        )
    normalized = source_location.casefold() if source_type == "local" else source_location
    digest = hashlib.sha256(
        f"{source_type}\0{normalized}\0{revision}".encode("utf-8")
    ).hexdigest()
    identity = f"source-sha256:{digest}"
    repo = root.name.removesuffix(".git") or "repository"
    snapshot = RepositorySnapshot(
        repo_url=source_location,
        owner="local" if source_type == "local" else (urlsplit(source_location).hostname or "public"),
        repo=repo,
        default_branch=branch,
        files=files,
        repository_revision=revision,
        source_type=source_type,
        source_location=source_location,
        source_identity=identity,
    )
    return ImportedRepository(snapshot, identity, root)


def _read_tracked_files(root: Path, settings: RepositorySettings) -> list[RepoFile]:
    raw = _git_bytes(root, "ls-files", "-z")
    paths = [value.decode("utf-8", errors="surrogateescape") for value in raw.split(b"\0") if value]
    files: list[RepoFile] = []
    total = 0
    for relative in paths:
        normalized = relative.replace("\\", "/")
        pure = PurePosixPath(normalized)
        raw_candidate = root / Path(*pure.parts)
        if raw_candidate.is_symlink():
            continue
        candidate = raw_candidate.resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if not candidate.is_file():
            continue
        size = candidate.stat().st_size
        if not is_interesting_text_file(normalized, min(size, 200_000)):
            continue
        if size > settings.max_file_bytes or total + size > settings.max_total_bytes:
            continue
        data = candidate.read_bytes()
        if b"\0" in data[:2000]:
            continue
        files.append(
            RepoFile(
                path=normalized,
                size=len(data),
                content=data.decode("utf-8", errors="replace"),
                extension=pure.suffix.lower(),
            )
        )
        total += len(data)
        if len(files) >= settings.max_files:
            break
    return files


def _git(root: Path, *arguments: str, allow_failure: bool = False) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=not allow_failure,
        capture_output=True,
        timeout=30,
        env=_git_environment(),
    )
    return result.stdout.decode("utf-8", errors="replace").strip()


def _git_bytes(root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        timeout=30,
        env=_git_environment(),
    ).stdout


def _git_environment(
    *, isolated: bool = False, git_proxy: str | None = None
) -> dict[str, str]:
    env = os.environ.copy()
    env.update({"GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "Never"})
    if isolated:
        for name in _PROXY_ENVIRONMENT_NAMES:
            env.pop(name, None)
        env.update(
            {
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": "NUL" if os.name == "nt" else "/dev/null",
            }
        )
        validated_proxy = validate_git_proxy(git_proxy)
        if validated_proxy is not None:
            env["HTTP_PROXY"] = validated_proxy
            env["HTTPS_PROXY"] = validated_proxy
    return env


def _remove_new_checkout(target: Path, clone_root: Path) -> CheckoutCleanupResult:
    status = _validate_cleanup_target(target, clone_root)
    if status == "target_absent":
        return CheckoutCleanupResult(False, status)
    if status != "eligible":
        logger.warning(
            "Temporary checkout cleanup rejected.",
            extra={"repository_cleanup": {"status": status, "cleanup_pending": True}},
        )
        return CheckoutCleanupResult(True, status)
    try:
        shutil.rmtree(target)
        return CheckoutCleanupResult(False, "removed")
    except FileNotFoundError:
        return CheckoutCleanupResult(False, "target_absent")
    except OSError:
        pass

    retry_status = _validate_cleanup_target(target, clone_root)
    if retry_status == "target_absent":
        return CheckoutCleanupResult(False, retry_status)
    if retry_status != "eligible":
        logger.warning(
            "Temporary checkout cleanup retry rejected.",
            extra={
                "repository_cleanup": {
                    "status": retry_status,
                    "cleanup_pending": True,
                }
            },
        )
        return CheckoutCleanupResult(True, retry_status)
    try:
        shutil.rmtree(target, onerror=_readonly_cleanup_handler(target))
        return CheckoutCleanupResult(False, "removed_after_retry")
    except FileNotFoundError:
        return CheckoutCleanupResult(False, "target_absent")
    except Exception:
        logger.warning(
            "Temporary checkout cleanup remains pending.",
            extra={
                "repository_cleanup": {
                    "status": "delete_failed",
                    "cleanup_pending": True,
                }
            },
        )
        return CheckoutCleanupResult(True, "delete_failed")


def _validate_cleanup_target(target: Path, clone_root: Path) -> str:
    if not _IMPORT_DIRECTORY_PATTERN.fullmatch(target.name):
        return "name_rejected"
    try:
        lexical_target = Path(os.path.abspath(os.fspath(target)))
        lexical_root = Path(os.path.abspath(os.fspath(clone_root)))
        if not _strictly_within(lexical_target, lexical_root):
            return "lexical_boundary_rejected"
        resolved_root = clone_root.resolve(strict=True)
        resolved_target = target.resolve(strict=False)
        if not _strictly_within(resolved_target, resolved_root):
            return "resolved_boundary_rejected"
        metadata = target.lstat()
    except FileNotFoundError:
        return "target_absent"
    except OSError:
        return "metadata_unavailable"
    if stat.S_ISLNK(metadata.st_mode):
        return "link_rejected"
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if getattr(metadata, "st_file_attributes", 0) & reparse_flag:
        return "reparse_point_rejected"
    if not stat.S_ISDIR(metadata.st_mode):
        return "not_directory"
    return "eligible"


def _strictly_within(target: Path, root: Path) -> bool:
    normalized_target = os.path.normcase(os.path.abspath(os.fspath(target)))
    normalized_root = os.path.normcase(os.path.abspath(os.fspath(root)))
    try:
        return (
            normalized_target != normalized_root
            and os.path.commonpath((normalized_target, normalized_root)) == normalized_root
        )
    except ValueError:
        return False


def _readonly_cleanup_handler(root: Path):
    def handle(function, path: str, exc_info) -> None:
        error = exc_info[1]
        candidate = Path(path)
        if not isinstance(error, PermissionError) or not _within_or_equal(candidate, root):
            raise error
        try:
            metadata = candidate.lstat()
        except OSError:
            raise error
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if stat.S_ISLNK(metadata.st_mode) or (
            getattr(metadata, "st_file_attributes", 0) & reparse_flag
        ):
            raise error
        candidate.chmod(metadata.st_mode | stat.S_IWRITE)
        function(path)

    return handle


def _within_or_equal(target: Path, root: Path) -> bool:
    normalized_target = os.path.normcase(os.path.abspath(os.fspath(target)))
    normalized_root = os.path.normcase(os.path.abspath(os.fspath(root)))
    try:
        return os.path.commonpath((normalized_target, normalized_root)) == normalized_root
    except ValueError:
        return False
