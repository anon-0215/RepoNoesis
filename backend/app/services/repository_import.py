from __future__ import annotations

import hashlib
import ipaddress
import os
import shutil
import socket
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit, urlunsplit

from app.config import RepositorySettings
from app.models import RepoFile, RepositorySnapshot
from app.services.github_client import is_interesting_text_file


@dataclass(frozen=True)
class RepositoryImportError(RuntimeError):
    code: str
    message: str
    status_code: int = 400
    retryable: bool = False

    def __str__(self) -> str:
        return self.message

    def to_safe_dict(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message, "retryable": self.retryable}


@dataclass(frozen=True)
class ImportedRepository:
    snapshot: RepositorySnapshot
    source_identity: str
    checkout_path: Path


def import_repository(
    source_type: str, source: str, settings: RepositorySettings
) -> ImportedRepository:
    if source_type == "local":
        return import_local_repository(source, settings)
    if source_type == "git_url":
        return import_public_git_repository(source, settings)
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
    source: str, settings: RepositorySettings
) -> ImportedRepository:
    url = validate_public_https_git_url(source)
    clone_root = settings.runtime_dir / "repositories"
    clone_root.mkdir(parents=True, exist_ok=True)
    temporary = clone_root / f".import-{uuid.uuid4().hex}"
    env = _git_environment(isolated=True)
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
            shutil.rmtree(temporary)
        else:
            temporary.replace(stable)
        return _snapshot_checkout(stable, "git_url", url, settings)
    except subprocess.TimeoutExpired as exc:
        _remove_new_checkout(temporary, clone_root)
        raise RepositoryImportError(
            "git_clone_timeout", "The public Git repository clone timed out.", 504, True
        ) from exc
    except (subprocess.CalledProcessError, OSError) as exc:
        _remove_new_checkout(temporary, clone_root)
        raise RepositoryImportError(
            "git_clone_failed", "The public Git repository could not be cloned.", 502, True
        ) from exc


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


def _git_environment(*, isolated: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    env.update({"GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "Never"})
    if isolated:
        env.update(
            {
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": "NUL" if os.name == "nt" else "/dev/null",
            }
        )
    return env


def _remove_new_checkout(target: Path, clone_root: Path) -> None:
    try:
        target.resolve(strict=False).relative_to(clone_root.resolve(strict=True))
    except (OSError, ValueError):
        return
    if target.name.startswith(".import-") and target.exists():
        shutil.rmtree(target)
