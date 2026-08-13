from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import PurePosixPath
from typing import Any

from app.config import get_env_value
from app.models import RepoFile, RepositorySnapshot


GITHUB_API = "https://api.github.com"
MAX_FILE_BYTES = 200_000
MAX_TEXT_FILES = 45
MAX_FETCH_WORKERS = 8

SKIP_PARTS = {
    ".git",
    ".github",
    ".idea",
    ".mypy_cache",
    ".next",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "node_modules",
    "target",
    "vendor",
}

TEXT_EXTENSIONS = {
    ".cjs",
    ".css",
    ".env",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".mdx",
    ".py",
    ".rs",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".vue",
    ".yaml",
    ".yml",
}

IMPORTANT_FILENAMES = {
    "Dockerfile",
    "Makefile",
    "README",
    "LICENSE",
    "requirements.txt",
    "package.json",
    "pyproject.toml",
    "vite.config.ts",
    "vite.config.js",
}

PRIORITY_ROOT_FILES = {
    ".gitignore",
    "Dockerfile",
    "LICENSE",
    "Makefile",
    "README.md",
    "README.rst",
    "README.txt",
    "package.json",
    "pnpm-lock.yaml",
    "poetry.lock",
    "pyproject.toml",
    "requirements.txt",
    "setup.cfg",
    "setup.py",
    "tsconfig.json",
    "vite.config.js",
    "vite.config.ts",
    "yarn.lock",
}

PRIORITY_DIRS = {
    "app",
    "backend",
    "frontend",
    "lib",
    "packages",
    "server",
    "src",
    "tests",
}


def parse_github_url(url: str) -> tuple[str, str]:
    cleaned = url.strip()
    match = re.match(r"^(?:https?://)?github\.com/([^/\s]+)/([^/\s#?]+)", cleaned)
    if not match:
        raise ValueError("请输入有效的 GitHub 仓库地址，例如 https://github.com/owner/repo")
    owner = match.group(1)
    repo = match.group(2).removesuffix(".git")
    if not owner or not repo:
        raise ValueError("GitHub 仓库地址缺少 owner 或 repo")
    return owner, repo


def should_skip_path(path: str) -> bool:
    parts = [part for part in PurePosixPath(path).parts if part]
    lowered = {part.lower() for part in parts}
    if lowered.intersection(SKIP_PARTS):
        return True
    return any(part.startswith(".") and part not in {".env"} for part in parts)


def is_interesting_text_file(path: str, size: int) -> bool:
    if size > MAX_FILE_BYTES or should_skip_path(path):
        return False
    name = PurePosixPath(path).name
    suffix = PurePosixPath(path).suffix.lower()
    if suffix in {".md", ".mdx"} and not name.lower().startswith("readme"):
        return False
    return suffix in TEXT_EXTENSIONS or name in IMPORTANT_FILENAMES


def fetch_repository(repo_url: str) -> RepositorySnapshot:
    owner, repo = parse_github_url(repo_url)
    metadata = _fetch_json(f"{GITHUB_API}/repos/{owner}/{repo}")
    default_branch = metadata.get("default_branch") or "main"
    branch = _fetch_json(
        f"{GITHUB_API}/repos/{owner}/{repo}/branches/{urllib.parse.quote(default_branch, safe='')}"
    )
    repository_revision = branch.get("commit", {}).get("sha", "")
    tree_sha = branch.get("commit", {}).get("commit", {}).get("tree", {}).get("sha") or default_branch
    tree_url = f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{tree_sha}?recursive=1"
    tree = _fetch_json(tree_url)
    files: list[RepoFile] = []

    candidates = [
        item
        for item in tree.get("tree", [])
        if item.get("type") == "blob"
        and is_interesting_text_file(item.get("path", ""), int(item.get("size") or 0))
    ]
    candidates.sort(key=lambda item: _candidate_priority(item.get("path", "")))

    selected_candidates = candidates[:MAX_TEXT_FILES]
    with ThreadPoolExecutor(max_workers=MAX_FETCH_WORKERS) as executor:
        futures = {
            executor.submit(_fetch_candidate_file, owner, repo, item): item
            for item in selected_candidates
        }
        for future in as_completed(futures):
            repo_file = future.result()
            if repo_file is not None:
                files.append(repo_file)

    files.sort(key=lambda file: _candidate_priority(file.path))

    if not files:
        raise RuntimeError("没有找到可分析的文本文件，仓库可能过大、为空或不是公开仓库。")

    return RepositorySnapshot(
        repo_url=f"https://github.com/{owner}/{repo}",
        owner=owner,
        repo=repo,
        default_branch=default_branch,
        files=files,
        repository_revision=repository_revision,
    )


def _request(url: str) -> bytes:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "RepoNoesis/0.1",
    }
    token = get_env_value("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="ignore")
        if exc.code == 401 and get_env_value("GITHUB_TOKEN"):
            raise RuntimeError(
                "GitHub Token 无效或已过期。请重新生成 token，并确认项目根目录 .env "
                "中的 GITHUB_TOKEN 只包含 token 本身，不要带引号、空格或注释。"
            ) from exc
        if exc.code == 403 and "rate limit" in message.lower() and not get_env_value("GITHUB_TOKEN"):
            raise RuntimeError(
                "GitHub 匿名 API 已达到限流。请在项目根目录 .env 中设置 "
                "GITHUB_TOKEN，然后重启后端。"
            ) from exc
        raise RuntimeError(f"GitHub 请求失败 {exc.code}: {message[:300]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"无法连接 GitHub: {exc.reason}") from exc


def _fetch_json(url: str) -> dict[str, Any]:
    return json.loads(_request(url).decode("utf-8"))


def _fetch_blob_file(owner: str, repo: str, sha: str) -> str | None:
    if not sha:
        return None
    try:
        blob = _fetch_json(f"{GITHUB_API}/repos/{owner}/{repo}/git/blobs/{sha}")
    except RuntimeError:
        return None
    if blob.get("encoding") != "base64":
        return None
    raw_content = blob.get("content", "")
    try:
        data = base64.b64decode(raw_content, validate=False)
    except ValueError:
        return None
    if b"\x00" in data[:2000]:
        return None
    return data.decode("utf-8", errors="replace")


def _fetch_candidate_file(owner: str, repo: str, item: dict[str, Any]) -> RepoFile | None:
    path = item.get("path", "")
    size = int(item.get("size") or 0)
    if not path or not is_interesting_text_file(path, size):
        return None
    content = _fetch_blob_file(owner, repo, item.get("sha", ""))
    if content is None:
        return None
    return RepoFile(
        path=path,
        size=size,
        content=content,
        extension=PurePosixPath(path).suffix.lower(),
    )


def _candidate_priority(path: str) -> tuple[int, int, str]:
    pure_path = PurePosixPath(path)
    name = pure_path.name
    parts = pure_path.parts
    suffix = pure_path.suffix.lower()
    lower_path = path.lower()

    score = 100
    if name in PRIORITY_ROOT_FILES or name.lower().startswith("readme"):
        score = min(score, 0)
    if parts and parts[0] in PRIORITY_DIRS:
        score = min(score, 12)
    if name.lower() in {"main.py", "app.py", "server.py", "index.js", "main.tsx", "app.tsx"}:
        score = min(score, 4)
    if suffix in {".py", ".js", ".jsx", ".ts", ".tsx"}:
        score = min(score, 20)
    if "test" in lower_path or "spec" in lower_path:
        score += 20
    return (score, len(parts), path)


def _fetch_raw_file(owner: str, repo: str, branch: str, path: str) -> str | None:
    quoted_path = urllib.parse.quote(path, safe="/")
    quoted_branch = urllib.parse.quote(branch, safe="")
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{quoted_branch}/{quoted_path}"
    try:
        data = _request(url)
    except RuntimeError:
        return None
    if b"\x00" in data[:2000]:
        return None
    return data.decode("utf-8", errors="replace")
