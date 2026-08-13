from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RepoFile:
    path: str
    size: int
    content: str
    extension: str = ""
    language: str = "Text"
    summary: str = ""
    importance: float = 0.0
    is_core: bool = False
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RepositorySnapshot:
    repo_url: str
    owner: str
    repo: str
    default_branch: str
    files: list[RepoFile]
    repository_revision: str = ""
    source_type: str = "legacy_github"
    source_location: str = ""
    source_identity: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_url": self.repo_url,
            "owner": self.owner,
            "repo": self.repo,
            "default_branch": self.default_branch,
            "repository_revision": self.repository_revision,
            "source_type": self.source_type,
            "source_location": self.source_location,
            "source_identity": self.source_identity,
            "files": [file.to_dict() for file in self.files],
        }

