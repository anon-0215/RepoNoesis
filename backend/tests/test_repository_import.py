from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from app.config import RepositorySettings
from app.services.repository_import import (
    RepositoryImportError,
    import_local_repository,
    validate_public_https_git_url,
)


class RepositoryImportTests(unittest.TestCase):
    def _repository(self, root: Path) -> Path:
        repo = root / "sample"
        repo.mkdir()
        subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
        (repo / "app.py").write_text("def answer():\n    return 42\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "add", "app.py"], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-m", "fixture"], check=True, capture_output=True)
        return repo

    def test_local_import_is_revisioned_and_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = self._repository(root)
            before = (repo / "app.py").read_bytes()
            result = import_local_repository(str(repo), RepositorySettings(runtime_dir=root / "runtime"))
            after = (repo / "app.py").read_bytes()
        self.assertEqual(before, after)
        self.assertEqual(result.snapshot.source_type, "local")
        self.assertEqual(len(result.snapshot.repository_revision), 40)
        self.assertTrue(result.source_identity.startswith("source-sha256:"))

    def test_dirty_local_repository_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = self._repository(root)
            (repo / "app.py").write_text("changed\n", encoding="utf-8")
            with self.assertRaises(RepositoryImportError) as raised:
                import_local_repository(str(repo), RepositorySettings(runtime_dir=root / "runtime"))
        self.assertEqual(raised.exception.code, "local_repository_dirty")

    def test_remote_url_rejects_credentials_and_private_hosts(self):
        for url in (
            "ssh://github.com/org/repo.git",
            "https://user:password@github.com/org/repo.git",
            "https://127.0.0.1/repo.git",
            "file:///tmp/repo",
        ):
            with self.subTest(url=url), self.assertRaises(RepositoryImportError):
                validate_public_https_git_url(url)


if __name__ == "__main__":
    unittest.main()
