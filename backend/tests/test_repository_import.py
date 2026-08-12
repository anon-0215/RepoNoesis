from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import RepositorySettings
from app.services.repository_import import (
    RepositoryImportError,
    classify_git_clone_failure,
    import_public_git_repository,
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

    def test_git_clone_classifier_uses_only_narrow_stable_codes(self):
        cases = (
            ("fatal: unable to access: Could not resolve host: example.invalid", "git_dns_failed", True),
            ("fatal: SSL certificate problem: unable to get local issuer certificate", "git_tls_failed", False),
            ("error: RPC failed; curl 56 Recv failure: Connection was reset", "git_connection_failed", True),
            ("remote: Repository not found.", "git_remote_not_found", False),
            ("fatal: Authentication failed for a redacted endpoint", "git_authentication_required", False),
            ("fatal: localized or otherwise unknown failure", "git_clone_failed", True),
        )
        for stderr, code, retryable in cases:
            with self.subTest(code=code):
                failure = classify_git_clone_failure(
                    stderr, exit_code=128, elapsed_ms=1250
                )
                self.assertEqual(failure.stable_code, code)
                self.assertEqual(failure.retryable, retryable)
                self.assertEqual(failure.safe_stage, "clone")
                self.assertEqual(failure.exit_code, 128)
                self.assertEqual(failure.elapsed_ms, 1250)

    def test_git_clone_classifier_distinguishes_timeout_and_missing_executable(self):
        timeout = classify_git_clone_failure(
            "ignored", exit_code=None, elapsed_ms=5000, timed_out=True
        )
        missing = classify_git_clone_failure(
            "ignored", exit_code=None, elapsed_ms=1, executable_unavailable=True
        )
        self.assertEqual(timeout.stable_code, "git_clone_timeout")
        self.assertEqual(timeout.retryable, True)
        self.assertEqual(missing.stable_code, "git_executable_unavailable")
        self.assertEqual(missing.retryable, False)

    def test_git_clone_classifier_bounds_exit_code_and_elapsed_time(self):
        low = classify_git_clone_failure("", exit_code=True, elapsed_ms=-10)
        high = classify_git_clone_failure("", exit_code=10**30, elapsed_ms=10**30)
        invalid = classify_git_clone_failure("", exit_code="128", elapsed_ms=float("nan"))
        infinite = classify_git_clone_failure("", exit_code=None, elapsed_ms=float("inf"))
        self.assertIsNone(low.exit_code)
        self.assertEqual(low.elapsed_ms, 0)
        self.assertIsNone(high.exit_code)
        self.assertEqual(high.elapsed_ms, 86_400_000)
        self.assertIsNone(invalid.exit_code)
        self.assertEqual(invalid.elapsed_ms, 0)
        self.assertEqual(infinite.elapsed_ms, 0)

    def test_clone_failure_does_not_expose_raw_output_url_or_local_path(self):
        marker = "PRIVATE_TOKEN_MARKER"
        raw = (
            f"fatal: unknown failure for https://user:{marker}@example.invalid/repo.git "
            f"at C:\\Users\\Private\\runtime\\repo"
        ).encode()
        called = subprocess.CalledProcessError(128, ["git", "clone"], stderr=raw)
        with tempfile.TemporaryDirectory() as temporary:
            settings = RepositorySettings(runtime_dir=Path(temporary) / "runtime")
            with (
                patch(
                    "app.services.repository_import.validate_public_https_git_url",
                    return_value="https://example.invalid/repo.git",
                ),
                patch("app.services.repository_import.subprocess.run", side_effect=called),
                self.assertLogs("app.services.repository_import", level="WARNING") as captured,
                self.assertRaises(RepositoryImportError) as raised,
            ):
                import_public_git_repository(
                    "https://example.invalid/repo.git",
                    settings,
                    request_id="request-safe-1",
                )
        safe = raised.exception.to_safe_dict(request_id="request-safe-1")
        serialized = repr(safe) + "\n".join(captured.output)
        self.assertEqual(raised.exception.code, "git_clone_failed")
        self.assertEqual(raised.exception.status_code, 502)
        self.assertNotIn(marker, serialized)
        self.assertNotIn("https://", serialized)
        self.assertNotIn("C:\\Users", serialized)
        self.assertEqual(safe["safe_stage"], "clone")
        self.assertEqual(safe["exit_code"], 128)


if __name__ == "__main__":
    unittest.main()
