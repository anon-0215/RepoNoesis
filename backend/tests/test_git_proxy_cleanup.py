from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.config import RepositorySettings
from app.services.repository_import import (
    CheckoutCleanupResult,
    RepositoryImportError,
    _git_environment,
    _remove_new_checkout,
    _validate_cleanup_target,
    import_public_git_repository,
)


class GitProxyContractTests(unittest.TestCase):
    def test_unconfigured_clone_environment_removes_ambient_proxies(self):
        ambient = {
            "HTTP_PROXY": "http://ambient.invalid",
            "HTTPS_PROXY": "http://ambient.invalid",
            "ALL_PROXY": "socks5://ambient.invalid",
            "http_proxy": "http://ambient.invalid",
            "https_proxy": "http://ambient.invalid",
            "all_proxy": "socks5://ambient.invalid",
        }
        with patch.dict(os.environ, ambient, clear=False):
            env = _git_environment(isolated=True)
        for name in ambient:
            self.assertNotIn(name, env)
        self.assertEqual(env["GIT_CONFIG_NOSYSTEM"], "1")
        self.assertIn(env["GIT_CONFIG_GLOBAL"], {"NUL", "/dev/null"})
        self.assertNotIn("GIT_SSL_NO_VERIFY", env)

    def test_http_and_https_proxy_enter_only_clone_environment(self):
        for scheme in ("http", "https"):
            marker = f"proxy-secret-{scheme}"
            proxy = f"{scheme}://user:{marker}@proxy.example.invalid:8443"
            observed: dict[str, object] = {}

            def fail_clone(command, **kwargs):
                observed["command"] = command
                observed["env"] = kwargs["env"]
                raise subprocess.CalledProcessError(128, command, stderr=b"unknown")

            with tempfile.TemporaryDirectory() as temporary:
                settings = RepositorySettings(Path(temporary), git_proxy=proxy)
                with (
                    patch(
                        "app.services.repository_import.validate_public_https_git_url",
                        return_value="https://public.example.invalid/repository.git",
                    ),
                    patch("app.services.repository_import.subprocess.run", side_effect=fail_clone),
                    self.assertLogs("app.services.repository_import", level="WARNING") as captured,
                    self.assertRaises(RepositoryImportError) as raised,
                ):
                    import_public_git_repository("ignored", settings, request_id="safe-request")

            env = observed["env"]
            self.assertIsInstance(env, dict)
            self.assertEqual(env["HTTP_PROXY"], proxy)
            self.assertEqual(env["HTTPS_PROXY"], proxy)
            serialized = repr(observed["command"]) + repr(raised.exception.to_safe_dict()) + "".join(captured.output)
            self.assertNotIn(marker, serialized)
            self.assertNotIn("proxy.example.invalid", serialized)
            self.assertNotIn("sslVerify=false", serialized)


class WindowsCheckoutCleanupTests(unittest.TestCase):
    def _target(self, root: Path, suffix: str = "a" * 32) -> Path:
        clone_root = root / "repositories"
        clone_root.mkdir()
        target = clone_root / f".import-{suffix}"
        target.mkdir()
        return target

    def test_normal_directory_is_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = self._target(root)
            (target / "object.tmp").write_text("x", encoding="utf-8")
            result = _remove_new_checkout(target, target.parent)
            self.assertFalse(result.cleanup_pending)
            self.assertEqual(result.status, "removed")
            self.assertFalse(target.exists())

    @unittest.skipUnless(os.name == "nt", "Windows file attributes are platform-specific")
    def test_readonly_file_is_removed_with_bounded_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = self._target(root)
            readonly = target / "readonly.tmp"
            readonly.write_text("x", encoding="utf-8")
            readonly.chmod(stat.S_IREAD)
            result = _remove_new_checkout(target, target.parent)
            self.assertFalse(result.cleanup_pending)
            self.assertFalse(target.exists())

    def test_first_failure_has_only_one_tree_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = self._target(root)
            original = shutil.rmtree
            calls = 0

            def flaky(path, *args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise PermissionError("first failure")
                return original(path, *args, **kwargs)

            with patch("app.services.repository_import.shutil.rmtree", side_effect=flaky):
                result = _remove_new_checkout(target, target.parent)
            self.assertFalse(result.cleanup_pending)
            self.assertEqual(result.status, "removed_after_retry")
            self.assertEqual(calls, 2)

    def test_persistent_permission_error_is_safe_and_pending(self):
        marker = "PRIVATE-PACK-NAME-MARKER"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = self._target(root)
            with (
                patch(
                    "app.services.repository_import.shutil.rmtree",
                    side_effect=PermissionError(marker),
                ) as remove,
                self.assertLogs("app.services.repository_import", level="WARNING") as captured,
            ):
                result = _remove_new_checkout(target, target.parent)
        self.assertTrue(result.cleanup_pending)
        self.assertEqual(result.status, "delete_failed")
        self.assertEqual(remove.call_count, 2)
        self.assertNotIn(marker, "".join(captured.output))
        self.assertNotIn(str(target), "".join(captured.output))

    def test_primary_git_failure_survives_cleanup_failure(self):
        called = subprocess.CalledProcessError(
            128, ["git", "clone"], stderr=b"failed to connect"
        )
        with tempfile.TemporaryDirectory() as temporary:
            settings = RepositorySettings(Path(temporary))
            with (
                patch(
                    "app.services.repository_import.validate_public_https_git_url",
                    return_value="https://public.example.invalid/repository.git",
                ),
                patch("app.services.repository_import.subprocess.run", side_effect=called),
                patch(
                    "app.services.repository_import._remove_new_checkout",
                    return_value=CheckoutCleanupResult(True, "delete_failed"),
                ),
                self.assertLogs("app.services.repository_import", level="WARNING"),
                self.assertRaises(RepositoryImportError) as raised,
            ):
                import_public_git_repository("ignored", settings, request_id="safe-request")
        safe = raised.exception.to_safe_dict(request_id="safe-request")
        self.assertEqual(raised.exception.code, "git_connection_failed")
        self.assertEqual(raised.exception.status_code, 502)
        self.assertTrue(raised.exception.retryable)
        self.assertEqual(safe["safe_stage"], "clone")
        self.assertEqual(safe["exit_code"], 128)
        self.assertTrue(safe["cleanup_pending"])

    def test_unsafe_targets_are_rejected_without_touching_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clone_root = root / "repositories"
            clone_root.mkdir()
            outside = root / (".import-" + "b" * 32)
            outside.mkdir()
            stable = clone_root / "stable-checkout"
            stable.mkdir()
            for target in (outside, clone_root, stable):
                with self.subTest(name=target.name):
                    result = _remove_new_checkout(target, clone_root)
                    self.assertTrue(result.cleanup_pending)
                    self.assertTrue(target.exists())

    def test_symlink_and_reparse_metadata_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = self._target(root)
            with patch.object(
                Path,
                "lstat",
                return_value=SimpleNamespace(st_mode=stat.S_IFLNK, st_file_attributes=0),
            ):
                self.assertEqual(_validate_cleanup_target(target, target.parent), "link_rejected")
            with patch.object(
                Path,
                "lstat",
                return_value=SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400),
            ):
                self.assertEqual(
                    _validate_cleanup_target(target, target.parent),
                    "reparse_point_rejected",
                )
            self.assertTrue(target.exists())

    def test_target_outside_files_are_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = self._target(root)
            external = root / "outside.txt"
            external.write_text("unchanged", encoding="utf-8")
            external.chmod(stat.S_IREAD)
            try:
                result = _remove_new_checkout(target, target.parent)
                self.assertFalse(result.cleanup_pending)
                self.assertEqual(external.read_text(encoding="utf-8"), "unchanged")
                self.assertFalse(external.stat().st_mode & stat.S_IWRITE)
            finally:
                external.chmod(stat.S_IWRITE)


if __name__ == "__main__":
    unittest.main()
