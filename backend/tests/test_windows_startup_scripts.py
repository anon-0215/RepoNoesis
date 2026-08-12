from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESOLVER = ROOT / "scripts" / "resolve_runtime.ps1"
STARTER = ROOT / "scripts" / "start_local.ps1"
POWERSHELL = Path(os.environ["SystemRoot"]) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"


class WindowsStartupScriptTests(unittest.TestCase):
    def _run_resolver(self, kind: str, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update(environment)
        return subprocess.run(
            [
                str(POWERSHELL),
                "-NoLogo",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(RESOLVER),
                "-Kind",
                kind,
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            timeout=10,
        )

    def test_explicit_python_override_supports_spaces(self):
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "runtime with spaces" / "python.exe"
            executable.parent.mkdir()
            executable.touch()
            result = self._run_resolver(
                "Python",
                {"REPONOESIS_PYTHON": str(executable), "CONDA_PREFIX": "", "PATH": ""},
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(result.stdout.strip()), executable)

    def test_conda_prefix_python_is_used_before_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "conda env" / "python.exe"
            executable.parent.mkdir()
            executable.touch()
            result = self._run_resolver(
                "Python",
                {"REPONOESIS_PYTHON": "", "CONDA_PREFIX": str(executable.parent), "PATH": ""},
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(result.stdout.strip()), executable)

    def test_missing_configured_python_fails_clearly(self):
        result = self._run_resolver(
            "Python",
            {
                "REPONOESIS_PYTHON": str(ROOT / "missing-python.exe"),
                "CONDA_PREFIX": "",
                "PATH": "",
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("REPONOESIS_PYTHON", result.stderr)

    def test_explicit_node_override_supports_spaces(self):
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "node runtime with spaces" / "node.exe"
            executable.parent.mkdir()
            executable.touch()
            result = self._run_resolver(
                "Node", {"REPONOESIS_NODE": str(executable), "PATH": ""}
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(result.stdout.strip()), executable)

    def test_missing_configured_node_fails_clearly(self):
        result = self._run_resolver(
            "Node",
            {"REPONOESIS_NODE": str(ROOT / "missing-node.exe"), "PATH": ""},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("REPONOESIS_NODE", result.stderr)

    def test_batch_entrypoints_do_not_depend_on_where_or_timeout(self):
        for relative in (
            "start_all.bat",
            "backend/run_backend.bat",
            "frontend/run_frontend.bat",
        ):
            content = (ROOT / relative).read_text(encoding="utf-8").casefold()
            with self.subTest(relative=relative):
                self.assertNotIn("where ", content)
                self.assertNotIn("timeout ", content)
                self.assertIn("exit /b", content)

    def test_startup_inherits_explicit_git_proxy_without_echo_or_system_mutation(self):
        scripts = (
            ROOT / "start_all.bat",
            ROOT / "backend" / "run_backend.bat",
            ROOT / "backend" / "run_backend.ps1",
            ROOT / "scripts" / "start_local.ps1",
        )
        for script in scripts:
            content = script.read_text(encoding="utf-8").casefold()
            with self.subTest(script=script.name):
                self.assertNotIn("usenewenvironment", content)
                self.assertNotIn("setx ", content)
                self.assertNotIn("environmentvariabletarget", content)
                self.assertNotIn("echo %reponoesis_git_proxy%", content)
                self.assertNotIn("write-output $env:reponoesis_git_proxy", content)

    def test_failed_backend_never_reports_overall_ready(self):
        env = os.environ.copy()
        env.update(
            {
                "REPONOESIS_PYTHON": str(POWERSHELL),
                "REPONOESIS_NODE": str(POWERSHELL),
            }
        )
        result = subprocess.run(
            [
                str(POWERSHELL),
                "-NoLogo",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(STARTER),
                "-HealthTimeoutSeconds",
                "1",
                "-Headless",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            timeout=10,
        )
        combined = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("backend and frontend are ready", combined.casefold())
        self.assertIn("startup failed", combined.casefold())


if __name__ == "__main__":
    unittest.main()
