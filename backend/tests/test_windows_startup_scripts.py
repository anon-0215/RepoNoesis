from __future__ import annotations

import os
import ctypes
import shutil
import socket
import subprocess
import sys
import tempfile
import time
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
        # An unrelated service must not satisfy the launcher's fixed health URLs.
        # Fail explicitly on occupied ports instead of passing an unrelated branch.
        ports = []
        for _ in range(2):
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                ports.append(reservation.getsockname()[1])

        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.update(
            {
                "REPONOESIS_PYTHON": sys.executable,
                "REPONOESIS_NODE": sys.executable,
                "REPONOESIS_SKIP_ENV_FILE": "1",
                "PYTHONNOUSERSITE": "1",
            }
        )
        # Copy only the launcher and resolver. No product modules or dependencies
        # are imported, and the frontend prerequisite exists even in a fresh ZIP.
        with tempfile.TemporaryDirectory(
            prefix="startup-failure-", dir=os.environ.get("REPONOESIS_STARTUP_TEST_TMP")
        ) as temporary:
            fixture = Path(temporary)
            scripts = fixture / "scripts"
            app = fixture / "backend" / "app"
            vite = fixture / "frontend" / "node_modules" / "vite" / "bin"
            for directory in (scripts, app, vite):
                directory.mkdir(parents=True)
            shutil.copy2(STARTER, scripts / STARTER.name)
            shutil.copy2(RESOLVER, scripts / RESOLVER.name)
            (app / "__init__.py").touch()
            (app / "run_server.py").write_text(
                "import os\nfrom pathlib import Path\n"
                "Path('backend-started.pid').write_text(str(os.getpid()))\n"
                "raise SystemExit(42)\n", encoding="utf-8"
            )
            # Python accepts this script filename; it would leave a marker if
            # the launcher incorrectly continued to the frontend component.
            (vite / "vite.js").write_text(
                "from pathlib import Path\nPath('frontend-started').touch()\n",
                encoding="utf-8",
            )
            output = fixture / "launcher-output.txt"
            started = time.monotonic()
            # Regular files cannot keep communicate() waiting on descendant pipe
            # handles after the launcher exits. The original 10s bound remains.
            with output.open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    [str(POWERSHELL), "-NoLogo", "-NoProfile", "-ExecutionPolicy",
                     "Bypass", "-File", str(scripts / STARTER.name),
                     "-HealthTimeoutSeconds", "1", "-Headless", "-BackendPort", str(ports[0]), "-FrontendPort", str(ports[1])],
                    check=False, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                    env=env, timeout=10,
                )
            elapsed = time.monotonic() - started
            combined = output.read_text(encoding="utf-8", errors="replace")
            marker = fixture / "backend" / "backend-started.pid"
            self.assertTrue(marker.is_file(), combined)
            self.assertFalse((fixture / "frontend" / "frontend-started").exists(), combined)
            pid = int(marker.read_text())
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
            kernel.OpenProcess.restype = ctypes.c_void_p
            kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
            kernel.CloseHandle.argtypes = [ctypes.c_void_p]
            handle = kernel.OpenProcess(0x00100000, False, pid)
            if handle:
                try:
                    self.assertEqual(kernel.WaitForSingleObject(handle, 0), 0, "fake backend still alive")
                finally:
                    kernel.CloseHandle(handle)
            else:
                self.assertEqual(ctypes.get_last_error(), 87, "cannot verify fake backend exit")
            self.assertLess(elapsed, 10)
            print(f"failure fixture: backend PID={pid} exited; launcher rc={result.returncode}; elapsed={elapsed:.3f}s")
        self.assertIn("Backend process started", combined)
        # The first bounded HTTP probe can outlast the 1s startup deadline while
        # the fake process exits, so either existing failure reason is valid.
        self.assertRegex(
            combined,
            r"Backend (process exited before becoming healthy|did not become healthy within the startup timeout)",
        )
        self.assertNotIn("Frontend process started", combined)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("backend and frontend are ready", combined.casefold())
        self.assertIn("startup failed", combined.casefold())


if __name__ == "__main__":
    unittest.main()
