import importlib.util
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("desktop_launcher", Path(__file__).with_name("launcher.py"))
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LaunchIsolationTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows hidden-host regression")
    def test_git_completes_while_hidden_host_waits_on_its_control_pipe(self):
        import shutil
        git = shutil.which("git")
        self.assertIsNotNone(git)
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary) / "repo"
            subprocess.run([git, "init", str(repo)], check=True, capture_output=True,
                           stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
            backend = Path(__file__).resolve().parents[1] / "backend"
            code = (
                "import sys,threading; sys.path.insert(0,sys.argv[1]); "
                "from pathlib import Path; from app.services.repository_import import _git; "
                "watch=threading.Thread(target=sys.stdin.buffer.readline,daemon=True); watch.start(); "
                "print(_git(Path(sys.argv[2]),'rev-parse','--git-dir'),flush=True); watch.join(10)"
            )
            job = launcher.OwnedJob()
            child = subprocess.Popen([sys.executable, "-I", "-B", "-c", code, str(backend), str(repo)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW)
            job.assign(child)
            line = []
            ready = threading.Event()
            def collect():
                line.append(child.stdout.readline())
                ready.set()
            reader = threading.Thread(target=collect, daemon=True)
            reader.start()
            try:
                self.assertTrue(ready.wait(8), "Git inherited the service control pipe and stalled")
                self.assertEqual(line, [b".git\r\n"])
                child.stdin.write(b"stop\n"); child.stdin.flush()
                self.assertEqual(child.wait(timeout=5), 0)
            finally:
                job.close()
                child.wait(timeout=5)
                for stream in (child.stdin, child.stdout, child.stderr):
                    stream.close()
                reader.join(timeout=2)

    def test_secrets_and_development_paths_do_not_reach_backend(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "test-only", "GITHUB_TOKEN": "test-only",
                "PYTHONPATH": "developer-code", "CONDA_PREFIX": "developer-python", "PATH": "developer-node"}):
            env = launcher.child_environment(Path("C:/portable"), {"git_directory": "C:/Git/cmd",
                "model_directory": "C:/models/bge-m3"}, Path("C:/user-data"), "test-instance")
        for key in ("LLM_API_KEY", "GITHUB_TOKEN", "PYTHONPATH", "CONDA_PREFIX"):
            self.assertNotIn(key, env)
        self.assertNotIn("developer", env["PATH"])
        self.assertEqual(env["REPONOESIS_SKIP_ENV_FILE"], "true")
        self.assertEqual(env["EMBEDDING_DEVICE"], "auto")
        self.assertIn("user-data", env["GITLEARN_DB"])

    def test_occupied_port_is_rejected_without_reusing_listener(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            with self.assertRaises(OSError):
                launcher.reserve_port(listener.getsockname()[1])
            self.assertGreater(listener.fileno(), 0)


if __name__ == "__main__":
    unittest.main()
