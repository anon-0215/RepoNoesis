from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


BACKEND_ROOT = Path(__file__).resolve().parents[1]
SENTINEL_KEY = "OPS_ENV1_SENTINEL"


class EnvironmentIsolationTests(unittest.TestCase):
    def test_get_env_value_does_not_implicitly_read_dotenv(self) -> None:
        from app import config

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text(f"{SENTINEL_KEY}=present\n", encoding="utf-8")
            with (
                patch.object(config, "PROJECT_ROOT", root),
                patch.dict(os.environ, {}, clear=True),
            ):
                self.assertEqual(config.get_env_value(SENTINEL_KEY, "missing"), "missing")

    def test_import_and_reload_main_do_not_load_dotenv_or_create_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database_path = root / "import-must-not-create.sqlite"
            (root / ".env").write_text(f"{SENTINEL_KEY}=present\n", encoding="utf-8")
            child_environment = os.environ.copy()
            child_environment.update(
                {
                    "GITLEARN_DB": str(database_path),
                    "OPS_ENV1_TEST_ROOT": str(root),
                    "EMBEDDING_ENABLED": "false",
                    "PYTHONPATH": str(BACKEND_ROOT),
                    "PYTHONUTF8": "1",
                }
            )
            child_environment.pop(SENTINEL_KEY, None)
            script = r'''
import importlib
import json
import os
import sys
import urllib.request
from pathlib import Path
from unittest.mock import patch

import app.config as config

root = Path(os.environ["OPS_ENV1_TEST_ROOT"])
candidate = root / ".env"
config.PROJECT_ROOT = root
original_read_text = Path.read_text
candidate_reads = []
provider_calls = []

def tracked_read_text(path, *args, **kwargs):
    if path == candidate:
        candidate_reads.append(True)
    return original_read_text(path, *args, **kwargs)

def reject_runtime_side_effects(event, _args):
    if event in {"socket.bind", "socket.connect"}:
        raise AssertionError(f"unexpected runtime side effect: {event}")

def reject_provider_call(*_args, **_kwargs):
    provider_calls.append(True)
    raise AssertionError("unexpected provider call")

sys.addaudithook(reject_runtime_side_effects)
with (
    patch.object(Path, "read_text", tracked_read_text),
    patch.object(urllib.request, "urlopen", reject_provider_call),
):
    main = importlib.import_module("app.main")
    main = importlib.reload(main)

payload = {
    "candidate_reads": len(candidate_reads),
    "database_created": Path(os.environ["GITLEARN_DB"]).exists(),
    "sentinel_loaded": "OPS_ENV1_SENTINEL" in os.environ,
    "route_count": len(main.app.routes),
    "openapi_paths": len(main.app.openapi()["paths"]),
    "provider_calls": len(provider_calls),
    "embedding_backend_loaded": main.embedding_service._backend is not None,
}
print(json.dumps(payload, sort_keys=True))
'''
            completed = subprocess.run(
                [sys.executable, "-B", "-c", script],
                cwd=BACKEND_ROOT,
                env=child_environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
            self.assertEqual(payload["candidate_reads"], 0)
            self.assertFalse(payload["database_created"])
            self.assertFalse(payload["sentinel_loaded"])
            self.assertEqual(payload["provider_calls"], 0)
            self.assertFalse(payload["embedding_backend_loaded"])
            self.assertGreater(payload["route_count"], 4)
            self.assertGreater(payload["openapi_paths"], 0)

    def test_explicit_environment_load_is_idempotent_and_process_values_win(self) -> None:
        from app import config

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text(f"{SENTINEL_KEY}=from-file\n", encoding="utf-8")
            with (
                patch.object(config, "PROJECT_ROOT", root),
                patch.dict(os.environ, {}, clear=True),
            ):
                config.load_environment()
                self.assertEqual(os.environ[SENTINEL_KEY], "from-file")
                os.environ[SENTINEL_KEY] = "from-process"
                config.load_environment()
                self.assertEqual(os.environ[SENTINEL_KEY], "from-process")

    def test_environment_load_failure_is_safe_and_stops_bootstrap(self) -> None:
        from app import config

        run_server = importlib.import_module("app.run_server")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text(f"{SENTINEL_KEY}=present\n", encoding="utf-8")
            with (
                patch.object(config, "PROJECT_ROOT", root),
                patch.object(
                    Path,
                    "read_text",
                    side_effect=UnicodeError("value-must-not-escape"),
                ),
            ):
                with self.assertRaises(config.EnvironmentLoadError) as raised:
                    config.load_environment()
            self.assertEqual(
                str(raised.exception),
                "Failed to load backend environment configuration.",
            )
            self.assertNotIn("value-must-not-escape", str(raised.exception))

        with (
            patch.object(
                run_server,
                "load_environment",
                side_effect=config.EnvironmentLoadError(
                    "Failed to load backend environment configuration."
                ),
            ),
            patch.object(run_server.uvicorn, "run") as uvicorn_run,
        ):
            with self.assertRaises(config.EnvironmentLoadError):
                run_server.main()
        uvicorn_run.assert_not_called()

    def test_run_server_loads_environment_before_consuming_configuration(self) -> None:
        run_server = importlib.import_module("app.run_server")
        events: list[str] = []

        def record_load() -> None:
            events.append("load")

        def record_get(key: str, default: str = "") -> str:
            events.append(f"get:{key}")
            return default

        def record_run(target: str, **kwargs: object) -> None:
            events.append(f"run:{target}")
            self.assertEqual(kwargs["host"], "127.0.0.1")
            self.assertEqual(kwargs["port"], 8000)

        with (
            patch.object(run_server, "load_environment", side_effect=record_load),
            patch.object(run_server, "get_env_value", side_effect=record_get),
            patch.object(run_server.uvicorn, "run", side_effect=record_run),
        ):
            run_server.main()

        self.assertEqual(
            events,
            [
                "load",
                "get:BACKEND_HOST",
                "get:BACKEND_PORT",
                "get:BACKEND_RELOAD",
                "run:app.main:app",
            ],
        )

    def test_run_server_bootstrap_loads_temporary_dotenv_before_app_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text(
                "\n".join(
                    (
                        "BACKEND_HOST=127.0.0.9",
                        "BACKEND_PORT=8123",
                        "LLM_MODEL=ops-env1-model",
                        "EMBEDDING_ENABLED=false",
                        f"GITLEARN_DB={root / 'bootstrap.sqlite'}",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            child_environment = os.environ.copy()
            child_environment.update(
                {
                    "OPS_ENV1_TEST_ROOT": str(root),
                    "PYTHONPATH": str(BACKEND_ROOT),
                    "PYTHONUTF8": "1",
                }
            )
            for key in ("BACKEND_HOST", "BACKEND_PORT", "LLM_MODEL"):
                child_environment.pop(key, None)
            script = r'''
import importlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import app.config as config
import app.run_server as run_server

config.PROJECT_ROOT = Path(os.environ["OPS_ENV1_TEST_ROOT"])
observed = {}

def inspect_application(target, **kwargs):
    main = importlib.import_module(target.rsplit(":", 1)[0])
    observed.update(
        {
            "target": target,
            "host": kwargs["host"],
            "port": kwargs["port"],
            "model": main.llm.model,
            "database_created": (config.PROJECT_ROOT / "bootstrap.sqlite").exists(),
        }
    )

with patch.object(run_server.uvicorn, "run", side_effect=inspect_application):
    run_server.main()
print(json.dumps(observed, sort_keys=True))
'''
            completed = subprocess.run(
                [sys.executable, "-B", "-c", script],
                cwd=BACKEND_ROOT,
                env=child_environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
            self.assertEqual(payload["target"], "app.main:app")
            self.assertEqual(payload["host"], "127.0.0.9")
            self.assertEqual(payload["port"], 8123)
            self.assertEqual(payload["model"], "ops-env1-model")
            self.assertFalse(payload["database_created"])


if __name__ == "__main__":
    unittest.main()
