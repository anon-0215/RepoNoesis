"""Portable Windows prototype. No developer environment or .env discovery."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
import webbrowser


ROOT = Path(__file__).resolve().parents[1]


def child_environment(root: Path, config: dict, data: Path, instance: str) -> dict[str, str]:
    # An allowlist prevents inherited API tokens, Python/Conda paths and Git config.
    env = {k: os.environ[k] for k in ("SystemRoot", "WINDIR", "COMSPEC", "TEMP", "TMP") if k in os.environ}
    runtime = root / "runtime"
    git_dir = Path(config["git_directory"]).resolve()
    env.update({
        "PATH": os.pathsep.join(map(str, (runtime, runtime / "Library/bin", git_dir, Path(env.get("SystemRoot", "C:/Windows")) / "System32"))),
        "USERPROFILE": str(data / "profile"), "HOME": str(data / "profile"),
        "USERNAME": "RepoNoesis",
        "APPDATA": str(data / "profile/AppData/Roaming"), "LOCALAPPDATA": str(data / "profile/AppData/Local"),
        "HF_HOME": str(data / "huggingface"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1", "PYTHONUTF8": "1",
        "REPONOESIS_SKIP_ENV_FILE": "true", "REPONOESIS_STARTUP_ID": instance,
        "GITLEARN_DB": str(data / "database/reponoesis.sqlite"),
        "RUNTIME_DATA_DIR": str(data / "runtime"), "EMBEDDING_CACHE_DIR": str(data / "embedding-cache"),
        "EMBEDDING_ENABLED": "true", "EMBEDDING_PROVIDER": "local_bge_m3",
        "EMBEDDING_MODEL": str(Path(config["model_directory"]).resolve()),
        "EMBEDDING_OFFLINE": "true", "EMBEDDING_DEVICE": config.get("device", "auto"),
        "EMBEDDING_BATCH_SIZE": str(config.get("batch_size", 1)),
        "EMBEDDING_MAX_LENGTH": str(config.get("max_length", 8192)),
        "RETRIEVAL_HYDE_ENABLED": str(bool(config.get("hyde_enabled", False))).lower(),
        "RETRIEVAL_RERANK_ENABLED": str(bool(config.get("rerank_enabled", False))).lower(),
        "RETRIEVAL_ENHANCEMENT_SECONDS": "15",
        "TOKENIZERS_PARALLELISM": "false", "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull, "GIT_TERMINAL_PROMPT": "0",
    })
    llm = config.get("llm", {})
    if llm:
        env.update(LLM_PROVIDER="openai_compatible", LLM_BASE_URL=llm.get("base_url", ""),
                   LLM_MODEL=llm.get("model", ""), LLM_API_KEY=llm.get("api_key", ""))
    return env


def reserve_port(port: int) -> None:
    with socket.socket() as sock:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        sock.bind(("127.0.0.1", port))


class OwnedJob:
    """Kill only this invocation's child tree if the launcher is closed abruptly."""
    def __init__(self):
        import ctypes
        from ctypes import wintypes
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self.api.CreateJobObjectW.restype = wintypes.HANDLE
        self.api.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.api.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.api.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        class Basic(ctypes.Structure):
            _fields_ = [("a", ctypes.c_int64), ("b", ctypes.c_int64), ("flags", wintypes.DWORD),
                        ("min", ctypes.c_size_t), ("max", ctypes.c_size_t), ("active", wintypes.DWORD),
                        ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(k, ctypes.c_uint64) for k in ("r", "w", "o", "rb", "wb", "ob")]
        class Extended(ctypes.Structure):
            _fields_ = [("basic", Basic), ("io", IO), ("process", ctypes.c_size_t), ("job", ctypes.c_size_t),
                        ("peakprocess", ctypes.c_size_t), ("peakjob", ctypes.c_size_t)]
        self.handle = self.api.CreateJobObjectW(None, None)
        limits = Extended()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.handle or not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            raise ctypes.WinError(ctypes.get_last_error())

    def assign(self, process):
        import ctypes
        if not self.api.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    config = json.loads((ROOT / "desktop-config.json").read_text(encoding="utf-8-sig"))
    data = (args.data_dir or ROOT.parent / "user-data").resolve()
    data.mkdir(parents=True, exist_ok=True)
    for folder in ("profile", "logs", "database"):
        (data / folder).mkdir(exist_ok=True)
    port = int(config.get("port", 18761))
    reserve_port(port)
    instance = uuid.uuid4().hex
    env = child_environment(ROOT, config, data, instance)
    if not Path(env["EMBEDDING_MODEL"]).is_dir():
        raise RuntimeError("Set model_directory in desktop-config.json to a complete BGE-M3 directory.")
    if not (Path(config["git_directory"]) / "git.exe").is_file():
        raise RuntimeError("Set git_directory in desktop-config.json to a directory containing git.exe.")
    job = OwnedJob()
    child = None
    state = data / "session.json"
    receipt = {"instance": instance, "status": "starting"}
    try:
        with (data / "logs/backend.log").open("a", encoding="utf-8") as log:
            child = subprocess.Popen([str(ROOT / "runtime/python.exe"), "-I", "-B", str(ROOT / "desktop/server.py"), str(port)],
                cwd=data, env=env, stdin=subprocess.PIPE, stdout=log, stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW)
            job.assign(child)
            url = f"http://127.0.0.1:{port}"
            receipt = {"instance": instance, "launcher_pid": os.getpid(), "backend_pid": child.pid,
                       "url": url, "data_directory": str(data), "release_directory": str(ROOT), "status": "starting"}
            state.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    raise RuntimeError("Backend exited; see user-data/logs/backend.log")
                try:
                    with opener.open(url + "/api/health", timeout=3) as response:
                        health = json.load(response)
                    if health.get("startup_instance_id") == instance and health.get("embedding_ready"):
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(0.25)
            else:
                raise RuntimeError("BGE startup did not become ready within 180 seconds; see backend.log")
            receipt.update(status="ready", health=health)
            state.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"RepoNoesis ready: {url}\nPress Enter in this window to stop. Closing the browser alone does not stop it.", flush=True)
            if not args.no_browser:
                webbrowser.open(url)
            input()
    finally:
        if child is not None and child.poll() is None:
            try:
                child.stdin.write(b"stop\n")
                child.stdin.flush()
                child.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                pass
        job.close()
        if child is not None:
            child.wait(timeout=10)
            if state.exists():
                receipt.update(status="stopped", backend_exit_code=child.returncode)
                state.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        print(f"RepoNoesis stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
