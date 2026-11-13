"""Single same-origin FastAPI + production React host; stdin owns lifetime."""
import json
import os
from pathlib import Path
import sys
import threading
import faulthandler
from threading import Event

faulthandler.enable()
faulthandler.dump_traceback_later(30, repeat=False)

ROOT = Path(__file__).resolve().parents[1]
_dll_handles = []
if os.name == "nt":
    _dll_handles.append(os.add_dll_directory(str(ROOT / "runtime/Library/bin")))
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).parent))

_stop = Event()
_server = None
def watch_parent():
    os.read(0, 1)
    _stop.set()
    if _server is not None:
        _server.should_exit = True

def progress(phase, message):
    path = Path("logs/boot-progress.json")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"phase": phase, "message": message}), encoding="utf-8")
    os.replace(temporary, path)

progress("runtime", "正在载入本地模型运行组件")

# Load native NumPy/Torch DLLs on the main thread before ASGI/semantic
# workers start. On Windows, simultaneous native import and worker creation
# can stall in the loader while Thread.start() waits for initialization.
from sentence_transformers import SentenceTransformer  # noqa: F401
faulthandler.cancel_dump_traceback_later()
if _stop.is_set():
    Path("logs/bootstrap-closeout.json").write_text('{"stopped_before_model":true}', encoding="utf-8")
    raise SystemExit(0)

import uvicorn
from fastapi.staticfiles import StaticFiles
from app.main import app
import app.main as main
from app.config import get_retrieval_enhancement_settings
from settings_lifecycle import install as install_settings_lifecycle

install_settings_lifecycle(app, main, os.environ.get("REPONOESIS_STARTUP_ID", ""))
app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="desktop-ui")
if os.environ.get("REPONOESIS_ACCEPTANCE_LEDGER"):
    from acceptance import install
    install(app, os.environ["REPONOESIS_ACCEPTANCE_LEDGER"])


def run():
    global _server
    import importlib.metadata
    receipt = {"executable": sys.executable, "prefix": sys.prefix, "sys_path": sys.path,
               "app_file": main.__file__, "cwd": os.getcwd(),
               "packages": {name: importlib.metadata.version(name) for name in (
                   "fastapi", "uvicorn", "pydantic", "sentence-transformers", "torch", "transformers", "numpy")}}
    flags = get_retrieval_enhancement_settings()
    receipt["retrieval_enhancement_config"] = {"hyde_enabled": flags.hyde_enabled, "rerank_enabled": flags.rerank_enabled}
    receipt["rerank_provider"] = flags.rerank_provider
    Path("logs/runtime-identity.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=int(sys.argv[1]), reload=False))
    _server = server
    # Keep native imports on the main thread before any stdin watcher starts.
    threading.Thread(target=watch_parent, daemon=True, name="desktop-lifetime").start()
    if _stop.is_set():
        return
    progress("model", "正在准备 BGE-M3，完成后自动进入工作台")
    server.run()
    external_modules = sorted({str(Path(module.__file__).resolve()) for module in list(sys.modules.values())
        if isinstance(getattr(module, "__file__", None), str)
        and not Path(module.__file__).resolve().is_relative_to(ROOT)})
    Path("logs/shutdown.json").write_text(json.dumps({
        "embedding": main.embedding_service.runtime_status(),
        "provider_workers": __import__("app.services.provider_transport", fromlist=["active_worker_count"]).active_worker_count(),
        "external_module_files": external_modules,
        "threads": [thread.name for thread in threading.enumerate()],
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
