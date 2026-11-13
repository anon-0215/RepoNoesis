"""Native window and pre-backend settings; reuses the portable process owner."""
import argparse
import ctypes
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
_dll = None  # Register packaged native DLLs only when launching, not during source tests.
from gui_settings import SettingsStore, SettingsValidationError, atomic_json
from launcher import OwnedJob, child_environment, reserve_port


class InstanceGuard:
    def __init__(self, data):
        import msvcrt
        self.file = (data / "instance.lock").open("a+b")
        if self.file.tell() == 0:
            self.file.write(b"0"); self.file.flush()
        self.file.seek(0)
        try:
            msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            self.file.close()
            raise RuntimeError("源鉴已经在运行，请切换到已打开的窗口。") from None

    def close(self):
        self.file.close()


class Controller:
    def __init__(self, data, defaults, *, acceptance=False):
        self.data = data
        self.store = SettingsStore(data, defaults)
        self.acceptance = acceptance
        self.window = None
        self.stop_event = threading.Event()
        self.reload_event = threading.Event()
        self.lock = threading.RLock()
        self.thread = None
        self.child = None
        self.job = None
        self.state = {"phase": "checking", "message": "正在检查本地资源", "backend_url": None,
                      "generation": 0, "api_ready": False}
        self.instance = uuid.uuid4().hex
        for name in ("logs", "profile", "database"):
            (data / name).mkdir(parents=True, exist_ok=True)
        atomic_json(data / "logs/window-runtime.json", {
            "executable": sys.executable, "prefix": sys.prefix, "sys_path": sys.path,
            "desktop_file": str(Path(__file__).resolve()), "data_directory": str(data),
        })

    def _status(self, phase, message, **extra):
        with self.lock:
            self.state.update(phase=phase, message=message, **extra)
            atomic_json(self.data / "logs/desktop-state.json", self.state | {
                "launcher_pid": os.getpid(), "backend_pid": self.child.pid if self.child else None,
                "instance": self.instance, "release": str(ROOT)})

    def get_status(self):
        with self.lock:
            result = dict(self.state)
        if self.acceptance:
            path = self.data / "acceptance-ledger.json"
            if path.exists():
                ledger = json.loads(path.read_text(encoding="utf-8"))
                result["acceptance"] = {k: ledger.get(k) for k in ("questions", "attempts_reserved", "halted")}
        return result

    def get_settings(self):
        return self.store.public()

    def save_settings(self, value):
        leased = False
        written = False
        lease_id = uuid.uuid4().hex
        try:
            with self.lock:
                before = self.store.load()
                prepared = self.store.prepare(value)
                result = self.store.public(prepared)
                changed = prepared != before
                if changed:
                    if self.child is not None and self.child.poll() is None:
                        if not self._settings_restart("acquire", lease_id):
                            self._settings_restart("release", lease_id)  # Lost acquire response may still own a lease.
                            return {"ok": False, "message": "未保存：正在问答、导入项目、应用设置或准备后台。请结束后再保存。"}
                        leased = True
                    atomic_json(self.store.path, prepared)
                    written = True
                    if leased and not self._settings_restart("acquire", lease_id):
                        # A slow disk write can outlive the lease. Re-check before
                        # restart so a newly admitted ask is never interrupted.
                        atomic_json(self.store.path, before)
                        written = False
                        self._settings_restart("release", lease_id)
                        return {"ok": False, "message": "未应用：后台状态已变化，原设置已恢复。请在回答结束后重试。"}
                    self.state.update(phase="restarting", message="设置已保存，正在重启本地服务；就绪后生效。", backend_url=None)
                    self.reload_event.set()
                    try:
                        self._status("restarting", "设置已保存，正在重启本地服务；就绪后生效。", backend_url=None)
                    except OSError:
                        # Settings are already durable. A diagnostic write failure
                        # must not falsely report them as unsaved or retain a lease.
                        self.state.update(phase="restarting", message="设置已保存，正在重启本地服务；就绪后生效。", backend_url=None)
            return {"ok": True, "settings": result, "changed": changed}
        except SettingsValidationError as exc:
            if leased:
                self._settings_restart("release", lease_id)
            return {"ok": False, "message": "未保存：" + exc.message}
        except ValueError:
            if leased:
                self._settings_restart("release", lease_id)
            return {"ok": False, "message": "未保存：请检查 API 地址、目录、检索增强布尔开关和 Agent 预算格式。"}
        except (OSError, RuntimeError):
            if leased:
                self._settings_restart("release", lease_id)
            # Do not return input or raw filesystem/crypto exceptions to JS.
            if written:
                return {"ok": False, "message": "设置可能已保存但未应用，恢复旧设置失败。请重开窗口并读取设置后确认。"}
            return {"ok": False, "message": "未保存：无法写入设置。请检查数据目录权限或可用空间。"}

    def _settings_restart(self, action, lease_id):
        """Acquire before writing: no new ask can race between idle check and restart."""
        port = int(self.store.load().get("port", 18771))
        request = urllib.request.Request(f"http://127.0.0.1:{port}/api/desktop/settings-restart",
            data=json.dumps({"action": action, "lease_id": lease_id}).encode(), method="POST",
            headers={"Content-Type": "application/json", "X-RepoNoesis-Instance": self.instance})
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=2) as response:
                result = json.load(response)
            return result.get("ok") is True and result.get("instance") == self.instance
        except (OSError, ValueError):
            return False

    def choose_folder(self, field):
        if field not in {"model_directory", "git_directory"}:
            return None
        import webview
        result = self.window.create_file_dialog(webview.FOLDER_DIALOG)
        return result[0] if result else None

    def open_data_directory(self):
        os.startfile(str(self.data))
        return True

    def retry(self):
        self.reload_event.set()
        return True

    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._run, name="desktop-owner", daemon=False)
            self.thread.start()

    def _stop_backend(self):
        child, job = self.child, self.job
        if child is not None:
            graceful = False
            if child.poll() is None:
                try:
                    child.stdin.write(b"stop\n"); child.stdin.flush()
                    child.wait(timeout=12)
                    graceful = child.returncode == 0
                except (OSError, subprocess.TimeoutExpired):
                    pass
            else:
                graceful = child.returncode == 0
            if job:
                job.close()
            child.wait(timeout=5)
            if child.stdin:
                child.stdin.close()
            atomic_json(self.data / "logs/desktop-closeout.json", {
                "backend_pid": child.pid, "backend_exit_code": child.returncode,
                "graceful": graceful, "closed_at": time.time(), "instance": self.instance})
        self.child = self.job = None

    def _serve(self):
        config = self.store.runtime()
        port = int(config.get("port", 18771))
        reserve_port(port)
        env = child_environment(ROOT, config, self.data, self.instance)
        if self.acceptance:
            env["REPONOESIS_ACCEPTANCE_LEDGER"] = str(self.data / "acceptance-ledger.json")
        progress = self.data / "logs/boot-progress.json"
        if progress.exists():
            progress.unlink()
        self.job = OwnedJob()
        with (self.data / "logs/backend.log").open("a", encoding="utf-8") as log:
            self.child = subprocess.Popen([str(ROOT / "runtime/python.exe"), "-I", "-B", str(HERE / "server.py"), str(port)],
                cwd=self.data, env=env, stdin=subprocess.PIPE, stdout=log, stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW)
            self.job.assign(self.child)
            self._status("runtime", "正在载入本地模型运行组件", backend_url=None)
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            deadline = time.monotonic() + 180
            last_phase = None
            while not self.stop_event.is_set() and not self.reload_event.is_set():
                if self.child.poll() is not None:
                    self._status("error", "后台启动失败。请检查模型目录与设备配置；详情见数据目录 logs/backend.log。", backend_url=None)
                    return
                if progress.exists():
                    try:
                        boot = json.loads(progress.read_text(encoding="utf-8"))
                        if boot.get("phase") != last_phase:
                            last_phase = boot.get("phase")
                            self._status(last_phase, boot["message"], backend_url=None)
                    except (OSError, ValueError):
                        pass
                if last_phase == "model":
                    try:
                        with opener.open(f"http://127.0.0.1:{port}/api/health", timeout=1) as response:
                            health = json.load(response)
                        if health.get("startup_instance_id") == self.instance and health.get("embedding_ready"):
                            self._status("ready", "本地服务已就绪", backend_url=f"http://127.0.0.1:{port}",
                                generation=self.state["generation"] + 1, api_ready=health.get("llm_available", False))
                            break
                        if health.get("embedding_runtime", {}).get("preparation_state") == "failed":
                            self._status("error", "BGE-M3 加载失败。请检查权重是否完整，以及当前设备是否有足够内存；可在数据目录查看模型日志后重试。", backend_url=None)
                            return
                    except (OSError, ValueError):
                        pass
                if time.monotonic() > deadline:
                    self._status("error", "模型准备超过 180 秒。请检查完整权重目录、设备与可用内存，再点击重试。", backend_url=None)
                    return
                self.stop_event.wait(.25)
            while not self.stop_event.wait(.25) and not self.reload_event.is_set() and self.state["phase"] == "ready":
                if self.child.poll() is not None:
                    self._status("error", "后台服务已退出。可打开数据目录查看日志，然后重试。", backend_url=None)
                    break

    def _run(self):
        try:
            while not self.stop_event.is_set():
                self.reload_event.clear()
                errors = self.store.resource_errors()
                if errors:
                    self._status("settings", "\n".join(errors), backend_url=None)
                else:
                    try:
                        self._serve()
                    except (OSError, RuntimeError, ValueError):
                        self._status("error", "无法启动：请检查端口是否被占用、资源目录及当前 Windows 用户保存的密钥。", backend_url=None)
                    finally:
                        self._stop_backend()
                while not self.stop_event.wait(.2) and not self.reload_event.is_set():
                    pass
        finally:
            self._stop_backend()
            self._status("stopped", "已退出", backend_url=None)

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=20)
        if self.job:
            self.job.close()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=6)


class WindowApi:
    """Expose only the intended bridge, never SettingsStore.runtime or key decryption."""
    def __init__(self, controller):
        self._controller = controller

    def get_status(self): return self._controller.get_status()
    def get_settings(self): return self._controller.get_settings()
    def save_settings(self, value): return self._controller.save_settings(value)
    def choose_folder(self, field): return self._controller.choose_folder(field)
    def open_data_directory(self): return self._controller.open_data_directory()
    def retry(self): return self._controller.retry()


def keep_webview_in_client(window):
    """Install on before_show (GUI thread); repair observed native child drift.

    WinForms can retain Bounds=(0, 0, ...) while the actual WebView HWND sits
    offscreen after minimize/restore. Comparing cached Bounds or repainting
    does not repair that mismatch. Keep the page and its current state alive.
    """
    from ctypes import wintypes
    from System.Windows.Forms import FormWindowState
    form, view = window.native, window.native.webview
    api = ctypes.WinDLL("user32", use_last_error=True)
    api.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int, wintypes.UINT]
    api.SetWindowPos.restype = wintypes.BOOL
    adjusting = False

    def fit(*_):
        nonlocal adjusting
        if adjusting or form.IsDisposed or view.IsDisposed or form.WindowState == FormWindowState.Minimized:
            return
        bounds = form.ClientRectangle
        if bounds.Width <= 0 or bounds.Height <= 0:
            return
        if view.RectangleToScreen(view.ClientRectangle) != form.RectangleToScreen(bounds):
            adjusting = True
            try:
                if not api.SetWindowPos(view.Handle.ToInt64(), 0, 0, 0, bounds.Width, bounds.Height, 0x14):
                    raise ctypes.WinError(ctypes.get_last_error())
            finally:
                adjusting = False

    form.Layout += fit
    form.ClientSizeChanged += fit
    form.Activated += fit
    form.Shown += fit
    fit()


def _registry_value(root, path, name):
    import winreg
    try:
        with winreg.OpenKey(getattr(winreg, root), path) as key:
            return winreg.QueryValueEx(key, name)[0]
    except OSError:
        return None


def window_prerequisite_errors():
    """Check before loading pythonnet; never fall back to legacy Internet Explorer."""
    errors = []
    release = _registry_value("HKEY_LOCAL_MACHINE", r"SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full", "Release")
    if not isinstance(release, int) or release < 394802:
        errors.append("未检测到 .NET Framework 4.6.2 或更高版本。请按《依赖说明.md》准备系统适用的 .NET Framework 后重试。")
    client = r"Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
    versions = [_registry_value(root, prefix + client, "pv") for root, prefix in (
        ("HKEY_CURRENT_USER", "SOFTWARE\\"),
        ("HKEY_LOCAL_MACHINE", "SOFTWARE\\WOW6432Node\\"),
        ("HKEY_LOCAL_MACHINE", "SOFTWARE\\"))]
    if not any(isinstance(value, str) and value.split(".")[0].isdigit()
               and int(value.split(".")[0]) >= 86 for value in versions):
        errors.append("未检测到 Microsoft Edge WebView2 Runtime。请按《依赖说明.md》安装微软官方 Evergreen Runtime 后重试。")
    return errors


def shell_document(data, source=None):
    """A content-addressed URL avoids reusing WebView's cached old shell.html."""
    content = (source or HERE / "shell.html").read_bytes()
    directory = data / "desktop-ui"
    directory.mkdir(exist_ok=True)
    document = directory / ("shell-" + hashlib.sha256(content).hexdigest() + ".html")
    if not document.exists():
        document.write_bytes(content)
    elif document.read_bytes() != content:
        raise RuntimeError("桌面界面文件校验失败，请检查用户目录权限后重开窗口。")
    return document


def main():
    global _dll
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--acceptance", action="store_true")
    parser.add_argument("--debug-port", type=int)
    args = parser.parse_args()
    _dll = os.add_dll_directory(str(ROOT / "runtime/Library/bin"))
    missing = window_prerequisite_errors()
    if missing:
        raise RuntimeError("\n\n".join(missing))
    data = (args.data_dir or ROOT.parent / "user-data").resolve()
    data.mkdir(parents=True, exist_ok=True)
    try:
        guard = InstanceGuard(data)
    except RuntimeError:
        # Repeated start never spawns another model. Bring the existing window forward.
        handle = ctypes.windll.user32.FindWindowW(None, "源鉴")
        if handle:
            ctypes.windll.user32.ShowWindow(handle, 9)
            ctypes.windll.user32.SetForegroundWindow(handle)
        return
    controller = None
    try:
        defaults = json.loads((ROOT / "desktop-config.json").read_text(encoding="utf-8-sig"))
        controller = Controller(data, defaults, acceptance=args.acceptance)
        import webview
        webview.logger.setLevel(logging.WARNING)
        if args.debug_port:
            webview.settings["REMOTE_DEBUGGING_PORT"] = args.debug_port
        controller.window = webview.create_window("源鉴", url=str(shell_document(data)),
            js_api=WindowApi(controller), width=1280, height=860, min_size=(940, 650), background_color="#f1f7f5")
        controller.window.events.before_show += keep_webview_in_client
        controller.window.events.shown += controller.start
        controller.window.events.closed += controller.stop_event.set
        webview.start(gui="edgechromium", http_server=True, private_mode=False, storage_path=str(data / "webview"))
    finally:
        if controller:
            controller.close()
        guard.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Fixed text only: no credentials in startup errors or log files.
        message = str(error) if isinstance(error, RuntimeError) else "桌面窗口无法启动。请确认 WebView2 Runtime 与 .NET Framework 已安装，或使用随包的排错启动入口查看错误。"
        ctypes.windll.user32.MessageBoxW(0, message, "源鉴", 0x10)
