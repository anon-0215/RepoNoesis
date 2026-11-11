"""Real WinForms/WebView2 regression, without backend/model/Provider calls.

The 32049/32083 offset was observed in the user's live blank window. This
injects that geometry deterministically; it is separate from the live repro.
"""
import argparse
import ctypes
from ctypes import wintypes
import importlib.util
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument("package", type=Path)
parser.add_argument("--baseline", action="store_true")
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
package = args.package.resolve()
sys.path.insert(0, str(package / "desktop"))
spec = importlib.util.spec_from_file_location("host", package / "desktop/desktop_host.py")
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)
import webview

result = {"baseline": args.baseline, "checks": []}
window = webview.create_window("源鉴布局回归", html="<h1>WebView layout regression</h1>", width=720, height=500)
if not args.baseline:
    window.events.before_show += host.keep_webview_in_client


def run():
    try:
        assert window.events.loaded.wait(20), "WebView not loaded"
        from System import Action
        from System.Windows.Forms import Form, FormWindowState
        native = window.native
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_int, ctypes.c_int, wintypes.UINT]
        user32.SetWindowPos.restype = wintypes.BOOL

        def check():
            view = native.webview
            expected = native.RectangleToScreen(native.ClientRectangle)
            actual = view.RectangleToScreen(view.ClientRectangle)
            result["checks"].append({"expected": str(expected), "actual": str(actual)})
            assert actual == expected, "WebView control is outside the native client area"

        def corrupt():
            size = native.ClientSize
            assert user32.SetWindowPos(native.webview.Handle.ToInt64(), 0, 32049, 32083,
                                       size.Width, size.Height, 0x14)

        def reactivate():
            other = Form()
            other.Text = "布局回归临时窗口"
            other.ShowInTaskbar = False
            other.Show()
            native.Activate()
            other.Close()
            other.Dispose()

        native.Invoke(Action(corrupt))
        native.Invoke(Action(reactivate))
        native.Invoke(Action(check))
        for state in (FormWindowState.Minimized, FormWindowState.Normal,
                      FormWindowState.Maximized, FormWindowState.Normal):
            native.Invoke(Action(lambda: setattr(native, "WindowState", state)))
            if state != FormWindowState.Minimized:
                native.Invoke(Action(check))
        result["passed"] = True
    except Exception as error:
        result["passed"] = False
        result["error"] = str(error)
    finally:
        window.destroy()


webview.start(run, gui="edgechromium", private_mode=True)
args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
raise SystemExit(0 if result.get("passed") else 1)
