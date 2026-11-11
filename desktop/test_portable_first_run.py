"""Offline first-run checks: no model, Provider, old configuration or real keys."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

package = Path(__file__).resolve().parents[1]
if __name__ == '__main__' and '--package' in sys.argv:
    index = sys.argv.index('--package')
    package = Path(sys.argv[index + 1]).resolve()
    del sys.argv[index:index + 2]
print('First-run host under test:', package)
spec = importlib.util.spec_from_file_location("portable_host", package / "desktop/desktop_host.py")
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)
import gui_settings


class PortableFirstRunTests(unittest.TestCase):
    def test_future_backend_paths_are_under_this_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "user-data"
            config = {"model_directory": str(root / "model"), "git_directory": str(root / "git"),
                      "llm": {"base_url": "", "model": "", "api_key": ""}}
            env = host.child_environment(root / "app", config, data, "offline-check")
            self.assertEqual(Path(env["GITLEARN_DB"]), data / "database/reponoesis.sqlite")
            self.assertTrue(Path(env["RUNTIME_DATA_DIR"]).is_relative_to(data))
            self.assertTrue(Path(env["EMBEDDING_CACHE_DIR"]).is_relative_to(data))
            self.assertEqual(env["REPONOESIS_SKIP_ENV_FILE"], "true")
            self.assertEqual(env["LLM_API_KEY"], "")

    def test_empty_save_and_restore_never_uses_dpapi(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(gui_settings, "protect", side_effect=AssertionError("DPAPI must not run")):
            data = Path(folder)
            defaults = {"model_directory": "", "git_directory": "", "api_base": "", "api_model": ""}
            store = gui_settings.SettingsStore(data, defaults)
            self.assertFalse(store.path.exists())
            self.assertFalse(store.public()["key_saved"])
            store.save({**defaults, "api_key": ""})
            restored = gui_settings.SettingsStore(data, defaults)
            self.assertFalse(restored.public()["key_saved"])
            self.assertNotIn("api_key_dpapi", restored.load())
            self.assertNotIn("api_key", restored.load())
            self.assertEqual(restored.runtime()["llm"]["api_key"], "")
            self.assertEqual(len(restored.resource_errors()), 2)

    def test_missing_resources_do_not_spawn_backend(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(host.subprocess, "Popen", side_effect=AssertionError("No backend")) as popen:
            controller = host.Controller(Path(folder), {"model_directory": "", "git_directory": ""})
            controller.start()
            try:
                deadline = time.monotonic() + 2
                while controller.get_status()["phase"] == "checking" and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertEqual(controller.get_status()["phase"], "settings")
                self.assertIsNone(controller.child)
                popen.assert_not_called()
                self.assertFalse(list((Path(folder) / "database").glob("*.sqlite")))
            finally:
                controller.close()

    def test_missing_native_dependencies_are_actionable(self):
        with patch.object(host, "_registry_value", return_value=None):
            errors = host.window_prerequisite_errors()
        self.assertEqual(len(errors), 2)
        self.assertIn(".NET Framework", errors[0])
        self.assertIn("WebView2", errors[1])
        self.assertTrue(all("依赖说明.md" in error for error in errors))

    def test_webview_missing_cannot_fall_back_to_internet_explorer(self):
        with patch.object(host, "_registry_value", side_effect=lambda root, path, name: 533320 if name == "Release" else None):
            self.assertEqual(len(host.window_prerequisite_errors()), 1)

    def test_existing_native_dependencies_pass(self):
        with patch.object(host, "_registry_value", side_effect=lambda root, path, name: 533320 if name == "Release" else "156.0.4314.56"):
            self.assertEqual(host.window_prerequisite_errors(), [])


if __name__ == "__main__":
    unittest.main()
