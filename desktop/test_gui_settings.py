import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gui_settings import SettingsStore, git_directory, protect
from launcher import child_environment


class SettingsTests(unittest.TestCase):
    def test_missing_resources_do_not_prevent_settings_save(self):
        with tempfile.TemporaryDirectory() as folder:
            store = SettingsStore(Path(folder), {"device": "auto", "batch_size": 1, "max_length": 8192})
            self.assertEqual(len(store.resource_errors()), 2)
            result = store.save({"model_directory": "missing", "git_directory": "missing"})
            self.assertFalse(result["key_saved"])
            self.assertEqual(result["device"], "auto")
            self.assertEqual(len(store.resource_errors()), 2)

    @unittest.skipUnless(os.name == "nt", "Windows DPAPI")
    def test_key_roundtrip_is_encrypted_masked_and_explicitly_clearable(self):
        with tempfile.TemporaryDirectory() as folder:
            store = SettingsStore(Path(folder), {})
            fake = "test-only-placeholder-secret"
            store.save({"api_base": "https://provider.invalid/v1", "api_model": "test", "api_key": fake})
            self.assertNotIn(fake, store.path.read_text())
            self.assertNotIn("api_key_dpapi", store.public())
            self.assertNotIn("api_key", store.public())
            self.assertTrue(store.public()["key_saved"])
            self.assertEqual(store.runtime()["llm"]["api_key"], fake)
            store.save({"api_base": "https://provider.invalid/v1", "api_model": "test", "api_key": ""})
            self.assertEqual(store.runtime()["llm"]["api_key"], fake)
            store.save({"clear_key": True})
            self.assertFalse(store.public()["key_saved"])

    def test_url_with_credentials_or_query_is_rejected_before_persistence(self):
        with tempfile.TemporaryDirectory() as folder:
            store = SettingsStore(Path(folder), {})
            for address in ("https://user:password@example.invalid/v1", "https://example.invalid?key=test", "file:///tmp"):
                with self.assertRaises(ValueError):
                    store.save({"api_base": address})
            self.assertFalse(store.path.exists())

    def test_git_executable_or_install_directory_is_normalized(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root / "mingw64/bin/git.exe"
            binary.parent.mkdir(parents=True)
            binary.touch()
            self.assertEqual(git_directory(str(root)), str(binary.parent))
            self.assertEqual(git_directory(str(binary)), str(binary.parent))

    def test_only_explicit_saved_provider_values_enter_child_environment(self):
        root = Path("C:/portable")
        config = {"git_directory": "C:/Git", "model_directory": "C:/model", "llm": {
            "base_url": "https://provider.invalid/v1", "model": "test", "api_key": "placeholder"}}
        env = child_environment(root, config, Path("C:/data"), "instance")
        self.assertEqual(env["LLM_API_KEY"], "placeholder")
        self.assertEqual(env["LLM_MODEL"], "test")
        self.assertNotIn("LLM_PLANNER_THINKING", env)
        self.assertNotIn("LLM_ANSWER_THINKING", env)


if __name__ == "__main__":
    unittest.main()
