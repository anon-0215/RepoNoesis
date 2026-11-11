import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("build_portable", Path(__file__).with_name("build_portable.py"))
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class PortablePackagePolicyTests(unittest.TestCase):
    def test_runtime_webview_is_program_not_profile(self):
        builder.check_relative(Path("app/runtime/Lib/site-packages/webview/errors.py"))

    def test_personal_artifacts_are_rejected(self):
        for relative in ("user-data/desktop-settings.json", "app/.env", "app/data/x.sqlite",
                         "app/profile/EBWebView/Cookies", "app/logs/backend.log"):
            with self.subTest(relative=relative), self.assertRaises(RuntimeError):
                builder.check_relative(Path(relative))

    def test_generated_cache_excluded_but_licenses_retained(self):
        self.assertIsNotNone(builder.runtime_exclusion(Path("Lib/site-packages/foo/__pycache__/x.pyc")))
        self.assertIsNotNone(builder.runtime_exclusion(Path("conda-meta/history")))
        self.assertIsNone(builder.runtime_exclusion(Path("LICENSE_PYTHON.txt")))
        self.assertIsNone(builder.runtime_exclusion(Path("Lib/site-packages/webview/__init__.py")))


if __name__ == "__main__":
    unittest.main()
