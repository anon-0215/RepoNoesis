import importlib.util
from pathlib import Path
import unittest
import json
import tempfile
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gui_settings import SettingsStore
spec = importlib.util.spec_from_file_location("launcher_enhancement", HERE / "launcher.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class EnhancementConfigTests(unittest.TestCase):
    def test_saved_user_flags_override_release_defaults_and_survive_settings_save(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            (data / 'desktop-settings.json').write_text(json.dumps({'hyde_enabled': False,
                'rerank_enabled': True}), encoding='utf-8')
            store = SettingsStore(data, {'hyde_enabled': True, 'rerank_enabled': True})
            store.save({'model_directory': '.', 'git_directory': '.', 'api_key': ''})
            restored = SettingsStore(data, {'hyde_enabled': True, 'rerank_enabled': False})
            env = launcher.child_environment(data / 'app', restored.runtime(), data, 'saved-flags')
            self.assertEqual(env['RETRIEVAL_HYDE_ENABLED'], 'false')
            self.assertEqual(env['RETRIEVAL_RERANK_ENABLED'], 'true')

    def test_missing_flags_preserve_original_disabled_default(self):
        env = launcher.child_environment(Path.cwd(), {'git_directory': '.', 'model_directory': '.'},
                                         Path.cwd() / 'test-data', 'defaults')
        self.assertEqual(env['RETRIEVAL_HYDE_ENABLED'], 'false')
        self.assertEqual(env['RETRIEVAL_RERANK_ENABLED'], 'false')

    def test_independent_flags_are_passed_to_owned_backend(self):
        for hyde, rerank in ((False,False),(True,False),(False,True),(True,True)):
            config = {"git_directory":".","model_directory":".","hyde_enabled":hyde,"rerank_enabled":rerank}
            env=launcher.child_environment(Path.cwd(),config,Path.cwd()/"test-data","test")
            self.assertEqual(env["RETRIEVAL_HYDE_ENABLED"],str(hyde).lower())
            self.assertEqual(env["RETRIEVAL_RERANK_ENABLED"],str(rerank).lower())
            self.assertEqual(env["RETRIEVAL_ENHANCEMENT_SECONDS"],"15")


if __name__ == "__main__":
    unittest.main()
