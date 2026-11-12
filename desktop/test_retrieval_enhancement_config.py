import importlib.util
from pathlib import Path
import unittest
import json
import tempfile
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'backend'))
from gui_settings import SettingsStore
from unittest.mock import patch, Mock
from app.config import get_retrieval_enhancement_settings
spec = importlib.util.spec_from_file_location("launcher_enhancement", HERE / "launcher.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class EnhancementConfigTests(unittest.TestCase):
    def test_public_save_reopen_environment_and_backend_all_four_combinations(self):
        for hyde, rerank in ((False,False),(True,False),(False,True),(True,True)):
            with self.subTest(hyde=hyde,rerank=rerank), tempfile.TemporaryDirectory() as folder:
                data=Path(folder)
                store=SettingsStore(data,{'hyde_enabled':True,'rerank_enabled':True})
                result=store.save({'hyde_enabled':hyde,'rerank_enabled':rerank})
                self.assertEqual((result['hyde_enabled'],result['rerank_enabled']),(hyde,rerank))
                reopened=SettingsStore(data,{'hyde_enabled':not hyde,'rerank_enabled':not rerank})
                env=launcher.child_environment(data,dict(reopened.load(),git_directory='.',model_directory='.'),data,'test')
                with patch.dict('os.environ',env,clear=True):
                    effective=get_retrieval_enhancement_settings()
                self.assertEqual((effective.hyde_enabled,effective.rerank_enabled),(hyde,rerank))

    def test_old_missing_values_and_invalid_strings_are_not_truthy(self):
        with tempfile.TemporaryDirectory() as folder:
            data=Path(folder)
            self.assertFalse(SettingsStore(data,{}).public()['hyde_enabled'])
            self.assertTrue(SettingsStore(data,{'hyde_enabled':True}).public()['hyde_enabled'])
            (data/'desktop-settings.json').write_text(json.dumps({'hyde_enabled':'false','rerank_enabled':0}),encoding='utf-8')
            store=SettingsStore(data,{'hyde_enabled':True,'rerank_enabled':True})
            self.assertEqual((store.public()['hyde_enabled'],store.public()['rerank_enabled']),(False,False))
            before=store.path.read_bytes()
            for value in ('false', 'true', 0, 1, None, []):
                with self.assertRaises(ValueError):store.save({'hyde_enabled':value})
                self.assertEqual(before,store.path.read_bytes())

    def test_flag_only_save_preserves_key_budgets_resources_and_unrelated_values(self):
        with tempfile.TemporaryDirectory() as folder:
            store=SettingsStore(Path(folder),{})
            original={'api_base':'https://provider.invalid/v1','api_model':'same-model',
                'model_directory':'same-model-path','git_directory':'same-git-path',
                'api_key_dpapi':'opaque-encrypted-do-not-decrypt','device':'cuda','batch_size':1,
                'max_length':8192,'custom':{'keep':True},'agent_runtime':{'LLM_PLANNER_MAX_GENERATION_TOKENS':16384}}
            store.path.write_text(json.dumps(original),encoding='utf-8')
            store.save({'hyde_enabled':True,'rerank_enabled':False})
            current=store.load()
            for key,value in original.items():self.assertEqual(current[key],value)
            store.save({'api_model':'same-model'})
            self.assertTrue(store.public()['hyde_enabled'])
            self.assertFalse(store.public()['rerank_enabled'])
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
