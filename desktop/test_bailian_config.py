import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from desktop.gui_settings import SettingsStore, protect
from desktop import launcher


class BailianConfigTests(unittest.TestCase):
    def test_legacy_default_and_bailian_independent_save_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            data=Path(directory);store=SettingsStore(data,{})
            self.assertEqual(store.public()['rerank_provider'],'generation')
            original={'api_model':'deepseek-flash','api_base':'https://api.deepseek.com',
                      'api_key_dpapi':protect('deepseek-only'),'batch_size':1,'device':'cuda',
                      'git_directory':'.','model_directory':'.','custom':{'retain':True},
                      'agent_runtime':{'LLM_TIMEOUT_SECONDS':90},'hyde_enabled':False}
            store.path.write_text(json.dumps(original),encoding='utf8')
            public=store.save({'rerank_provider':'bailian','rerank_enabled':True,'bailian_api_key':'bailian-only'})
            self.assertTrue(public['bailian_key_saved']);self.assertNotIn('bailian-only',str(public))
            reopened=SettingsStore(data,{})
            config=reopened.runtime()
            self.assertEqual(config['bailian_api_key'],'bailian-only')
            for key,value in original.items():self.assertEqual(reopened.load()[key],value)
            env=launcher.child_environment(data,config,data,'test')
            self.assertEqual(env['DASHSCOPE_API_KEY'],'bailian-only')
            self.assertEqual(env['LLM_API_KEY'],'deepseek-only')
            self.assertEqual(env['LLM_MODEL'],'deepseek-flash')
            self.assertEqual(env['AGENT_TOTAL_DEADLINE_MS'],'300000')
            self.assertEqual(env['RETRIEVAL_HYDE_ENABLED'],'false')
            saved=reopened.load()['bailian_api_key_dpapi']
            reopened.save({'bailian_api_key':''})
            self.assertEqual(reopened.load()['bailian_api_key_dpapi'],saved)
            reopened.save({'bailian_clear_key':True})
            self.assertFalse(reopened.public()['bailian_key_saved'])

    def test_environment_fallback_does_not_change_original_key(self):
        with tempfile.TemporaryDirectory() as directory,patch('desktop.gui_settings.bailian_environment_key',return_value='bailian-env'):
            store=SettingsStore(Path(directory),{'api_base':'','api_model':''})
            self.assertTrue(store.public()['bailian_environment_key_present'])
            config=store.runtime();self.assertEqual(config['bailian_api_key'],'bailian-env')
            self.assertEqual(config['llm']['api_key'],'')

    def test_invalid_settings_do_not_write(self):
        with tempfile.TemporaryDirectory() as directory:
            store=SettingsStore(Path(directory),{})
            for incoming in [{'rerank_provider':'other'},{'bailian_endpoint':'https://evil.invalid'},
                             {'bailian_timeout_seconds':float('nan')},{'bailian_timeout_seconds':True},
                             {'bailian_api_key':False}]:
                with self.assertRaises(ValueError):store.save(incoming)
                self.assertFalse(store.path.exists())

    def test_migration_preserves_explicit_bailian_and_new_mode_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            data=Path(directory)
            (data/'desktop-settings.json').write_text(json.dumps({'rerank_provider':'bailian','rerank_enabled':True}),encoding='utf8')
            before=(data/'desktop-settings.json').read_bytes()
            store=SettingsStore(data,{'rerank_mode':'generation'})
            self.assertEqual(store.public()['rerank_mode'],'independent')
            self.assertEqual(before,store.path.read_bytes())
            store.save({'rerank_mode':'generation'})
            self.assertEqual(SettingsStore(data,{}).public()['rerank_mode'],'generation')
            store.save({'rerank_mode':'independent'})
            self.assertEqual(SettingsStore(data,{}).public()['rerank_provider'],'bailian')

    def test_switching_and_disabling_preserves_independent_settings_and_key(self):
        with tempfile.TemporaryDirectory() as directory:
            store=SettingsStore(Path(directory),{})
            store.save({'rerank_mode':'independent','bailian_api_key':'independent-fixture','bailian_timeout_seconds':5})
            original=store.load()
            for incoming in [{'rerank_enabled':False},{'rerank_mode':'generation'},{'rerank_mode':'independent','rerank_enabled':True}]:
                store.save(incoming)
                for key in ['bailian_api_key_dpapi','bailian_endpoint','bailian_model','bailian_timeout_seconds']:
                    self.assertEqual(store.load()[key],original[key])

    def test_missing_key_clear_and_priority_are_public_without_secrets(self):
        with tempfile.TemporaryDirectory() as directory,patch('desktop.gui_settings.bailian_environment_key',return_value=''):
            store=SettingsStore(Path(directory),{})
            value=store.save({'rerank_mode':'independent','rerank_enabled':True})
            self.assertEqual(value['rerank_missing_fields'],['百炼 API Key'])
            store.save({'bailian_api_key':'private-fixture'})
            with patch.dict('os.environ',{'DASHSCOPE_API_KEY':'environment-fixture'}):
                self.assertEqual(store.public()['bailian_key_source'],'saved')
                self.assertNotIn('private-fixture',str(store.public()))
                store.save({'bailian_api_key':''})
                self.assertTrue(store.public()['bailian_key_saved'])
                store.save({'bailian_clear_key':True})
                self.assertEqual(store.public()['bailian_key_source'],'process_environment')
            self.assertEqual(store.public()['bailian_key_source'],'missing')
