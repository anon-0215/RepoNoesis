"""Desktop -> environment -> real serializer -> bounded Agent (no network/model)."""
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'desktop'), str(ROOT / 'backend')]
from gui_settings import SettingsStore, desktop_runtime
from launcher import child_environment
from app.config import get_agent_limits, get_llm_settings
from app.database import Database
from app.services.agent_core import run_bounded_agent, _base_request_top_k
from app.services.llm_client import LLMClient, ProviderError
from app.services.smoke_diagnostics import SmokeDiagnosticsRecorder
from tests.m1_helpers import make_project, disabled_embedding_service
from tests.test_m2_agent import decision


class AgentRuntimeTests(unittest.TestCase):
    def environment(self, data, runtime=None, **overrides):
        store = SettingsStore(data, {'api_base': 'https://api.deepseek.com/v1',
            'api_model': 'deepseek-flash', 'model_directory': '.', 'git_directory': '.', **overrides})
        if runtime is not None:
            store.save({'api_base': 'https://api.deepseek.com/v1', 'api_model': 'deepseek-flash',
                'agent_runtime': runtime})
        config = store.runtime()
        config['llm']['api_key'] = 'fixture-only'
        return child_environment(ROOT, config, data, 'test')

    def test_old_settings_defaults_reach_stage_requests_without_inherited_env(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ,
                {'LLM_PLANNER_MAX_GENERATION_TOKENS': '512', 'LLM_PLANNER_THINKING': 'disabled'}):
            env = self.environment(Path(folder))
        self.assertEqual(env['LLM_PLANNER_THINKING'], 'enabled')
        self.assertEqual(env['LLM_PLANNER_REASONING_EFFORT'], 'high')
        with patch.dict(os.environ, env, clear=True):
            settings, limits = get_llm_settings(), get_agent_limits()
        self.assertEqual((settings.planner_max_generation_tokens, settings.answer_max_generation_tokens), (8192,8192))
        self.assertEqual((limits.max_agent_steps, limits.max_tool_calls), (5,8))
        self.assertIsNone(limits.base_evidence_count)
        for mode,cap,expected in [('agent',8,5),('rag',8,8),('agent',5,5),('rag',5,5)]:
            self.assertEqual(_base_request_top_k(cap,limits,mode),expected)
        self.assertEqual((limits.total_deadline_ms, limits.min_final_answer_budget_ms), (300000,90000))
        self.assertEqual(settings.timeout_seconds, 90)
        self.assertEqual(limits.max_planner_output_tokens_per_step, 1024)
        self.assertEqual(limits.max_final_answer_tokens, 4096)
        sent=[]
        def opener(req, **kwargs):
            sent.append(json.loads(req.data))
            return io.BytesIO(json.dumps({'choices':[{'finish_reason':'stop','message':{'content':'ok'}}]}).encode())
        client=LLMClient(settings, opener=opener)
        with self.assertLogs('uvicorn.error',level='INFO') as logs:
            for purpose, thinking in [('planner',settings.planner_thinking),('final_answer',settings.answer_thinking)]:
                client.chat([{'role':'user','content':'private-prompt'}], purpose=purpose, thinking=thinking, max_tokens=512)
        self.assertNotIn('fixture-only',''.join(logs.output))
        self.assertNotIn('private-prompt',''.join(logs.output))
        self.assertTrue(all(p['max_tokens']==8192 and p['thinking']=={'type':'enabled'} and p['reasoning_effort']=='high' for p in sent))

    def test_explicit_values_unknown_fields_and_encrypted_key_survive_other_save(self):
        with tempfile.TemporaryDirectory() as folder:
            data=Path(folder)
            saved={'api_base':'https://api.deepseek.com/v1','api_model':'deepseek-flash',
                'api_key_dpapi':'opaque-do-not-decrypt','unrelated':{'keep':True},
                'agent_runtime':{'LLM_PLANNER_MAX_GENERATION_TOKENS':16384,
                'LLM_PLANNER_THINKING':'disabled','LLM_PLANNER_REASONING_EFFORT':'low'}}
            (data/'desktop-settings.json').write_text(json.dumps(saved),encoding='utf-8')
            store=SettingsStore(data,{})
            store.save({'api_base':saved['api_base'],'api_model':saved['api_model'],'api_key':''})
            current=store.load()
            self.assertEqual(current,saved|{'port':18771,'hyde_enabled':False,'rerank_enabled':False})
            runtime=desktop_runtime(current)
            self.assertEqual(runtime['LLM_PLANNER_MAX_GENERATION_TOKENS'],16384)
            self.assertEqual(runtime['LLM_PLANNER_REASONING_EFFORT'],'')

    def test_generic_defaults_omit_vendor_extensions_and_off_suppresses_effort(self):
        for model,base in [('other','https://provider.invalid/v1'),('deepseek-flash','https://other.invalid')]:
            config={'api_base':base,'api_model':model,'git_directory':'.','model_directory':'.'}
            env=child_environment(ROOT,config,ROOT/'unused','test')
            self.assertNotIn('LLM_PLANNER_THINKING',env)
            self.assertNotIn('LLM_ANSWER_REASONING_EFFORT',env)
        with tempfile.TemporaryDirectory() as folder:
            env=self.environment(Path(folder), {'LLM_PLANNER_THINKING':'disabled', 'LLM_PLANNER_REASONING_EFFORT':'high'})
        self.assertEqual(env['LLM_PLANNER_THINKING'],'disabled')
        self.assertNotIn('LLM_PLANNER_REASONING_EFFORT',env)

    def test_invalid_and_conflicting_budgets_fail_before_save(self):
        for runtime in [{'LLM_PLANNER_MAX_GENERATION_TOKENS':True}, {'LLM_ANSWER_MAX_GENERATION_TOKENS':0},
            {'LLM_PLANNER_THINKING':'maybe'}, {'AGENT_TOTAL_DEADLINE_MS':90000},
            {'AGENT_FINAL_ANSWER_RESERVE_MS':5000}, {'AGENT_MAX_TOTAL_PLANNER_TOKENS':64}]:
            with self.subTest(runtime=runtime), tempfile.TemporaryDirectory() as folder:
                store=SettingsStore(Path(folder),{})
                with self.assertRaises(ValueError):store.save({'agent_runtime':runtime})
                self.assertFalse(store.path.exists())

    def test_full_tool_second_planner_and_final_with_serialized_requests(self):
        with tempfile.TemporaryDirectory() as folder:
            data=Path(folder); env=self.environment(data)
            with patch.dict(os.environ,env,clear=True):
                limits,settings=get_agent_limits(),get_llm_settings()
            db=Database(data/'fixture.sqlite')
            _,bundle=make_project(db,[('entry.py','entry','def entry():\n    return helper()\n'),
                ('helper.py','helper','def helper():\n    return 1\n')])
            contents=[decision('continue','lookup_symbol',{'symbol':'helper'}),decision('answer'),
                {'parts':[{'text':'The source returns one.','evidence_aliases':['A1']}]}]
            sent=[]
            def opener(req,**kwargs):
                sent.append((json.loads(req.data),kwargs['timeout']))
                return io.BytesIO(json.dumps({'choices':[{'finish_reason':'stop','message':{
                    'content':json.dumps(contents[len(sent)-1]),'reasoning_content':'not visible'}}]}).encode())
            recorder=SmokeDiagnosticsRecorder()
            result=run_bounded_agent('entry',bundle,LLMClient(settings,opener=opener,diagnostics_recorder=recorder),db,
                disabled_embedding_service(),limits=limits,diagnostics_recorder=recorder)
            self.assertEqual(result['answer_mode'],'llm_grounded')
            self.assertTrue(result['citations'])
            self.assertEqual(len(sent),3)
            self.assertTrue(all(p['max_tokens']==8192 and p['reasoning_effort']=='high' for p,_ in sent))
            self.assertTrue(all(0<t<=90 for _,t in sent))
            snap=recorder.snapshot()
            self.assertEqual(len(snap['planner_attempts']),2)
            self.assertEqual(snap['tool_executions'][0]['status'],'succeeded')
            self.assertIn('The source returns one.',result['answer'])

    def test_partial_user_runtime_merges_release_values_by_key(self):
        with tempfile.TemporaryDirectory() as folder:
            data=Path(folder)
            (data/'desktop-settings.json').write_text(json.dumps({'agent_runtime':{
                'LLM_ANSWER_THINKING':'disabled'}}),encoding='utf-8')
            store=SettingsStore(data,{'agent_runtime':{'LLM_PLANNER_MAX_GENERATION_TOKENS':16384}})
            self.assertEqual(desktop_runtime(store.load())['LLM_PLANNER_MAX_GENERATION_TOKENS'],16384)
            store.save({'api_key':''})
            self.assertEqual(store.load()['agent_runtime']['LLM_ANSWER_THINKING'],'disabled')

    def test_automatic_modes_do_not_stick_after_provider_change(self):
        with tempfile.TemporaryDirectory() as folder:
            store=SettingsStore(Path(folder),{})
            store.save({'api_base':'https://api.deepseek.com','api_model':'deepseek-flash'})
            self.assertEqual(desktop_runtime(store.load())['LLM_PLANNER_THINKING'],'enabled')
            store.save({'api_base':'https://provider.invalid/v1','api_model':'other'})
            self.assertEqual(desktop_runtime(store.load())['LLM_PLANNER_THINKING'],'')


if __name__=='__main__':unittest.main()
