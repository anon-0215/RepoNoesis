"""Actual coordinator + LLMPlanner inputs; scripted reachability, not model quality."""
from dataclasses import replace
import json,unittest
from unittest.mock import patch
from app.services import agent_core
from app.services.agent_contracts import AgentLimits
from tests.test_source_read_information_flow import SourceReadInformationFlowTests
from tests.test_m2_agent import decision
from app.services.code_chunker import extract_python_code_chunks

def rows(model,index=-1):
    return model.payload(index).get('planner_evidence_view',{}).get('rows',[])

class PlannerUnifiedViewTests(unittest.TestCase):
    def setUp(self):
        self.flow=SourceReadInformationFlowTests();self.flow.setUp();self.addCleanup(self.flow.doCleanups)
        self.observations=[]
        execute=agent_core._execute_agent_tool
        def observe(*args,**kwargs):
            value=execute(*args,**kwargs);self.observations.append(value[1]);return value
        mock=patch.object(agent_core,'_execute_agent_tool',side_effect=observe);mock.start();self.addCleanup(mock.stop)

    def neutral_chain(self):
        from tests.test_source_read_information_flow import REVISION,FILLERS
        sources={'api/entry.py':'def entry(active=True):\n    if active:\n        return bridge()\n    return None\n',
            'impl/bridge.py':'def bridge():\n    """'+('Stored source padding for precise coverage only.\n    '*60)+'"""\n    return Terminal()\n',
            'storage/terminal.py':'class Terminal:\n    def close(self):\n        return "terminal-committed"\n',
            'other/fillers.py':FILLERS}
        pid=self.flow.db.create_project({'repo_url':'https://example.invalid/neutral-chain','owner':'offline','repo':'neutral','repository_revision':REVISION,'default_branch':'main'})
        files=[{'path':p,'content':s,'language':'Python','extension':'.py','size':len(s.encode()),'summary':'fixture','importance':1,'is_core':True,'imports':[],'exports':[],'symbols':[]} for p,s in sources.items()]
        chunks=[c.to_dict() for p,s in sources.items() for c in extract_python_code_chunks(p,s,REVISION).chunks]
        self.flow.db.save_analysis(pid,{'overview':'offline','primary_language':'Python'},files,[],chunks)
        self.flow.pid=pid;self.flow.bundle=self.flow.db.get_bundle(pid)
        self.flow.chunks={c['qualified_name']:c for c in self.flow.db.get_code_chunks(pid)}
        self.flow.base=['entry','filler_0','filler_1','filler_2','filler_3']

    def test_neutral_cross_file_search_lookup_and_recovery(self):
        for search in [False,True]:
            self.neutral_chain();middle=self.flow.chunks['bridge']
            locate=decision('continue','search_code',{'query':'bridge','symbol':'bridge'}) if search else decision('continue','lookup_symbol',{'symbol':'bridge','match_mode':'exact'})
            actions=[decision('continue','lookup_symbol',{'symbol':'missing_optional_context'}),locate,self.flow.read(middle),decision('continue','lookup_symbol',{'symbol':'Terminal','match_mode':'exact'})]
            windows=[self.flow.base,['bridge']] if search else None
            model,events,result=self.flow.run_flow(actions,retrieval_windows=windows,label='neutral-full-chain-'+str(search))
            self.assertIn('return bridge()',json.dumps(rows(model,0)))
            partial=next(r for r in rows(model,2) if r['symbol']=='bridge')
            self.assertTrue(partial['excerpt_coverage']['truncated'])
            complete=next(r for r in rows(model,3) if r['symbol']=='bridge')
            self.assertEqual(complete['untrusted_source_excerpt'],middle['content'])
            self.assertIn('terminal-committed',json.dumps(rows(model,4)))
            self.assertEqual(result['budget_usage']['steps_used'],5)
            self.assertEqual(result['budget_usage']['tool_calls_used'],4)
            self.assertEqual({r['qualified_name'] for r in result['citations']} & {'entry','bridge','Terminal'},{'entry','bridge','Terminal'})
            for name in ['entry','bridge','Terminal']:
                self.assertEqual(next(r for r in model.final_rows() if r['symbol']==name)['untrusted_source_excerpt'],self.flow.chunks[name]['content'])

    def test_authority_cache_preview_and_final_budget_independence(self):
        self.flow.fixture('click');states=[];get=agent_core._get_valid_decision
        self.flow.base=['filler_0','filler_1','open_file','_LazyFile','_LazyFile.open']
        def inspect(planner,state,*args):
            before=dict(state.delivered_source_prefixes)
            with patch.object(self.flow.db,'get_evidence_source',side_effect=AssertionError('preview queried source')):
                views=[agent_core._planner_state(state)['planner_evidence_view'] for _ in range(3)]
            self.assertEqual(views[0],views[-1]);self.assertEqual(before,state.delivered_source_prefixes)
            states.append(state);return get(planner,state,*args)
        with patch.object(agent_core,'_get_valid_decision',side_effect=inspect):
            model,events,result=self.flow.run_flow([],limits=replace(AgentLimits(),max_accumulated_evidence_context_bytes=100),label='base-visible-final-excluded')
        self.assertIn('f, should_close = open_stream',json.dumps(rows(model,0)))
        self.assertNotIn('open_file',{r['symbol'] for r in model.final_rows()})
        self.assertNotIn('open_file',{r['qualified_name'] for r in result['citations']})
        self.assertEqual(states[-1].progress_by_call,{})

    def test_projection_omissions_do_not_create_unrelated_progress(self):
        self.flow.fixture('click');states=[];get=agent_core._get_valid_decision
        def inspect(planner,state,*args):states.append(state);return get(planner,state,*args)
        with patch.object(agent_core,'_get_valid_decision',side_effect=inspect):
            model,events,result=self.flow.run_flow([decision('continue','lookup_symbol',{'symbol':'missing_a'}),decision('continue','lookup_symbol',{'symbol':'missing_b'})],limits=replace(AgentLimits(),max_source_read_bytes=128),label='unified-projection-omitted')
        self.assertEqual(rows(model,0),[])
        self.assertTrue(all(r['reason']=='projection_byte_budget' for r in model.payload(0)['planner_evidence_view']['omissions']))
        self.assertEqual(states[-1].delivered_source_prefixes,{})
        self.assertEqual(states[-1].no_progress_count,2)

    def test_invalid_base_source_and_stale_binding_cannot_project(self):
        self.flow.fixture('click');entry=self.flow.chunks['open_file'];source=self.flow.db.get_evidence_source
        def corrupt(project,chunk,path):
            value=source(project,chunk,path)
            if chunk==entry['id']:
                value=dict(value);value['file_content']=value['file_content'].replace('return f','return forged')
            return value
        with patch.object(self.flow.db,'get_evidence_source',side_effect=corrupt):
            model,events,result=self.flow.run_flow([],label='invalid-base-authority')
        self.assertNotIn('open_file',{r['symbol'] for r in rows(model,0)})
        self.assertTrue(any(r['reason']=='not_valid_current_evidence' for r in model.payload(0)['planner_evidence_view']['omissions']))

    def test_promotion_followed_by_file_authority_failure_is_not_progress(self):
        self.flow.fixture('click');target=self.flow.chunks['_AtomicFile'];source=self.flow.db.get_evidence_source
        states=[];get=agent_core._get_valid_decision
        def inspect(planner,state,*args):states.append(state);return get(planner,state,*args)
        def corrupt(project,chunk,path):
            value=source(project,chunk,path)
            if chunk==target['id']:
                value=dict(value);value['file_content']=value['file_content'].replace('os.replace','os.invalid')
            return value
        with patch.object(self.flow.db,'get_evidence_source',side_effect=corrupt),patch.object(agent_core,'_get_valid_decision',side_effect=inspect):
            model,events,result=self.flow.run_flow([decision('continue','lookup_symbol',{'symbol':'_AtomicFile','match_mode':'exact'})],label='post-promotion-authority-failed')
        self.assertEqual(sum(states[-1].progress_by_call.values()),0)
        self.assertNotIn('_AtomicFile',{r['symbol'] for r in rows(model,1)})
        self.assertNotIn('_AtomicFile',{r['qualified_name'] for r in result['citations']})

    def test_base_first_late_display_and_window_reentry_have_no_tool_credit(self):
        self.flow.fixture('click');c=self.flow.chunks['open_file'];project=agent_core._planner_state
        for window in [False,True]:
            states=[];get=agent_core._get_valid_decision
            def inspect(planner,state,*args):states.append(state);return get(planner,state,*args)
            def filter_view(state):
                value=project(state)
                omit=(len(state.steps)==0) if not window else (1<=len(state.steps)<=2)
                if omit:value['planner_evidence_view']['rows']=[r for r in value['planner_evidence_view']['rows'] if r['symbol']!=c['qualified_name']]
                return value
            actions=[decision('continue','lookup_symbol',{'symbol':'missing_'+str(i)}) for i in range(3 if window else 1)]
            with patch.object(agent_core,'_planner_state',side_effect=filter_view),patch.object(agent_core,'_get_valid_decision',side_effect=inspect):
                model,events,result=self.flow.run_flow(actions,limits=replace(AgentLimits(),max_no_progress_steps=10),label='base-late-or-reentry-'+str(window))
            self.assertIn(c['content'],[r['untrusted_source_excerpt'] for r in rows(model)])
            self.assertEqual(sum(states[-1].progress_by_call.values()),0)
            self.assertEqual(states[-1].no_progress_count,len(actions))
            self.assertEqual(len(states[-1].delivered_source_prefixes),5)

    def test_default_click_complete_chain_and_detour(self):
        for detour in [False,True]:
            self.flow.fixture('click')
            middle=self.flow.chunks['open_stream'];end=self.flow.chunks['_AtomicFile']
            actions=([self.flow.read(self.flow.chunks['_LazyFile'])] if detour else [])+[
                decision('continue','lookup_symbol',{'symbol':'open_stream','match_mode':'exact'}),
                self.flow.read(middle),
                decision('continue','lookup_symbol',{'symbol':'_AtomicFile','match_mode':'exact'})]
            model,events,result=self.flow.run_flow(actions,label='default-full-click'+('-detour' if detour else ''))
            self.assertIn('f, should_close = open_stream',json.dumps(rows(model,0)))
            lookup_index=2 if detour else 1
            partial=next(r for r in rows(model,lookup_index) if r['symbol']=='open_stream')
            self.assertTrue(partial['excerpt_coverage']['truncated'])
            completed=next(r for r in rows(model,lookup_index+1) if r['symbol']=='open_stream')
            self.assertEqual(completed['untrusted_source_excerpt'],middle['content'])
            self.assertIn('af = _AtomicFile(',completed['untrusted_source_excerpt'])
            atomic=next(r for r in rows(model) if r['symbol']=='_AtomicFile')
            self.assertEqual(atomic['untrusted_source_excerpt'],end['content'])
            self.assertIn('os.replace',atomic['untrusted_source_excerpt'])
            self.assertIn('self.close(delete=exc_type is not None)',atomic['untrusted_source_excerpt'])
            self.assertLessEqual(result['budget_usage']['steps_used'],5)
            self.assertEqual(len(model.planner_messages),len(actions)+1)
            self.assertEqual(len(events[0]['after']),5)
            self.assertEqual(events[0]['policy']['capacity'],8)
            for name in ['open_file','open_stream','_AtomicFile']:
                c=self.flow.chunks[name]
                final=next(r for r in model.final_rows() if r['symbol']==name)
                self.assertEqual(final['untrusted_source_excerpt'],c['content'])
                cite=next(r for r in result['citations'] if r['qualified_name']==name)
                self.assertEqual((cite['path'],cite['start_line'],cite['end_line']),(c['path'],c['start_line'],c['end_line']))

    def test_initial_base_and_lookup_have_body_without_unlock_read(self):
        self.flow.fixture('click')
        model,events,result=self.flow.run_flow([decision('continue','lookup_symbol',{'symbol':'_AtomicFile','match_mode':'exact'})],label='base-lookup-direct')
        self.assertIn('f, should_close = open_stream',json.dumps(rows(model,0)))
        self.assertIn('os.replace',json.dumps(rows(model,1)))
        self.assertFalse(any('canonical_evidence_excerpts' in o.get('source_read',{}) for o in model.payload(1)['untrusted_observation_summaries']))
        self.assertEqual(result['budget_usage']['tool_calls_used'],1)

    def test_tool_quota_visible_and_lookup_still_available(self):
        c=self.flow.fixture('neutral');end=self.flow.chunks['PacketVault.close']['end_line']
        actions=[self.flow.read(c,end=end),self.flow.read(c),self.flow.read(self.flow.chunks['filler_3']),
            decision('continue','lookup_symbol',{'symbol':'missing'}),self.flow.read(c,start=c['start_line']+1)]
        model,events,result=self.flow.run_flow(actions,label='tool-quota')
        tools=model.payload(3)['server_constraints']['tool_availability']
        self.assertEqual((tools['read_source']['used_attempts'],tools['read_source']['limit'],tools['read_source']['remaining']),(3,3,0))
        self.assertFalse(tools['read_source']['available']);self.assertTrue(tools['lookup_symbol']['available'])
        self.assertNotIn('read_source',model.payload(3)['server_constraints']['allowed_actions']['tool_names'])
        call=result['agent_trace'][-1]['tool_calls'][0]
        self.assertEqual(call['status'],'rejected')
        self.assertEqual(self.observations[-1].error['reason'],'same_tool_limit')

    def test_duplicate_reason_and_initial_delivery_not_tool_progress(self):
        self.flow.fixture('click');c=self.flow.chunks['open_file'];states=[]
        original=agent_core._get_valid_decision
        def record(planner,state,*args):
            states.append(state);return original(planner,state,*args)
        with patch.object(agent_core,'_get_valid_decision',side_effect=record):
            model,events,result=self.flow.run_flow([self.flow.read(c)]*3,label='already-visible-duplicate')
        self.assertEqual(len(model.planner_messages),2)
        self.assertEqual(sum(states[-1].progress_by_call.values()),0)
        self.assertEqual(states[-1].no_progress_count,2)
        self.assertEqual(self.observations[-1].error['reason'],'identical_successful_call')
