"""Scripted Planner, actual coordinator and message inputs; no model quality claim."""
from dataclasses import replace
import json, os, time
from pathlib import Path
import unittest
from unittest.mock import patch

from app.services import agent_core
from app.services.agent_contracts import AgentLimits
from tests.test_source_read_information_flow import SourceReadInformationFlowTests
from tests.test_m2_agent import decision


class PlannerDeliveryProgressTests(unittest.TestCase):
    def setUp(self):
        self.flow=SourceReadInformationFlowTests(); self.flow.setUp()
        self.addCleanup(self.flow.doCleanups)

    def rows(self, model, index):
        return model.payload(index)['planner_evidence_view']['rows']

    def run_flow(self, actions, **kwargs):
        states=[]; commits=[]; current=[]
        original_get=agent_core._get_valid_decision
        original_commit=agent_core._record_planner_source_delivery
        def snapshot(state):
            return {'no_progress_count':state.no_progress_count,
                    'progress_by_call':dict(state.progress_by_call),
                    'delivered_prefixes':[{'binding':list(k),'text':v} for k,v in state.delivered_source_prefixes.items()]}
        def observe_get(planner,state,*args):
            current[:]=[state]; before=snapshot(state)
            for _ in range(3): agent_core._planner_state(state)
            self.assertEqual(snapshot(state),before,'Previews must not consume delivery')
            states.append(before)
            return original_get(planner,state,*args)
        def observe_commit(state,payload):
            before=snapshot(state); original_commit(state,payload)
            commits.append({'before':before,'after':snapshot(state)})
        with patch.object(agent_core,'_get_valid_decision',side_effect=observe_get),patch.object(agent_core,'_record_planner_source_delivery',side_effect=observe_commit):
            model,events,result=self.flow.run_flow(actions,**kwargs)
        self.state=current[-1]; self.states=states; self.commits=commits
        destination=os.environ.get('SOURCE_FLOW_CAPTURE_DIR')
        if destination:
            (Path(destination)/(kwargs.get('label','progress')+'-progress.json')).write_text(json.dumps({'states_before_invocation':states,'actual_invocation_commits':commits,'final_state':snapshot(self.state)},ensure_ascii=False,indent=2),encoding='utf-8')
        return model,events,result

    def test_default_complete_A_B_then_new_C(self):
        for kind,names in [('neutral',['PacketVault','filler_3','PacketVault.close']),
                           ('click',['_LazyFile','_AtomicFile.close','_AtomicFile.__exit__'])]:
            with self.subTest(kind=kind):
                self.flow.fixture(kind); chunks=[self.flow.chunks[n] for n in names]
                self.assertEqual((AgentLimits().max_no_progress_steps,AgentLimits().max_same_tool_calls,AgentLimits().max_agent_steps),(2,3,5))
                model,events,result=self.run_flow([self.flow.read(c) for c in chunks],label='default-A-B-C-'+kind)
                self.assertEqual(len(model.planner_messages),4,'B must be delivered before no_progress stop; C must be reachable')
                for i,c in enumerate(chunks,1):
                    row=next(r for r in self.rows(model,i) if r['symbol']==c['qualified_name'])
                    self.assertEqual(row['untrusted_source_excerpt'],c['content'])
                    self.assertEqual((row['path'],row['start_line'],row['end_line']),(c['path'],c['start_line'],c['end_line']))
                    self.assertFalse(row['excerpt_coverage']['truncated'])
                self.assertNotIn(chunks[0]['content'],json.dumps(model.planner_messages[:1]))
                self.assertEqual([e['metrics']['new_evidence_count'] for e in events[1:]],[0,1,1])
                self.assertEqual([len(e['after']) for e in events],[5,5,6,7])
                final=next(r for r in model.final_rows() if r['symbol']==names[2])
                self.assertEqual(final['untrusted_source_excerpt'],chunks[2]['content'])
                cite=next(c for c in result['citations'] if c['qualified_name']==names[2])
                self.assertEqual((cite['path'],cite['start_line'],cite['end_line']),(chunks[2]['path'],chunks[2]['start_line'],chunks[2]['end_line']))
                self.assertFalse(any('no-progress' in w for w in result.get('warnings',[])))
                self.assertEqual([c['after']['no_progress_count'] for c in self.commits],[0,0,0,0])
                self.assertEqual(len(self.state.delivered_source_prefixes),7)
                self.assertEqual(sum(self.state.progress_by_call.values()),3)

    def test_threshold_waits_for_real_first_delivery_not_preview(self):
        self.flow.fixture('neutral'); c=self.flow.chunks['PacketVault.open']
        actions=[decision('continue','lookup_symbol',{'symbol':'absent_alpha'}),self.flow.read(c)]
        project=agent_core._planner_state
        def omit_initial(state):
            value=project(state)
            if len(state.steps)<2:
                value['planner_evidence_view']['rows']=[r for r in value['planner_evidence_view']['rows'] if r['symbol']!=c['qualified_name']]
            return value
        with patch.object(agent_core,'_planner_state',side_effect=omit_initial):
            model,events,result=self.run_flow(actions,label='threshold-pending')
        self.assertEqual(len(model.planner_messages),3)
        self.assertEqual(self.commits[-1]['before']['no_progress_count'],2)
        self.assertEqual(self.commits[-1]['after']['no_progress_count'],0)
        self.assertEqual(next(r for r in self.rows(model,2) if r['symbol']==c['qualified_name'])['untrusted_source_excerpt'],c['content'])

    def test_repeat_same_call_and_changed_ranges_do_not_recount(self):
        self.flow.fixture('click'); c=self.flow.chunks['open_file']
        for label,actions,statuses in [
                ('same',[self.flow.read(c)]*3,['succeeded','rejected']),
                ('range',[self.flow.read(c,start=c['start_line']+i) for i in range(3)],['succeeded']*2)]:
            with self.subTest(label=label):
                model,events,result=self.run_flow(actions,label='repeat-'+label)
                self.assertEqual(len(model.planner_messages),2)
                self.assertEqual(self.state.no_progress_count,2)
                self.assertEqual(sum(self.state.progress_by_call.values()),0)
                self.assertEqual(len(self.state.delivered_source_prefixes),5)
                self.assertEqual([c['status'] for s in result['agent_trace'] for c in s['tool_calls']],statuses)
                self.assertTrue(any('no-progress' in w for w in result.get('warnings',[])))

    def test_coverage_and_new_evidence_each_progress_once(self):
        c=self.flow.fixture('neutral'); new=self.flow.chunks['filler_3']
        end=self.flow.chunks['PacketVault.close']['end_line']
        model,events,result=self.run_flow([self.flow.read(c,end=end),self.flow.read(c),self.flow.read(new)],label='coverage-then-new')
        self.assertEqual(len(model.planner_messages),4)
        self.assertIn('coverage-tail-731',json.dumps(self.rows(model,1)))
        self.assertNotIn('return self.close()',json.dumps(self.rows(model,1)))
        self.assertIn('return self.close()',json.dumps(self.rows(model,2)))
        self.assertEqual(sum(self.state.progress_by_call.values()),3)
        self.assertEqual([c['after']['no_progress_count'] for c in self.commits],[0]*4)
        # Acquisition already counted these steps; delivery records bytes without
        # another progress event or changing their call IDs.
        self.assertTrue(all(not e['metrics']['planner_source_delivery']['progress_counted'] for e in events[1:]))
        self.assertEqual(len(self.state.delivered_source_prefixes),len({r['symbol'] for i in range(1,4) for r in self.rows(model,i)}))

    def test_protocol_repair_reuses_delivery_ledger(self):
        self.flow.fixture('neutral'); c=self.flow.chunks['PacketVault.open']
        model,events,result=self.run_flow([self.flow.read(c),{'not_a_decision':True},decision('answer')],label='repair-resend')
        self.assertEqual(len(model.planner_messages),3)
        self.assertIn('repair_request',model.payload(2))
        self.assertEqual(self.rows(model,1),self.rows(model,2))
        self.assertEqual(self.commits[-1]['before']['delivered_prefixes'],self.commits[-1]['after']['delivered_prefixes'])
        self.assertEqual(sum(self.state.progress_by_call.values()),0)

    def test_old_prefix_metadata_and_window_rotation_do_not_refresh_progress(self):
        self.flow.fixture('click'); c=self.flow.chunks['open_file']
        limits=replace(AgentLimits(),max_agent_steps=10,max_tool_calls=12,max_same_tool_calls=10,max_no_progress_steps=20)
        model,events,result=self.run_flow([self.flow.read(c,start=c['start_line']+i) for i in range(7)],limits=limits,label='old-prefix-window')
        self.assertEqual(sum(self.state.progress_by_call.values()),0)
        self.assertEqual(self.state.no_progress_count,7)
        self.assertEqual(len(model.payload()['untrusted_observation_summaries']),5)
        # Reintroduce a legitimate older prefix with a fresh authority proof
        # before an actual coordinator-owned Planner call, not a preview commit.
        from app.services.evidence import CitationValidator
        original_projection=agent_core._planner_state
        prefix=''.join(c['content'].splitlines(keepends=True)[:3])
        def old_prefix_projection(state):
            if len(state.steps)==2:
                owned=next(e for e in state.context.evidence_store.all(state.request_id) if e.qualified_name=='open_file')
                owned.excerpt=prefix
                CitationValidator(self.flow.db).validate(owned)
                owned.selection_reason='different display metadata'; owned.fusion_rank=99
            return original_projection(state)
        with patch.object(agent_core,'_planner_state',side_effect=old_prefix_projection):
            model,events,result=self.run_flow([self.flow.read(c),decision('continue','lookup_symbol',{'symbol':'_AtomicFile'})],label='old-prefix-actual-input')
        self.assertEqual(next(r for r in self.rows(model,2) if r['symbol']==c['qualified_name'])['untrusted_source_excerpt'],prefix)
        self.assertIn(c['content'],self.state.delivered_source_prefixes.values())
        self.assertEqual(self.state.no_progress_count,0)
        self.assertEqual(sum(self.state.progress_by_call.values()),1)

    def test_projection_exclusion_partial_association_and_final_exclusion(self):
        self.flow.fixture('neutral'); c=self.flow.chunks['PacketVault.open']
        limits=replace(AgentLimits(),max_source_read_bytes=128)
        model,events,result=self.run_flow([self.flow.read(c),self.flow.read(c,start=c['end_line'])],limits=limits,label='projection-no-progress')
        self.assertEqual(len(model.planner_messages),2)
        self.assertEqual(self.state.no_progress_count,2)
        self.assertEqual(self.state.delivered_source_prefixes,{})
        source=model.payload()['untrusted_observation_summaries'][-1]['source_read']
        self.assertEqual(source['canonical_excerpt_delivery_status'],'omitted')
        self.assertTrue(any(r['untrusted_source_excerpt']==c['content'] for r in model.final_rows()))
        # One inclusive source read associates several small canonical blocks.
        filler=self.flow.chunks['filler_0']; last=self.flow.chunks['filler_3']
        model,events,result=self.run_flow([self.flow.read(filler,end=last['end_line'])],limits=replace(AgentLimits(),max_source_read_bytes=800),label='partial-associations')
        source=model.payload()['untrusted_observation_summaries'][-1]['source_read']
        self.assertEqual(source['canonical_excerpt_delivery_status'],'partial')
        projected=set(source['projected_evidence_ids'])
        self.assertTrue(projected); self.assertLess(len(projected),len(source['evidence_ids']))
        self.assertEqual(len(self.state.delivered_source_prefixes),3)  # earlier BASE + two read associations
        self.assertLessEqual(model.payload()['planner_evidence_view']['used_bytes'],800)
        self.flow.base=['filler_0','filler_1','filler_2','PacketVault','PacketVault.open']
        model,events,result=self.run_flow([self.flow.read(c)],limits=replace(AgentLimits(),max_accumulated_evidence_context_bytes=200),label='planner-seen-final-excluded')
        self.assertEqual(sum(self.state.progress_by_call.values()),0)
        self.assertEqual(next(r for r in self.rows(model,1) if r['symbol']==c['qualified_name'])['untrusted_source_excerpt'],c['content'])
        self.assertFalse(any(r['symbol']==c['qualified_name'] for r in model.final_rows()))
        self.assertFalse(any(r['qualified_name']==c['qualified_name'] for r in result['citations']))

    def test_invalid_mapping_and_changed_binding_never_delivery_progress(self):
        self.flow.fixture('neutral'); c=self.flow.chunks['PacketVault.open']
        for field,value in [('project_id','foreign'),('repository_revision','old'),('content_hash','0'*64),('id',99999),('end_line',9999)]:
            with self.subTest(field=field):
                bad=dict(c); bad[field]=value
                with patch.object(self.flow.db,'get_code_chunks_for_hierarchy',return_value=[bad]):
                    model,events,result=self.run_flow([self.flow.read(c)],label='invalid-progress-'+field)
                self.assertEqual(len(self.state.delivered_source_prefixes),5)
                self.assertEqual(self.state.no_progress_count,1)
                self.assertIn(c['content'],[r['untrusted_source_excerpt'] for r in self.rows(model,1)])

    def test_default_genuine_no_information_still_stops(self):
        self.flow.fixture('neutral')
        actions=[decision('continue','lookup_symbol',{'symbol':'missing_'+str(i)}) for i in range(3)]
        model,events,result=self.run_flow(actions,label='genuine-no-information')
        self.assertEqual(len(model.planner_messages),2)
        self.assertEqual(self.state.no_progress_count,2)
        self.assertEqual(sum(self.state.progress_by_call.values()),0)

    def test_pending_source_cannot_bypass_hard_stops(self):
        self.flow.fixture('neutral'); c=self.flow.chunks['filler_3']
        original=agent_core._update_state_from_observation
        for stop in ['cancel','deadline','reserve','tokens','steps','tools']:
            with self.subTest(stop=stop):
                limits=replace(AgentLimits(),**({'max_agent_steps':2} if stop=='steps' else {'max_tool_calls':2} if stop=='tools' else {}))
                def inject(state,action,observation):
                    original(state,action,observation)
                    if action!='read_source':return
                    if stop=='cancel':state.context.cancellation.cancel()
                    if stop=='deadline':state.budget=replace(state.budget,request_deadline_at=time.monotonic()-1)
                    if stop=='reserve':state.budget=replace(state.budget,request_deadline_at=time.monotonic()+30,final_answer_reserve_ms=31000)
                    if stop=='tokens':state.planner_token_usage=state.limits.max_total_planner_output_tokens
                with patch.object(agent_core,'_update_state_from_observation',side_effect=inject):
                    model,events,result=self.run_flow([decision('continue','lookup_symbol',{'symbol':'absent_before_hard_stop'}),self.flow.read(c)],limits=limits,label='hard-'+stop)
                self.assertEqual(len(model.planner_messages),2)
                self.assertEqual(len(self.state.delivered_source_prefixes),5)
                self.assertFalse(any(c['content'] in json.dumps(m) for m in model.planner_messages))

    def test_same_tool_limit_and_cancellation_at_actual_boundary(self):
        self.flow.fixture('neutral'); a=self.flow.chunks['PacketVault.open']; b=self.flow.chunks['filler_0']
        model,events,result=self.run_flow([self.flow.read(a),self.flow.read(b)],limits=replace(AgentLimits(),max_same_tool_calls=1),label='same-tool-hard')
        self.assertEqual([c['status'] for s in result['agent_trace'] for c in s['tool_calls']],['succeeded','rejected'])
        self.assertEqual(len(self.state.delivered_source_prefixes),5)
        original=agent_core._planner_state
        def cancel_during_projection(state):
            payload=original(state)
            if state.steps:state.context.cancellation.cancel()
            return payload
        with patch.object(agent_core,'_planner_state',side_effect=cancel_during_projection):
            model,events,result=self.run_flow([self.flow.read(a)],label='cancel-at-boundary')
        self.assertEqual(len(model.planner_messages),1)
        self.assertEqual(result['agent_status'],'cancelled')
        self.assertEqual(len(self.state.delivered_source_prefixes),5)

    def test_request_local_ledger_and_pure_previews(self):
        self.flow.fixture('neutral'); c=self.flow.chunks['PacketVault.open']
        for i in range(2):
            model,events,result=self.run_flow([self.flow.read(c)],label='request-isolation-'+str(i))
            self.assertEqual(self.commits[0]['before']['delivered_prefixes'],[])
            self.assertEqual(len(self.state.delivered_source_prefixes),5)
            self.assertEqual(sum(self.state.progress_by_call.values()),0)

    def test_custom_planner_without_delivery_confirmation_keeps_stop_rule(self):
        from tests.test_m2_agent import ScriptedPlanner
        self.flow.fixture('neutral'); c=self.flow.chunks['PacketVault.open']
        actions=[self.flow.read(c),self.flow.read(c,start=c['end_line']),decision('answer')]
        for inherited in [False,True]:
            with self.subTest(inherited=inherited):
                scripted=ScriptedPlanner(actions)
                class UnconfirmedPlanner(agent_core.LLMPlanner):
                    def decide(self,state,*,repair_hint=None):
                        return scripted.decide(state,repair_hint=repair_hint)
                adapter=UnconfirmedPlanner(None,None,AgentLimits()) if inherited else scripted
                with patch('tests.test_source_read_information_flow.LLMPlanner',return_value=adapter):
                    model,events,result=self.run_flow(actions,label='legacy-planner-stop-'+str(inherited))
                self.assertEqual(self.state.no_progress_count,2)
                self.assertEqual(len(self.state.progress_by_call),2)
                self.assertEqual(self.state.delivered_source_prefixes,{})
                self.assertEqual(model.planner_messages,[])
                self.assertTrue(any('no-progress' in w for w in result.get('warnings',[])))

    def test_body_leaves_window_and_returns_without_new_progress(self):
        self.flow.fixture('neutral'); c=self.flow.chunks['PacketVault.open']
        limits=replace(AgentLimits(),max_agent_steps=9,max_tool_calls=10,max_same_tool_calls=8,max_no_progress_steps=20)
        actions=[self.flow.read(c)]+[decision('continue','lookup_symbol',{'symbol':'absent_window_'+str(i)}) for i in range(5)]+[self.flow.read(c,start=c['end_line'])]
        model,events,result=self.run_flow(actions,limits=limits,label='window-leave-return')
        self.assertIn(c['content'],[r['untrusted_source_excerpt'] for r in self.rows(model,6)])
        self.assertEqual(next(r for r in self.rows(model,7) if r['symbol']==c['qualified_name'])['untrusted_source_excerpt'],c['content'])
        self.assertEqual(self.state.no_progress_count,7)
        self.assertEqual(sum(self.state.progress_by_call.values()),0)
        self.assertEqual(len(self.state.delivered_source_prefixes),5)

    def test_new_acquisition_then_same_body_does_not_extend_progress_again(self):
        self.flow.fixture('neutral'); c=self.flow.chunks['filler_3']
        repeated=self.flow.read(c,start=c['end_line'])
        model,events,result=self.run_flow([self.flow.read(c),repeated,repeated],label='new-then-repeat')
        self.assertEqual(len(model.planner_messages),3)
        self.assertEqual(sum(self.state.progress_by_call.values()),1)
        self.assertEqual(self.state.no_progress_count,2)
        self.assertFalse(events[1]['metrics']['planner_source_delivery']['progress_counted'])
        self.assertEqual(next(r for r in self.rows(model,1) if r['symbol']==c['qualified_name'])['untrusted_source_excerpt'],c['content'])
        self.assertTrue(any(r['untrusted_source_excerpt']==c['content'] for r in model.final_rows()))

    def test_promotion_failure_and_stale_valid_flag_cannot_deliver(self):
        self.flow.fixture('neutral'); c=self.flow.chunks['filler_3']
        original=self.flow.db.get_code_chunks_by_ids_bounded
        def fail_target(project,revision,ids,**kwargs):
            return [] if c['id'] in ids else original(project,revision,ids,**kwargs)
        with patch.object(self.flow.db,'get_code_chunks_by_ids_bounded',side_effect=fail_target):
            model,events,result=self.run_flow([self.flow.read(c)],label='promotion-failure')
        self.assertEqual(len(self.state.delivered_source_prefixes),5)
        self.assertEqual(self.state.no_progress_count,1)
        self.assertNotIn(c['qualified_name'],[r['symbol'] for r in self.rows(model,1)])
        self.assertFalse(any(r['symbol']==c['qualified_name'] for r in model.final_rows()))
        self.assertFalse(any(r['qualified_name']==c['qualified_name'] for r in result['citations']))
        c=self.flow.chunks['PacketVault.open']; original_update=agent_core._update_state_from_observation
        def corrupt_after_validation(state,action,observation):
            original_update(state,action,observation)
            if action=='read_source':
                held=next(e for e in state.context.evidence_store.all(state.request_id) if e.qualified_name==c['qualified_name'])
                self.assertEqual(held.validation_status,'valid')
                held.content_hash='0'*64
        with patch.object(agent_core,'_update_state_from_observation',side_effect=corrupt_after_validation):
            model,events,result=self.run_flow([self.flow.read(c)],label='stale-valid-proof')
        self.assertEqual(len(self.state.delivered_source_prefixes),5)
        self.assertNotIn(c['qualified_name'],[r['symbol'] for r in self.rows(model,1)])
        omission=model.payload(1)['untrusted_observation_summaries'][-1]['source_read']
        self.assertEqual(omission['canonical_excerpt_omissions'][0]['reason'],'not_valid_current_evidence')
        self.assertFalse(omission['canonical_excerpt_projection_truncated'])
