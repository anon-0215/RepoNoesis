"""Admission and actual message integration; scripted decisions are not model quality."""
from dataclasses import replace
import hashlib,json,os,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.database import Database
from app.services.agent_contracts import AgentLimits,CancellationToken,ToolCall
from app.services.agent_core import LLMPlanner,run_bounded_agent
from app.services.agent_tools import EvidenceStore,ToolRegistry,build_m2_tool_registry,build_tool_context
from app.services.code_chunker import extract_python_code_chunks
from app.services.evidence import EvidenceBuilder
from app.services.hybrid_retriever import HybridSearchResult
from tests.m1_helpers import REVISION,disabled_embedding_service,make_chunk
from tests.test_m2_agent import decision


class MessageModel:
    available=True
    def __init__(self,actions):
        self.actions=iter(actions); self.planner_messages=[]; self.final_messages=[]
    def require_available(self): pass
    def chat(self,messages,**kwargs):
        if 'bounded_repository_planner' in messages[0]['content']:
            self.planner_messages.append(messages)
            return json.dumps(next(self.actions))
        self.final_messages.append(messages)
        rows=self.final_rows()
        return json.dumps({'parts':[{'text':'Stored fixture snippet '+str(i)+'.','evidence_aliases':[x['alias']]} for i,x in enumerate(rows,1)]})
    def payload(self,index=-1): return json.loads(self.planner_messages[index][1]['content'])
    def final_rows(self):
        raw=self.final_messages[-1][1]['content'].split('VALIDATED_UNTRUSTED_EVIDENCE_BEGIN\n',1)[1].split('\nVALIDATED_UNTRUSTED_EVIDENCE_END',1)[0]
        return json.loads(raw)


class SupplementalEvidenceFlowTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.db=Database(Path(temp.name)/'flow.sqlite')
        self.source=('class MailDock:\n    """'+('Untrusted padding without implementations.\n    '*75)+'"""\n'
            '    def scope(self):\n        return self\n\n'
            '    def exit_scope(self):\n        """Leave scope."""\n\n        return self._release_queue()\n\n'
            '    def resource(self):\n        return 1\n\n'
            '    def register(self, fn):\n        return self.queue.append(fn)\n\n'
            '    def close(self):\n        return self._release_queue()\n\n'
            '    def unrelated(self):\n        return 0\n\n'
            '    def _release_queue(self):\n        """Release the pending queue."""\n\n        self.queue.clear()\n        return 4\n\n'
            '    def enter_scope(self):\n        """Enter scope."""\n\n        return self.activate()\n\n'
            'def query_active():\n    return 1\n')
        self.source=self.source.replace('\n','\r\n')
        self.project={'repo_url':'https://example.invalid/mail-dock','owner':'offline','repo':'mail-dock','repository_revision':REVISION,'default_branch':'main'}
        self.pid=self.db.create_project(self.project)
        files=[{'path':'lib/dock.py','content':self.source,'language':'Python','extension':'.py','size':len(self.source.encode()),'summary':'fixture','importance':1,'is_core':True,'imports':[],'exports':[],'symbols':[]}]
        chunks=[x.to_dict() for x in extract_python_code_chunks('lib/dock.py',self.source,REVISION).chunks]
        self.db.save_analysis(self.pid,{'primary_language':'Python','overview':'fixture'},files,[],chunks)
        self.bundle=self.db.get_bundle(self.pid)
        self.chunks={c['qualified_name']:c for c in self.db.get_code_chunks(self.pid)}
        self.base=['MailDock.scope','MailDock.exit_scope','MailDock.resource','MailDock','query_active']
        self.wide=['MailDock.register','MailDock.close','MailDock.unrelated','MailDock._release_queue','MailDock.resource']
        self.limits=replace(AgentLimits(),base_evidence_count=5,max_agent_steps=6,max_tool_calls=10)

    def hit(self,name,rank=1,**changes):
        c=self.chunks[name]
        value=HybridSearchResult(project_id=self.pid,repository_revision=REVISION,code_chunk_id=c['id'],
            language=c['language'],path=c['path'],chunk_type=c['chunk_type'],symbol_name=c['symbol_name'],qualified_name=c['qualified_name'],
            start_line=c['start_line'],end_line=c['end_line'],content='UNTRUSTED CANDIDATE SOURCE',content_hash=c['content_hash'],
            retrieval_sources=['lexical'],fusion_rank=rank,fusion_score=1/rank)
        return replace(value,**changes)

    def run_flow(self,actions,windows,*,limits=None,capacity=8,mode='agent',label='flow'):
        limits=limits or self.limits
        model=MessageModel(actions+[decision('answer')]); events=[]; windows=iter(windows)
        def retrieve(*args,**kwargs):
            values=next(windows)
            hits=[self.hit(n,i) if isinstance(n,str) else n for i,n in enumerate(values,1)]
            return SimpleNamespace(results=hits,retrieval_mode='lexical',retrieval_strategy_version='weighted-rrf-v1',warnings=[],audit={})
        original=ToolRegistry._execute_resolved
        def observe(registry,context,call,spec):
            before=[x.to_dict() for x in context.evidence_store.all(context.request_id)]
            result=original(registry,context,call,spec)
            events.append({'tool':call.tool_name,'parameters':call.parameters,'status':result.status,'error':result.error,
                'metrics':result.metrics,'before':before,'after':[x.to_dict() for x in context.evidence_store.all(context.request_id)]})
            return result
        with patch('app.services.agent_tools.retrieve_code',side_effect=retrieve),patch.object(ToolRegistry,'_execute_resolved',observe):
            result=run_bounded_agent('Explain stored lifecycle.',self.bundle,model,self.db,disabled_embedding_service(),
                planner=LLMPlanner(model,build_m2_tool_registry(limits),limits),limits=limits,evidence_count=capacity,
                execution_mode=mode,request_id='fixture-'+label,hierarchy_mode='off',relation_mode='off')
        destination=os.environ.get('SUPPLEMENTAL_FLOW_CAPTURE_DIR')
        if destination:
            path=Path(destination)/(label+'-messages.json')
            path.write_text(json.dumps({'planner_messages':model.planner_messages,'final_messages':model.final_messages,
                'tool_events':events,'result':result,'scripted_model_not_quality_evidence':True},ensure_ascii=False,indent=2),encoding='utf-8')
        return model,events,result

    def legal_actions(self):
        target=self.chunks['MailDock.enter_scope']
        return [decision('continue','search_code',{'query':'def register','top_k':5}),
            decision('continue','search_code',{'query':'_release_queue','top_k':3}),
            decision('continue','read_source',{'path':'lib/dock.py','start_line':target['start_line'],'end_line':target['end_line']})]

    def test_legal_chain_candidate_admission_planner_final_and_citations(self):
        model,events,result=self.run_flow(self.legal_actions(),[self.base,self.wide,['MailDock._release_queue','MailDock.close']],label='legal-chain')
        self.assertEqual([len(e['after']) for e in events],[5,6,7,8])
        payload=model.payload()
        self.assertEqual(payload['evidence_count'],8)
        self.assertEqual(payload['known_evidence_ids'],['E'+str(i) for i in range(1,9)])
        self.assertEqual(len(payload['evidence_locators']),8)
        target=[x for x in payload['evidence_locators'] if x['qualified_name']=='MailDock._release_queue'][0]
        self.assertFalse(target['excerpt_coverage']['truncated'])
        self.assertEqual(target['final_context_selection'],'selected_by_byte_budget')
        final={x['symbol']:x['untrusted_source_excerpt'] for x in model.final_rows()}
        for symbol in ['MailDock._release_queue','MailDock.enter_scope']:
            self.assertEqual(final[symbol],self.chunks[symbol]['content'])
        self.assertEqual({c['evidence_id'] for c in result['citations']},{e['evidence_id'] for e in result['evidence']})
        self.assertNotIn('UNTRUSTED CANDIDATE SOURCE',json.dumps(model.final_messages))
        self.assertEqual(result['answer_mode'],'llm_grounded')

    def test_crlf_hint_at_actual_boundary_retains_text_hash_and_original_line(self):
        model,events,result=self.run_flow([], [self.base],label='crlf')
        loc=next(x for x in model.payload()['evidence_locators'] if x['qualified_name']=='MailDock.exit_scope')
        self.assertEqual(loc['direct_call_hints']['status'],'available')
        call=loc['direct_call_hints']['calls'][0]
        self.assertEqual(call['callee_expression'],'self._release_queue')
        self.assertEqual(call['resolution_status'],'unresolved')
        self.assertEqual(call['start_line'],self.chunks['MailDock.exit_scope']['end_line'])
        original=self.chunks['MailDock.exit_scope']['content']
        evidence=next(e for e in result['evidence'] if e['qualified_name']=='MailDock.exit_scope')
        self.assertEqual(evidence['excerpt'],original)
        self.assertEqual(evidence['content_hash'],hashlib.sha256(original.encode()).hexdigest())
        self.assertNotIn('direct_call_hints',evidence)

    def test_initial_after_tool_and_protocol_repair_have_limits_and_feedback(self):
        actions=[decision('continue','search_code',{'query':'def register'}),{'status':'invalid'},decision('answer')]
        model,events,result=self.run_flow(actions,[self.base,self.wide],label='repair-feedback')
        first,after,repair=[model.payload(i) for i in range(3)]
        for p in [first,after,repair]:
            self.assertEqual(p['server_constraints']['tool_limits']['read_source']['max_inclusive_lines'],200)
            self.assertIn('200',p['server_constraints']['tool_descriptions']['read_source'])
            self.assertEqual(p['server_constraints']['evidence_admission']['max_new_per_supplement'],1)
        self.assertEqual(first['server_constraints']['evidence_admission']['remaining_capacity'],3)
        self.assertEqual(after['server_constraints']['evidence_admission']['remaining_capacity'],2)
        self.assertGreater(after['untrusted_observation_summaries'][-1]['candidate_metrics']['admission_deferred_count'],0)
        self.assertEqual(after['evidence_locators'],repair['evidence_locators'])
        self.assertIn('repair_request',repair)
        self.assertEqual(first['server_constraints']['evidence_admission']['max_new_per_read_source'],2)
        self.assertIn('rn-lwb-03-v11',model.planner_messages[0][0]['content'])

    def test_reserved_slots_then_capacity_exhaustion_have_honest_feedback(self):
        actions=[decision('continue','search_code',{'query':'wide first'}),
            decision('continue','search_code',{'query':'wide second'}),
            decision('continue','lookup_symbol',{'symbol':'MailDock._release_queue'}),
            decision('continue','lookup_symbol',{'symbol':'MailDock.enter_scope'}),
            decision('continue','lookup_symbol',{'symbol':'MailDock.close'})]
        model,events,result=self.run_flow(actions,[self.base,self.wide,['MailDock.unrelated']],label='capacity-bound')
        self.assertEqual(events[2]['metrics']['result_count'],0)
        self.assertEqual(events[2]['metrics']['admission_reason'],'admission_deferred')
        self.assertEqual(events[2]['metrics']['remaining_capacity'],2)
        self.assertEqual(events[-1]['metrics']['admission_reason'],'capacity_exhausted')
        self.assertEqual(events[-1]['metrics']['remaining_capacity'],0)
        self.assertEqual(model.payload()['evidence_count'],8)
        self.assertEqual(model.payload()['server_constraints']['evidence_admission']['remaining_capacity'],0)
        self.assertEqual([e['evidence_id'] for e in result['evidence']],['E'+str(i) for i in range(1,9)])

    def test_zero_new_distinguishes_duplicate_invalid_and_no_hit(self):
        cases=[('duplicate',[self.hit('MailDock.scope')],'duplicate'),
            ('nohit',[],'no_hits'),
            ('foreign',[self.hit('MailDock._release_queue',project_id='foreign')],'invalid_candidates'),
            ('stale',[self.hit('MailDock._release_queue',repository_revision='stale')],'invalid_candidates'),
            ('forged',[self.hit('MailDock._release_queue',content_hash='0'*64)],'invalid_candidates')]
        for label,hits,expected in cases:
            with self.subTest(label=label):
                model,events,result=self.run_flow([decision('continue','search_code',{'query':'_release_queue'})],[self.base,hits],label=label)
                metrics=events[-1]['metrics']
                self.assertEqual(metrics['result_count'],0)
                self.assertEqual(metrics['admission_reason'],expected)
                self.assertEqual(model.payload()['evidence_count'],5)
                self.assertEqual(model.payload()['untrusted_observation_summaries'][-1]['candidate_metrics']['admission_reason'],expected)

    def test_byte_exclusion_keeps_held_identity_and_actual_final_citations_subset(self):
        model,events,result=self.run_flow(self.legal_actions(),[self.base,self.wide,['MailDock._release_queue']],
            limits=replace(self.limits,max_accumulated_evidence_context_bytes=200),label='byte-exclusion')
        p=model.payload(); ids={e['evidence_id'] for e in result['evidence']}
        self.assertEqual(p['evidence_count'],8)
        excluded={e['evidence_id'] for e in p['evidence_locators'] if e['final_context_selection']=='excluded_by_byte_budget'}
        selected={e['evidence_id'] for e in p['evidence_locators'] if e['final_context_selection']=='selected_by_byte_budget'}
        self.assertTrue(excluded)
        self.assertEqual(ids,selected)
        self.assertFalse(excluded & {c['evidence_id'] for c in result['citations']})
        self.assertEqual(len(model.final_rows()),len(selected))

    def test_200_line_read_legal_and_221_rejected_then_legal_adjustment(self):
        # A long readable file; range limit is enforced before any source read.
        actions=[decision('continue','read_source',{'path':'lib/dock.py','start_line':1,'end_line':221}),
            decision('continue','read_source',{'path':'lib/dock.py','start_line':1,'end_line':200})]
        # Ensure a >200-line fixture without executing it.
        content=self.source+'\r\n'*240
        with self.db.connect() as con:
            con.execute('UPDATE repo_files SET content=? WHERE project_id=? AND path=?',(content,self.pid,'lib/dock.py'))
        model,events,result=self.run_flow(actions,[self.base],label='read-range')
        self.assertEqual(events[1]['error']['code'],'source_range_exceeded')
        self.assertIn('200',events[1]['error']['message'])
        self.assertEqual(len(events[1]['after']),5)
        self.assertEqual(events[2]['status'],'succeeded')
        feedback=model.payload(1)['untrusted_observation_summaries'][-1]
        self.assertIn('200',feedback['read_range_feedback'])

    def test_crlf_half_line_and_multiline_truncation_lf_equivalence(self):
        source='    def transfer(self):\r\n        """Note."""\r\n\r\n        return self.write(\r\n            3,\r\n        )\r\n'
        project=self.bundle['project']
        for newline in ['\r\n','\n']:
            text=source.replace('\r\n',newline); chunk=make_chunk('lib/transfer.py','MailDock.transfer',text,start_line=20)
            chunk.update(id=999,project_id=self.pid)
            full=EvidenceBuilder().build_from_code_chunks([chunk],project)[0]
            self.assertEqual(full.planner_direct_call_hints()['calls'][0],{'callee_expression':'self.write','start_line':23,'end_line':25,'resolution_status':'unresolved'})
            for cutoff in [text.index('3,')+1,len(text)-1,text.index('return')+12]:
                e=EvidenceBuilder(cutoff).build_from_code_chunks([chunk],project)[0]
                h=e.planner_direct_call_hints()
                # Removing the whole partial return line leaves a parseable
                # docstring-only prefix; it is available but explicitly partial.
                expected='available' if cutoff==text.index('return')+12 else 'unknown'
                self.assertEqual(h['status'],expected)
                self.assertTrue(h['may_be_incomplete'])
                self.assertEqual(h['calls'],[])
                self.assertEqual(e.excerpt,text[:cutoff])

    def test_rag_keeps_base_and_never_invokes_planner(self):
        model,events,result=self.run_flow([], [self.base],mode='rag',label='rag')
        self.assertEqual(model.planner_messages,[])
        self.assertEqual(len(result['evidence']),5)
        self.assertEqual(result['budget_usage']['tool_calls_used'],0)


if __name__=='__main__': unittest.main()
