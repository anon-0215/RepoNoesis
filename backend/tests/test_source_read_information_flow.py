"""Scripted decisions + fake output, never evidence of real Planner quality."""
from dataclasses import replace
import hashlib,json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.database import Database
from app.services.agent_contracts import AgentLimits
from app.services.agent_core import LLMPlanner,run_bounded_agent
from app.services.agent_tools import ToolRegistry,build_m2_tool_registry
from app.services.code_chunker import extract_python_code_chunks
from tests.m1_helpers import disabled_embedding_service
from tests.test_m2_agent import decision
from tests.test_supplemental_evidence_flow import MessageModel

SNAPSHOT = os.environ.get('SOURCE_FLOW_SNAPSHOT_DIR')
CLICK_FIXTURE = Path(__file__).resolve().parent/'fixtures/source_read_click.json'
REVISION = '6aabf099bfdd4c1e75fe8d0e0d4241372b988ab1'
NEUTRAL = ('class PacketVault:\n    """'+('Neutral fixture padding only. '*5+'\n    ')*32+'"""\n'
    '    def open(self):\n        return 7\n\n'
    '    def close(self):\n        return "coverage-tail-731"\n\n'
    '    def __exit__(self, *args):\n        return self.close()\n')
FILLERS = ''.join(f'def filler_{i}():\n    return {i}\n\n' for i in range(4))


class SourceReadInformationFlowTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.db=Database(Path(tmp.name)/'source-flow.sqlite')

    def fixture(self, kind):
        if kind=='neutral':
            sources={'pkg/vault.py':NEUTRAL,'pkg/filler.py':FILLERS}
            target='PacketVault';base=[target,'PacketVault.open','filler_0','filler_1','filler_2']
            marker='return self.close()'
        else:
            if SNAPSHOT:
                sources={f'src/click/{name}':(Path(SNAPSHOT)/name).read_bytes().decode('utf-8') for name in ['utils.py','_compat.py']}
            else:
                sources=json.loads(CLICK_FIXTURE.read_text(encoding='utf-8'))['sources']
            sources['pkg/filler.py']=FILLERS
            target='_LazyFile';base=['open_file',target,'_LazyFile.open','filler_0','filler_1']
            marker='self.close_intelligently()'
        pid=self.db.create_project({'repo_url':'https://example.invalid/offline-flow','owner':'offline','repo':'read-flow','repository_revision':REVISION,'default_branch':'main'})
        files=[{'path':p,'content':s,'language':'Python','extension':'.py','size':len(s.encode()),'summary':'fixture','importance':1,'is_core':True,'imports':[],'exports':[],'symbols':[]} for p,s in sources.items()]
        chunks=[x.to_dict() for p,s in sources.items() for x in extract_python_code_chunks(p,s,REVISION).chunks]
        self.db.save_analysis(pid,{'overview':'offline','primary_language':'Python'},files,[],chunks)
        bundle=self.db.get_bundle(pid)
        chunks={c['qualified_name']:c for c in self.db.get_code_chunks(pid)}
        self.pid=pid;self.bundle=bundle;self.chunks=chunks;self.base=base;self.target=target;self.marker=marker
        return chunks[target]

    def run_flow(self, actions, *, limits=None, capacity=None, mode='agent', mutation=None, label='flow', retrieval_windows=None, request_constraints=None, question='Inspect the stored implementation, without executing it.'):
        from app.services.agent_tools import _chunk_as_read_candidate
        limits=limits or AgentLimits();model=MessageModel(actions+[decision('answer')]);events=[]
        hits=[_chunk_as_read_candidate(self.chunks[n],rank=i) for i,n in enumerate(self.base,1)]
        windows=iter(retrieval_windows) if retrieval_windows is not None else None
        def retrieve(*args,**kwargs):
            current=hits if windows is None else [_chunk_as_read_candidate(self.chunks[n],rank=i) for i,n in enumerate(next(windows),1)]
            return SimpleNamespace(results=current[:kwargs['evidence_count']],retrieval_mode='lexical',retrieval_strategy_version='weighted-rrf-v1',warnings=[],audit={})
        execute=ToolRegistry._execute_resolved
        def snapshots(context):
            return [e.to_dict()|{'coverage':e.planner_excerpt_coverage()} for e in context.evidence_store.all(context.request_id)]
        def observe(registry,context,call,spec):
            before=snapshots(context)
            if mutation and call.tool_name=='read_source':mutation(context)
            result=execute(registry,context,call,spec)
            events.append({'tool':call.tool_name,'parameters':call.parameters,'status':result.status,
                'call':call.to_dict(),
                'metrics':result.metrics,'results':result.structured_results,'error':result.error,
                'before':before,'after':snapshots(context),'policy':context.evidence_admission_policy()})
            return result
        args={'execution_mode':mode,'request_id':'offline-'+label,'limits':limits,
            'planner':LLMPlanner(model,build_m2_tool_registry(limits),limits)}
        if capacity is not None:args['evidence_count']=capacity
        if request_constraints:args.update(request_constraints)
        with patch('app.services.agent_tools.retrieve_code',side_effect=retrieve),patch.object(ToolRegistry,'_execute_resolved',observe):
            result=run_bounded_agent(question,self.bundle,model,self.db,disabled_embedding_service(),**args)
        capture={'label':label,'scripted_planner_fake_model_not_quality_evidence':True,
            'planner_messages':model.planner_messages,'final_messages':model.final_messages,'tools':events,'result':result}
        if os.environ.get('SOURCE_FLOW_CAPTURE_DIR'):
            out=Path(os.environ['SOURCE_FLOW_CAPTURE_DIR']);out.mkdir(parents=True,exist_ok=True)
            (out/(label+'.json')).write_text(json.dumps(capture,ensure_ascii=False,indent=2),encoding='utf-8')
        return model,events,result

    def read(self, chunk, start=None, end=None):
        return decision('continue','read_source',{'path':chunk['path'],'start_line':chunk['start_line'] if start is None else start,'end_line':chunk['end_line'] if end is None else end})

    def assert_in_messages(self, model, marker):
        self.assertIn(marker,json.dumps(model.planner_messages[1:],ensure_ascii=False))
        self.assertIn(marker,json.dumps(model.final_rows(),ensure_ascii=False))

    def test_complete_held_block_and_missing_tail_click_and_neutral(self):
        for kind in ['click','neutral']:
            for tail in [False,True]:
                with self.subTest(kind=kind,tail=tail):
                    chunk=self.fixture(kind)
                    start=self.chunks[self.target+'.close']['start_line'] if tail else chunk['start_line']
                    model,events,result=self.run_flow([self.read(chunk,start)],label=kind+('-tail' if tail else '-full'))
                    read=events[-1]
                    self.assertIn(self.marker,read['results']['content'])
                    self.assertNotIn(self.marker,json.dumps(events[0]['after']))
                    self.assert_in_messages(model,self.marker)
                    held=next(e for e in read['after'] if e['qualified_name']==self.target)
                    original=next(e for e in read['before'] if e['qualified_name']==self.target)
                    for field in ['evidence_id','chunk_identity','content_hash','project_id','repository_revision','start_line','end_line']:
                        self.assertEqual(held[field],original[field])
                    self.assertFalse(held['coverage']['truncated'])
                    self.assertEqual(held['coverage']['complete_end_line'],chunk['end_line'])
                    self.assertEqual(held['excerpt'],chunk['content'])
                    self.assertGreater(read['metrics']['information_delta_bytes'],0)
                    feedback=model.payload()['untrusted_observation_summaries'][-1]['candidate_metrics']
                    self.assertGreater(feedback['information_delta_bytes'],0)
                    self.assertEqual(result['answer_mode'],'llm_grounded')
                    self.assertEqual(len(result['citations']),len(result['evidence']))

    def test_correct_atomic_methods_enter_final_with_accurate_citations(self):
        self.fixture('click')
        close=self.chunks['_AtomicFile.close'];exit=self.chunks['_AtomicFile.__exit__']
        model,events,result=self.run_flow([self.read(close),self.read(exit)],label='atomic-correct-targets')
        final={r['symbol']:r['untrusted_source_excerpt'] for r in model.final_rows()}
        self.assertEqual(final['_AtomicFile.close'],close['content'])
        self.assertEqual(final['_AtomicFile.__exit__'],exit['content'])
        self.assertIn('os.replace',final['_AtomicFile.close'])
        self.assertIn('self.close(delete=exc_type is not None)',final['_AtomicFile.__exit__'])
        self.assertIn('os.replace',json.dumps(model.planner_messages[1:]))
        self.assertIn('self.close(delete=exc_type is not None)',json.dumps(model.planner_messages[1:]))
        for name in ['_AtomicFile.close','_AtomicFile.__exit__']:
            c=next(x for x in result['citations'] if x['qualified_name']==name)
            self.assertEqual((c['path'],c['start_line'],c['end_line']),
                (self.chunks[name]['path'],self.chunks[name]['start_line'],self.chunks[name]['end_line']))
        self.assertEqual([len(e['after']) for e in events],[5,6,7])

    def test_real_duplicate_read_still_rejected_without_false_progress(self):
        chunk=self.fixture('neutral');action=self.read(chunk)
        model,events,result=self.run_flow([action,action,action],label='repeat-read')
        statuses=[c['status'] for s in result['agent_trace'] for c in s['tool_calls']]
        self.assertEqual(statuses,['succeeded','rejected','rejected'])
        self.assertEqual(len(events),2)
        self.assert_in_messages(model,self.marker)
        self.assertEqual(result['answer_mode'],'llm_grounded')

    def test_full_capacity_allows_coverage_growth_but_not_new_identity(self):
        chunk=self.fixture('neutral')
        model,events,result=self.run_flow([self.read(chunk)],capacity=5,label='full-cap-coverage')
        self.assertEqual(len(events[-1]['after']),5)
        self.assertEqual(events[-1]['policy']['remaining_capacity'],0)
        self.assert_in_messages(model,self.marker)

    def test_already_complete_excerpt_zero_store_increment_first_delivery_progress(self):
        self.fixture('neutral');chunk=self.chunks['PacketVault.open'];action=self.read(chunk)
        model,events,result=self.run_flow([action,action],label='complete-duplicate')
        self.assertEqual(events[-1]['metrics']['new_evidence_count'],0)
        self.assertEqual(events[-1]['metrics']['information_delta_bytes'],0)
        self.assertEqual(events[-1]['metrics']['coverage_updated_count'],0)
        self.assertEqual(events[-1]['metrics']['admission_reason'],'duplicate')
        self.assertEqual(result['agent_status'],'completed')
        self.assertEqual(len(model.planner_messages),2)
        self.assertEqual(result['budget_usage']['steps_used'],2)
        self.assertEqual([c['status'] for s in result['agent_trace'] for c in s['tool_calls']][-1],'rejected')

    def test_final_byte_exclusion_and_read_byte_limit_are_honest(self):
        chunk=self.fixture('neutral')
        self.base=self.base[2:]+self.base[:2]
        model,events,result=self.run_flow([self.read(chunk)],limits=replace(AgentLimits(),max_accumulated_evidence_context_bytes=200),label='final-byte-exclusion')
        self.assertIn(self.marker,json.dumps(model.planner_messages[1:]))
        self.assertNotIn(self.marker,json.dumps(model.final_rows()))
        target=next(x for x in model.payload()['evidence_locators'] if x['qualified_name']==self.target)
        self.assertEqual(target['final_context_selection'],'excluded_by_byte_budget')
        self.assertNotIn(target['evidence_id'],{c['evidence_id'] for c in result['citations']})
        original=self.chunks[self.target]['content']
        model,events,result=self.run_flow([self.read(chunk)],limits=replace(AgentLimits(),max_source_read_bytes=100),label='source-byte-limit')
        self.assertNotIn(self.marker,json.dumps(model.planner_messages[1:]))
        self.assertNotIn(self.marker,json.dumps(model.final_rows()))
        self.assertNotEqual(next(x for x in events[-1]['after'] if x['qualified_name']==self.target)['excerpt'],original)

    def test_invalid_held_identity_hash_revision_and_span_cannot_be_completed(self):
        for field,value in [('project_id','foreign'),('repository_revision','stale'),('content_hash','0'*64),('end_line',9999),('chunk_identity','forged')]:
            with self.subTest(field=field):
                chunk=self.fixture('neutral')
                def corrupt(context):
                    held=next(e for e in context.evidence_store.all(context.request_id) if e.qualified_name==self.target)
                    setattr(held,field,value)
                model,events,result=self.run_flow([self.read(chunk)],mutation=corrupt,label='invalid-'+field)
                target=next(e for e in events[-1]['after'] if e['qualified_name']==self.target)
                self.assertNotIn(self.marker,target['excerpt'])
                self.assertNotIn(self.marker,json.dumps(model.final_messages))
                rows=[e for message in model.planner_messages[1:] for e in json.loads(message[1]['content'])['planner_evidence_view']['rows']]
                self.assertFalse(any(e['evidence_id']==target['evidence_id'] and self.marker in e['untrusted_source_excerpt'] for e in rows))

    def test_invalid_read_range_does_not_complete_held_block(self):
        chunk=self.fixture('neutral')
        for start,end in [(1,9999),(50,20),(0,20)]:
            model,events,result=self.run_flow([self.read(chunk,start,end)],label='invalid-range-'+str(start))
            self.assertNotIn(self.marker,json.dumps(model.final_rows()))

    def test_partial_completion_preserves_exact_visible_coverage(self):
        chunk=self.fixture('neutral');end=self.chunks['PacketVault.close']['end_line']
        model,events,result=self.run_flow([self.read(chunk,end=end)],label='partial-completion')
        held=next(e for e in events[-1]['after'] if e['qualified_name']==self.target)
        self.assertEqual(held['end_line'],chunk['end_line'])
        self.assertTrue(held['coverage']['truncated'])
        self.assertEqual(held['coverage']['complete_end_line'],end)
        self.assertIsNone(held['coverage']['partial_end_line'])
        self.assertIn('coverage-tail-731',held['excerpt'])
        self.assertNotIn(self.marker,held['excerpt'])
        self.assertIn('coverage-tail-731',json.dumps(model.planner_messages[1:]))
        self.assertNotIn(self.marker,json.dumps(model.final_rows()))

    def test_bad_mapping_metadata_cannot_enable_completion(self):
        for field,value in [('project_id','foreign'),('repository_revision','stale'),('content_hash','0'*64),('end_line',9999),('id',99999),('path','other.py')]:
            with self.subTest(field=field):
                chunk=self.fixture('neutral');bad=dict(chunk);bad[field]=value
                with patch.object(self.db,'get_code_chunks_for_hierarchy',return_value=[bad]):
                    model,events,result=self.run_flow([self.read(chunk)],label='bad-mapping-'+field)
                self.assertEqual(events[-1]['metrics']['coverage_updated_count'],0)
                self.assertEqual(events[-1]['metrics']['information_delta_bytes'],0)
                self.assertNotIn(self.marker,json.dumps(model.final_messages))
                self.assertNotIn(self.marker,json.dumps(model.planner_messages[1:]))

    def test_old_serialized_evidence_snapshot_keeps_previous_excerpt(self):
        chunk=self.fixture('neutral')
        model,events,result=self.run_flow([self.read(chunk)],label='old-snapshot')
        old=next(e for e in events[0]['after'] if e['qualified_name']==self.target)
        new=next(e for e in result['evidence'] if e['qualified_name']==self.target)
        self.assertNotIn(self.marker,old['excerpt']);self.assertIn(self.marker,new['excerpt'])
        self.assertEqual(old['evidence_id'],new['evidence_id'])
        self.assertEqual(old['chunk_identity'],new['chunk_identity'])


if __name__=='__main__':unittest.main()
