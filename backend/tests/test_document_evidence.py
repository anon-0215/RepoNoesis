"""Stored documents are sources, never synthetic Python chunks."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from app.database import Database
from app.services.agent_contracts import AgentLimits, CancellationToken, ToolCall
from app.services.agent_tools import EvidenceStore, build_m2_tool_registry, build_tool_context
from app.services.agent_core import LLMPlanner, run_bounded_agent
from app.services.citation_protocol import CanonicalCitationDescriptor
from app.services.chat_history import prepare_recovery_metadata, RecoveryMetadata, bindings_match
from app.services.document_evidence import build_document, document_inventory, retrieve_documents, planner_document_inventory
from app.services.evidence import CitationValidator
from app.services.qa_agent import answer_question
from app.services.smoke_diagnostics import SmokeDiagnosticsRecorder, _bounded_smoke_diagnostics
from tests.m1_helpers import make_project, disabled_embedding_service, REVISION
from tests.test_m2_agent import decision
from tests.test_supplemental_evidence_flow import MessageModel


class DocumentEvidenceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.db = Database(Path(temp.name) / 'docs.sqlite')
        self.pid, _ = make_project(self.db, [('a.py', 'compute', 'def compute():\n    return 4\n')])
        self.text = '# Acorn\r\nA tool for deterministic tasks.\r\n\r\n## Installation\r\nUse a compatible Python runtime.\r\n'
        with self.db.connect() as conn:
            conn.execute('INSERT INTO repo_files (project_id,path,content,language) VALUES (?,?,?,?)',
                         (self.pid, 'README.md', self.text, 'Markdown'))
        self.bundle = self.db.get_bundle(self.pid)
        self.limits = AgentLimits()
        self.recorder = SmokeDiagnosticsRecorder()
        self.recorder.begin_agent(['read_source'], request_id='test-docs')
        self.context = build_tool_context(request_id='test-docs', bundle=self.bundle, database=self.db,
            embedding_service=disabled_embedding_service(), evidence_store=EvidenceStore(5),
            limits=self.limits, cancellation=CancellationToken(), deadline_monotonic=time.monotonic()+60,
            diagnostics_recorder=self.recorder)
        self.registry = build_m2_tool_registry(self.limits)

    def read(self, path='README.md', start=1, end=5):
        return self.registry.execute(self.context, ToolCall(call_id='T1', step_id='S1',
            tool_name='read_source', tool_version='1', parameters={'path':path, 'start_line':start, 'end_line':end},
            timeout_ms=15000, budget={'max_results':1, 'max_bytes':65536}))

    def test_discover_and_read_document_without_chunks(self):
        with self.db.connect() as conn:
            conn.execute('DELETE FROM code_chunks WHERE project_id=?', (self.pid,))
        self.context.bundle = self.db.get_bundle(self.pid)
        inventory = document_inventory(self.context.bundle)
        self.assertEqual(inventory[0]['total_lines'], 5)
        result = self.read()
        self.assertEqual(result.status, 'succeeded', result.error)
        self.assertEqual(result.metrics['new_evidence_count'], 1)
        item = self.context.evidence_store.all('test-docs')[0]
        self.assertEqual(item.excerpt, self.text)
        self.assertIsNone(item.code_chunk_id)
        self.assertEqual(item.source_kind, 'document')
        self.assertEqual(item.planner_excerpt_coverage()['complete_end_line'], 5)
        self.assertEqual(self.read().metrics['new_evidence_count'], 0)
        subset = self.read(end=4)
        self.assertEqual(subset.metrics['new_evidence_count'],0)
        self.assertEqual(subset.metrics['information_delta_bytes'],0)
        self.assertEqual(len(self.context.evidence_store.all('test-docs')),1)

    def test_short_file_range_and_path_failures_have_corrective_receipts(self):
        for path, end, code in [('README.md',100,'source_line_out_of_range'),
                                ('MISSING.md',5,'source_file_not_found')]:
            observation = self.read(path, end=end)
            self.assertEqual(observation.error['code'],code)
        self.assertIn('1..5', self.read(end=99).error['message'])
        snapshot = self.recorder.snapshot()
        receipts = snapshot['tool_failures']
        self.assertEqual(receipts[0]['target'], {'status':'known','path':'README.md','start_line':1,'end_line':100})
        self.assertEqual(receipts[0]['request_id'], 'test-docs')
        self.assertEqual(receipts[1]['order'], 2)
        noisy = dict(snapshot, provider_calls=[{'padding':'x'*10000}]*16, tool_executions=[{'padding':'x'*10000}]*16)
        bounded = _bounded_smoke_diagnostics(noisy)
        self.assertEqual(bounded['tool_failures'], receipts)
        self.assertLessEqual(len(json.dumps(bounded, ensure_ascii=True)),4096)
        from app.services.tool_failure_receipts import sanitize_receipts
        for bad in ['bad', [], 3]:
            self.assertEqual(sanitize_receipts([{**receipts[0], 'target':bad}])[0]['target']['status'],'omitted')
        long_target = {**receipts[0], 'target':{'status':'truncated','path_sha256':'a'*64}}
        self.assertEqual(sanitize_receipts(sanitize_receipts([long_target])),sanitize_receipts([long_target]))

    def test_inventory_is_bounded_and_broad_search_keeps_read_reserve(self):
        large = {'files':[{'path':('长'*200) + f'/{i}/README.md', 'content':'x\n'} for i in range(32)]}
        inventory = planner_document_inventory(large, 2048)
        self.assertLessEqual(len(json.dumps(inventory,ensure_ascii=False).encode()),2048)
        self.assertTrue(inventory['may_be_incomplete'])
        self.context.evidence_store = EvidenceStore(3)
        self.context.supplemental_admission_enabled = True
        self.context.evidence_store.add('test-docs',[
            build_document(self.db,self.pid,REVISION,'README.md',1,1),
            build_document(self.db,self.pid,REVISION,'README.md',2,2)])
        observation = self.registry.execute(self.context,ToolCall(call_id='T2', step_id='S2',tool_name='search_code',
            tool_version='1', parameters={'query':'Acorn project','top_k':5},timeout_ms=15000,budget={}))
        self.assertEqual(observation.status,'succeeded',observation.error)
        self.assertEqual(len(self.context.evidence_store.all('test-docs')),2)

    def test_document_only_base_and_four_enhancement_combinations(self):
        from app.services.agent_tools import retrieve_code
        for hyde,rerank in [(False,False),(True,False),(False,True),(True,True)]:
            with self.subTest(hyde=hyde,rerank=rerank), patch.dict('os.environ', {
                'RETRIEVAL_HYDE_ENABLED':str(hyde).lower(),'RETRIEVAL_RERANK_ENABLED':str(rerank).lower()}), \
                    patch('app.services.agent_tools.retrieve_initial_code',side_effect=lambda *a,**kw:
                          retrieve_code(*a,**{k:v for k,v in kw.items() if k not in {'llm','enhancement_deadline_at'}})):
                model=MessageModel([decision('answer')])
                result=run_bounded_agent('Acorn',self.bundle,model,self.db,disabled_embedding_service(),
                    execution_mode='agent',limits=self.limits,request_id='combo-docs',evidence_count=5,
                    hierarchy_mode='off',relation_mode='off')
                self.assertTrue(any(e['source_kind']=='document' for e in result['evidence']))
        with self.db.connect() as conn:
            conn.execute('DELETE FROM code_chunks WHERE project_id=?',(self.pid,))
        model=MessageModel([])
        recorder=SmokeDiagnosticsRecorder()
        result=run_bounded_agent('Acorn',self.db.get_bundle(self.pid),model,self.db,disabled_embedding_service(),
            execution_mode='rag',limits=self.limits,request_id='doc-only-base',evidence_count=5,
            diagnostics_recorder=recorder,hierarchy_mode='off',relation_mode='off')
        self.assertEqual(recorder.snapshot()['base_retrieval']['status'],'succeeded')
        self.assertEqual(result['evidence'][0]['source_kind'],'document')

    def test_tampering_rejected_and_code_path_remains_strict(self):
        original = build_document(self.db,self.pid,REVISION,'README.md',1,5)
        for field,value in [('path','../README.md'),('repository_revision','other'),
                ('content_hash','0'*64),('start_line',0),('end_line',99),('document_file_id',999),
                ('source_kind','code'),('code_chunk_id',1),('chunk_type','function'),('language','python'),
                ('qualified_name','fake_function'),('excerpt','forged')]:
            with self.subTest(field=field):
                item = replace(original, **{field:value})
                self.assertEqual(CitationValidator(self.db).validate(item).validation_status,'invalid')
        with self.db.connect() as conn:
            conn.execute('UPDATE repo_files SET content=? WHERE path=?', ('changed\n','README.md'))
        self.assertEqual(CitationValidator(self.db).validate(original).validation_status,'invalid')

    def test_read_extends_held_document_and_counts_information_not_new_chunk(self):
        long_text = '# Acorn\n' + ('Long untrusted 文档 text\n' * 30)
        with self.db.connect() as conn:
            conn.execute('UPDATE repo_files SET content=? WHERE path=?',(long_text,'README.md'))
        held = build_document(self.db,self.pid,REVISION,'README.md',1,31,excerpt_limit=100)
        self.context.evidence_store.add('test-docs',[held])
        result = self.read(end=31)
        self.assertEqual(result.status,'succeeded',result.error)
        self.assertEqual(result.metrics['new_evidence_count'],0)
        self.assertEqual(result.metrics['coverage_updated_count'],1)
        self.assertGreater(result.metrics['information_delta_bytes'],0)
        self.assertEqual(held.excerpt,long_text)
        self.assertEqual(result.structured_results['coverage_updates'][0]['evidence_id'],held.evidence_id)
        self.assertEqual(self.read(end=31).metrics['information_delta_bytes'],0)

    def test_document_seed_cannot_expand_arbitrary_code_relations(self):
        from app.services.relation_analysis import index_project_relations
        index_project_relations(self.db,self.pid)
        self.read()
        observation=self.registry.execute(self.context,ToolCall(call_id='T3',step_id='S3',
            tool_name='expand_relations',tool_version='1',parameters={'seed_evidence_ids':['E1'],
            'relation_types':['calls']},timeout_ms=15000,budget={}))
        self.assertEqual(observation.status,'failed')
        self.assertEqual(observation.error['code'],'document_relation_unsupported')

    def test_retrieval_bounded_relevant_and_missing_safe(self):
        self.assertEqual(len(retrieve_documents(self.db,self.pid,REVISION,'Acorn tasks')),1)
        self.assertEqual(retrieve_documents(self.db,self.pid,REVISION,'nonmatching_xyz'),[])
        self.assertEqual(retrieve_documents(self.db,self.pid,REVISION,'Acorn',path='a.py'),[])
        with self.assertRaisesRegex(ValueError,'revision_mismatch'):
            build_document(self.db,self.pid,'stale','README.md',1,5)

    def test_document_is_in_real_planner_and_final_messages_and_progress(self):
        model = MessageModel([decision('continue','read_source',{'path':'README.md','start_line':1,'end_line':5}), decision('answer')])
        result = run_bounded_agent('compute behavior and author deployment guidance',self.bundle,model,
            self.db,disabled_embedding_service(),limits=self.limits,evidence_count=8,
            planner=LLMPlanner(model,self.registry,self.limits),request_id='test-live-contract',
            execution_mode='agent',hierarchy_mode='off',relation_mode='off')
        self.assertEqual(len(model.planner_messages),2)
        self.assertEqual(model.payload(0)['stored_project_documents']['files'][0]['path'],'README.md')
        rows = model.payload(1)['planner_evidence_view']['rows']
        self.assertTrue(any(row['source_kind']=='document' and 'compatible Python' in row['untrusted_source_excerpt'] for row in rows))
        self.assertTrue(any(row.get('source_kind')=='document' and 'compatible Python' in row['untrusted_source_excerpt'] for row in model.final_rows()))
        self.assertNotEqual(result.get('agent_status'),'no_progress')
        self.assertEqual(result['answer_mode'],'llm_grounded')

    def test_rag_mixed_citations_and_history_identity_full_file(self):
        model = MessageModel([])
        result = answer_question('Acorn compute',self.bundle,model,database=self.db,
                                 embedding_service=disabled_embedding_service())
        self.assertEqual({x['source_kind'] for x in result['evidence']},{'code','document'})
        self.assertEqual(result['evidence'][0]['source_kind'],'code')
        self.assertTrue(any(c['source_kind']=='document' for c in result['citations']))
        metadata = prepare_recovery_metadata(result,self.pid,REVISION)
        recovered = RecoveryMetadata.model_validate_json(json.dumps(metadata))
        self.assertTrue(bindings_match(recovered,result['citations'],self.pid,REVISION))
        saved=self.db.save_chat_answer(self.pid,'Acorn compute',result['answer'],result['citations'],
            repository_revision=REVISION,recovery_metadata=metadata)
        workspace=self.db.get_workspace_for_project(self.pid)['id']
        page=self.db.read_chat_history(workspace,self.pid,REVISION)
        record=next(r for r in page['items'] if r['record_id']==saved['record_id'])
        self.assertEqual(record['metadata_status'],'available')
        self.assertTrue(record['full_file_available'])
        self.assertTrue(any(c['source_kind']=='document' for c in record['citations']))
        bad = [dict(c,source_kind='code') for c in result['citations']]
        self.assertFalse(bindings_match(recovered,bad,self.pid,REVISION))
        workspace = self.db.get_workspace_for_project(self.pid)['id']
        file = self.db.read_source_file(self.pid,REVISION,'README.md')
        self.assertEqual(file['content'],self.text)
        descriptor = CanonicalCitationDescriptor.from_evidence('A1',build_document(self.db,self.pid,REVISION,'README.md',1,5))
        self.assertEqual(descriptor.prompt_evidence()['source_kind'],'document')
        self.assertIn('not proof of implementation',descriptor.prompt_evidence()['scope'])


if __name__ == '__main__':
    unittest.main()
