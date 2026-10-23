"""Offline acquisition and actual final request coverage, not Planner quality."""
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.database import Database
from app.services.agent_contracts import AgentLimits
from app.services.agent_core import LLMPlanner, run_bounded_agent
from app.services.agent_tools import ToolRegistry, build_m2_tool_registry
from app.services.agent_tools import _chunk_as_read_candidate
from app.services.candidate_evidence import CandidateEvidencePool
from app.services.code_chunker import extract_python_code_chunks
from app.services.evidence import EvidenceBuilder
from app.services.hybrid_retriever import HybridSearchResult
from tests.m1_helpers import REVISION, disabled_embedding_service
from tests.test_m2_agent import decision
from tests.test_supplemental_evidence_flow import MessageModel


class DirectedReadSelectionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.db = Database(Path(directory.name) / 'selection.sqlite')

    def scenario(self, *, owner='Dock', path='pkg/dock.py', target='release',
                 slots=2, mode='agent', calls=None, extra='', label='generic'):
        calls = calls if calls is not None else f'self.open_gate() + self.{target}()'
        source = (f'class {owner}:\n'
                  f'    def leave(self):\n        return {calls}\n\n'
                  '    def irrelevant(self):\n        return 0\n\n'
                  '    def open_gate(self):\n        return 1\n\n'
                  f'    def {target}(self):\n        marker = 7\n        return marker\n\n'
                  + extra + '\n'.join(f'def ballast_{i}():\n    return {i}\n' for i in range(7)))
        pid = self.db.create_project({'repo_url':'https://example.invalid/fixture',
            'owner':'offline','repo':'fixture','repository_revision':REVISION,'default_branch':'main'})
        chunks = [x.to_dict() for x in extract_python_code_chunks(path, source, REVISION).chunks]
        self.db.save_analysis(pid, {'primary_language':'Python','overview':'fixture'},
            [{'path':path,'content':source,'language':'Python','extension':'.py','size':len(source.encode())}], [], chunks)
        bundle = self.db.get_bundle(pid)
        by_name = {c['qualified_name']:c for c in self.db.get_code_chunks(pid)}
        base = [f'{owner}.leave'] + [f'ballast_{i}' for i in range(7-slots)]
        self.case = (pid, bundle['project'], by_name, base, self.db.get_code_chunks(pid))
        limits = replace(AgentLimits(), base_evidence_count=len(base))
        def hit(name, rank):
            c = by_name[name]
            return HybridSearchResult(project_id=pid, repository_revision=REVISION,
                code_chunk_id=c['id'], language=c['language'], path=path, chunk_type=c['chunk_type'],
                symbol_name=c['symbol_name'], qualified_name=c['qualified_name'],
                start_line=c['start_line'],end_line=c['end_line'],content='UNTRUSTED',content_hash=c['content_hash'],
                retrieval_sources=['lexical'],fusion_rank=rank,fusion_score=1/rank)
        def retrieve(*args, **kwargs):
            return SimpleNamespace(results=[hit(n,i) for i,n in enumerate(base,1)],
                retrieval_mode='lexical',retrieval_strategy_version='weighted-rrf-v1',warnings=[],audit={})
        start = by_name[f'{owner}.irrelevant']['start_line']
        end = by_name[f'{owner}.{target}']['end_line']
        model = MessageModel([decision('continue','read_source',{'path':path,'start_line':start,'end_line':end}),decision('answer')])
        events = []
        original = ToolRegistry._execute_resolved
        def observe(registry, context, call, spec):
            before = [x.to_dict() for x in context.evidence_store.all(context.request_id)]
            observation = original(registry, context, call, spec)
            events.append({'tool':call.tool_name,'before':before,'after':[x.to_dict() for x in context.evidence_store.all(context.request_id)],
                           'metrics':observation.metrics, 'results':observation.structured_results})
            return observation
        with patch('app.services.agent_tools.retrieve_code', side_effect=retrieve), patch.object(ToolRegistry,'_execute_resolved',observe):
            result = run_bounded_agent('Explain acquired fixture methods.',bundle,model,self.db,disabled_embedding_service(),
                planner=LLMPlanner(model,build_m2_tool_registry(limits),limits),limits=limits,
                evidence_count=8,execution_mode=mode,request_id='selection-'+label,hierarchy_mode='off',relation_mode='off')
        capture = os.environ.get('SUPPLEMENTAL_FLOW_CAPTURE_DIR')
        if capture:
            (Path(capture)/(label+'.json')).write_text(json.dumps({'events':events,'final_messages':model.final_messages,
                'planner_messages':model.planner_messages,'result':result},ensure_ascii=False,indent=2),encoding='utf-8')
        return model, events, result, by_name

    def test_two_static_targets_survive_irrelevant_high_rank_and_reach_final_request(self):
        model, events, result, chunks = self.scenario()
        final = {x['symbol']:x['untrusted_source_excerpt'] for x in model.final_rows()}
        self.assertEqual(events[-1]['metrics']['new_evidence_count'],2)
        for name in ('Dock.open_gate','Dock.release'):
            self.assertEqual(final.get(name),chunks[name]['content'])
        self.assertNotIn('Dock.irrelevant',final)
        self.assertEqual({x['evidence_id'] for x in result['citations']},{x['evidence_id'] for x in result['evidence']})
        self.assertEqual(result['answer_mode'],'llm_grounded')
        metrics = model.payload()['untrusted_observation_summaries'][-1]['candidate_metrics']
        self.assertEqual(metrics['max_new_per_supplement'],2)
        self.assertEqual(metrics['directed_priority_count'],2)
        for locator in model.payload()['evidence_locators']:
            self.assertEqual(locator['excerpt_coverage']['complete_end_line'],locator['end_line'])
            self.assertFalse(locator['excerpt_coverage']['truncated'])
            self.assertEqual(locator['final_context_selection'],'selected_by_byte_budget')

    def test_rename_one_slot_full_and_rag_boundaries(self):
        for slots in (2,1,0):
            with self.subTest(slots=slots):
                model,events,result,chunks = self.scenario(owner='Renamed',path='lib/renamed.py',target='flush',slots=slots,label=f'renamed-{slots}')
                final = {x['symbol'] for x in model.final_rows()}
                self.assertEqual(events[-1]['metrics']['new_evidence_count'],min(2,slots))
                self.assertNotIn('Renamed.irrelevant',final)
                if slots==2: self.assertIn('Renamed.flush',final)
        model,events,result,chunks = self.scenario(mode='rag',label='rag')
        self.assertEqual(model.planner_messages,[])
        self.assertEqual(len(events),1)
        model,events,result,chunks = self.scenario(slots=1,calls='self.release()',label='one-slot-key')
        self.assertEqual(events[-1]['metrics']['new_evidence_count'],1)
        self.assertIn('Dock.release',{r['symbol'] for r in model.final_rows()})

    def promote_fixture(self, *, mutations=(), ranks=(1,2,3), reverse=False, held_change=None,
                        scope_change=None, duplicates=False, directed=True, new_limit=2):
        pid, project, chunks, base, scope = self.case
        pool = CandidateEvidencePool(8)
        held = EvidenceBuilder().build_from_code_chunks([chunks[n] for n in base],project)
        if held_change: held_change(held[0])
        names = ['Dock.irrelevant','Dock.open_gate','Dock.release']
        candidates = pool.normalize_retrieval_results(
            [_chunk_as_read_candidate(chunks[n],rank=r) for n,r in zip(names,ranks)],provenance='planner_read_source')
        if duplicates: candidates += [candidates[-1]]
        target = next(c for c in candidates if c.qualified_name=='Dock.release')
        candidates = [replace(target, **mutation) for mutation in mutations] + candidates
        if reverse: candidates.reverse()
        return pool.promote(candidates,database=self.db,project=project,project_id=pid,
            repository_revision=REVISION,retrieval_strategy_version='weighted-rrf-v1',
            new_evidence_limit=new_limit,held_chunk_identities={x.chunk_identity for x in held},
            directed_read_evidence=held if directed else None,
            directed_read_scope_chunks=scope_change(scope) if scope_change else scope)

    def test_order_duplicates_invalid_identity_and_priority_are_separate(self):
        self.scenario(mode='rag',label='pool-fixture')
        mutations = ({'project_id':'foreign'}, {'repository_revision':'stale'}, {'path':'other/file.py'},
                     {'content_hash':'0'*64}, {'chunk_identity':'forged'}, {'fusion_score':float('nan')})
        for ranks in ((1,2,3),(1,3,2),(50,9,20)):
            for reverse in (False,True):
                with self.subTest(ranks=ranks,reverse=reverse):
                    p = self.promote_fixture(mutations=mutations,ranks=ranks,reverse=reverse,duplicates=True)
                    self.assertEqual({x.qualified_name for x in p.evidence},{'Dock.open_gate','Dock.release'})
                    self.assertEqual(sum(p.rejection_counts.values()),len(mutations))
                    self.assertEqual(p.new_candidate_count,2)
                    self.assertEqual({r['original_fusion_rank'] for r in p.selection_audit},set(ranks))
                    self.assertTrue(all('runtime target unresolved' in x.selection_reason for x in p.evidence))
        # Duplicate of an already held method consumes no new allowance.
        pid,project,chunks,base,scope = self.case
        pool = CandidateEvidencePool(8)
        candidates = pool.normalize_retrieval_results([_chunk_as_read_candidate(chunks[n],rank=i)
            for i,n in enumerate(['Dock.leave','Dock.irrelevant','Dock.open_gate','Dock.release'],1)],provenance='planner_read_source')
        held = EvidenceBuilder().build_from_code_chunks([chunks[n] for n in base],project)
        p = pool.promote(candidates,database=self.db,project=project,project_id=pid,repository_revision=REVISION,
            retrieval_strategy_version='weighted-rrf-v1',new_evidence_limit=2,held_chunk_identities={e.chunk_identity for e in held},
            directed_read_evidence=held,directed_read_scope_chunks=scope)
        self.assertEqual(p.duplicate_candidate_count,1)
        self.assertEqual({e.qualified_name for e in p.evidence},{'Dock.open_gate','Dock.release'})

    def test_alias_nested_scope_rebinding_and_missing_targets_fall_back(self):
        for label,calls in [('alias','alias.release()'),('missing','self.missing()'),
                            ('cls','cls.release()'),('other_owner','Other.release()')]:
            with self.subTest(label=label):
                model,events,result,chunks = self.scenario(calls=calls,label=label)
                self.assertEqual(events[-1]['metrics']['directed_priority_count'],0)
                self.assertIn('Dock.irrelevant',{r['symbol'] for r in model.final_rows()})
        # No source proof, changed identity, truncated input, ambiguous class or
        # target, and incomplete hierarchy all keep the original ordering.
        self.scenario(mode='rag',label='proof-fixture')
        changes = [lambda e:setattr(e,'excerpt',e.excerpt[:20]),
                   lambda e:setattr(e,'path','other.py'),
                   lambda e:setattr(e,'repository_revision','stale'),
                   lambda e:setattr(e,'_excerpt_coverage',None)]
        for change in changes:
            p=self.promote_fixture(held_change=change)
            self.assertTrue(all(r['selection_reason']=='original_order_fallback' for r in p.selection_audit))
        for name in ('Dock.release','Dock','Dock.leave'):
            def ambiguous(scope):
                c=next(c for c in scope if c['qualified_name']==name)
                return scope+[dict(c,id=9999)]
            p=self.promote_fixture(scope_change=ambiguous)
            if name=='Dock.release':
                self.assertNotIn('Dock.release',{e.qualified_name for e in p.evidence})
            else:
                self.assertTrue(all(r['selection_reason']=='original_order_fallback' for r in p.selection_audit))
        p=self.promote_fixture(scope_change=lambda scope:scope*2049)
        self.assertTrue(all(r['selection_reason']=='original_order_fallback' for r in p.selection_audit))

    def test_broad_or_non_agent_promotion_keeps_original_order_and_single_limit(self):
        self.scenario(mode='rag',label='unchanged-fixture')
        p=self.promote_fixture(directed=False,new_limit=1)
        self.assertEqual([e.qualified_name for e in p.evidence],['Dock.irrelevant'])
        self.assertEqual(p.selection_audit,[])

    def test_ambiguous_duplicate_method_never_gets_static_priority(self):
        model,events,result,chunks=self.scenario(calls='self.release()',
            extra='    def release(self):\n        return 88\n\n',label='ambiguous-method')
        self.assertEqual(events[-1]['metrics']['directed_priority_count'],0)

    def test_truncated_target_remains_incomplete(self):
        self.scenario(mode='rag',label='truncation-fixture')
        pid,project,chunks,base,scope=self.case
        chunk=chunks['Dock.release']
        evidence=EvidenceBuilder(excerpt_limit=30).build_from_code_chunks([chunk],project)[0]
        self.assertTrue(evidence.planner_excerpt_coverage()['truncated'])
        self.assertLess(evidence.planner_excerpt_coverage()['complete_end_line'],chunk['end_line'])

    def test_nested_scope_and_rebound_receiver_are_not_static_targets(self):
        from app.services.candidate_evidence import _method_ast
        for source in ('def leave(self):\n    def inner(self):\n        return self.release()\n    return 1\n',
                       'def leave(self):\n    self = other\n    return self.release()\n',
                       'def leave(self):\n    import self.other\n    return self.release()\n',
                       'def leave(self):\n    try:\n        pass\n    except Exception as self:\n        return self.release()\n'):
            self.assertIsNone(_method_ast(source,'leave'))

    def test_mapping_rest_rebinding_self_falls_back_but_other_rest_is_allowed(self):
        from app.services.candidate_evidence import _method_ast
        source = ('def leave(self, value):\n'
                  '    match value:\n'
                  '        case {**self}:\n'
                  '            return self.release()\n')
        self.assertIsNone(_method_ast(source, 'leave'))
        self.assertIsNotNone(_method_ast(source.replace('{**self}', '{**rest}'), 'leave'))


if __name__ == '__main__':
    unittest.main()
