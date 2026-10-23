"""Actual relation/acquisition/message flow; scripted actions do not test quality."""
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
from app.services.hybrid_retriever import HybridSearchResult
from app.services.symbol_retriever import SymbolRetriever
from tests.m1_helpers import disabled_embedding_service
from tests.m3_helpers import REVISION, make_relation_project
from tests.test_m2_agent import decision
from tests.test_supplemental_evidence_flow import MessageModel


class SupplementalEvidenceConsistencyTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.db = Database(Path(directory.name) / 'consistency.sqlite')
        source = ('def entry():\n    return relay_a() + relay_b()\n\n'
                  'def relay_a():\n    return 11\n\n'
                  'def relay_b():\n    return 22\n\n'
                  'def extra_b():\n    return 33\n\n'
                  'def extra_c():\n    return 44\n\n')
        source += ''.join(f'def ballast_{i}():\n    return {i}\n\n' for i in range(4))
        self.pid, self.bundle = make_relation_project(self.db, {'pkg/relay.py': source})
        self.chunks = {c['qualified_name']: c for c in self.db.get_code_chunks(self.pid)}
        self.base = ['entry', 'ballast_0', 'ballast_1', 'ballast_2', 'ballast_3']
        self.limits = replace(AgentLimits(), base_evidence_count=5, max_agent_steps=6, max_tool_calls=10)

    def hit(self, symbol, rank=1, **changes):
        chunk = self.chunks[symbol]
        hit = HybridSearchResult(
            project_id=self.pid, repository_revision=REVISION, code_chunk_id=chunk['id'],
            language=chunk['language'], path=chunk['path'], chunk_type=chunk['chunk_type'],
            symbol_name=chunk['symbol_name'], qualified_name=chunk['qualified_name'],
            start_line=chunk['start_line'], end_line=chunk['end_line'],
            content='UNTRUSTED CANDIDATE BODY', content_hash=chunk['content_hash'],
            retrieval_sources=['lexical'], fusion_rank=rank, fusion_score=1/rank,
        )
        return replace(hit, **changes)

    def run_flow(self, actions, windows, label, symbol_window=None):
        model = MessageModel(actions + [decision('answer')])
        windows = iter(windows)
        events = []
        def retrieve(*args, **kwargs):
            hits = [self.hit(x, i) if isinstance(x, str) else x
                    for i, x in enumerate(next(windows), 1)]
            return SimpleNamespace(results=hits, retrieval_mode='lexical',
                retrieval_strategy_version='weighted-rrf-v1', warnings=[], audit={})
        execute = ToolRegistry._execute_resolved
        def observe(registry, context, call, spec):
            before = [x.to_dict() for x in context.evidence_store.all(context.request_id)]
            observation = execute(registry, context, call, spec)
            events.append({'tool': call.tool_name, 'parameters': call.parameters,
                'status': observation.status, 'error': observation.error,
                'metrics': observation.metrics, 'before': before,
                'after': [x.to_dict() for x in context.evidence_store.all(context.request_id)],
                'pool_identities': [x.chunk_identity for x in context.candidate_pool.accepted_candidates()]})
            return observation
        search = SymbolRetriever.search
        def symbols(retriever, *args, **kwargs):
            return symbol_window if symbol_window is not None else search(retriever, *args, **kwargs)
        with patch('app.services.agent_tools.retrieve_code', side_effect=retrieve), \
                patch.object(ToolRegistry, '_execute_resolved', observe), \
                patch.object(SymbolRetriever, 'search', symbols):
            result = run_bounded_agent(
                'Explain the stored relay.', self.bundle, model, self.db, disabled_embedding_service(),
                planner=LLMPlanner(model, build_m2_tool_registry(self.limits), self.limits),
                limits=self.limits, evidence_count=8, execution_mode='agent',
                request_id='consistency-'+label, hierarchy_mode='off', relation_mode='off',
            )
        folder = os.environ.get('SUPPLEMENTAL_FLOW_CAPTURE_DIR')
        if folder:
            (Path(folder)/(label+'-messages.json')).write_text(json.dumps({
                'planner_messages': model.planner_messages, 'final_messages': model.final_messages,
                'tool_events': events, 'result': result,
                'scripted_model_not_quality_evidence': True,
            }, ensure_ascii=False, indent=2), encoding='utf-8')
        return model, events, result

    @staticmethod
    def relation_action():
        return decision('continue', 'expand_relations', {
            'seed_evidence_ids': ['E1'], 'relation_types': ['calls'],
            'direction': 'outbound', 'max_depth': 1,
        })

    def assert_final_binding(self, model, events, result):
        held = {x['chunk_identity']: x for x in events[-1]['after']}
        for event in events:
            for item in event['after']:
                self.assertEqual(item['evidence_id'], held[item['chunk_identity']]['evidence_id'])
        self.assertTrue(set(events[-1]['pool_identities']) <= set(held))
        self.assertEqual({x['evidence_id'] for x in result['citations']},
                         {x['evidence_id'] for x in result['evidence']})
        final = {x['symbol']: x['untrusted_source_excerpt'] for x in model.final_rows()}
        for symbol, excerpt in final.items():
            self.assertEqual(excerpt, self.chunks[symbol]['content'])
        self.assertNotIn('UNTRUSTED CANDIDATE BODY', json.dumps(model.final_messages))

    def test_relation_then_broad_search_respects_actual_occupancy_and_full_capacity(self):
        actions = [self.relation_action(),
            decision('continue', 'search_code', {'query': 'wide extra'}),
            decision('continue', 'lookup_symbol', {'symbol': 'extra_b'}),
            decision('continue', 'lookup_symbol', {'symbol': 'extra_c'})]
        model, events, result = self.run_flow(actions, [self.base, ['extra_b']], 'relation-broad')
        self.assertEqual([len(x['after']) for x in events], [5, 7, 7, 8, 8])
        self.assertEqual(events[1]['status'], 'succeeded')
        broad = events[2]['metrics']
        self.assertEqual(broad['result_count'], 0)
        self.assertEqual(broad['remaining_capacity'], 1)
        self.assertEqual(broad['admission_reason'], 'admission_deferred')
        full = events[-1]['metrics']
        self.assertEqual(full['result_count'], 0)
        self.assertEqual(full['remaining_capacity'], 0)
        self.assertEqual(full['admission_reason'], 'capacity_exhausted')
        self.assertEqual(model.payload()['evidence_count'], 8)
        self.assert_final_binding(model, events, result)

    def test_relation_duplicate_a_does_not_consume_directed_new_b_allowance(self):
        actions = [self.relation_action(), decision('continue', 'search_code', {'query': 'extra_b', 'top_k': 2})]
        model, events, result = self.run_flow(actions, [self.base, ['relay_a', 'extra_b']], 'relation-duplicate')
        self.assertEqual([len(x['after']) for x in events], [5, 7, 8])
        self.assertEqual(events[-1]['metrics']['duplicate_candidate_count'], 1)
        self.assertEqual(events[-1]['metrics']['result_count'], 1)
        self.assertEqual(events[-1]['metrics']['admission_deferred_count'], 0)
        self.assertIn('extra_b', {x['qualified_name'] for x in result['evidence']})
        self.assert_final_binding(model, events, result)

    def test_held_claims_still_need_canonical_validation_and_do_not_leave_ghosts(self):
        for mutation in ({'content_hash': '0'*64}, {'start_line': 999},
                         {'project_id': 'other'}, {'repository_revision': 'stale'}):
            with self.subTest(mutation=mutation):
                actions = [self.relation_action(),
                    decision('continue', 'search_code', {'query': 'extra_b', 'top_k': 2}),
                    decision('continue', 'search_code', {'query': 'extra_c', 'top_k': 2})]
                forged = self.hit('relay_a', 1, **mutation)
                model, events, result = self.run_flow(actions,
                    [self.base, [forged, self.hit('extra_b', 2)], ['relay_a', 'extra_c']],
                    'forged-'+next(iter(mutation)))
                self.assertEqual([len(x['after']) for x in events], [5, 7, 8, 8])
                self.assertEqual(events[2]['metrics']['rejected_candidate_count'], 1)
                self.assertEqual(events[2]['metrics']['duplicate_candidate_count'], 0)
                self.assertEqual(events[2]['metrics']['result_count'], 1)
                self.assertEqual(events[-1]['metrics']['duplicate_candidate_count'], 1)
                self.assertEqual(events[-1]['metrics']['capacity_rejected_count'], 1)
                self.assert_final_binding(model, events, result)

    def test_failed_read_then_legal_acquisition_and_repair_keep_historical_snapshots(self):
        chunk = self.chunks['extra_b']
        actions = [decision('continue', 'read_source', {'path': 'pkg/relay.py', 'start_line': 1, 'end_line': 221}),
            decision('continue', 'read_source', {'path': chunk['path'], 'start_line': chunk['start_line'], 'end_line': chunk['end_line']}),
            {'status': 'invalid'}, decision('continue', 'search_code', {'query': 'extra_c'})]
        model, events, result = self.run_flow(actions, [self.base, ['extra_c']], 'failure-snapshot')
        self.assertEqual([len(x['after']) for x in events], [5, 5, 6, 7])
        self.assertEqual(events[1]['error']['code'], 'source_range_exceeded')
        initial, failed, success, repair, final = [model.payload(i) for i in range(5)]
        self.assertEqual(initial['server_constraints']['evidence_admission']['remaining_capacity'], 3)
        self.assertEqual(failed['server_constraints']['evidence_admission']['remaining_capacity'], 3)
        failure = failed['untrusted_observation_summaries'][-1]
        self.assertEqual(failure['result_count'], 0)
        self.assertIn('200', failure['read_range_feedback'])
        self.assertNotIn('capacity', failure['candidate_metrics'])
        self.assertNotIn('remaining_capacity', failure['candidate_metrics'])
        self.assertNotIn('max_new_per_supplement', failure['candidate_metrics'])
        self.assertIn('repair_request', repair)
        self.assertEqual(success['server_constraints']['evidence_admission']['remaining_capacity'], 2)
        self.assertEqual(repair['server_constraints']['evidence_admission']['remaining_capacity'], 2)
        self.assertEqual(final['server_constraints']['evidence_admission']['remaining_capacity'], 1)
        past = success['untrusted_observation_summaries'][-1]['candidate_metrics']
        self.assertEqual(past['remaining_capacity'], 2)
        self.assertEqual(final['untrusted_observation_summaries'][1]['candidate_metrics'], past)
        self.assertEqual(repair['untrusted_observation_summaries'], success['untrusted_observation_summaries'])
        self.assert_final_binding(model, events, result)

    def test_forged_locator_claiming_held_identity_is_rejected_before_duplicate_check(self):
        retriever = SymbolRetriever(self.db)
        a = retriever.search(self.pid, 'relay_a', explicit_symbol=True, repository_revision=REVISION)[0]
        b = retriever.search(self.pid, 'extra_b', explicit_symbol=True, repository_revision=REVISION)[0]
        for label, forged in (
            ('hash', replace(a, content_hash='0'*64, symbol_rank=1)),
            ('identity', replace(b, chunk_identity=a.chunk_identity, symbol_rank=1)),
        ):
            with self.subTest(label=label):
                window = [forged, replace(a, symbol_rank=2), replace(b, symbol_rank=3)]
                actions = [self.relation_action(), decision('continue', 'lookup_symbol', {'symbol': 'relay_a', 'top_k': 2})]
                model, events, result = self.run_flow(actions, [self.base], 'held-claim-'+label, window)
                self.assertEqual([len(x['after']) for x in events], [5, 7, 8])
                metrics = events[-1]['metrics']
                self.assertEqual(metrics['rejected_candidate_count'], 1)
                self.assertEqual(metrics['duplicate_candidate_count'], 1)
                self.assertEqual(metrics['result_count'], 1)
                self.assertEqual(metrics['admission_deferred_count'], 0)
                self.assert_final_binding(model, events, result)


if __name__ == '__main__':
    unittest.main()
