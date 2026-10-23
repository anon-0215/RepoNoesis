"""Real request boundary and authoritative evidence flow; no default DB or Provider."""
import asyncio
from dataclasses import replace
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pydantic import ValidationError
from app import main
from app.services.agent_contracts import AgentLimits
from app.services.agent_tools import ToolRegistry
from tests.m1_helpers import disabled_embedding_service
from tests.test_m2_agent import decision
from tests import test_supplemental_evidence_flow as fixtures
from tests.test_supplemental_evidence_flow import MessageModel


class AgentEvidenceHeadroomTests(unittest.TestCase):
    setUp = fixtures.SupplementalEvidenceFlowTests.setUp
    hit = fixtures.SupplementalEvidenceFlowTests.hit
    legal_actions = fixtures.SupplementalEvidenceFlowTests.legal_actions

    def request_flow(self, payload=None, *, actions=(), windows=None, limits=None):
        payload = {'question': 'Explain stored lifecycle.', **(payload or {})}
        request = main.AskRequest.model_validate(payload)
        model = MessageModel(list(actions) + [decision('answer')])
        events = []
        windows = iter(windows or [self.base])
        limits = limits or AgentLimits()
        def retrieve(*args, **kwargs):
            values = next(windows)
            # Respect the real retrieval limit; candidates carry forged text,
            # so only authoritative rereads can enter the final request.
            hits = [self.hit(n, i) if isinstance(n, str) else n for i, n in enumerate(values, 1)]
            return SimpleNamespace(results=hits[:kwargs['evidence_count']], retrieval_mode='lexical',
                retrieval_strategy_version='weighted-rrf-v1', warnings=[], audit={})
        original = ToolRegistry._execute_resolved
        def observe(registry, context, call, spec):
            result = original(registry, context, call, spec)
            events.append({'tool': call.tool_name, 'parameters': call.parameters, 'status': result.status,
                'metrics': result.metrics, 'policy': context.evidence_admission_policy(),
                'held': [e.to_dict() for e in context.evidence_store.all(context.request_id)]})
            return result
        async def consume():
            response = await main.ask_project_stream(self.pid, request)
            frames = []
            async for line in response.body_iterator:
                frames.append(json.loads(line))
            return frames
        with (patch.object(main, 'db', self.db), patch.object(main, 'llm', model),
              patch.object(main, 'embedding_service', disabled_embedding_service()),
              patch.object(main, 'learning_service', SimpleNamespace(get_learning_context=lambda _: None)),
              patch.object(main, '_bundle_or_404', return_value=self.bundle),
              patch.object(main, 'agent_limits', limits),
              patch('app.services.agent_tools.retrieve_code', side_effect=retrieve),
              patch.object(ToolRegistry, '_execute_resolved', observe)):
            frames = asyncio.run(consume())
        self.assertEqual(frames[-1]['type'], 'completed', frames[-1])
        return model, events, frames[-1]['result']

    def assert_capacity(self, model, events, held, capacity):
        p = model.payload(0)
        self.assertEqual(p['evidence_count'], held)
        self.assertEqual(p['known_evidence_ids'], ['E'+str(i) for i in range(1, held+1)])
        self.assertEqual(p['server_constraints']['evidence_admission']['capacity'], capacity)
        self.assertEqual(p['server_constraints']['evidence_admission']['remaining_capacity'], capacity-held)
        self.assertEqual(events[0]['policy']['remaining_capacity'], capacity-held)
        self.assertEqual(len(events[0]['held']), held)

    def test_default_desktop_agent_five_of_eight_and_rag_stays_five(self):
        self.assertEqual(main.AskRequest(question='Explain.').evidence_count, 8)
        self.assertNotIn('evidence_count', main.AskRequest(question='Explain.').model_fields_set)
        explicit = main.AskRequest(question='Explain.', evidence_count=5)
        self.assertIn('evidence_count', explicit.model_fields_set)
        self.assertEqual(explicit.evidence_count, 5)
        for mode, expected in [('agent', 8), ('rag', 5)]:
            request = main.AskRequest(question='Explain.', execution_mode=mode)
            self.assertEqual(main.AskRequest.model_validate(request.model_dump()).evidence_count, expected)
        model, events, result = self.request_flow()
        self.assert_capacity(model, events, 5, 8)
        self.assertEqual(events[0]['parameters']['top_k'], 5)
        self.assertEqual(result['budget_usage']['base_evidence_count_effective'], 5)
        model, events, result = self.request_flow({'execution_mode': 'rag'})
        self.assertEqual(model.planner_messages, [])
        self.assertEqual(events[0]['policy']['capacity'], 5)
        self.assertEqual(events[0]['policy']['remaining_capacity'], 0)
        self.assertEqual(len(model.final_rows()), 5)

    def test_explicit_caps_and_actual_short_initial_hits(self):
        for capacity in [1, 3, 5, 8]:
            for short in [False, True]:
                with self.subTest(capacity=capacity, short=short):
                    base = self.base[:2] if short else self.base
                    model, events, result = self.request_flow({'evidence_count': capacity}, windows=[base])
                    self.assert_capacity(model, events, min(len(base), capacity), capacity)
                    self.assertLessEqual(len(model.final_rows()), capacity)
        model, events, result = self.request_flow(windows=[self.base[:2]])
        self.assert_capacity(model, events, 2, 8)

    def test_explicit_base_configuration_is_clamped_without_expanding_cap(self):
        for base, capacity, expected in [(2, None, 2), (8, None, 8), (8, 3, 3), (2, 5, 2)]:
            with self.subTest(base=base, capacity=capacity):
                payload = {} if capacity is None else {'evidence_count': capacity}
                all_hits = self.base + ['MailDock.register','MailDock.close','MailDock.unrelated']
                model, events, result = self.request_flow(payload, windows=[all_hits],
                    limits=replace(AgentLimits(), base_evidence_count=base))
                self.assert_capacity(model, events, expected, capacity or 8)
                self.assertEqual(result['budget_usage']['base_evidence_count_effective'], expected)

    def test_six_to_eight_enter_actual_final_input_ninth_rejected_and_existing_readable(self):
        actions = [decision('continue','lookup_symbol',{'symbol':symbol}) for symbol in
            ['MailDock.register','MailDock._release_queue','MailDock.enter_scope','MailDock.unrelated']]
        target = self.chunks['MailDock.scope']
        actions.append(decision('continue','read_source',{'path':target['path'],
            'start_line':target['start_line'],'end_line':target['end_line']}))
        model, events, result = self.request_flow(actions=actions, limits=replace(AgentLimits(),max_agent_steps=6,max_same_tool_calls=5))
        self.assert_capacity(model, events, 5, 8)
        self.assertEqual([len(e['held']) for e in events], [5,6,7,8,8,8])
        self.assertEqual(events[4]['metrics']['admission_reason'], 'capacity_exhausted')
        self.assertEqual(events[5]['status'], 'succeeded')
        self.assertEqual(events[5]['metrics']['admission_reason'], 'duplicate')
        self.assertEqual(len(model.final_rows()), 8)
        self.assertEqual([x['evidence_id'] for x in result['evidence']], ['E'+str(i) for i in range(1,9)])
        self.assertEqual({c['evidence_id'] for c in result['citations']}, {e['evidence_id'] for e in result['evidence']})
        final = {r['symbol']:r['untrusted_source_excerpt'] for r in model.final_rows()}
        for symbol in ['MailDock.register','MailDock._release_queue','MailDock.enter_scope']:
            self.assertEqual(final[symbol], self.chunks[symbol]['content'])
        self.assertNotIn('UNTRUSTED CANDIDATE SOURCE', json.dumps(model.final_messages))
        self.assertEqual(model.payload()['server_constraints']['evidence_admission']['remaining_capacity'], 0)

    def test_duplicate_invalid_and_failed_validation_leave_actual_slots(self):
        for hit, reason in [(self.hit('MailDock.scope'), 'duplicate'),
                (self.hit('MailDock.register', project_id='foreign'), 'invalid_candidates'),
                (self.hit('MailDock.register', content_hash='0'*64), 'invalid_candidates'),
                (self.hit('MailDock.register', repository_revision='stale'), 'invalid_candidates')]:
            with self.subTest(reason=reason):
                model, events, result = self.request_flow(
                    actions=[decision('continue','search_code',{'query':'register'})], windows=[self.base,[hit]])
                self.assertEqual(events[-1]['metrics']['admission_reason'], reason)
                self.assertEqual(events[-1]['policy']['remaining_capacity'], 3)
                self.assertEqual(model.payload()['evidence_count'], 5)
                self.assertEqual(model.payload()['untrusted_observation_summaries'][-1]['candidate_metrics']['admission_reason'],reason)

    def test_byte_budget_still_excludes_evidence_from_final_citations(self):
        model, events, result = self.request_flow(actions=[decision('continue','lookup_symbol',{'symbol':'MailDock._release_queue'})],
            limits=replace(AgentLimits(),max_accumulated_evidence_context_bytes=200))
        self.assertEqual(model.payload()['evidence_count'], 6)
        locators = model.payload()['evidence_locators']
        selected = {x['evidence_id'] for x in locators if x['final_context_selection']=='selected_by_byte_budget'}
        excluded = {x['evidence_id'] for x in locators if x['final_context_selection']=='excluded_by_byte_budget'}
        self.assertTrue(excluded)
        self.assertEqual({x['evidence_id'] for x in result['evidence']}, selected)
        self.assertFalse(excluded & {x['evidence_id'] for x in result['citations']})
        self.assertEqual(len(model.final_rows()), len(selected))

    def test_null_and_out_of_range_do_not_mean_omission(self):
        for value in [None, 0, 9]:
            with self.subTest(value=value), self.assertRaises(ValidationError):
                main.AskRequest.model_validate({'question':'Explain.', 'evidence_count':value})


if __name__ == '__main__':
    unittest.main()
