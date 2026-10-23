"""Actual message boundary regressions; scripted Planner and fake answer only."""
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

from app.services import agent_core
from app.services.agent_contracts import AgentLimits
from tests.test_source_read_information_flow import SourceReadInformationFlowTests


class ReadSourceDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.flow = SourceReadInformationFlowTests()
        self.flow.setUp()
        self.addCleanup(self.flow.doCleanups)

    def projections(self, model):
        return [row for message in model.planner_messages[1:]
                for row in json.loads(message[1]['content'])['planner_evidence_view']['rows']]

    def assert_delivery(self, model, chunk, result):
        # A complete BASE excerpt is already delivered in the initial view.
        rows = [row for row in self.projections(model) if row['symbol'] == chunk['qualified_name']]
        self.assertTrue(rows, 'Canonical body absent from actual next Planner message')
        row = rows[-1]
        self.assertEqual(row['untrusted_source_excerpt'], chunk['content'])
        self.assertEqual((row['path'], row['start_line'], row['end_line']),
                         (chunk['path'], chunk['start_line'], chunk['end_line']))
        self.assertFalse(row['excerpt_coverage']['truncated'])
        self.assertEqual(row['excerpt_coverage']['complete_end_line'], chunk['end_line'])
        final = next(x for x in model.final_rows() if x['symbol'] == chunk['qualified_name'])
        self.assertEqual(final['untrusted_source_excerpt'], chunk['content'])
        citation = next(x for x in result['citations'] if x['qualified_name'] == chunk['qualified_name'])
        self.assertEqual((citation['path'], citation['start_line'], citation['end_line']),
                         (chunk['path'], chunk['start_line'], chunk['end_line']))

    def test_existing_complete_read_is_delivered_click_and_neutral(self):
        for kind, name in [('click', 'open_file'), ('neutral', 'PacketVault.open')]:
            with self.subTest(kind=kind):
                self.flow.fixture(kind)
                chunk = self.flow.chunks[name]
                model, events, result = self.flow.run_flow([self.flow.read(chunk)], label='A-complete-'+kind)
                read = events[-1]
                self.assertEqual(read['results']['content'], chunk['content'])
                self.assertEqual(read['metrics']['new_evidence_count'], 0)
                self.assertEqual(read['results']['coverage_updates'], [])
                self.assert_delivery(model, chunk, result)
                associated = read['results']['associated_evidence']
                self.assertEqual(len(associated), 1)
                item = next(iter(associated.values()))
                self.assertEqual((item['path'], item['content_hash'], item['repository_revision']),
                                 (chunk['path'], chunk['content_hash'], chunk['repository_revision']))
                self.assertEqual(item['read_overlap_start_line'], chunk['start_line'])
                self.assertEqual(item['read_overlap_end_line'], chunk['end_line'])

    def test_historical_lazy_then_complete_open_file_sequence(self):
        lazy = self.flow.fixture('click')
        opened = self.flow.chunks['open_file']
        model, events, result = self.flow.run_flow(
            [self.flow.read(lazy), self.flow.read(opened), self.flow.read(opened)], label='historical-sequence')
        self.assertIn('self.close_intelligently()', json.dumps(model.planner_messages[1:]))
        third = json.loads(model.planner_messages[2][1]['content'])
        source = next(o['source_read'] for o in third['untrusted_observation_summaries']
                      if o.get('source_read', {}).get('start_line') == opened['start_line'])
        self.assertTrue(source['projected_evidence_ids'])
        self.assertIn('f, should_close = open_stream', json.dumps(third['planner_evidence_view']['rows']))
        self.assertEqual(events[-1]['results']['coverage_updates'], [])
        self.assertEqual([c['status'] for s in result['agent_trace'] for c in s['tool_calls']][-1], 'rejected')

    def test_truncated_held_evidence_completion_still_delivered(self):
        for kind in ['click', 'neutral']:
            with self.subTest(kind=kind):
                chunk = self.flow.fixture(kind)
                model, events, result = self.flow.run_flow([self.flow.read(chunk)], label='B-truncated-'+kind)
                self.assertGreater(events[-1]['metrics']['information_delta_bytes'], 0)
                self.assert_delivery(model, chunk, result)
                self.assertTrue(events[-1]['results']['associated_evidence'])

    def test_new_atomic_methods_and_neutral_function_delivered(self):
        for kind, names in [('click', ['_AtomicFile.close', '_AtomicFile.__exit__']), ('neutral', ['filler_3'])]:
            with self.subTest(kind=kind):
                self.flow.fixture(kind)
                chunks = [self.flow.chunks[name] for name in names]
                model, events, result = self.flow.run_flow([self.flow.read(chunk) for chunk in chunks], label='C-new-'+kind)
                for chunk in chunks:
                    self.assert_delivery(model, chunk, result)
                self.assertEqual(len(result['evidence']), 5+len(chunks))
                for event in events[1:]:
                    self.assertEqual(event['metrics']['new_evidence_count'], 1)
                    self.assertTrue(event['results']['associated_evidence'])

    def test_complete_delivery_is_planning_progress_without_store_change(self):
        self.flow.fixture('neutral')
        chunk = self.flow.chunks['PacketVault.open']
        states = []
        original = agent_core._planner_state
        def observe(state):
            states.append((state.no_progress_count, set(state.progress_keys)))
            return original(state)
        with patch.object(agent_core, '_planner_state', side_effect=observe):
            model, events, result = self.flow.run_flow([self.flow.read(chunk), self.flow.read(chunk)], label='already-delivered-repeat')
        self.assert_delivery(model, chunk, result)
        self.assertEqual(states[1][0], 1)
        self.assertEqual(states[1][1], states[0][1])
        self.assertEqual(events[-1]['metrics']['information_delta_bytes'], 0)
        self.assertEqual([c['status'] for s in result['agent_trace'] for c in s['tool_calls']][-1], 'rejected')
        self.assertEqual(len(model.planner_messages), 2)
        self.assertEqual(states[-1][0], 2)

    def test_window_rotation_does_not_recount_old_body_as_progress(self):
        self.flow.fixture('click'); chunk = self.flow.chunks['open_file']
        states = []
        original = agent_core._planner_state
        def observe(state):
            states.append((state.no_progress_count, set(state.progress_keys)))
            return original(state)
        # Test-only explicit bounds let seven zero-increment observations rotate
        # through the five-entry projection window; production defaults unchanged.
        limits = replace(AgentLimits(), max_agent_steps=9, max_tool_calls=10,
                         max_same_tool_calls=9, max_no_progress_steps=20)
        actions = [self.flow.read(chunk, start=chunk['start_line']+i) for i in range(7)]
        with patch.object(agent_core, '_planner_state', side_effect=observe):
            model, events, result = self.flow.run_flow(actions, limits=limits, label='window-rotation')
        self.assertEqual(states[-1][0], 7)
        self.assertEqual(states[-1][1], states[0][1])
        self.assertEqual(len(model.payload()['untrusted_observation_summaries']), 5)
        self.assertEqual(len(events), 8)
        self.assertTrue(all(e['metrics']['information_delta_bytes'] == 0 for e in events[1:]))
        self.assert_delivery(model, chunk, result)

    def test_projection_budget_omission_is_explicit(self):
        self.flow.fixture('neutral'); chunk = self.flow.chunks['PacketVault.open']
        limits = replace(AgentLimits(), max_source_read_bytes=128)
        model, events, result = self.flow.run_flow([self.flow.read(chunk)], limits=limits, label='projection-omitted')
        self.assertEqual(events[-1]['results']['content'], chunk['content'])
        source = model.payload()['untrusted_observation_summaries'][-1]['source_read']
        self.assertTrue(source['evidence_ids'])
        self.assertEqual(source['projected_evidence_ids'], [])
        self.assertTrue(source['canonical_excerpt_projection_truncated'])
        self.assertEqual(source['canonical_excerpt_delivery_status'], 'omitted')
        self.assertEqual(source['canonical_excerpt_omissions'][0]['reason'], 'projection_byte_budget')
        self.assertNotIn(chunk['content'], json.dumps(model.planner_messages[1:]))
        self.assertEqual(next(x for x in model.final_rows() if x['symbol'] == chunk['qualified_name'])['untrusted_source_excerpt'], chunk['content'])

    def test_planner_delivery_does_not_override_final_byte_exclusion(self):
        self.flow.fixture('neutral'); chunk = self.flow.chunks['PacketVault.open']
        self.flow.base = ['filler_0', 'filler_1', 'filler_2', 'PacketVault', 'PacketVault.open']
        limits = replace(AgentLimits(), max_accumulated_evidence_context_bytes=200)
        model, events, result = self.flow.run_flow([self.flow.read(chunk)], limits=limits, label='delivered-final-excluded')
        row = next(x for x in self.projections(model) if x['symbol'] == chunk['qualified_name'])
        self.assertEqual(row['untrusted_source_excerpt'], chunk['content'])
        self.assertEqual(row['final_context_selection'], 'excluded_by_byte_budget')
        self.assertFalse(any(x['symbol'] == chunk['qualified_name'] for x in model.final_rows()))
        self.assertFalse(any(x['qualified_name'] == chunk['qualified_name'] for x in result['citations']))

    def test_rejected_candidate_does_not_associate_existing_evidence(self):
        self.flow.fixture('neutral'); chunk = self.flow.chunks['PacketVault.open']
        for field, value in [('project_id', 'foreign'), ('repository_revision', 'old'),
                             ('content_hash', '0'*64), ('id', 99999), ('end_line', 9999)]:
            with self.subTest(field=field):
                bad = dict(chunk); bad[field] = value
                with patch.object(self.flow.db, 'get_code_chunks_for_hierarchy', return_value=[bad]):
                    model, events, result = self.flow.run_flow([self.flow.read(chunk)], label='association-invalid-'+field)
                self.assertEqual(events[-1]['results']['associated_evidence'], {})
                # Failed association cannot revoke already validated BASE text.
                self.assertIn(chunk['content'],[row['untrusted_source_excerpt'] for row in self.projections(model)])
                source = model.payload()['untrusted_observation_summaries'][-1]['source_read']
                self.assertEqual(source['canonical_excerpt_delivery_status'], 'no_valid_association')


if __name__ == '__main__':
    unittest.main()
