from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from app.database import Database
from app.services.agent_contracts import AgentLimits
from app.services.agent_core import LLMPlanner, _planner_state, run_bounded_agent
from app.services.agent_tools import EvidenceStore, build_m2_tool_registry
from app.services.evidence import EvidenceBuilder
from tests.m1_helpers import disabled_embedding_service, make_project
from tests.test_m2_agent import decision
from tests import test_planner_excerpt_coverage as coverage_tests

CapturingModel = coverage_tests.CapturingModel


class PlannerFinalContextSelectionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.database = Database(Path(directory.name) / 'selection.sqlite')
        self.first = 'def collect_note():\n    return "票"\n'
        self.second = 'def dispatch_note():\n    return "件"\n'
        self.project_id, self.bundle = make_project(self.database, [
            ('src/collect.py', 'collect_note', self.first),
            ('src/dispatch.py', 'dispatch_note', self.second),
        ])
        self.limits = replace(AgentLimits(), base_evidence_count=2,
                              max_accumulated_evidence_context_bytes=len(self.first.encode('utf-8')))

    def run_agent(self, model, *, limits=None, question='collect_note dispatch_note'):
        limits = limits or self.limits
        return run_bounded_agent(
            question, self.bundle, model, self.database, disabled_embedding_service(),
            planner=LLMPlanner(model, build_m2_tool_registry(limits), limits),
            limits=limits, evidence_count=8, execution_mode='agent', request_id='selection-request')

    @staticmethod
    def payload(model, index=0):
        return json.loads(model.planner_messages[index][1]['content'])

    def assert_selection_and_answer(self, model, result, *, index=0):
        payload = self.payload(model, index)
        self.assertEqual(payload['evidence_count'], 2)
        self.assertEqual(payload['known_evidence_ids'], ['E1', 'E2'])
        locators = payload['evidence_locators']
        self.assertEqual([x['evidence_id'] for x in locators], ['E1', 'E2'])
        self.assertEqual([x['qualified_name'] for x in locators], ['collect_note', 'dispatch_note'])
        self.assertEqual([x['final_context_selection'] for x in locators],
                         ['selected_by_byte_budget', 'excluded_by_byte_budget'])
        for locator in locators:
            self.assertFalse(locator['excerpt_coverage']['truncated'])
            self.assertEqual(locator['excerpt_coverage']['complete_end_line'], locator['end_line'])
        final = coverage_tests.PlannerExcerptCoverageTests.answer_excerpts(model)
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0]['symbol'], 'collect_note')
        self.assertEqual(final[0]['untrusted_source_excerpt'], self.first)
        self.assertNotIn('dispatch_note', json.dumps(final))
        self.assertEqual(result['agent_status'], 'completed')
        self.assertEqual([x['evidence_id'] for x in result['evidence']], ['E1'])
        self.assertEqual(result['citations'][0]['evidence_id'], 'E1')
        self.assertNotIn('final_context_selection', result['evidence'][0])
        self.assertIn('snapshot prediction', model.planner_messages[index][0]['content'])
        self.assertIn('cannot bypass', model.planner_messages[index][0]['content'])

    def test_complete_base_evidence_outside_byte_budget_is_held_but_not_answer_input(self):
        model = CapturingModel([decision('answer')])
        result = self.run_agent(model)
        self.assert_selection_and_answer(model, result)
        self.assertEqual(result['budget_usage']['tool_calls_used'], 0)

    def test_complete_tool_addition_updates_prediction_without_losing_owned_identity(self):
        model = CapturingModel([
            decision('continue', 'lookup_symbol', {'symbol': 'dispatch_note'}), decision('answer')])
        result = self.run_agent(model, limits=replace(self.limits, base_evidence_count=1),
                                question='collect_note')
        first = self.payload(model)
        self.assertEqual(first['known_evidence_ids'], ['E1'])
        self.assertEqual(first['evidence_locators'][0]['final_context_selection'], 'selected_by_byte_budget')
        self.assert_selection_and_answer(model, result, index=1)

    def test_byte_filter_stops_after_overflow_and_missing_budget_stays_unknown_in_repair(self):
        chunks = self.database.get_code_chunks(self.project_id)
        evidence = EvidenceBuilder().build_from_code_chunks(chunks, self.bundle['project'])
        store = EvidenceStore(capacity=8)
        # The first item is too large, while the later item would fit alone.
        # The existing filter stops at overflow instead of skipping that item.
        self.assertGreater(len(self.second.encode('utf-8')), len(self.first.encode('utf-8')))
        store.add('current', list(reversed(evidence)))
        state = SimpleNamespace(
            steps=[], remaining_budget=lambda: {'work_time_ms': 10000, 'tool_calls': 3},
            budget=SimpleNamespace(work_cutoff_at=None), request_id='current', user_goal='fixture',
            limits=self.limits,
            context=SimpleNamespace(evidence_store=store, project_id=self.project_id,
                                    repository_revision=self.bundle['project']['repository_revision']))
        model = CapturingModel([decision('answer')] * 2)
        planner = LLMPlanner(model, build_m2_tool_registry(self.limits), self.limits)
        planner.decide(_planner_state(state), repair_hint={'stable_code': 'invalid_json'})
        self.assertEqual([x['final_context_selection'] for x in self.payload(model)['evidence_locators']],
                         ['excluded_by_byte_budget'] * 2)
        del state.limits
        planner.decide(_planner_state(state), repair_hint={'stable_code': 'invalid_json'})
        self.assertEqual([x['final_context_selection'] for x in self.payload(model, 1)['evidence_locators']],
                         ['unknown'] * 2)
        self.assertEqual(self.payload(model, 1)['known_evidence_ids'], ['E1', 'E2'])


if __name__ == '__main__':
    unittest.main()
