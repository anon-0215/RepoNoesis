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
from app.services.evidence import CitationValidator, Evidence, EvidenceBuilder
from tests.m1_helpers import disabled_embedding_service, make_chunk, make_project
from tests.test_m2_agent import decision
from tests import test_planner_excerpt_coverage as coverage_tests

CapturingModel = coverage_tests.CapturingModel


class PlannerDirectCallHintsTests(unittest.TestCase):
    """Assertions capture actual chat inputs, not a model's simulated judgement."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.database = Database(Path(directory.name) / 'call-hints.sqlite')
        self.wrapper = ('def send_note(self):\n'
                        '    """A note wrapper; self.imaginary_note() is documentation."""\n'
                        '    # self.comment_only()\n'
                        '    return self.persist_note()\n')
        self.target = 'def persist_note(self):\n    return write_note()\n'
        self.project_id, self.bundle = make_project(self.database, [
            ('src/notes.py', 'NotePort.send_note', self.wrapper),
            ('src/storage.py', 'NotePort.persist_note', self.target),
        ])
        self.limits = replace(AgentLimits(), base_evidence_count=1)

    @staticmethod
    def payload(model, index=0):
        return json.loads(model.planner_messages[index][1]['content'])

    @staticmethod
    def hints(locator):
        return locator['direct_call_hints']

    def run_agent(self, model, *, limits=None, mode='agent'):
        limits = limits or self.limits
        return run_bounded_agent(
            'NotePort.send_note', self.bundle, model, self.database, disabled_embedding_service(),
            planner=LLMPlanner(model, build_m2_tool_registry(limits), limits),
            limits=limits, evidence_count=8, execution_mode=mode,
            hierarchy_mode='off', relation_mode='off', retrieval_version='v1',
            request_id='note-call-request')

    def built(self, source, limit=2000, start=30, **changes):
        chunk = make_chunk('src/independent.py', 'relay_note', source, start_line=start)
        chunk.update({'id': 91, 'project_id': self.project_id, **changes})
        return EvidenceBuilder(limit).build_from_code_chunks([chunk], self.bundle['project'])[0]

    def send_store(self, items, *, repair=False, limits=None):
        store = EvidenceStore(capacity=30)
        for owner, evidence in items:
            store.add(owner, [evidence])
        state = SimpleNamespace(
            steps=[], remaining_budget=lambda: {'work_time_ms': 10000, 'tool_calls': 3},
            budget=SimpleNamespace(work_cutoff_at=None), request_id='current', user_goal='fixture',
            limits=limits or self.limits,
            context=SimpleNamespace(evidence_store=store, project_id=self.project_id,
                                    repository_revision=self.bundle['project']['repository_revision']))
        model = CapturingModel([decision('answer')])
        LLMPlanner(model, build_m2_tool_registry(self.limits), self.limits).decide(
            _planner_state(state), repair_hint={'stable_code': 'invalid_json'} if repair else None)
        return model, self.payload(model)

    def test_initial_wrapper_has_call_source_but_target_is_not_evidence(self):
        model = CapturingModel([decision('answer')])
        result = self.run_agent(model)
        payload = self.payload(model)
        self.assertEqual(payload['evidence_count'], 1)
        self.assertEqual(payload['known_evidence_ids'], ['E1'])
        self.assertFalse(payload['direct_call_hints_may_be_incomplete'])
        locator = payload['evidence_locators'][0]
        self.assertEqual(locator['path'], 'src/notes.py')
        self.assertEqual(self.hints(locator), {'status': 'available', 'may_be_incomplete': False,
            'calls': [{'callee_expression': 'self.persist_note', 'start_line': 4, 'end_line': 4,
                       'resolution_status': 'unresolved'}]})
        self.assertEqual([x['qualified_name'] for x in payload['evidence_locators']], ['NotePort.send_note'])
        final = coverage_tests.PlannerExcerptCoverageTests.answer_excerpts(model)
        self.assertEqual([x['symbol'] for x in final], ['NotePort.send_note'])
        self.assertNotIn('def persist_note', json.dumps(final))
        self.assertEqual(len(result['evidence']), 1)
        self.assertEqual(result['citations'][0]['evidence_id'], 'E1')
        self.assertNotIn('direct_call_hints', result['evidence'][0])
        self.assertNotIn('source_identity', result['evidence'][0])
        self.assertEqual(result['budget_usage']['tool_calls_used'], 0)
        self.assertIn('rn-lwb-03-v11', model.planner_messages[0][0]['content'])
        self.assertIn('no target identity or runtime binding', model.planner_messages[0][0]['content'])

    def test_tool_promotion_then_actual_protocol_repair_keeps_calls_and_updates_final_evidence(self):
        model = CapturingModel([
            decision('continue', 'lookup_symbol', {'symbol': 'NotePort.persist_note'}),
            {'status': 'invalid-status'}, decision('answer')])
        result = self.run_agent(model)
        first, second, repair = [self.payload(model, i) for i in range(3)]
        self.assertEqual(first['evidence_count'], 1)
        self.assertEqual(second['evidence_count'], 2)
        self.assertEqual(second['known_evidence_ids'], ['E1', 'E2'])
        self.assertEqual(first['evidence_locators'][0], second['evidence_locators'][0])
        self.assertEqual(second['evidence_locators'], repair['evidence_locators'])
        self.assertIn('repair_request', repair)
        self.assertEqual(self.hints(second['evidence_locators'][1])['calls'], [
            {'callee_expression': 'write_note', 'start_line': 2, 'end_line': 2,
             'resolution_status': 'unresolved'}])
        final = coverage_tests.PlannerExcerptCoverageTests.answer_excerpts(model)
        self.assertEqual([x['symbol'] for x in final], ['NotePort.send_note', 'NotePort.persist_note'])
        self.assertEqual(final[1]['untrusted_source_excerpt'], self.target)
        self.assertEqual([x['evidence_id'] for x in result['evidence']], ['E1', 'E2'])
        self.assertEqual(result['budget_usage']['tool_calls_used'], 1)
        self.assertTrue(all('direct_call_hints' not in x for x in result['evidence']))

    def test_read_source_addition_has_new_call_clue_without_following_its_target(self):
        model=CapturingModel([decision('continue','read_source',
            {'path':'src/storage.py','start_line':1,'end_line':2}),decision('answer')])
        result=self.run_agent(model)
        payload=self.payload(model,1)
        self.assertEqual(payload['evidence_count'],2)
        self.assertEqual(payload['known_evidence_ids'],['E1','E2'])
        self.assertEqual(self.hints(payload['evidence_locators'][1])['calls'][0]['callee_expression'],'write_note')
        self.assertNotIn('write_note',[x['qualified_name'] for x in payload['evidence_locators']])
        self.assertEqual([x['evidence_id'] for x in result['evidence']],['E1','E2'])

    def test_comments_docstrings_and_string_literals_are_not_calls(self):
        item = self.built('def relay_note():\n    """fake_doc()"""\n    # fake_comment()\n'
                          '    text = "fake_literal()"\n    return actual_note()\n')
        model, payload = self.send_store([('current', item)])
        hints = self.hints(payload['evidence_locators'][0])
        self.assertEqual([x['callee_expression'] for x in hints['calls']], ['actual_note'])
        self.assertNotIn('fake_', model.planner_messages[0][1]['content'])

    def test_long_class_prefix_does_not_claim_tail_calls(self):
        source = ('class NoteCrate:\n    def relay(self):\n        return self.local_note()\n'
                  + '    # reserve space\n' * 30 +
                  '    def later(self):\n        return self.hidden_note()\n')
        cut = source.index('    def later')
        item = self.built(source, limit=cut)
        _, payload = self.send_store([('current', item)])
        locator = payload['evidence_locators'][0]
        self.assertGreater(locator['end_line'], locator['excerpt_coverage']['complete_end_line'])
        self.assertEqual([x['callee_expression'] for x in self.hints(locator)['calls']], ['self.local_note'])
        self.assertTrue(self.hints(locator)['may_be_incomplete'])

    def test_line_midline_crlf_and_multiline_cutoffs_do_not_overclaim(self):
        source = 'def relay_note():\r\n    return first_note(); second_note()\r\n'
        boundary = source.index('\n') + 1
        for cut in (boundary - 1, boundary, source.index(';') + 1, len(source) - 1, len(source)):
            with self.subTest(cut=cut):
                _, payload = self.send_store([('current', self.built(source, cut))])
                hints = self.hints(payload['evidence_locators'][0])
                if cut < len(source):
                    self.assertEqual(hints['status'], 'unknown')
                    self.assertEqual(hints['calls'], [])
                else:
                    self.assertEqual([x['callee_expression'] for x in hints['calls']], ['first_note', 'second_note'])
        source = 'def relay_note():\n    return first_note(\n        3\n    )\n'
        _, payload = self.send_store([('current', self.built(source, source.index('    )')))])
        self.assertEqual(self.hints(payload['evidence_locators'][0])['status'], 'unknown')
        _, payload = self.send_store([('current', self.built('def relay_note():\n    return first_note()', 2000))])
        self.assertEqual(self.hints(payload['evidence_locators'][0])['calls'][0]['end_line'], 31)

    def test_parse_failure_and_language_or_input_limits_are_explicit(self):
        cases=[('def relay_note():\n    return (\n', 2000, {}, 'unknown'),
               ('not python\n', 2000, {'language': 'javascript'}, 'unavailable'),
               ('def relay_note():\n    text = "' + 'x' * 9000 + '"\n    return actual_note()\n', 20000, {}, 'unknown')]
        for source, limit, changes, status in cases:
            with self.subTest(status=status, limit=limit):
                _, payload = self.send_store([('current', self.built(source, limit, **changes))])
                hints = self.hints(payload['evidence_locators'][0])
                self.assertEqual(hints['status'], status)
                self.assertTrue(hints['may_be_incomplete'])
                self.assertEqual(hints['calls'], [])

    def test_dynamic_alias_and_attribute_targets_remain_unresolved(self):
        source = ('def relay_note(self, alias, name):\n    self.callback()\n'
                  '    alias()\n    getattr(self, name)()\n')
        _, payload = self.send_store([('current', self.built(source))], repair=True)
        calls = self.hints(payload['evidence_locators'][0])['calls']
        self.assertEqual({x['callee_expression'] for x in calls},
                         {'self.callback', 'alias', 'getattr(self, name)', 'getattr'})
        self.assertTrue(all(x['resolution_status']=='unresolved' for x in calls))
        self.assertTrue(all('target_path' not in x and 'target_evidence_id' not in x for x in calls))

    def test_legacy_changed_identity_and_changed_excerpt_cannot_keep_current_hints(self):
        item = self.built('def relay_note():\n    return actual_note()\n')
        original = item.to_dict()
        legacy = Evidence(**original)
        for stale in [legacy, replace(item, path='src/renamed.py'),
                      replace(item, content_hash='changed'), replace(item, code_chunk_id=99),
                      replace(item, chunk_identity='changed'), replace(item, qualified_name='other_note'),
                      replace(item, excerpt=item.excerpt[:10])]:
            with self.subTest(path=stale.path, chunk=stale.code_chunk_id):
                _, payload = self.send_store([('current', stale)], repair=True)
                self.assertEqual(self.hints(payload['evidence_locators'][0])['status'], 'unknown')
        self.assertEqual(legacy.to_dict(), original)
        # The existing authority validator refreshes both coverage and identity.
        chunk = self.database.get_code_chunks(self.project_id)[0]
        current = EvidenceBuilder().build_from_code_chunks([chunk], self.bundle['project'])[0]
        current.excerpt = self.wrapper[:self.wrapper.index('    #')]
        CitationValidator(self.database).validate(current)
        _, payload = self.send_store([('current', current)])
        self.assertNotIn('self.persist_note', json.dumps(self.hints(payload['evidence_locators'][0])))

    def test_owner_project_revision_isolation_and_summary_bounds_at_chat(self):
        item = self.built('def relay_note():\n    return actual_note()\n')
        _, payload = self.send_store([
            ('old', replace(item, chunk_identity='foreign-request')),
            ('current', replace(item, project_id='foreign', chunk_identity='foreign-project')),
            ('current', replace(item, repository_revision='old', chunk_identity='foreign-revision')),
            ('current', item)], repair=True)
        self.assertEqual(payload['evidence_count'], 1)
        self.assertEqual(payload['known_evidence_ids'], ['E4'])
        self.assertEqual(len(self.hints(payload['evidence_locators'][0])['calls']), 1)
        items=[]
        for i in range(12):
            source='def relay_note():\n'+''.join('    note_'+str(n)+'()\n' for n in range(9))
            items.append(('current', self.built(source, id=100+i, path=f'src/notes_{i}.py')))
        _, payload = self.send_store(items)
        locators=payload['evidence_locators']
        self.assertEqual(payload['evidence_count'], 12)
        self.assertTrue(payload['direct_call_hints_may_be_incomplete'])
        self.assertEqual(len(locators), 8)
        self.assertEqual(sum(len(self.hints(x)['calls']) for x in locators), 16)
        self.assertTrue(all(len(self.hints(x)['calls'])<=4 for x in locators))
        self.assertTrue(all(self.hints(x)['may_be_incomplete'] for x in locators))
        self.assertLessEqual(len(json.dumps(locators,ensure_ascii=False).encode()), 16384)
        self.assertNotIn('    note_', json.dumps(locators))
        _, bounded=self.send_store([('current',self.built('def relay_note():\n    return actual_note()\n',
            id=200+i,path='界'*490+'.py',qualified_name='界'*500)) for i in range(8)])
        self.assertLess(len(bounded['evidence_locators']),8)
        self.assertTrue(bounded['direct_call_hints_may_be_incomplete'])
        self.assertTrue(all(not self.hints(x)['may_be_incomplete'] for x in bounded['evidence_locators']))
        self.assertLessEqual(len(json.dumps(bounded['evidence_locators'],ensure_ascii=False).encode()),16384)
        oversized = self.built('def relay_note():\n    return ' + '界'*170 + '()\n')
        _, payload = self.send_store([('current', oversized)])
        hints=self.hints(payload['evidence_locators'][0])
        self.assertEqual(hints['calls'], [])
        self.assertTrue(hints['may_be_incomplete'])

    def test_tool_target_stays_held_but_byte_excluded_from_final_answer(self):
        limits=replace(self.limits,max_accumulated_evidence_context_bytes=len(self.wrapper.encode()))
        model=CapturingModel([decision('continue','lookup_symbol',{'symbol':'NotePort.persist_note'}),decision('answer')])
        result=self.run_agent(model,limits=limits)
        payload=self.payload(model,1)
        self.assertEqual(payload['evidence_count'],2)
        self.assertEqual(payload['known_evidence_ids'],['E1','E2'])
        self.assertEqual([x['final_context_selection'] for x in payload['evidence_locators']],
                         ['selected_by_byte_budget','excluded_by_byte_budget'])
        self.assertEqual(len(self.hints(payload['evidence_locators'][0])['calls']),1)
        final=coverage_tests.PlannerExcerptCoverageTests.answer_excerpts(model)
        self.assertEqual([x['symbol'] for x in final],['NotePort.send_note'])
        self.assertEqual([x['evidence_id'] for x in result['evidence']],['E1'])
        self.assertEqual({x['evidence_id'] for x in result['citations']},{'E1'})

    def test_rag_does_not_send_new_planner_hints(self):
        model=CapturingModel([])
        result=self.run_agent(model,mode='rag')
        self.assertEqual(model.planner_messages,[])
        self.assertNotIn('direct_call_hints',json.dumps(result['evidence']))


if __name__=='__main__':
    unittest.main()
