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
from app.services.citation_protocol import build_canonical_citation_descriptors
from app.services.code_chunker import extract_python_code_chunks
from app.services.evidence import CitationValidator, Evidence, EvidenceBuilder
from app.services.qa_agent import _answer_with_grounded_llm
from tests.m1_helpers import REVISION, disabled_embedding_service, make_chunk
from tests.test_m2_agent import decision


class CapturingModel:
    """Script protocol decisions only; assertions concern inputs, not model quality."""
    available = True

    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.planner_messages = []
        self.answer_messages = []

    def require_available(self):
        pass

    def chat(self, messages, **kwargs):
        if 'bounded_repository_planner' in messages[0]['content']:
            self.planner_messages.append(messages)
            return json.dumps(next(self.decisions))
        self.answer_messages.append(messages)
        return json.dumps({'parts': [{'text': 'Fixture projection only', 'evidence_aliases': ['A1']}]})


class PlannerExcerptCoverageTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.db = Database(Path(directory.name) / 'coverage.sqlite')
        self.project = {'repo_url': 'https://example.invalid/archive-fixture',
                        'owner': 'fixture', 'repo': 'archive', 'default_branch': 'main',
                        'repository_revision': REVISION}
        self.project_id = self.db.create_project(self.project)
        self.source = ('class ArchiveShelf:\n    """ArchiveShelf overview.\n' +
                       ('    Padding for the class overview, with no method implementation.\n' * 80) +
                       '    """\n\n    def release_stock(self):\n        return 3\n')
        chunks = extract_python_code_chunks('src/archive_shelf.py', self.source, REVISION)
        self.assertFalse(chunks.warnings)
        files = [{'path': 'src/archive_shelf.py', 'content': self.source, 'language': 'Python',
                  'extension': '.py', 'summary': 'ArchiveShelf overview', 'importance': 10,
                  'is_core': True, 'imports': [], 'exports': [], 'symbols': ['ArchiveShelf']}]
        self.db.save_analysis(self.project_id, {'files': files, 'modules': []}, files, [],
                              [x.to_dict() for x in chunks.chunks])
        self.bundle = self.db.get_bundle(self.project_id)
        self.limits = replace(AgentLimits(), base_evidence_count=1)

    @staticmethod
    def payload(model, index=0):
        return json.loads(model.planner_messages[index][1]['content'])

    @staticmethod
    def answer_excerpts(model):
        text = model.answer_messages[-1][1]['content']
        content = text.split('VALIDATED_UNTRUSTED_EVIDENCE_BEGIN\n', 1)[1].split(
            '\nVALIDATED_UNTRUSTED_EVIDENCE_END', 1)[0]
        return json.loads(content)

    def run_agent(self, model, mode='agent'):
        return run_bounded_agent(
            'ArchiveShelf overview', self.bundle, model, self.db, disabled_embedding_service(),
            planner=LLMPlanner(model, build_m2_tool_registry(self.limits), self.limits),
            limits=self.limits, evidence_count=8, execution_mode=mode,
            request_id='coverage-request',
        )

    def send_evidence(self, evidence, *, repair=False):
        store = EvidenceStore(capacity=8)
        store.add('current', [evidence])
        state = SimpleNamespace(
            steps=[], remaining_budget=lambda: {'work_time_ms': 10000, 'tool_calls': 3},
            budget=SimpleNamespace(work_cutoff_at=None), request_id='current', user_goal='fixture',
            context=SimpleNamespace(evidence_store=store, project_id=evidence.project_id,
                                    repository_revision=evidence.repository_revision),
        )
        model = CapturingModel([decision('answer')])
        LLMPlanner(model, build_m2_tool_registry(self.limits), self.limits).decide(
            _planner_state(state), repair_hint={'stable_code': 'invalid_json'} if repair else None)
        _answer_with_grounded_llm('fixture', build_canonical_citation_descriptors([evidence]), model)
        return model, self.payload(model)['evidence_locators'][0]['excerpt_coverage']

    def built(self, source, limit, start=20, end=None):
        chunk = make_chunk('src/tickets.py', 'issue_ticket', source, start_line=start)
        chunk.update(id=1, project_id=self.project_id)
        if end is not None:
            chunk['end_line'] = end
        return EvidenceBuilder(limit).build_from_code_chunks([chunk], self.project)[0]

    def test_initial_long_class_has_full_locator_but_only_prefix_coverage(self):
        model = CapturingModel([decision('answer')])
        result = self.run_agent(model)
        locator = self.payload(model)['evidence_locators'][0]
        self.assertEqual(locator['qualified_name'], 'ArchiveShelf')
        coverage = locator['excerpt_coverage']
        tail = next(c for c in self.db.get_code_chunks(self.project_id)
                    if c['qualified_name'] == 'ArchiveShelf.release_stock')
        self.assertLess(coverage['complete_end_line'], tail['start_line'])
        self.assertLess(coverage['partial_end_line'], tail['start_line'])
        self.assertGreaterEqual(locator['end_line'], tail['end_line'])
        self.assertTrue(coverage['truncated'])
        excerpt = self.answer_excerpts(model)[0]['untrusted_source_excerpt']
        self.assertEqual(excerpt, result['evidence'][0]['excerpt'])
        self.assertEqual(excerpt, self.source[:2000])
        self.assertNotIn('release_stock', excerpt)
        self.assertIn('Padding for', model.planner_messages[0][1]['content'])
        self.assertNotIn('_excerpt_coverage', result['evidence'][0])
        self.assertIn('rn-lwb-03-v11', model.planner_messages[0][0]['content'])

    def test_lookup_then_actual_request_and_answer_include_independent_complete_method(self):
        model = CapturingModel([
            decision('continue', 'lookup_symbol', {'symbol': 'ArchiveShelf.release_stock'}),
            decision('answer'),
        ])
        result = self.run_agent(model)
        first, second = self.payload(model), self.payload(model, 1)
        self.assertEqual(first['evidence_count'], 1)
        self.assertEqual(second['evidence_count'], 2)
        self.assertEqual(second['evidence_locators'][0], first['evidence_locators'][0])
        method = second['evidence_locators'][1]
        self.assertEqual(method['qualified_name'], 'ArchiveShelf.release_stock')
        self.assertEqual(method['excerpt_coverage'], {
            'status': 'known', 'start_line': method['start_line'],
            'complete_end_line': method['end_line'], 'partial_end_line': None, 'truncated': False})
        final = self.answer_excerpts(model)
        self.assertEqual(final[1]['untrusted_source_excerpt'], result['evidence'][1]['excerpt'])
        self.assertIn('def release_stock', final[1]['untrusted_source_excerpt'])

    def test_read_source_promotes_the_same_method_through_real_canonical_path(self):
        method = next(c for c in self.db.get_code_chunks(self.project_id)
                      if c['qualified_name'] == 'ArchiveShelf.release_stock')
        model = CapturingModel([
            decision('continue', 'read_source', {'path': method['path'],
                'start_line': method['start_line'], 'end_line': method['end_line']}),
            decision('answer'),
        ])
        self.run_agent(model)
        locators = self.payload(model, 1)['evidence_locators']
        actual = next(x for x in locators if x['qualified_name'] == 'ArchiveShelf.release_stock')
        self.assertFalse(actual['excerpt_coverage']['truncated'])
        self.assertEqual(actual['excerpt_coverage']['complete_end_line'], method['end_line'])

    def test_every_cut_in_crlf_source_is_conservative_at_actual_message_boundary(self):
        source = 'def issue_ticket():\r\n    return 3\r\n'
        line_boundary = source.index('\n') + 1
        for cut in range(1, len(source) + 1):
            with self.subTest(cut=cut):
                item = self.built(source, cut)
                model, coverage = self.send_evidence(item)
                if cut < line_boundary:
                    end, partial = None, 20
                elif cut == line_boundary:
                    end, partial = 20, None
                elif cut < len(source):
                    end, partial = 20, 21
                else:
                    end, partial = 21, None
                self.assertEqual(coverage, {'status': 'known', 'start_line': 20,
                    'complete_end_line': end, 'partial_end_line': partial, 'truncated': cut < len(source)})
                self.assertEqual(self.answer_excerpts(model)[0]['untrusted_source_excerpt'], source[:cut])

    def test_lf_boundary_partial_first_line_and_final_line_without_newline(self):
        source = 'def issue_ticket():\n    return 3'
        boundary = source.index('\n') + 1
        for cut,end,partial,truncated in [(1,None,20,True), (boundary-1,None,20,True),
                                          (boundary,20,None,True), (len(source),21,None,False)]:
            with self.subTest(cut=cut):
                _, coverage = self.send_evidence(self.built(source, cut))
                self.assertEqual((coverage['complete_end_line'],coverage['partial_end_line'],
                                  coverage['truncated']), (end,partial,truncated))

    def test_old_projection_and_inconsistent_ranges_do_not_assume_full_coverage(self):
        item = self.built('def issue_ticket():\n    return 3\n', 2000)
        public = item.to_dict()
        legacy = Evidence(**public)
        _, coverage = self.send_evidence(legacy, repair=True)
        self.assertEqual(coverage, {'status': 'unknown'})
        self.assertEqual(legacy.to_dict(), public)
        _, coverage = self.send_evidence(self.built('def issue_ticket():\n    return 3\n', 2000, end=999))
        self.assertEqual(coverage, {'status': 'unknown'})

    def test_later_excerpt_change_cannot_keep_a_larger_claim_and_validation_refreshes_it(self):
        method = next(c for c in self.db.get_code_chunks(self.project_id)
                      if c['qualified_name'] == 'ArchiveShelf.release_stock')
        item = EvidenceBuilder().build_from_code_chunks([method], self.project)[0]
        item.excerpt = item.excerpt[:7]
        _, coverage = self.send_evidence(item)
        self.assertEqual(coverage, {'status': 'unknown'})
        CitationValidator(self.db).validate(item)
        self.assertEqual(item.validation_status, 'valid')
        model, coverage = self.send_evidence(item)
        self.assertEqual(coverage['complete_end_line'], None)
        self.assertEqual(coverage['partial_end_line'], item.start_line)
        self.assertTrue(coverage['truncated'])
        self.assertEqual(self.answer_excerpts(model)[0]['untrusted_source_excerpt'], item.excerpt)
        item.excerpt = 'not a source prefix'
        CitationValidator(self.db).validate(item)
        self.assertEqual(item.planner_excerpt_coverage(), {'status': 'unknown'})

    def test_rag_still_skips_planner_and_uses_the_same_excerpt(self):
        model = CapturingModel([])
        result = self.run_agent(model, 'rag')
        self.assertEqual(model.planner_messages, [])
        self.assertEqual(self.answer_excerpts(model)[0]['untrusted_source_excerpt'], result['evidence'][0]['excerpt'])

    def test_known_coverage_keeps_request_isolation_and_summary_bounds(self):
        item = self.built('def issue_ticket():\n    return 3\n', 2000)

        def capture(items):
            store = EvidenceStore(capacity=30)
            for owner, value in items:
                store.add(owner, [value])
            state = SimpleNamespace(
                steps=[], remaining_budget=lambda: {'work_time_ms': 10000, 'tool_calls': 3},
                budget=SimpleNamespace(work_cutoff_at=None), request_id='current', user_goal='fixture',
                context=SimpleNamespace(evidence_store=store, project_id=item.project_id,
                                        repository_revision=item.repository_revision))
            model = CapturingModel([decision('answer')])
            LLMPlanner(model, build_m2_tool_registry(self.limits), self.limits).decide(_planner_state(state))
            return self.payload(model)

        payload = capture([
            ('old', replace(item, chunk_identity='old', qualified_name='Other.request')),
            ('current', replace(item, chunk_identity='project', project_id='other', qualified_name='Other.project')),
            ('current', replace(item, chunk_identity='revision', repository_revision='old', qualified_name='Other.revision')),
            ('current', item),
        ])
        self.assertEqual(payload['evidence_count'], 1)
        self.assertEqual(payload['evidence_locators'][0]['qualified_name'], 'issue_ticket')
        self.assertEqual(payload['evidence_locators'][0]['excerpt_coverage']['status'], 'known')
        items = [('current', replace(item, chunk_identity=str(i), path='界' * 490 + '.py',
                                     qualified_name='界' * 500)) for i in range(12)]
        locators = capture(items)['evidence_locators']
        self.assertLess(len(locators), 8)
        self.assertTrue(all(x['excerpt_coverage']['status'] == 'known' for x in locators))
        self.assertLessEqual(len(json.dumps(locators, ensure_ascii=False).encode('utf-8')), 16384)


if __name__ == '__main__':
    unittest.main()
