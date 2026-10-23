"""Scripted message contracts; these tests do not measure model planning quality."""
from dataclasses import replace
import json
import os
import unittest
from unittest.mock import patch

from app.services.agent_contracts import AgentLimits
from app.services.code_chunker import extract_python_code_chunks
from tests import test_source_read_information_flow as source_flow
REVISION, FILLERS = source_flow.REVISION, source_flow.FILLERS


class DirectedEvidencePrototypeTests(unittest.TestCase):
    def setUp(self):
        self.flow = source_flow.SourceReadInformationFlowTests()
        self.flow.setUp()
        self.addCleanup(self.flow.doCleanups)

    def fixture(self, entry=None, bridge=None, extra=None):
        sources = {
            'pkg/api/entry.py': entry or 'from ..impl.bridge import bridge\ndef entry():\n    return bridge()\n',
            'pkg/impl/bridge.py': bridge or ('from ..storage.terminal import Terminal\ndef bridge():\n    """' + 'Neutral padding for full source coverage.\n    ' * 65 + '"""\n    return Terminal()\n'),
            'pkg/storage/terminal.py': 'class Terminal:\n    def close(self):\n        return "neutral-committed"\n',
            'pkg/fillers.py': FILLERS,
        }
        sources.update(extra or {})
        pid = self.flow.db.create_project({'repo_url': 'https://example.invalid/directed-neutral',
            'owner': 'offline', 'repo': 'neutral', 'repository_revision': REVISION, 'default_branch': 'main'})
        files = [{'path': p, 'content': s, 'language': 'Python', 'extension': '.py', 'size': len(s.encode()),
            'summary': 'fixture', 'importance': 1, 'is_core': True, 'imports': [], 'exports': [], 'symbols': []} for p, s in sources.items()]
        chunks = [c.to_dict() for p, s in sources.items() for c in extract_python_code_chunks(p, s, REVISION).chunks]
        self.flow.db.save_analysis(pid, {'overview': 'offline', 'primary_language': 'Python'}, files, [], chunks)
        self.flow.pid = pid
        self.flow.bundle = self.flow.db.get_bundle(pid)
        self.flow.chunks = {c['qualified_name']: c for c in self.flow.db.get_code_chunks(pid)}
        self.flow.base = ['entry', 'filler_0', 'filler_1', 'filler_2', 'filler_3']

    def run_flow(self, enabled=True, actions=(), **kwargs):
        kwargs.setdefault('question', 'Explain entry().')
        with patch.dict(os.environ, {'AGENT_DIRECTED_EVIDENCE_ENABLED': '1' if enabled else '0'}):
            return self.flow.run_flow(list(actions), **kwargs)

    def test_neutral_actual_planner_final_and_citation_chain(self):
        self.fixture()
        model, events, result = self.run_flow(label='prototype-neutral-chain')
        rows = model.payload(0)['planner_evidence_view']['rows']
        for name in ['entry', 'bridge', 'Terminal']:
            chunk = self.flow.chunks[name]
            self.assertEqual(next(r for r in rows if r['symbol'] == name)['untrusted_source_excerpt'], chunk['content'])
            self.assertEqual(next(r for r in model.final_rows() if r['symbol'] == name)['untrusted_source_excerpt'], chunk['content'])
            cite = next(c for c in result['citations'] if c['qualified_name'] == name)
            self.assertEqual((cite['path'], cite['start_line'], cite['end_line']), (chunk['path'], chunk['start_line'], chunk['end_line']))
        self.assertEqual(result['budget_usage']['tool_calls_used'], 0)
        self.assertEqual(len(model.planner_messages), 1)
        self.assertTrue(all(e['validation_status'] == 'valid' for e in result['evidence']))

    def test_disabled_is_original_path(self):
        self.fixture()
        model, events, result = self.run_flow(False, label='prototype-disabled')
        self.assertNotIn('bridge', {r['symbol'] for r in model.final_rows()})
        self.assertNotIn('directed_evidence_audit', result)
        self.assertEqual(len(result['evidence']), 5)

    def test_same_file_and_module_import(self):
        for entry in ['def bridge():\n    return 73\ndef entry():\n    return bridge()\n',
                      'import pkg.impl.bridge as component\ndef entry():\n    return component.bridge()\n']:
            self.fixture(entry=entry, bridge='def bridge():\n    return 73\n')
            model, _, result = self.run_flow(label='prototype-same-or-module')
            self.assertIn('return 73', json.dumps(model.final_rows()))
            self.assertTrue(any(e['status'] == 'committed' for e in result['directed_evidence_audit']['events']))

    def test_unsafe_bindings_fall_back(self):
        entries = [
            'from ..impl.bridge import bridge\ndef entry(bridge):\n    return bridge()\n',
            'from ..impl.bridge import bridge\ndef entry():\n    bridge = factory()\n    return bridge()\n',
            'from ..impl.bridge import bridge\nbridge = factory()\ndef entry():\n    return bridge()\n',
            'from ..impl.bridge import bridge\ndef entry(obj):\n    return obj.bridge()\n',
            'from ..impl.bridge import bridge\ndef entry():\n    return getattr(thing, "bridge")()\n',
            'from ..impl.bridge import bridge\nfrom other import *\ndef entry():\n    return bridge()\n',
            'from ..impl.bridge import bridge\ndef entry():\n    exec("bridge = other")\n    return bridge()\n',
            'from ..impl.bridge import bridge\ndef entry(value):\n    match value:\n        case {"x": bridge}:\n            return bridge()\n',
            'from ..impl.bridge import bridge\nexec("bridge = other")\ndef entry():\n    return bridge()\n',
            'from ..impl.bridge import bridge\nmatch value:\n    case {"x": bridge}:\n        pass\ndef entry():\n    return bridge()\n',
            'import pkg.impl.bridge as component\ndef entry():\n    component.bridge = replacement\n    return component.bridge()\n',
        ]
        for entry in entries:
            with self.subTest(entry=entry):
                self.fixture(entry=entry)
                model, _, result = self.run_flow(label='prototype-unsafe')
                self.assertNotIn('bridge', {r['symbol'] for r in model.final_rows()})
                self.assertEqual(result['directed_evidence_audit']['read_count'], 0)

    def test_duplicate_definitions_decorators_inheritance_and_monkeypatch(self):
        for bridge in [
            'def bridge():\n    return 1\ndef bridge():\n    return 2\n',
            '@wrapper\ndef bridge():\n    return 1\n',
            'class bridge(Parent):\n    pass\n',
            'def bridge():\n    return 1\nbridge.replacement = True\n',
            'def bridge():\n    return 1\nsetattr(bridge, "replacement", True)\n',
            'def bridge():\n    return 1\nfrom somewhere import *\n',
            'def bridge():\n    return 1\ndef patch():\n    global bridge\n    bridge = replacement\n',
        ]:
            with self.subTest(bridge=bridge):
                self.fixture(bridge=bridge)
                model, _, result = self.run_flow(label='prototype-ambiguous')
                self.assertFalse(any(e['committed'] for e in result['directed_evidence_audit']['events']))

    def test_import_same_name_in_unrelated_file_not_selected(self):
        self.fixture(bridge='def bridge():\n    return "correct-target"\n', extra={'other/bridge.py': 'def bridge():\n    return "wrong-target"\n'})
        model, _, _ = self.run_flow(label='prototype-same-name')
        rows = model.final_rows()
        self.assertIn('correct-target', json.dumps(rows))
        self.assertNotIn('wrong-target', json.dumps(rows))

    def test_long_held_source_completed_same_identity_once(self):
        self.fixture()
        self.flow.base = ['entry', 'bridge', 'filler_0', 'filler_1', 'filler_2']
        from tests.test_m2_agent import decision
        model, events, result = self.run_flow(actions=[decision('continue', 'lookup_symbol', {'symbol': 'missing'})], label='prototype-held-complete')
        old = next(e for e in events[0]['after'] if e['qualified_name'] == 'bridge')
        new = next(e for e in result['evidence'] if e['qualified_name'] == 'bridge')
        self.assertEqual(old['evidence_id'], new['evidence_id'])
        self.assertEqual(old['chunk_identity'], new['chunk_identity'])
        self.assertEqual(new['excerpt'], self.flow.chunks['bridge']['content'])
        audit = result['directed_evidence_audit']
        self.assertEqual(sum(e.get('symbol') == 'bridge' and e['read_count'] for e in audit['events']), 1)
        self.assertTrue(all(e.get('planner_delivery') == 'complete' for e in audit['events'] if e.get('committed')))

    def test_overlong_function_does_not_claim_complete(self):
        self.fixture(bridge='def bridge():\n' + '    # long source\n' * 210 + '    return 99\n')
        self.flow.base[1] = 'bridge'
        model, _, result = self.run_flow(label='prototype-line-limit')
        row = next(r for r in model.payload()['planner_evidence_view']['rows'] if r['symbol'] == 'bridge')
        self.assertTrue(row['excerpt_coverage']['truncated'])
        self.assertNotIn('return 99', row['untrusted_source_excerpt'])
        self.assertIn('line_limit', {e['status'] for e in result['directed_evidence_audit']['events']})

    def test_plain_large_class_reads_own_constructor_without_claiming_class_complete(self):
        self.fixture(bridge='class bridge:\n    """' + 'Padding only.\n' * 205 + '"""\n    def __init__(self):\n        self.value = 19\n')
        model, _, result = self.run_flow(label='prototype-constructor')
        self.assertIn('bridge.__init__', {r['symbol'] for r in model.final_rows()})
        self.assertNotIn('bridge', {r['symbol'] for r in model.final_rows()})

    def test_held_inherited_class_does_not_infer_constructor(self):
        self.fixture(bridge='class bridge(Parent):\n    """' + 'Padding only.\n' * 205 + '"""\n    def __init__(self):\n        self.value = 19\n')
        self.flow.base[1] = 'bridge'
        model, _, result = self.run_flow(label='prototype-inherited-held')
        self.assertNotIn('bridge.__init__', {r['symbol'] for r in model.final_rows()})

    def test_capacity_and_delivery_byte_guards(self):
        for changes, capacity, expected in [({}, 5, 'capacity_limit'),
            ({'max_accumulated_evidence_context_bytes': 150}, 8, 'delivery_budget_limit'),
            ({'max_source_read_bytes': 900}, 8, 'read_budget_limit')]:
            self.fixture()
            model, _, result = self.run_flow(capacity=capacity, limits=replace(AgentLimits(), **changes), label='prototype-budgets-' + expected)
            self.assertLessEqual(len(result['evidence']), capacity)
            self.assertIn(expected, {e['status'] for e in result['directed_evidence_audit']['events']})
            self.assertNotIn('neutral-committed', json.dumps(model.final_rows()))

    def test_canonical_corruption_never_reaches_planner_or_final(self):
        self.fixture()
        original = self.flow.db.get_evidence_source
        def corrupt(project, chunk, path):
            value = original(project, chunk, path)
            if path == 'pkg/impl/bridge.py':
                value = dict(value)
                value['file_content'] = value['file_content'].replace('return Terminal()', 'return Forged()')
            return value
        with patch.object(self.flow.db, 'get_evidence_source', side_effect=corrupt):
            model, _, result = self.run_flow(label='prototype-authority-failure')
        self.assertNotIn('bridge', {r['symbol'] for r in model.final_rows()})
        self.assertNotIn('bridge', {r['symbol'] for r in model.payload()['planner_evidence_view']['rows']})
        self.assertEqual(result['directed_evidence_audit']['inspection_rejections']['pkg/impl/bridge.py'], 'binding_snapshot_mismatch')

    def test_held_method_identity_and_utf8_crlf_completion(self):
        source = 'class Worker:\r\n    def finish(self):\r\n        """' + '中文覆盖核对。\r\n        ' * 120 + '"""\r\n        return 41\r\n'
        self.fixture(extra={'pkg/worker.py': source})
        self.flow.base[1] = 'Worker.finish'
        model, events, result = self.run_flow(question='Explain Worker.finish().', label='prototype-method-crlf')
        row = next(r for r in model.payload()['planner_evidence_view']['rows'] if r['symbol'] == 'Worker.finish')
        self.assertEqual(row['untrusted_source_excerpt'], self.flow.chunks['Worker.finish']['content'])
        self.assertFalse(row['excerpt_coverage']['truncated'])
        audit = result['directed_evidence_audit']
        self.assertEqual(sum(e['read_count'] for e in audit['events'] if e.get('symbol') == 'Worker.finish'), 1)

    def test_read_count_depth_and_deadline_bounds(self):
        self.fixture(entry='from ..impl.bridge import bridge\ndef entry():\n    return bridge()\n',
            bridge='from ..storage.terminal import Terminal\ndef bridge():\n    return Terminal()\n')
        model, _, result = self.run_flow(limits=replace(AgentLimits(), max_relation_depth=1), label='prototype-depth')
        self.assertIn('bridge', {r['symbol'] for r in model.final_rows()})
        self.assertNotIn('Terminal', {r['symbol'] for r in model.final_rows()})
        model, _, result = self.run_flow(limits=replace(AgentLimits(), max_same_tool_calls=1), label='prototype-read-count')
        self.assertEqual(result['directed_evidence_audit']['read_count'], 1)
        self.assertNotIn('Terminal', {r['symbol'] for r in model.final_rows()})
        from app.services import directed_evidence
        original = directed_evidence.StaticLocator.module
        def expire(locator, path):
            value = original(locator, path)
            from app.services.agent_tools import ToolDeadlineExceeded
            raise ToolDeadlineExceeded('tool_timeout', 'test cutoff')
        with patch.object(directed_evidence.StaticLocator, 'module', expire):
            model, _, result = self.run_flow(label='prototype-deadline')
        self.assertEqual(result['directed_evidence_audit']['read_count'], 0)
        self.assertIn('tool_timeout', {e['status'] for e in result['directed_evidence_audit']['events']})

    def test_automatic_reads_share_existing_tool_quota(self):
        self.fixture()
        from tests.test_m2_agent import decision
        model, _, result = self.run_flow(actions=[decision('continue', 'lookup_symbol', {'symbol': 'missing'})], label='prototype-shared-quota')
        availability = model.payload(0)['server_constraints']['tool_availability']
        self.assertEqual(availability['read_source']['remaining'], 1)
        self.assertEqual(model.payload(0)['server_constraints']['remaining_budget']['tool_calls'], 6)

    def test_actual_preparation_cutoff_prevents_commit(self):
        self.fixture()
        from app.services import directed_evidence
        import time
        start = time.monotonic()
        ticks = iter([start, start, start + 100, start + 100])
        from types import SimpleNamespace
        with patch.object(directed_evidence, 'time', SimpleNamespace(monotonic=lambda: next(ticks, start + 100))):
            model, _, result = self.run_flow(label='prototype-clock-cutoff')
        self.assertEqual(result['directed_evidence_audit']['read_count'], 0)
        self.assertNotIn('bridge', {r['symbol'] for r in model.final_rows()})
        self.assertIn('tool_timeout', {e['status'] for e in result['directed_evidence_audit']['events']})


if __name__ == '__main__':
    unittest.main()
