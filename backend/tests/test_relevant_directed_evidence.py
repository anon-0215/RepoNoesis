"""Request relevance and actual serialized delivery, using isolated fixtures."""
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

from app.services.agent_contracts import AgentLimits
from app.services import directed_evidence as directed
from tests import test_directed_evidence_prototype as prototype
from tests.test_supplemental_evidence_flow import MessageModel
from tests.test_m2_agent import decision


def final_diagnostics(model, index=-1):
    text = model.final_messages[index][1]['content']
    if 'BOUNDED_SOURCE_DELIVERY_DIAGNOSTICS_BEGIN\n' in text:
        return json.loads(text.split('BOUNDED_SOURCE_DELIVERY_DIAGNOSTICS_BEGIN\n', 1)[1].split('\nBOUNDED_SOURCE_DELIVERY_DIAGNOSTICS_END', 1)[0])
    return json.loads(text).get('source_delivery_diagnostics') if text.startswith('{') else None


class RelevantDirectedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.proto = prototype.DirectedEvidencePrototypeTests()
        self.proto.setUp()
        self.addCleanup(self.proto.doCleanups)
        self.flow = self.proto.flow

    def fixture(self, **kwargs):
        self.proto.fixture(**kwargs)

    def run_flow(self, **kwargs):
        return self.proto.run_flow(**kwargs)

    def test_direct_import_and_necessary_indirect_target_survive(self):
        self.fixture()
        model, _, result = self.run_flow()
        self.assertEqual(result['directed_evidence_audit']['read_count'], 2)
        self.assertTrue({'entry', 'bridge', 'Terminal'} <= {r['symbol'] for r in model.final_rows()})
        for report in [model.payload()['source_delivery_diagnostics'], final_diagnostics(model)]:
            self.assertFalse(report['is_evidence'])
            self.assertTrue(report['may_be_incomplete'])
            self.assertTrue(all(r['delivery_status'] == 'body_complete' for r in report['rows'] if r.get('symbol') in {'entry', 'bridge', 'Terminal'}))

    def test_irrelevant_held_class_and_path_data_never_seed(self):
        self.fixture(bridge='def bridge():\n    return 73\n', extra={
            'pkg/noise.py': 'def custom():\n    return 1\nclass Noise:\n    """' + 'padding\n' * 205 + '"""\n    def __init__(self):\n        self.value = 42\n'})
        self.flow.base[1] = 'Noise'
        model, _, result = self.run_flow(question='Explain entry(). Assume DATA=/custom/config and label="custom".')
        symbols = {r['symbol'] for r in model.final_rows()}
        self.assertIn('Noise', symbols)  # BASE is preserved, with its partial body.
        self.assertNotIn('Noise.__init__', symbols)
        self.assertNotIn('custom', symbols)
        self.assertEqual(result['directed_evidence_audit']['read_count'], 1)

    def test_multiple_explicit_entries_and_distinct_indirect_helpers(self):
        self.fixture(bridge='def bridge():\n    return 73\n', extra={
            'pkg/other.py': 'def tail():\n    return 81\ndef second():\n    return tail()\n'})
        model, _, result = self.run_flow(question='Compare entry() and pkg.other.second().')
        self.assertTrue({'bridge', 'second', 'tail'} <= {r['symbol'] for r in model.final_rows()})
        self.assertLessEqual(result['directed_evidence_audit']['read_count'], 3)

    def test_bare_collision_is_not_bound_qualified_entry_is(self):
        self.fixture(bridge='def bridge():\n    return 73\n', extra={'unrelated/entry.py': 'def entry():\n    return 91\n'})
        model, _, result = self.run_flow(question='Explain entry().')
        self.assertEqual(result['directed_evidence_audit']['read_count'], 0)
        self.assertEqual(model.payload()['source_delivery_diagnostics']['selection_status'], 'no_reliable_entry')
        self.assertIn('not_statically_located', {r['delivery_status'] for r in final_diagnostics(model)['rows']})
        model, _, _ = self.run_flow(question='Explain pkg.api.entry.entry().')
        self.assertIn('bridge', {r['symbol'] for r in model.final_rows()})

    def test_no_reliable_entry_retains_base_and_planner(self):
        self.fixture()
        model, _, result = self.run_flow(question='Explain what this library does.')
        self.assertEqual(result['directed_evidence_audit']['read_count'], 0)
        self.assertEqual(len(result['evidence']), 5)
        self.assertEqual(len(model.planner_messages), 1)
        self.assertEqual(result['answer_mode'], 'llm_grounded')

    def test_explicit_successful_tool_target_can_seed_without_question_entry(self):
        self.fixture(bridge='from ..storage.terminal import Terminal\ndef bridge():\n    return Terminal()\n')
        model, _, result = self.run_flow(question='Explain the stored implementation.', actions=[
            decision('continue', 'lookup_symbol', {'symbol': 'bridge'})])
        self.assertEqual(result['directed_evidence_audit']['read_count'], 1)
        self.assertIn('Terminal', {r['symbol'] for r in model.final_rows()})
        self.assertEqual(model.payload(0)['source_delivery_diagnostics']['selection_status'], 'no_reliable_entry')
        self.assertEqual(model.payload(1)['source_delivery_diagnostics']['selection_status'], 'explicit_targets')

    def test_unresolved_call_delivered_as_diagnostic_not_source_or_citation(self):
        self.fixture(entry='def entry(obj):\n    return obj.run()\n')
        model, _, result = self.run_flow()
        self.assertEqual(len(result['evidence']), 5)
        self.assertEqual(len(result['citations']), 5)
        for report in [model.payload()['source_delivery_diagnostics'], final_diagnostics(model)]:
            row = next(r for r in report['rows'] if r.get('expression') == 'obj.run')
            self.assertEqual(row['delivery_status'], 'not_statically_located')
            self.assertEqual((row['source_path'], row['call_line']), ('pkg/api/entry.py', 2))
            self.assertNotIn('alias', row)

    def test_partial_body_and_read_line_budget_are_distinguished(self):
        self.fixture(bridge='def bridge():\n' + '    # long source\n' * 210 + '    return 99\n')
        self.flow.base[1] = 'bridge'
        model, _, _ = self.run_flow()
        report = final_diagnostics(model)
        self.assertTrue(any(r.get('symbol') == 'bridge' and r['delivery_status'] == 'body_partial' and r['reason'] == 'line_limit' for r in report['rows']))
        self.flow.base[1] = 'filler_0'
        model, _, _ = self.run_flow()
        self.assertTrue(any(r.get('symbol') == 'bridge' and r['delivery_status'] == 'blocked_by_budget' and r['reason'] == 'line_limit' for r in final_diagnostics(model)['rows']))

    def test_capacity_and_shared_read_budget_diagnostics(self):
        for kwargs, reason in [({'capacity': 5}, 'capacity_limit'),
                               ({'limits': replace(AgentLimits(), max_same_tool_calls=1)}, 'read_count_limit')]:
            self.fixture()
            model, _, result = self.run_flow(**kwargs)
            self.assertLessEqual(len(result['evidence']), kwargs.get('capacity', 8))
            self.assertTrue(any(r['delivery_status'] == 'blocked_by_budget' and r['reason'] == reason for r in final_diagnostics(model)['rows']))

    def test_depth_bound_reports_only_checked_next_hop(self):
        self.fixture(bridge='from ..storage.terminal import Terminal\ndef bridge():\n    return Terminal()\n')
        model, _, _ = self.run_flow(limits=replace(AgentLimits(), max_relation_depth=1))
        self.assertNotIn('Terminal', {r['symbol'] for r in model.final_rows()})
        self.assertTrue(any(r.get('symbol') == 'Terminal' and r['delivery_status'] == 'blocked_by_budget' and r['reason'] == 'depth_limit' for r in final_diagnostics(model)['rows']))

    def test_final_byte_exclusion_is_based_on_selected_bodies(self):
        self.fixture(bridge='def bridge():\n    """' + 'padding only\n' * 140 + '"""\n    return 99\n')
        self.flow.base = ['entry', 'filler_0', 'filler_1', 'filler_2', 'bridge']
        # The held long body is excluded by the final byte filter; metadata
        # uses spare bytes after the prefix and must not claim its delivery.
        model, _, _ = self.run_flow(question='Explain bridge().', limits=replace(AgentLimits(), max_accumulated_evidence_context_bytes=1700))
        self.assertNotIn('bridge', {r['symbol'] for r in model.final_rows()})
        report = final_diagnostics(model)
        self.assertTrue(any(r.get('symbol') == 'bridge' and r['delivery_status'] == 'blocked_by_budget'
            and r['reason'] == 'final_answer_body_selection' for r in report['rows']), report)

    def test_metadata_shares_message_budget_and_never_displaces_body(self):
        self.fixture(bridge='def bridge():\n    return 73\n')
        model, _, _ = self.run_flow(limits=replace(AgentLimits(), max_observation_bytes=1800, max_accumulated_evidence_context_bytes=200))
        payload = model.payload()
        size = lambda value: len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode())
        used = size(payload['untrusted_observation_summaries']) + payload['planner_evidence_view']['used_bytes']
        if 'source_delivery_diagnostics' in payload:
            used += size(payload['source_delivery_diagnostics']) + len(directed.DELIVERY_GUIDANCE.encode()) + 64
        self.assertLessEqual(used, 1800)
        self.assertIsNone(final_diagnostics(model))
        self.assertIn('entry', {r['symbol'] for r in model.final_rows()})

    def test_disabled_messages_equal_enabled_baseline_contract(self):
        self.fixture()
        model, _, result = self.run_flow(enabled=False)
        self.assertNotIn('source_delivery_diagnostics', model.payload())
        self.assertIsNone(final_diagnostics(model))
        self.assertIn('Prompt version: rn-lwb-03-v11.', model.planner_messages[0][0]['content'])
        self.assertIn('Prompt version: m1-v3.', model.final_messages[0][0]['content'])
        self.assertNotIn('directed_evidence_audit', result)

    def test_rag_enabled_diagnostics_and_default_off(self):
        self.fixture(bridge='def bridge():\n    return 73\n')
        model, _, result = self.run_flow(mode='rag', capacity=8)
        self.assertEqual(len(model.planner_messages), 0)
        self.assertIn('bridge', {r['symbol'] for r in model.final_rows()})
        self.assertIsNotNone(final_diagnostics(model))
        model, _, result = self.run_flow(mode='rag', enabled=False, capacity=8)
        self.assertNotIn('bridge', {r['symbol'] for r in model.final_rows()})
        self.assertIsNone(final_diagnostics(model))

    def test_rebound_or_conditional_entry_is_not_a_reliable_seed(self):
        for entry in ['from ..impl.bridge import bridge\ndef entry():\n    return bridge()\nentry = replacement\n',
                      'from ..impl.bridge import bridge\nif FLAG:\n    def entry():\n        return bridge()\n']:
            self.fixture(entry=entry)
            model, _, result = self.run_flow()
            self.assertEqual(result['directed_evidence_audit']['read_count'], 0)
            self.assertEqual(final_diagnostics(model)['selection_status'], 'no_reliable_entry')
            self.assertTrue(any(r['delivery_status'] == 'not_statically_located' for r in final_diagnostics(model)['rows']))

    def test_explicit_missing_tool_target_reaches_final_and_planner(self):
        self.fixture()
        model, _, _ = self.run_flow(question='Explain this library.', actions=[
            decision('continue', 'lookup_symbol', {'symbol': 'not_available_here'})])
        for report in [model.payload(1)['source_delivery_diagnostics'], final_diagnostics(model)]:
            row = next(r for r in report['rows'] if r.get('expression') == 'not_available_here')
            self.assertEqual(row['delivery_status'], 'not_statically_located')
            self.assertEqual(row['reason'], 'explicit_target_not_statically_located')

    def test_cutoff_inside_acquire_has_target_specific_budget_status(self):
        self.fixture(bridge='def bridge():\n    return 73\n')
        original = self.flow.db.get_code_chunks_for_hierarchy
        def expire(*args, **kwargs):
            from app.services.agent_tools import ToolDeadlineExceeded
            if kwargs.get('limit') == 2049:
                raise ToolDeadlineExceeded('tool_timeout', 'offline acquisition cutoff')
            return original(*args, **kwargs)
        with patch.object(self.flow.db, 'get_code_chunks_for_hierarchy', side_effect=expire):
            model, _, result = self.run_flow()
        self.assertTrue(any(r.get('symbol') == 'bridge' and r['delivery_status'] == 'blocked_by_budget'
            and r['reason'] == 'tool_timeout' for r in final_diagnostics(model)['rows']))
        self.assertEqual(result['directed_evidence_audit']['read_count'], 0)

    def test_planner_and_final_protocol_repairs_receive_same_delivery(self):
        self.fixture(entry='def entry(obj):\n    return obj.run()\n')
        original = MessageModel.chat
        counts = {'planner': 0, 'final': 0}
        def repair_model(model, messages, **kwargs):
            is_planner = 'bounded_repository_planner' in messages[0]['content']
            key = 'planner' if is_planner else 'final'
            counts[key] += 1
            if counts[key] == 1:
                (model.planner_messages if is_planner else model.final_messages).append(messages)
                return '{"invalid":true}'
            if not is_planner and 'answer_repair' in messages[0]['content']:
                model.final_messages.append(messages)
                payload = json.loads(messages[1]['content'])
                return json.dumps({'parts': [{'text': 'The call target remains unresolved.', 'evidence_aliases': [payload['evidence_aliases'][0]]}]})
            return original(model, messages, **kwargs)
        with patch.object(MessageModel, 'chat', repair_model):
            model, _, result = self.run_flow()
        self.assertEqual(counts, {'planner': 2, 'final': 2})
        self.assertEqual(model.payload(0)['source_delivery_diagnostics'], model.payload(1)['source_delivery_diagnostics'])
        self.assertEqual(final_diagnostics(model, 0), final_diagnostics(model, 1))
        self.assertEqual(result['answer_mode'], 'llm_grounded')


if __name__ == '__main__':
    unittest.main()
