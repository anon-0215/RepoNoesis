"""Product-selected narrow source candidates, never scripted Planner reads."""
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

from app.services.agent_contracts import AgentLimits
from app.services import agent_tools
from tests.test_candidate_source_review import CandidateSourceReviewTests
from tests.test_main_method_completion import diagnostics, final_rows
from tests.test_m2_agent import decision
from tests.test_supplemental_evidence_flow import MessageModel


class CandidateAutoReadTests(unittest.TestCase):
    setUp = CandidateSourceReviewTests.setUp
    fixture = CandidateSourceReviewTests.fixture
    neutral = CandidateSourceReviewTests.neutral
    frozen = CandidateSourceReviewTests.frozen
    run_flow = CandidateSourceReviewTests.run_flow
    reads = CandidateSourceReviewTests.reads
    assert_body = CandidateSourceReviewTests.assert_body
    assert_unknown_binding = CandidateSourceReviewTests.assert_unknown_binding

    def state_case(self, *, body=None, owner=None, question=None, extra=None, base=None):
        main = ('from .storage import Cabinet\nclass Dispatch:\n'
            '    def route(self, shelf: Cabinet, key, value):\n        """' +
            ('Unrelated neutral documentation padding '*3+'\n        ')*24+'"""\n')
        main += body or ('        before = shelf.values[key]\n        saved = shelf.inspect(key)\n'
            '        shelf.record(key, value)\n        if saved is None:\n'
            '            shelf.values[key] = value\n        else:\n'
            '            shelf.record(key, saved)\n        return before\n')
        storage = owner or ('class Cabinet:\n    def inspect(self, label):\n        return self.origins.get(label)\n\n'
            '    def record(self, label, token):\n        self.origins[label] = token\n')
        self.neutral(source=main, extra={'domain/storage.py': storage, **(extra or {})},
            question=question or 'Explain Dispatch.route, including the final shelf.values[key] and saved state.',
            base=base or ['Dispatch.route','filler_0','filler_1','filler_2','filler_3'])

    def assert_selected(self, model, result, required):
        for name in required:
            self.assert_body(model, name, self.flow.chunks[name]['content'])
        self.assert_unknown_binding(model)
        self.assert_unknown_binding(model, 'final')
        self.assertEqual(len(model.planner_messages), 1)
        self.assertEqual([c['tool_name'] for s in result['agent_trace'] for c in s['tool_calls']], [])
        for name in required:
            c = self.flow.chunks[name]
            item = next(e for e in result['evidence'] if e['qualified_name']==name)
            citation = next(e for e in result['citations'] if e['qualified_name']==name)
            for field in ['path','start_line','end_line','evidence_id']:
                self.assertEqual(citation[field], item[field])
            self.assertEqual((item['start_line'],item['end_line']), (c['start_line'],c['end_line']))

    def test_frozen_n_main_is_automatic_and_main_enum_are_retained(self):
        self.frozen('N-main')
        model, events, result = self.run_flow(label='auto-n-main')
        names = ['Parameter.handle_parse_result','ParameterSource',
                 'Context.get_parameter_source','Context.set_parameter_source']
        self.assert_selected(model, result, names)
        audit = result['directed_evidence_audit']
        self.assertEqual(audit['read_count'],3)
        self.assertEqual(audit['read_lines'],150)
        self.assertEqual(len(result['evidence']),8)
        self.assertEqual(model.payload()['server_constraints']['tool_availability']['read_source']['remaining'],0)
        self.assertEqual(len(events),1)  # BASE only; Planner did not prescribe reads.

    def test_neutral_pair_auto_acquires_both_without_planner_actions(self):
        self.state_case()
        model, _, result = self.run_flow(label='auto-neutral')
        self.assert_selected(model, result, ['Dispatch.route','Cabinet.inspect','Cabinet.record'])
        self.assertEqual(result['directed_evidence_audit']['read_count'],2)
        self.assertEqual(len(result['evidence']),7)

    def test_same_key_pair_unrelated_to_asked_state_arbitration_is_not_read(self):
        self.state_case(body='        before = shelf.values[key]\n        saved = shelf.inspect(key)\n'
            '        if saved is None:\n            shelf.record(key, saved)\n'
            '        shelf.values[key] = value\n        return before\n')
        model, _, result = self.run_flow(label='auto-unrelated-same-key')
        self.assertEqual([r['symbol'] for r in self.reads(result)],['Dispatch.route'])
        self.assertNotIn('Cabinet.inspect',{r['symbol'] for r in final_rows(model)})

    def test_restore_call_before_snapshot_does_not_make_a_relevant_pair(self):
        self.state_case(body='        shelf.record(key, saved)\n        before = shelf.values[key]\n'
            '        saved = shelf.inspect(key)\n        if saved is None:\n            shelf.values[key] = value\n'
            '        return before\n')
        model, _, result = self.run_flow(label='auto-forward-restore')
        self.assertEqual([r['symbol'] for r in self.reads(result)],['Dispatch.route'])

    def test_unasked_state_key_and_non_accessor_candidates_are_not_read(self):
        variants = [
            {'question':'Explain Dispatch.route on an already established Cabinet.'},
            {'question':'Explain Dispatch.route. The unrelated display label is "shelf.values".'},
            {'body':'        before = shelf.values[key]\n        saved = shelf.inspect("another")\n'
                '        if saved is None:\n            shelf.record("another", saved)\n        return before\n'},
            {'owner':'class Cabinet:\n    def inspect(self, label):\n        log(label)\n        return self.origins.get(label)\n\n'
                '    def record(self, label, token):\n        self.origins[label] = token\n'},
            {'owner':'class Cabinet:\n    def inspect(self, label):\n        return self.origins.get(label)\n\n'
                '    def record(self, label, token):\n        self.audit[label] = token\n'},
            {'body':'        before = shelf.values[key]\n        saved = shelf.inspect(key)\n'
                '        shelf.record(key, value)\n        return before\n'},
            {'body':'        before = shelf.values[key]\n        saved = shelf.inspect(key)\n'
                '        if saved is None:\n            shelf.audit[key] = value\n            shelf.record(key, saved)\n'
                '        return before\n'},
        ]
        for i, variant in enumerate(variants):
            with self.subTest(i=i):
                self.state_case(**variant)
                model, _, result = self.run_flow(label='auto-unrelated-'+str(i))
                self.assertEqual([r['symbol'] for r in self.reads(result)],['Dispatch.route'])
                self.assertNotIn('Cabinet.inspect',{r['symbol'] for r in final_rows(model)})
                self.assert_unknown_binding(model)

    def test_reassigned_receiver_ambiguity_inheritance_and_nested_scope_fall_back(self):
        normal=('        before = shelf.values[key]\n        saved = shelf.inspect(key)\n'
            '        if saved is None:\n            shelf.record(key, saved)\n        return before\n')
        variants = [
            {'body':'        shelf = replacement\n'+normal},
            {'body':'        setattr(shelf, "inspect", replacement)\n'+normal},
            {'body':'        shelf.alias.inspect = replacement\n'+normal},
            {'body':'        key = replacement\n'+normal},
            {'body':'        def child():\n            return shelf\n'+normal},
            {'owner':'class Cabinet:\n    def inspect(self, label):\n        return self.origins.get(label)\n'
                '    def inspect(self, label):\n        return self.origins.get(label)\n'
                '    def record(self, label, token):\n        self.origins[label] = token\n'},
            {'owner':'class Base: pass\nclass Cabinet(Base):\n'
                '    def inspect(self, label):\n        return self.origins.get(label)\n'
                '    def record(self, label, token):\n        self.origins[label] = token\n'},
        ]
        for i, variant in enumerate(variants):
            with self.subTest(i=i):
                self.state_case(**variant)
                model, _, result = self.run_flow(label='auto-unsafe-'+str(i))
                self.assertEqual([r['symbol'] for r in self.reads(result)],['Dispatch.route'])
                self.assert_unknown_binding(model, 'final')

    def test_only_selected_whitespace_adjacent_methods_share_a_read(self):
        storage=('class Cabinet:\n    def inspect(self, label):\n        return self.origins.get(label)\n\n'
            '    def unrelated(self, value):\n        return value + 97\n\n'
            '    def record(self, label, token):\n        self.origins[label] = token\n')
        self.state_case(owner=storage)
        model, _, result = self.run_flow(limits=replace(AgentLimits(),max_same_tool_calls=2),label='auto-gap-definition')
        self.assert_body(model,'Dispatch.route',self.flow.chunks['Dispatch.route']['content'])
        self.assert_body(model,'Cabinet.inspect',self.flow.chunks['Cabinet.inspect']['content'])
        self.assertNotIn('Cabinet.record',{r['symbol'] for r in final_rows(model)})
        self.assertNotIn('Cabinet.unrelated',{r['symbol'] for r in final_rows(model)})
        audit=result['directed_evidence_audit']
        self.assertEqual(audit['read_count'],2)
        event=next(e for e in audit['events'] if e.get('symbol')=='Cabinet.inspect')
        c=self.flow.chunks['Cabinet.inspect']
        self.assertEqual(event['read_range'],{'path':c['path'],'start_line':c['start_line'],'end_line':c['end_line']})

    def test_multiple_pairs_have_stable_priority_and_do_not_evict_main(self):
        body=('        before = shelf.values[key]\n        saved = shelf.inspect(key)\n'
            '        other = shelf.peek(key)\n        if saved is None or other is None:\n'
            '            shelf.values[key] = value\n            shelf.record(key, saved)\n'
            '            shelf.restore(key, other)\n        return before\n')
        owner=('class Cabinet:\n    def inspect(self, label):\n        return self.origins.get(label)\n\n'
            '    def record(self, label, token):\n        self.origins[label] = token\n\n'
            '    def peek(self, label):\n        return self.labels.get(label)\n\n'
            '    def restore(self, label, token):\n        self.labels[label] = token\n')
        self.state_case(body=body,owner=owner)
        model, _, result = self.run_flow(capacity=7,label='auto-competing')
        self.assert_selected(model,result,['Dispatch.route','Cabinet.inspect','Cabinet.record'])
        self.assertEqual(result['directed_evidence_audit']['read_count'],2)
        self.assertEqual(len(result['evidence']),7)
        self.assertFalse({'Cabinet.peek','Cabinet.restore'} & {r['symbol'] for r in final_rows(model)})

    def test_held_complete_pair_and_repeated_preparation_do_not_read_or_charge(self):
        self.state_case(base=['Dispatch.route','Cabinet.inspect','Cabinet.record','filler_0','filler_1'])
        model, _, result = self.run_flow(actions=[decision('continue','validate_evidence',{'evidence_ids':['E1']})],
            label='auto-owned-complete')
        self.assertEqual([r['symbol'] for r in self.reads(result)],['Dispatch.route'])
        self.assertEqual(len(result['evidence']),5)
        for name in ['Dispatch.route','Cabinet.inspect','Cabinet.record']:
            self.assert_body(model,name,self.flow.chunks[name]['content'])
        self.assert_unknown_binding(model,'final')

    def test_partial_held_pair_completes_in_place_at_full_capacity(self):
        padding=('state method source padding '*10+'\n        ')*12
        owner=('class Cabinet:\n    def inspect(self, label):\n        """'+padding+'"""\n'
            '        return self.origins.get(label)\n\n    def record(self, label, token):\n        """'+padding+'"""\n'
            '        self.origins[label] = token\n')
        self.state_case(owner=owner,base=['Dispatch.route','Cabinet.inspect','Cabinet.record','filler_0','filler_1'])
        model, events, result = self.run_flow(capacity=5,label='auto-owned-partial')
        self.assert_selected(model,result,['Dispatch.route','Cabinet.inspect','Cabinet.record'])
        self.assertEqual(result['directed_evidence_audit']['read_count'],2)
        self.assertEqual(len(result['evidence']),5)
        for original in events[0]['after']:
            current=next(e for e in result['evidence'] if e['evidence_id']==original['evidence_id'])
            for k in ['evidence_id','content_hash','chunk_identity','start_line','end_line']:
                self.assertEqual(current[k],original[k])

    def test_shared_quota_capacity_and_depth_shortfalls_keep_primary(self):
        for kind, kwargs in [('quota',{'limits':replace(AgentLimits(),max_same_tool_calls=2)}),
                             ('capacity',{'capacity':6}),
                             ('depth',{'limits':replace(AgentLimits(),max_relation_depth=0)})]:
            with self.subTest(kind=kind):
                self.frozen('N-main')
                model, _, result = self.run_flow(label='auto-shortfall-'+kind,**kwargs)
                for name in ['Parameter.handle_parse_result','ParameterSource']:
                    self.assert_body(model,name,self.flow.chunks[name]['content'])
                self.assertFalse({'Context.get_parameter_source','Context.set_parameter_source'} &
                    {r['symbol'] for r in final_rows(model)})
                self.assertEqual(result['directed_evidence_audit']['read_count'],2)
                self.assert_unknown_binding(model,'final')

    def test_message_budget_rejection_charges_read_but_does_not_evict_or_promote(self):
        self.frozen('N-main')
        # 12432 is the frozen full main/enum + original BASE body size. Leave
        # room for them and metadata descriptors, but not both extra methods.
        model, _, result = self.run_flow(limits=replace(AgentLimits(),max_accumulated_evidence_context_bytes=13000),
            label='auto-message-budget')
        for name in ['Parameter.handle_parse_result','ParameterSource']:
            self.assert_body(model,name,self.flow.chunks[name]['content'])
        self.assertEqual(result['directed_evidence_audit']['read_count'],3)
        self.assertEqual(len(result['evidence']),6)
        self.assertNotIn('Context.get_parameter_source',{r['symbol'] for r in final_rows(model)})
        candidates=[e for e in result['directed_evidence_audit']['events'] if e.get('selection_reason')]
        self.assertEqual({e['status'] for e in candidates},{'delivery_budget_limit'})
        self.assertTrue(all(e['body_read'] and not e['committed'] for e in candidates))

    def test_read_byte_and_line_shortfalls_decline_candidate_body(self):
        for kind in ['bytes','lines']:
            padding=('oversized neutral candidate documentation '*10+'\n        ')*(30 if kind=='bytes' else 201)
            owner=('class Cabinet:\n    def inspect(self, label):\n        """'+padding+'"""\n'
                '        return self.origins.get(label)\n\n    def record(self, label, token):\n'
                '        self.origins[label] = token\n')
            self.state_case(owner=owner)
            kwargs={'limits':replace(AgentLimits(),max_source_read_bytes=8192)} if kind=='bytes' else {}
            model, _, result = self.run_flow(label='auto-read-limit-'+kind,**kwargs)
            self.assert_body(model,'Dispatch.route',self.flow.chunks['Dispatch.route']['content'])
            self.assertNotIn('Cabinet.inspect',{r['symbol'] for r in final_rows(model)})
            self.assertLessEqual(result['directed_evidence_audit']['read_count'],3)

    def test_cutoff_or_cancelled_range_cannot_commit_staged_evidence(self):
        for exc in [agent_tools.ToolDeadlineExceeded('tool_timeout','offline cutoff'),agent_tools.ToolCancelled('offline cancel')]:
            with self.subTest(exc=exc):
                self.state_case()
                with patch.object(agent_tools,'_read_source',side_effect=exc):
                    model, _, result = self.run_flow(label='auto-cutoff-'+exc.code)
                self.assert_body(model,'Dispatch.route',self.flow.chunks['Dispatch.route']['content'])
                self.assertEqual(result['directed_evidence_audit']['read_count'],2)
                self.assertEqual(len(result['evidence']),5)
                self.assertNotIn('Cabinet.inspect',{r['symbol'] for r in final_rows(model)})

    def test_actual_planner_final_and_repairs_keep_sources_and_unknown_binding(self):
        self.state_case()
        original=MessageModel.chat
        counts={'planner':0,'final':0}
        def fake(model,messages,**kwargs):
            kind='planner' if 'bounded_repository_planner' in messages[0]['content'] else 'final'
            counts[kind]+=1
            if counts[kind]==1:
                (model.planner_messages if kind=='planner' else model.final_messages).append(messages)
                return '{"invalid":true}'
            if kind=='final' and 'answer_repair' in messages[0]['content']:
                model.final_messages.append(messages)
                return json.dumps({'parts':[{'text':'Offline fixture only.','evidence_aliases':['A1']}]})
            return original(model,messages,**kwargs)
        with patch.object(MessageModel,'chat',fake):
            model, _, result = self.run_flow(label='auto-repairs')
        self.assertEqual(counts,{'planner':2,'final':2})
        for i in [0,1]:
            self.assert_unknown_binding(model,index=i)
            self.assert_unknown_binding(model,'final',i)
            for rows in [model.payload(i)['planner_evidence_view']['rows'],final_rows(model,i)]:
                for name in ['Dispatch.route','Cabinet.inspect','Cabinet.record']:
                    self.assertEqual(next(r['untrusted_source_excerpt'] for r in rows if r['symbol']==name),self.flow.chunks[name]['content'])
        self.assertTrue(all(e['validation_status']=='valid' for e in result['evidence']))

    def test_disabled_has_no_auto_read_diagnostics_or_new_contract(self):
        self.state_case()
        model, _, result = self.run_flow(enabled=False,label='auto-disabled')
        self.assertEqual(len(result['evidence']),5)
        self.assertNotIn('directed_evidence_audit',result)
        self.assertIsNone(diagnostics(model))
        self.assertNotIn('source_delivery_diagnostics',model.payload())
        self.assertNotIn('Cabinet.inspect',{r['symbol'] for r in final_rows(model)})

    def test_authority_rejection_does_not_commit_staged_bodies_or_ids(self):
        self.state_case()
        original = agent_tools._read_source
        def damaged(context, parameters):
            result = original(context, parameters)
            # Simulate canonical reread refusing one required identity. The
            # other staged definition must not leak from a partially valid pair.
            item=next(e for e in context.evidence_store.all(context.request_id) if e.qualified_name=='Cabinet.record')
            item.validation_status='invalid'
            return result
        with patch.object(agent_tools,'_read_source',side_effect=damaged):
            model, _, result = self.run_flow(label='auto-authority-rejection')
        self.assertEqual(len(result['evidence']),5)
        self.assertEqual([e['evidence_id'] for e in result['evidence']],['E1','E2','E3','E4','E5'])
        self.assertEqual(result['directed_evidence_audit']['read_count'],2)
        self.assert_body(model,'Dispatch.route',self.flow.chunks['Dispatch.route']['content'])
        self.assertFalse({'Cabinet.inspect','Cabinet.record'} & {r['qualified_name'] for r in result['citations']})

    def test_small_diagnostic_budget_keeps_source_and_does_not_claim_completeness(self):
        self.state_case()
        model, _, result = self.run_flow(limits=replace(AgentLimits(),max_observation_bytes=10000),label='auto-bounded-diagnostic')
        self.assert_selected(model,result,['Dispatch.route','Cabinet.inspect','Cabinet.record'])
        report=diagnostics(model,'planner')
        self.assertTrue(report['may_be_incomplete'])
        self.assertFalse(report['is_evidence'])
        self.assertFalse(report['runtime_binding_proven'])
        self.assertEqual(len(result['evidence']),7)

    def test_local_condition_dependencies_are_bounded_and_rebinding_declines(self):
        prefix='        before = shelf.values[key]\n        saved = shelf.inspect(key)\n'
        for kind in ['short','deep','rebound','forward']:
            if kind=='short':
                middle='        decision = saved is None\n'
            elif kind=='deep':
                middle='        flag0 = saved is None\n'+''.join('        flag'+str(i)+' = flag'+str(i-1)+'\n' for i in range(1,10))+'        decision = flag9\n'
            elif kind=='rebound':
                middle='        decision = saved is None\n        decision = unrelated\n'
            else:
                middle=''
            body=prefix+middle+'        if decision:\n            shelf.values[key] = value\n            shelf.record(key, saved)\n'
            body+=('        decision = saved is None\n' if kind=='forward' else '')+'        return before\n'
            self.state_case(body=body)
            model, _, result = self.run_flow(label='auto-condition-'+kind)
            self.assert_body(model,'Dispatch.route',self.flow.chunks['Dispatch.route']['content'])
            if kind=='short':
                self.assert_selected(model,result,['Dispatch.route','Cabinet.inspect','Cabinet.record'])
            else:
                self.assertEqual([r['symbol'] for r in self.reads(result)],['Dispatch.route'])

    def test_overload_unique_state_implementation_is_source_not_binding_proof(self):
        owner=('from typing import overload\nclass Cabinet:\n    @overload\n    def inspect(self, label: str): ...\n'
            '    def inspect(self, label):\n        return self.origins.get(label)\n\n'
            '    def record(self, label, token):\n        self.origins[label] = token\n')
        self.state_case(owner=owner)
        model, _, result = self.run_flow(label='auto-overload')
        self.assert_selected(model,result,['Dispatch.route','Cabinet.inspect','Cabinet.record'])
        self.assertNotIn('@overload',next(r['untrusted_source_excerpt'] for r in final_rows(model) if r['symbol']=='Cabinet.inspect'))


del CandidateSourceReviewTests
if __name__ == '__main__':
    unittest.main()
