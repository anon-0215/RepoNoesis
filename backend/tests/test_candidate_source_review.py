"""Candidate source reachability is independent of runtime dispatch proof."""
from dataclasses import replace
from pathlib import Path
import json,unittest
from unittest.mock import patch

from app.services.agent_contracts import AgentLimits
from app.services import directed_evidence
from tests.test_auxiliary_evidence_selection import AuxiliaryEvidenceSelectionTests
from tests.test_main_method_completion import final_rows,diagnostics
from tests.test_m2_agent import decision
from tests.test_supplemental_evidence_flow import MessageModel


class CandidateSourceReviewTests(unittest.TestCase):
    setUp=AuxiliaryEvidenceSelectionTests.setUp
    fixture=AuxiliaryEvidenceSelectionTests.fixture
    neutral=AuxiliaryEvidenceSelectionTests.neutral
    run_flow=AuxiliaryEvidenceSelectionTests.run_flow
    reads=AuxiliaryEvidenceSelectionTests.reads
    assert_body=AuxiliaryEvidenceSelectionTests.assert_body
    frozen=AuxiliaryEvidenceSelectionTests.frozen

    def joint_read(self,*symbols):
        chunks=[self.flow.chunks[s] for s in symbols]
        self.assertEqual(len({(c['path'],c['repository_revision']) for c in chunks}),1)
        return decision('continue','read_source',{'path':chunks[0]['path'],
            'start_line':min(c['start_line'] for c in chunks),'end_line':max(c['end_line'] for c in chunks)})

    def candidates(self,model,index=0):
        report=diagnostics(model,'planner',index)
        return [c for r in report['rows'] for c in r.get('candidate_definitions',[])] if report else []

    def typed(self,*,owner=None,annotation='Mailbox',extra=None,body=None):
        owner=owner or ('class Mailbox:\n    def take(self, name):\n        return self.values.get(name)\n\n'
            '    def put(self, name, value):\n        self.values[name] = value\n')
        source='from .storage import Mailbox\ndef noisy_branch():\n    return 909\nclass Dispatch:\n'
        source+='    def route(self, box: '+annotation+', value):\n        """'+('Different neutral source padding '*5+'\n        ')*22+'"""\n'
        source+=body or '        current = box.take("topic")\n        box.put("topic", value)\n        if value is None:\n            noisy_branch()\n        return current\n'
        self.neutral(source=source,extra={'domain/storage.py':owner,**(extra or {})},
            question='Explain Dispatch.route and the source needed for its state changes.',
            base=['Dispatch.route','filler_0','filler_1','filler_2','filler_3'])

    def assert_unknown_binding(self,model,stage='planner',index=0):
        report=diagnostics(model,stage,index)
        self.assertFalse(report['runtime_binding_proven'])
        hints=[r for r in report['rows'] if r.get('candidate_definitions')]
        self.assertTrue(hints)
        for row in hints:
            self.assertEqual(row['binding_status'],'unconfirmed')
            for c in row['candidate_definitions']:
                self.assertEqual(c['discovery_basis'],'parameter_annotation')
                self.assertFalse(c['runtime_binding_proven'])

    def test_existing_symbol_and_combined_read_are_reachable_when_off(self):
        for mode in ['lookup','range']:
            with self.subTest(mode=mode):
                self.frozen('N-main')
                symbols=['Context.get_parameter_source','Context.set_parameter_source']
                actions=([decision('continue','lookup_symbol',{'symbol':s,'match_mode':'exact','top_k':2}) for s in symbols]
                    if mode=='lookup' else [self.joint_read(*symbols)])
                model,events,result=self.run_flow(enabled=False,actions=actions,label='reachable-off-'+mode)
                for s in symbols:self.assert_body(model,s,self.flow.chunks[s]['content'])
                self.assertEqual([e['status'] for e in events],['succeeded']*(len(actions)+1))
                if mode=='range':
                    read=events[1]
                    self.assertEqual(read['parameters']['end_line']-read['parameters']['start_line']+1,26)
                    self.assertEqual(read['metrics']['new_evidence_count'],2)
                    self.assertEqual(set(a['qualified_name'] for a in read['results']['associated_evidence'].values()),set(symbols))
                    for s in symbols:self.assertIn(self.flow.chunks[s]['content'].strip(),read['results']['content'])
                self.assertIsNone(diagnostics(model))

    def test_frozen_n_main_has_candidate_feedback_and_bounded_automatic_bodies(self):
        self.frozen('N-main')
        model,_,result=self.run_flow(label='frozen-no-choice')
        self.assertEqual([r['symbol'] for r in self.reads(result)][:2],['Parameter.handle_parse_result','ParameterSource'])
        self.assertEqual(result['directed_evidence_audit']['read_count'],3)
        self.assertEqual(model.payload(0)['server_constraints']['tool_availability']['read_source']['remaining'],0)
        self.assertEqual(model.payload(0)['server_constraints']['evidence_admission']['remaining_capacity'],0)
        defs={c['symbol']:c for c in self.candidates(model)}
        for s in ['Context.get_parameter_source','Context.set_parameter_source']:
            self.assertIn(s,defs)
            c=self.flow.chunks[s]
            self.assertEqual((defs[s]['path'],defs[s]['start_line'],defs[s]['end_line'],defs[s]['content_hash']),
                (c['path'],c['start_line'],c['end_line'],c['content_hash']))
            self.assert_body(model,s,self.flow.chunks[s]['content'])
        self.assert_unknown_binding(model)

    def test_frozen_n_main_combined_existing_read_delivers_both_within_original_limits(self):
        self.frozen('N-main');symbols=['Context.get_parameter_source','Context.set_parameter_source']
        model,events,result=self.run_flow(actions=[self.joint_read(*symbols)],label='frozen-choice-range')
        # The automatic range has already used the last legal read. An explicit
        # attempted repetition must be rejected rather than expanding the quota.
        self.assertEqual([e['status'] for e in events],['succeeded'])
        self.assertEqual(result['agent_trace'][0]['tool_calls'][0]['status'],'rejected')
        for s in ['Parameter.handle_parse_result','ParameterSource',*symbols]:
            self.assert_body(model,s,self.flow.chunks[s]['content'])
        self.assertEqual(len(result['evidence']),8)
        self.assertEqual(result['directed_evidence_audit']['read_count'],3)
        self.assertEqual(model.payload()['server_constraints']['tool_availability']['read_source']['remaining'],0)
        self.assert_unknown_binding(model,index=-1)
        self.assert_unknown_binding(model,'final',index=-1)
        self.assertNotIn('_format_deprecated_suffix',{r['symbol'] for r in final_rows(model)})
        for s in symbols:
            evidence=next(e for e in result['evidence'] if e['qualified_name']==s)
            citation=next(c for c in result['citations'] if c['qualified_name']==s)
            for k in ['path','start_line','end_line','evidence_id']:self.assertEqual(citation[k],evidence[k])
            self.assertEqual(evidence['validation_status'],'valid')

    def test_neutral_annotation_definitions_are_candidates_and_not_auto_evidence(self):
        self.typed(extra={'unrelated.py':'class Random:\n    def unknown(self):\n        return 77\n'})
        model,_,result=self.run_flow(label='neutral-candidates-only')
        self.assertEqual([r['symbol'] for r in self.reads(result)],['Dispatch.route'])
        self.assertEqual({c['symbol'] for c in self.candidates(model)},{'Mailbox.take','Mailbox.put'})
        self.assertEqual({e['qualified_name'] for e in result['evidence']},
            {self.flow.chunks[s]['qualified_name'] for s in self.flow.base})
        self.assertNotIn('noisy_branch',{r['symbol'] for r in final_rows(model)})
        self.assert_unknown_binding(model)
        model,_,result=self.run_flow(actions=[self.joint_read('Mailbox.take','Mailbox.put')],label='neutral-candidates-read')
        for s in ['Dispatch.route','Mailbox.take','Mailbox.put']:self.assert_body(model,s,self.flow.chunks[s]['content'])
        self.assert_unknown_binding(model,'final',-1)

    def test_same_name_and_rebound_receiver_do_not_prove_runtime_target(self):
        owner=('class Mailbox:\n    def take(self, name): return 1\n    def take(self, name): return 2\n'
            '    def put(self, name, value): return 3\nclass Override(Mailbox):\n    def take(self, name): return 4\n')
        self.typed(owner=owner,body='        box = external_object\n        return box.take("topic")\n')
        model,_,result=self.run_flow(label='multiple-candidate-declarations')
        hints=[c for c in self.candidates(model) if c['symbol']=='Mailbox.take']
        self.assertEqual(len(hints),2)
        self.assertEqual(len({(c['start_line'],c['end_line']) for c in hints}),2)
        self.assert_unknown_binding(model)
        self.assertEqual([r['symbol'] for r in self.reads(result)],['Dispatch.route'])
        self.assertNotIn('Override.take',{c['symbol'] for c in self.candidates(model)})
        # A manually chosen declaration can pass source/citation validation,
        # while the duplicated definition and rebound receiver stay uncertain.
        chunk=min((c for c in self.flow.db.get_code_chunks(self.flow.pid) if c['qualified_name']=='Mailbox.take'),
            key=lambda c:c['start_line'])
        action=decision('continue','read_source',{'path':chunk['path'],'start_line':chunk['start_line'],'end_line':chunk['end_line']})
        model,_,result=self.run_flow(actions=[action],label='ambiguous-but-valid-source')
        row=next(r for r in final_rows(model) if r['symbol']=='Mailbox.take')
        self.assertEqual(row['untrusted_source_excerpt'],chunk['content'])
        citation=next(c for c in result['citations'] if c['qualified_name']=='Mailbox.take')
        self.assertEqual((citation['start_line'],citation['end_line']),(chunk['start_line'],chunk['end_line']))
        self.assert_unknown_binding(model,'final',-1)

    def test_many_declarations_are_bounded_and_mark_omission(self):
        self.typed(owner='class Mailbox:\n'+''.join('    def take(self, name): return '+str(i)+'\n' for i in range(4)),
            body='        return box.take("topic")\n')
        model,_,result=self.run_flow(label='candidate-definition-cap')
        self.assertEqual(len(self.candidates(model)),2)
        self.assertTrue(diagnostics(model,'planner')['records_omitted'])
        row=next(r for r in diagnostics(model,'planner')['rows'] if r.get('candidate_definitions'))
        self.assertTrue(row['candidate_definitions_omitted'])
        self.assertEqual([r['symbol'] for r in self.reads(result)],['Dispatch.route'])

    def test_full_capacity_read_does_not_lend_citation_or_final_delivery(self):
        self.frozen('N-main');symbols=['Context.get_parameter_source','Context.set_parameter_source']
        model,events,result=self.run_flow(capacity=6,actions=[self.joint_read(*symbols)],label='candidate-capacity-limit')
        self.assertEqual(events[-1]['status'],'succeeded')
        self.assertEqual(events[-1]['metrics']['new_evidence_count'],0)
        self.assertEqual(len(result['evidence']),6)
        for name in ['Parameter.handle_parse_result','ParameterSource']:
            self.assert_body(model,name,self.flow.chunks[name]['content'])
        for name in symbols:
            self.assertNotIn(name,{r['symbol'] for r in final_rows(model)})
            self.assertNotIn(name,{r['qualified_name'] for r in result['citations']})
            row=next(r for r in diagnostics(model)['rows'] if r.get('symbol')==name)
            self.assertEqual(row['held_body_status'],'not_held')
            self.assertNotEqual(row['delivery_status'],'body_complete')
        self.assert_unknown_binding(model,'final',-1)

    def test_inherited_member_is_not_guessed_and_unknown_annotation_falls_back(self):
        for annotation,owner in [('Mailbox','class Base:\n    def take(self, name): return 1\nclass Mailbox(Base):\n    pass\n'),
                                 ('Unknown','class Mailbox:\n    def take(self, name): return 1\n')]:
            with self.subTest(annotation=annotation):
                self.typed(owner=owner,annotation=annotation,body='        return box.take("topic")\n')
                model,_,result=self.run_flow(label='inherited-or-unknown-'+annotation)
                self.assertEqual(self.candidates(model),[])
                self.assertNotIn('Base.take',{r['symbol'] for r in final_rows(model)})

    def test_budget_shortfall_keeps_main_and_explicit_enum_without_fake_completion(self):
        self.frozen('N-main')
        model,events,result=self.run_flow(limits=replace(AgentLimits(),max_same_tool_calls=2),label='frozen-quota-shortfall')
        for s in ['Parameter.handle_parse_result','ParameterSource']:self.assert_body(model,s,self.flow.chunks[s]['content'])
        self.assertEqual(model.payload()['server_constraints']['tool_availability']['read_source']['remaining'],0)
        self.assertEqual(len(result['evidence']),6)
        self.assert_unknown_binding(model)
        self.assertNotIn('Context.get_parameter_source',{r['symbol'] for r in final_rows(model)})

    def test_normal_and_repair_requests_keep_candidate_uncertainty_and_acquired_bodies(self):
        self.typed();chat=MessageModel.chat;calls={'planner':0,'final':0}
        def fake(model,messages,**kwargs):
            kind='planner' if 'bounded_repository_planner' in messages[0]['content'] else 'final'
            calls[kind]+=1
            if calls[kind]==1:
                (model.planner_messages if kind=='planner' else model.final_messages).append(messages)
                return '{"invalid":true}'
            if kind=='final' and 'answer_repair' in messages[0]['content']:
                model.final_messages.append(messages)
                return json.dumps({'parts':[{'text':'Scripted candidate source.', 'evidence_aliases':['A1']}]})
            return chat(model,messages,**kwargs)
        with patch.object(MessageModel,'chat',fake):
            model,_,result=self.run_flow(actions=[self.joint_read('Mailbox.take','Mailbox.put')],label='candidate-repairs')
        self.assertEqual(calls,{'planner':3,'final':2})
        for i in [0,1]:self.assert_unknown_binding(model,index=i)
        for i in [0,1]:
            self.assert_unknown_binding(model,'final',i)
            for name in ['Dispatch.route','Mailbox.take','Mailbox.put']:
                self.assertEqual(next(r['untrusted_source_excerpt'] for r in final_rows(model,i) if r['symbol']==name),self.flow.chunks[name]['content'])

    def test_small_final_budget_retains_original_diagnostic_row(self):
        self.neutral(source='class Processor:\n    def dispatch(self):\n        """'+'full source padding '*85+'"""\n        return 417\n',
            base=['filler_0','filler_1','filler_2','filler_3','Processor.dispatch'])
        model,_,result=self.run_flow(limits=replace(AgentLimits(),max_accumulated_evidence_context_bytes=1700),label='small-guidance-budget')
        self.assertLessEqual(len(directed_evidence.DELIVERY_GUIDANCE.encode()),644)
        row=next(r for r in diagnostics(model)['rows'] if r.get('symbol')=='Processor.dispatch')
        self.assertEqual(row['held_body_status'],'body_complete')
        self.assertEqual(row['delivery_status'],'blocked_by_budget')
        self.assertNotIn('Processor.dispatch',{r['symbol'] for r in final_rows(model)})

    def test_deferred_expansion_resumes_after_lookup_without_repeated_read(self):
        self.typed(body='        value = noisy_branch()\n        return box.take("topic")\n')
        lookup=decision('continue','lookup_symbol',{'symbol':'Mailbox.take','match_mode':'exact','top_k':2})
        model,events,result=self.run_flow(actions=[lookup,lookup],label='deferred-expansion-resumes')
        first={r['symbol'] for r in model.payload(0)['planner_evidence_view']['rows']}
        self.assertNotIn('noisy_branch',first)
        self.assertIn('noisy_branch',{r['symbol'] for r in model.payload()['planner_evidence_view']['rows']})
        for name in ['Dispatch.route','Mailbox.take','noisy_branch']:
            self.assert_body(model,name,self.flow.chunks[name]['content'])
        reads=[r['symbol'] for r in self.reads(result)]
        self.assertEqual(reads,['Dispatch.route','noisy_branch'])
        self.assertEqual(sum(e['qualified_name']=='noisy_branch' for e in result['evidence']),1)
        self.assertEqual(len(result['evidence']),7)
        self.assertEqual(model.payload()['server_constraints']['tool_availability']['read_source']['remaining'],1)
        self.assert_unknown_binding(model,'final',-1)


del AuxiliaryEvidenceSelectionTests
if __name__=='__main__':unittest.main()
