"""Explicit data-member source selection, through actual serialized requests.

Fixtures are parsed into explicit temporary databases; target code is never run.
Scripted model outputs exercise delivery and citations, not model answer quality.
"""
from dataclasses import replace
from pathlib import Path
import json, unittest
from unittest.mock import patch

from app.services.agent_contracts import AgentLimits
from tests.test_main_method_completion import MainMethodCompletionTests, final_rows, diagnostics
from tests.test_supplemental_evidence_flow import MessageModel

FIXTURES=Path(__file__).resolve().parent/'fixtures'


class AuxiliaryEvidenceSelectionTests(unittest.TestCase):
    # Reuse only fixture helpers; the existing regression class runs separately.
    setUp=MainMethodCompletionTests.setUp
    fixture=MainMethodCompletionTests.fixture
    neutral=MainMethodCompletionTests.neutral
    run_flow=MainMethodCompletionTests.run_flow
    reads=MainMethodCompletionTests.reads
    assert_body=MainMethodCompletionTests.assert_body

    def frozen(self, qid):
        case=json.loads((FIXTURES/'auxiliary_source_cases.json').read_text(encoding='utf-8'))[qid]
        sources=json.loads((FIXTURES/'main_method_click.json').read_text(encoding='utf-8'))['sources']
        sources={**sources,**json.loads((FIXTURES/'main_method_regression_click.json').read_text(encoding='utf-8'))['sources']}
        self.fixture(sources,case['base'],case['question'],case['revision'])

    def neutral_member(self, *, question=None, enum=None, extra=None, source=None):
        main=('class Processor:\n    def dispatch(self, value):\n        """'+
            ('Neutral selection padding '*6+'\n        ')*23+'"""\n'+
            '        if value is None:\n            noise()\n        return value < Bias.LOW\n')
        head='from .ranking import Bias\ndef noise():\n    return "secondary branch"\n'
        self.neutral(source=source or head+main,extra={
            'domain/ranking.py':enum or 'import enum\nclass Bias(enum.IntEnum):\n    HIGH = enum.auto()\n    LOW = enum.auto()\n',
            **(extra or {})},question=question or 'Explain Processor.dispatch using Bias.LOW and Bias.HIGH.')

    def test_frozen_n_main_explicit_source_order_precedes_secondary_calls(self):
        self.frozen('N-main')
        model,events,result=self.run_flow(label='frozen-n-main')
        self.assertEqual([e['symbol'] for e in self.reads(result)][:2],
            ['Parameter.handle_parse_result','ParameterSource'])
        for symbol in ['Parameter.handle_parse_result','ParameterSource']:
            self.assert_body(model,symbol,self.flow.chunks[symbol]['content'])
        self.assertNotIn('echo',[e['symbol'] for e in self.reads(result)])
        original=events[0]['after'][0]
        updated=next(e for e in result['evidence'] if e['qualified_name']=='Parameter.handle_parse_result')
        for key in ['evidence_id','chunk_identity','content_hash','code_chunk_id']:
            self.assertEqual(original[key],updated[key])
        for stage in ['planner','final']:
            report=diagnostics(model,stage)
            self.assertFalse(report['runtime_binding_proven'])
            row=next(r for r in report['rows'] if r.get('symbol')=='ParameterSource')
            self.assertEqual(row['delivery_status'],'body_complete')
        # The question-linked state pair may be acquired mechanically. Definition
        # source is delivered without claiming the annotation proves dispatch.
        for symbol in ['Context.get_parameter_source','Context.set_parameter_source']:
            self.assert_body(model,symbol,self.flow.chunks[symbol]['content'])
        self.assertTrue(any(r.get('expression')=='ctx.get_parameter_source' and
            r['delivery_status']=='not_statically_located' for r in diagnostics(model)['rows']))
        self.assertLessEqual(len(self.reads(result)),3)

    def test_neutral_members_share_one_body_and_keep_main_first(self):
        self.neutral_member(extra={'unrelated.py':'class Other:\n    SECRET = 123\n    def extra(self):\n        return 18\n'})
        model,events,result=self.run_flow(label='neutral-data-member')
        self.assertEqual([e['symbol'] for e in self.reads(result)],['Processor.dispatch','Bias','noise'])
        for symbol in ['Processor.dispatch','Bias']:
            self.assert_body(model,symbol,self.flow.chunks[symbol]['content'])
        self.assertNotIn('Other',{e['qualified_name'] for e in result['evidence']})
        self.assertEqual(sum(e['qualified_name']=='Bias' for e in result['evidence']),1)
        source=next(e for e in result['evidence'] if e['qualified_name']=='Bias')
        citation=next(c for c in result['citations'] if c['qualified_name']=='Bias')
        for k in ['path','start_line','end_line','evidence_id']:
            self.assertEqual(citation[k],source[k])

    def test_multiple_partial_main_methods_get_budget_before_member_body(self):
        self.neutral_member()
        original=next(f['content'] for f in self.flow.bundle['files'] if f['path']=='domain/processor.py')
        source=original.replace('    def dispatch(self, value):','    def second(self, value):')
        body=source.split('class Processor:\n',1)[1]
        self.neutral_member(source=original+body,question='Explain Processor.dispatch and Processor.second with Bias.LOW.')
        self.flow.base=['Processor.dispatch','Processor.second','filler_0','filler_1','filler_2']
        model,_,result=self.run_flow(label='multiple-main-member')
        self.assertEqual([e['symbol'] for e in self.reads(result)],['Processor.dispatch','Processor.second','Bias'])
        for name in ['Processor.dispatch','Processor.second','Bias']:
            self.assert_body(model,name,self.flow.chunks[name]['content'])

    def test_bare_class_and_unmentioned_member_do_not_expand(self):
        self.neutral_member(question='Explain Processor.dispatch on an established object of Processor; Bias is background.')
        model,_,result=self.run_flow(label='bare-class-not-data-member')
        self.assertNotIn('Bias',[e['symbol'] for e in self.reads(result)])
        self.assertNotIn('Bias',{r['symbol'] for r in final_rows(model)})

    def test_data_owner_does_not_expand_its_unrelated_methods(self):
        self.neutral_member(source='class Processor:\n    def dispatch(self, value):\n        return value\n',
            enum='def unrelated():\n    return 976\nclass Bias:\n    LOW = 3\n    def ignored(self):\n        return unrelated()\n')
        model,_,result=self.run_flow(label='data-owner-terminal')
        self.assertEqual([r['symbol'] for r in self.reads(result)],['Bias'])
        self.assert_body(model,'Bias',self.flow.chunks['Bias']['content'])
        self.assertNotIn('unrelated',{r['symbol'] for r in final_rows(model)})

    def test_separate_construction_request_keeps_constructor_call_expansion(self):
        source='def build_state():\n    return 99\nclass Bias:\n    LOW = 7\n    def __init__(self):\n        self.ready = build_state()\n'
        source+=''.join('    field_'+str(i)+' = '+str(i)+'\n' for i in range(205))
        self.neutral_member(enum=source,question='Explain Processor.dispatch, Bias.LOW and Bias() construction.')
        model,_,result=self.run_flow(label='construction-and-data-member')
        self.assertEqual([r['symbol'] for r in self.reads(result)],['Processor.dispatch','Bias.__init__','noise'])
        self.assert_body(model,'Bias.__init__',self.flow.chunks['Bias.__init__']['content'])
        # noise uses the last read in BFS order; the constructor edge must still
        # be inspected and report the shared quota, rather than disappearing.
        self.assertTrue(any(r.get('symbol')=='build_state' and r['reason']=='read_count_limit'
            for r in diagnostics(model)['rows']))

    def test_same_named_owner_is_ambiguous_until_module_qualified(self):
        extra={'elsewhere.py':'class Bias:\n    LOW = -400\n    HIGH = 900\n'}
        self.neutral_member(extra=extra)
        model,_,result=self.run_flow(label='ambiguous-owner')
        self.assertNotIn('Bias',[e['symbol'] for e in self.reads(result)])
        self.assertTrue(any(r.get('expression')=='Bias.LOW' and r['delivery_status']=='not_statically_located'
            for r in diagnostics(model)['rows']))
        self.neutral_member(extra=extra,question='Explain Processor.dispatch with domain.ranking.Bias.LOW.')
        model,_,result=self.run_flow(label='qualified-owner')
        self.assert_body(model,'Bias',next(c['content'] for c in self.flow.db.get_code_chunks(self.flow.pid)
            if c['path']=='domain/ranking.py' and c['qualified_name']=='Bias'))
        self.assertFalse(any(e['path']=='elsewhere.py' for e in result['evidence']))

    def test_conditional_rebound_inherited_decorated_and_dynamic_owners_fall_back(self):
        variants=[
            'class Bias:\n    if FLAG:\n        LOW = 7\n',
            'class Bias:\n    LOW = 1\n    LOW = 2\n',
            'class Base:\n    LOW = 4\nclass Bias(Base):\n    pass\n',
            '@unknown\nclass Bias:\n    LOW = 7\n',
            'class Bias(metaclass=unknown):\n    LOW = 7\n',
            'class Bias:\n    LOW = 7\nBias.LOW = 9\n',
            'class Bias:\n    LOW = 7\nBias = replacement\n',
            'from unknown import Palette as Bias\n',
        ]
        for index,source in enumerate(variants):
            with self.subTest(source=source):
                self.neutral_member(enum=source)
                model,_,result=self.run_flow(label='unsafe-data-member-'+str(index))
                self.assertNotIn('Bias',[e['symbol'] for e in self.reads(result)])
                self.assert_body(model,'Processor.dispatch',self.target['content'])

    def test_shared_quota_full_capacity_and_byte_selection_keep_main(self):
        for kind in ['quota','capacity','delivery','read-bytes']:
            with self.subTest(kind=kind):
                self.neutral_member()
                kwargs={}
                if kind=='quota':kwargs['limits']=replace(AgentLimits(),max_same_tool_calls=1)
                elif kind=='capacity':kwargs['capacity']=5
                elif kind=='delivery':
                    self.neutral_member(enum='class Bias:\n    LOW = 8\n    """'+('delivery padding '*9+'\n    ')*20+'"""\n')
                    base_bytes=sum(len(self.flow.chunks[n]['content'].encode()) for n in self.flow.base)
                    kwargs['limits']=replace(AgentLimits(),max_accumulated_evidence_context_bytes=base_bytes+1000)
                else:
                    self.neutral_member(enum='class Bias:\n    LOW = 8\n    """'+('large enum documentation '*15+'\n    ')*30+'"""\n')
                    kwargs['limits']=replace(AgentLimits(),max_source_read_bytes=8192)
                model,_,result=self.run_flow(label='member-budget-'+kind,**kwargs)
                self.assert_body(model,'Processor.dispatch',self.target['content'])
                self.assertNotIn('Bias',{r['symbol'] for r in final_rows(model)})
                self.assertNotIn('Bias',{r['qualified_name'] for r in result['citations']})
                event=next(r for r in result['directed_evidence_audit']['events'] if r.get('symbol')=='Bias')
                self.assertIn(event['status'],{'read_count_limit','capacity_limit','delivery_budget_limit','read_budget_limit'})
                report=diagnostics(model)
                row=next((r for r in report['rows'] if r.get('symbol')=='Bias'),None) if report else None
                if row:self.assertEqual(row['held_body_status'],'not_held')
                else:self.assertTrue(report is None or report['records_omitted'])

    def test_large_data_owner_does_not_redirect_to_constructor(self):
        self.neutral_member(enum='class Bias:\n    LOW = 7\n    def __init__(self):\n        self.built = True\n'+
            ''.join('    field_'+str(i)+' = '+str(i)+'\n' for i in range(205)))
        model,_,result=self.run_flow(label='large-data-owner')
        self.assert_body(model,'Processor.dispatch',self.target['content'])
        self.assertNotIn('Bias.__init__',[r['symbol'] for r in self.reads(result)])
        self.assertEqual(next(r for r in diagnostics(model)['rows'] if r.get('symbol')=='Bias')['reason'],'line_limit')

    def test_existing_planner_and_final_repairs_receive_member_and_main(self):
        self.neutral_member();chat=MessageModel.chat;counts={'planner':0,'final':0}
        def fake(model,messages,**kwargs):
            kind='planner' if 'bounded_repository_planner' in messages[0]['content'] else 'final'
            counts[kind]+=1
            if counts[kind]==1:
                (model.planner_messages if kind=='planner' else model.final_messages).append(messages)
                return '{"invalid":true}'
            if kind=='final' and 'answer_repair' in messages[0]['content']:
                model.final_messages.append(messages)
                return json.dumps({'parts':[{'text':'Scripted fixture.', 'evidence_aliases':['A1']}]})
            return chat(model,messages,**kwargs)
        with patch.object(MessageModel,'chat',fake):
            model,_,result=self.run_flow(label='member-repairs')
        self.assertEqual(counts,{'planner':2,'final':2})
        for i in [0,1]:
            for rows in [model.payload(i)['planner_evidence_view']['rows'],final_rows(model,i)]:
                for name in ['Processor.dispatch','Bias']:
                    self.assertEqual(next(r['untrusted_source_excerpt'] for r in rows if r['symbol']==name),self.flow.chunks[name]['content'])

    def test_frozen_boundary_off_gap_and_enabled_helper_are_unchanged(self):
        self.frozen('U-boundary')
        for on in [False,True]:
            model,_,result=self.run_flow(enabled=on,label='boundary-'+str(on))
            for symbol in ['Parameter.get_default','Context.lookup_default']:
                # get_default has overload declarations; base-0 is the selected implementation.
                chunk=next(c for c in self.flow.db.get_code_chunks(self.flow.pid) if c['qualified_name']==symbol
                    and 'return value' in c['content'])
                self.assert_body(model,symbol,chunk['content'])
            self.assertEqual('Context._default_map_has' in {r['symbol'] for r in final_rows(model)},on)
            if on:
                self.assertEqual([r['symbol'] for r in self.reads(result)],['Context._default_map_has'])
                self.assert_body(model,'Context._default_map_has',self.flow.chunks['Context._default_map_has']['content'])
            else:self.assertIsNone(diagnostics(model))


# Avoid loading the helper's test class a second time from this module.
del MainMethodCompletionTests

if __name__=='__main__':unittest.main()
