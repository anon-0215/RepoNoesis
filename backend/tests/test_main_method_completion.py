"""Identity-bound method completion; scripted outputs are not model quality."""
from dataclasses import replace
from pathlib import Path
import json, os, unittest
from unittest.mock import patch

from app.services.agent_contracts import AgentLimits
from app.services.code_chunker import extract_python_code_chunks
from app.services import directed_evidence as directed
from app.services.evidence import CitationValidator, Evidence
from tests import test_source_read_information_flow as source_flow
from tests.test_supplemental_evidence_flow import MessageModel

FIXTURE=Path(__file__).resolve().parent/'fixtures/main_method_click.json'

def final_rows(model, index=-1):
    text=model.final_messages[index][1]['content']
    return (json.loads(text)['evidence'] if text.startswith('{') else json.loads(
        text.split('VALIDATED_UNTRUSTED_EVIDENCE_BEGIN\n',1)[1].split('\nVALIDATED_UNTRUSTED_EVIDENCE_END',1)[0]))

def diagnostics(model, stage='final', index=-1):
    if stage=='planner': return model.payload(index).get('source_delivery_diagnostics')
    text=model.final_messages[index][1]['content']
    return (json.loads(text).get('source_delivery_diagnostics') if text.startswith('{') else
        json.loads(text.split('BOUNDED_SOURCE_DELIVERY_DIAGNOSTICS_BEGIN\n',1)[1].split('\nBOUNDED_SOURCE_DELIVERY_DIAGNOSTICS_END',1)[0])
        if 'BOUNDED_SOURCE_DELIVERY_DIAGNOSTICS_BEGIN\n' in text else None)


class MainMethodCompletionTests(unittest.TestCase):
    def setUp(self):
        self.flow=source_flow.SourceReadInformationFlowTests();self.flow.setUp()
        self.addCleanup(self.flow.doCleanups)
        self.fixture_count=0

    def fixture(self, sources, base, question, revision=source_flow.REVISION):
        self.question=question
        db=self.flow.db
        # Stable project ID in explicit temporary databases makes before/after
        # source identities comparable; it is fixture metadata, never a target.
        self.fixture_count+=1
        fixture_id='offline-method-completion'+('' if self.fixture_count==1 else '-'+str(self.fixture_count))
        with patch('app.database.uuid.uuid4', return_value=fixture_id):
            pid=db.create_project({'repo_url':'https://example.invalid/methods','owner':'offline','repo':'methods',
                'repository_revision':revision,'default_branch':'main'})
        files=[{'path':p,'content':s,'language':'Python','extension':'.py','size':len(s.encode()),
            'summary':'fixture','importance':1,'is_core':True,'imports':[],'exports':[],'symbols':[]} for p,s in sources.items()]
        chunks=[c.to_dict() for p,s in sources.items() for c in extract_python_code_chunks(p,s,revision).chunks]
        db.save_analysis(pid,{'overview':'offline','primary_language':'Python'},files,[],chunks)
        values=db.get_code_chunks(pid)
        self.flow.pid=pid;self.flow.bundle=db.get_bundle(pid)
        self.flow.chunks={c['qualified_name']:c for c in values}
        self.flow.base=[]
        for i,ref in enumerate(base):
            matching=[c for c in values if c['qualified_name']==ref] if isinstance(ref,str) else [c for c in values
                if (c['path'],c['qualified_name'],c['start_line'],c['end_line'])==(ref['path'],ref['symbol'],ref['start_line'],ref['end_line'])]
            self.assertEqual(len(matching),1,ref)
            key=f'base-{i}';self.flow.chunks[key]=matching[0];self.flow.base.append(key)
        return values

    def click(self):
        data=json.loads(FIXTURE.read_text(encoding='utf-8'))
        self.fixture(data['sources'],data['base'],data['question'],data['revision'])
        self.target=self.flow.chunks['base-0']

    def run_flow(self, *, enabled=True, actions=(), **kwargs):
        with patch.dict(os.environ,{'AGENT_DIRECTED_EVIDENCE_ENABLED':'1' if enabled else '0'}):
            return self.flow.run_flow(list(actions),question=self.question,**kwargs)

    def assert_click_tail(self, model):
        for rows in [model.payload(0)['planner_evidence_view']['rows'],final_rows(model)]:
            body=next(r['untrusted_source_excerpt'] for r in rows if r['symbol']=='Context.invoke')
            self.assertEqual(body,self.target['content'])
            self.assertIn('return callback(*args, **kwargs)',body)

    def neutral(self, *, source=None, question='Explain Processor.dispatch on an established Processor.', base=None, extra=None):
        source=source or ('class Processor:\n    def __init__(self):\n        self.ready = True\n'
            '    def dispatch(self, value):\n        """' + 'Neutral source padding. '*5+'\n        ')
        if source.endswith('\n        '):
            source += ('Different source text for method coverage. '*3+'\n        ')*26+'"""\n        return value + 417\n'
        sources={'domain/processor.py':source,'support/filler.py':source_flow.FILLERS}
        sources.update(extra or {})
        if base is None:
            target=next(c for c in extract_python_code_chunks('domain/processor.py',source,source_flow.REVISION).chunks if c.qualified_name=='Processor.dispatch')
            base=[{'path':target.path,'symbol':target.qualified_name,'start_line':target.start_line,'end_line':target.end_line},'filler_0','filler_1','filler_2','filler_3']
        self.fixture(sources,base,question)
        self.target=self.flow.chunks['base-0']

    def reads(self,result):
        return [e for e in result['directed_evidence_audit']['events'] if e.get('read_count')]

    def row(self,model,symbol='Processor.dispatch',stage='final'):
        return next(r for r in diagnostics(model,stage)['rows'] if r.get('symbol')==symbol)

    def assert_body(self, model, symbol, content):
        for rows in [model.payload()['planner_evidence_view']['rows'],final_rows(model)]:
            self.assertEqual(next(r['untrusted_source_excerpt'] for r in rows if r['symbol']==symbol),content)

    def test_frozen_g02_completes_exact_body_before_any_constructor(self):
        self.click()
        model,events,result=self.run_flow(label='frozen-g02')
        original=next(e for e in events[0]['after'] if e['qualified_name']=='Context.invoke')
        self.assertEqual(original['coverage']['complete_end_line'],902)
        self.assertEqual(original['coverage']['partial_end_line'],903)
        updated=next(e for e in result['evidence'] if e['qualified_name']=='Context.invoke')
        for key in ['evidence_id','chunk_identity','content_hash','code_chunk_id','start_line','end_line']:
            self.assertEqual(original[key],updated[key])
        self.assert_click_tail(model)
        targets=[e['symbol'] for e in result['directed_evidence_audit']['events'] if e.get('read_count')]
        self.assertTrue(targets and targets[0]=='Context.invoke',targets)
        self.assertNotIn('Context.__init__',targets);self.assertNotIn('Command.__init__',targets)
        for report in [diagnostics(model,'planner'),diagnostics(model)]:
            row=next(r for r in report['rows'] if r.get('symbol')=='Context.invoke')
            self.assertEqual(row['delivery_status'],'body_complete')

    def test_frozen_g02_existing_repair_messages_deliver_the_same_body(self):
        self.click();original=MessageModel.chat;counts={'planner':0,'final':0}
        def fake(model,messages,**kwargs):
            kind='planner' if 'bounded_repository_planner' in messages[0]['content'] else 'final'
            counts[kind]+=1
            if counts[kind]==1:
                (model.planner_messages if kind=='planner' else model.final_messages).append(messages)
                return '{"invalid":true}'
            if kind=='final' and 'answer_repair' in messages[0]['content']:
                model.final_messages.append(messages)
                return json.dumps({'parts':[{'text':'Scripted fixture statement.', 'evidence_aliases':['A1']}]})
            return original(model,messages,**kwargs)
        with patch.object(MessageModel,'chat',fake):
            model,events,result=self.run_flow(label='frozen-g02-repair')
        self.assertEqual(counts,{'planner':2,'final':2})
        for index in [0,1]:
            row=next(r for r in model.payload(index)['planner_evidence_view']['rows'] if r['symbol']=='Context.invoke')
            self.assertEqual(row['untrusted_source_excerpt'],self.target['content'])
            row=next(r for r in final_rows(model,index) if r['symbol']=='Context.invoke')
            self.assertEqual(row['untrusted_source_excerpt'],self.target['content'])
        self.assertEqual(diagnostics(model,'planner',0),diagnostics(model,'planner',1))
        self.assertEqual(diagnostics(model,index=0),diagnostics(model,index=1))

    def test_neutral_completion_retains_identity_at_full_capacity_and_counts_read(self):
        self.neutral()
        model,events,result=self.run_flow(capacity=5,label='neutral-full-capacity')
        self.assert_body(model,'Processor.dispatch',self.target['content'])
        self.assertEqual(len(result['evidence']),5)
        self.assertEqual([r['symbol'] for r in self.reads(result)],['Processor.dispatch'])
        original=events[0]['after'][0];updated=result['evidence'][0]
        for key in ['evidence_id','chunk_identity','content_hash','code_chunk_id']:
            self.assertEqual(original[key],updated[key])
        self.assertEqual(result['directed_evidence_audit']['read_bytes'],len(self.target['content'].encode()))
        self.assertEqual(len(result['citations']),5)
        for stage in ['planner','final']:
            row=self.row(model,stage=stage)
            self.assertEqual(row['held_body_status'],'body_complete')
            self.assertEqual(row['delivered_coverage']['complete_end_line'],self.target['end_line'])

    def test_all_held_explicit_methods_precede_constructor_and_outward_reads(self):
        one='    def first(self):\n        """'+('first padding '*12+'\n        ')*24+'"""\n        return helper()\n'
        two='    def second(self):\n        """'+('second padding '*12+'\n        ')*22+'"""\n        return 663\n'
        source='def helper():\n    return 829\nclass Runner:\n'+one+two
        extra={'other/build.py':'class Builder:\n    """'+'class padding\n'*205+'"""\n    def __init__(self):\n        self.built = True\n'}
        self.neutral(source=source,base=['Runner.first','Runner.second','filler_0','filler_1','filler_2'],extra=extra,
            question='Explain Builder construction, Runner.first and Runner.second.')
        model,_,result=self.run_flow(limits=replace(AgentLimits(),max_same_tool_calls=2),label='two-primary-methods')
        self.assertEqual([r['symbol'] for r in self.reads(result)],['Runner.first','Runner.second'])
        for symbol in ['Runner.first','Runner.second']:
            self.assert_body(model,symbol,self.flow.chunks[symbol]['content'])
        self.assertTrue(any(r.get('symbol')=='Builder.__init__' and r['reason']=='read_count_limit' for r in diagnostics(model)['rows']))

    def test_no_name_binding_does_not_complete_all_same_named_held_methods(self):
        long='    def dispatch(self):\n        """'+('ambiguous padding '*12+'\n        ')*20+'"""\n        return 887\n'
        self.neutral(source='class Processor:\n'+long,extra={'another.py':'class Other:\n'+long},
            base=['Processor.dispatch','Other.dispatch','filler_0','filler_1','filler_2'],question='Explain dispatch().')
        model,_,result=self.run_flow(label='ambiguous-held-identities')
        self.assertEqual(self.reads(result),[])
        for symbol in ['Processor.dispatch','Other.dispatch']:
            row=self.row(model,symbol)
            self.assertEqual(row['held_body_status'],'body_partial')
            self.assertEqual(row['delivery_status'],'body_partial')
            self.assertEqual(row['binding_status'],'ambiguous_explicit_name')
            self.assertEqual(row['chunk_identity'],next(e['chunk_identity'] for e in result['evidence'] if e['qualified_name']==symbol))
        self.assertTrue(any(r['delivery_status']=='not_statically_located' for r in diagnostics(model)['rows']))

    def test_qualified_identity_survives_other_file_same_method_name(self):
        self.neutral(extra={'elsewhere.py':'class Processor:\n    def dispatch(self):\n        return 998\n'},
            question='Explain domain.processor.Processor.dispatch on an established object.')
        model,_,result=self.run_flow()
        self.assertEqual([r['symbol'] for r in self.reads(result)],['Processor.dispatch'])
        self.assert_body(model,'Processor.dispatch',self.target['content'])
        self.assertFalse(any(e['path']=='elsewhere.py' for e in result['evidence']))

    def test_overload_unique_implementation_with_import_aliases(self):
        for imported,decorator in [('from typing import overload','overload'),
                ('from typing import overload as signature','signature'),('import typing as t','t.overload')]:
            with self.subTest(imported=imported):
                source=imported+'\nclass Dispatcher:\n    @'+decorator+'\n    def send(self, x: int): ...\n    @'+decorator+'\n    def send(self, x: str): ...\n'
                source+='    def send(self, x):\n        """'+('implementation padding '*8+'\n        ')*20+'"""\n        return x\n'
                body=next(c for c in extract_python_code_chunks('domain/processor.py',source,source_flow.REVISION).chunks if 'return x' in c.content and c.qualified_name=='Dispatcher.send')
                ref={'path':'domain/processor.py','symbol':body.qualified_name,'start_line':body.start_line,'end_line':body.end_line}
                self.neutral(source=source,base=[ref,'filler_0','filler_1','filler_2','filler_3'],question='Explain Dispatcher.send.')
                model,_,result=self.run_flow()
                self.assertEqual([r['symbol'] for r in self.reads(result)],['Dispatcher.send'])
                self.assert_body(model,'Dispatcher.send',body.content)
                self.assertEqual(self.row(model,'Dispatcher.send')['binding_status'],'overload_unique_implementation')

    def test_unsafe_declarations_remain_partial_with_known_identity(self):
        body='    def send(self, x):\n        """'+('uncertain source '*12+'\n        ')*20+'"""\n        return x\n'
        variants=[
            'class Dispatcher:\n    def send(self, x): return 0\n'+body,
            'class Dispatcher:\n'+body+'    send = replacement\n',
            'class Dispatcher:\n    if FLAG:\n'+''.join('    '+line+'\n' for line in body.splitlines()),
            'class Dispatcher:\n    @unknown\n'+body,
            'from typing import overload\noverload = custom\nclass Dispatcher:\n    @overload\n    def send(self, x): ...\n'+body,
            'from typing import overload\nclass Dispatcher:\n    overload = custom\n    @overload\n    def send(self, x): ...\n'+body,
            'from typing import overload\nclass Dispatcher:\n    @overload\n    def send(self, x): return x\n'+body,
            'class Dispatcher:\n'+body+'Dispatcher.send = replacement\n',
        ]
        for source in variants:
            with self.subTest(source=source[:100]):
                candidate=next(c for c in extract_python_code_chunks('domain/processor.py',source,source_flow.REVISION).chunks if c.qualified_name=='Dispatcher.send' and 'uncertain source' in c.content)
                ref={'path':'domain/processor.py','symbol':candidate.qualified_name,'start_line':candidate.start_line,'end_line':candidate.end_line}
                self.neutral(source=source,base=[ref,'filler_0','filler_1','filler_2','filler_3'],question='Explain Dispatcher.send.')
                model,_,result=self.run_flow()
                self.assertEqual(self.reads(result),[])
                self.assertEqual(self.row(model,'Dispatcher.send')['held_body_status'],'body_partial')
                self.assertEqual(self.row(model,'Dispatcher.send')['delivery_status'],'body_partial')
                self.assertTrue(any(r['delivery_status']=='not_statically_located' for r in diagnostics(model)['rows']))

    def test_necessary_constructor_paths_survive(self):
        source='class Builder:\n    """'+'class padding\n'*205+'"""\n    def __init__(self):\n        self.value = 716\n'
        for question in ['Explain Builder construction.','Explain Builder.__init__.','Explain Builder().','说明 Builder 的构造行为。','Explain entry().']:
            with self.subTest(question=question):
                self.neutral(source=source+'def entry():\n    return Builder()\n',base=['entry','filler_0','filler_1','filler_2','filler_3'],question=question)
                model,_,result=self.run_flow()
                self.assertEqual([r['symbol'] for r in self.reads(result)],['Builder.__init__'])
                self.assert_body(model,'Builder.__init__',self.flow.chunks['Builder.__init__']['content'])

    def test_bare_class_and_method_owner_do_not_seed_constructor(self):
        source='class Builder:\n    """'+'class padding\n'*205+'"""\n    def __init__(self):\n        self.value = 716\n    def run(self):\n        return self.value\n'
        for question in ['Explain Builder.','Explain Builder.run on the established Builder object.']:
            self.neutral(source=source,base=['Builder.run','filler_0','filler_1','filler_2','filler_3'],question=question)
            _,_,result=self.run_flow()
            self.assertEqual(self.reads(result),[])

    def test_one_construction_request_does_not_activate_another_owner_class(self):
        source='class Builder:\n    """'+'class padding\n'*205+'"""\n    def __init__(self):\n        self.value = 716\n    def run(self):\n        return self.value\n'
        extra={'other.py':'class Factory:\n    """'+'class padding\n'*205+'"""\n    def __init__(self):\n        self.product = 523\n'}
        self.neutral(source=source,extra=extra,base=['Builder.run','filler_0','filler_1','filler_2','filler_3'],
            question='Explain Factory construction and Builder.run on an established Builder.')
        _,_,result=self.run_flow()
        self.assertEqual([r['symbol'] for r in self.reads(result)],['Factory.__init__'])

    def test_rebound_conditional_or_decorated_constructor_is_not_guessed(self):
        head='class Builder:\n    """'+'class padding\n'*205+'"""\n'
        constructor='    def __init__(self):\n        self.value = 716\n'
        for tail in [constructor+'    __init__ = replacement\n',
                constructor+'    if FLAG:\n        def __init__(self): pass\n',
                '    @custom\n'+constructor]:
            for question in ['Explain Builder construction.','Explain entry().']:
                self.neutral(source=head+tail+'def entry():\n    return Builder()\n',
                    base=['entry','filler_0','filler_1','filler_2','filler_3'],question=question)
                model,_,result=self.run_flow()
                self.assertEqual(self.reads(result),[])
                self.assertTrue(any(r.get('expression')=='Builder.__init__' and
                    r['delivery_status']=='not_statically_located' for r in diagnostics(model)['rows']))

    def test_line_byte_and_read_count_bounds_keep_partial_body(self):
        cases=[('lines','class Processor:\n    def dispatch(self):\n'+('        # long neutral source\n'*205)+'        return 917\n',AgentLimits(),'line_limit'),
            ('bytes',None,replace(AgentLimits(),max_source_read_bytes=100),'read_budget_limit'),
            ('reads',None,replace(AgentLimits(),max_same_tool_calls=0),'read_count_limit')]
        for label,source,limits,reason in cases:
            with self.subTest(label=label):
                self.neutral(source=source)
                model,_,result=self.run_flow(limits=limits,label='bounded-'+label)
                self.assertEqual(self.reads(result),[])
                row=self.row(model)
                self.assertEqual(row['held_body_status'],'body_partial')
                self.assertEqual(row['delivery_status'],'body_partial')
                self.assertEqual(row['reason'],reason)
                self.assertNotEqual(next(r['untrusted_source_excerpt'] for r in final_rows(model) if r['symbol']=='Processor.dispatch'),self.target['content'])

    def test_cutoff_and_cancellation_do_not_promote_or_lose_held_coverage(self):
        from app.services.agent_tools import ToolDeadlineExceeded, ToolCancelled
        for exception,code in [(ToolDeadlineExceeded,'tool_timeout'),(ToolCancelled,'cancelled')]:
            self.neutral()
            original=self.flow.db.get_code_chunks_for_hierarchy
            def cutoff(*args,**kwargs):
                if kwargs.get('limit')==2049: raise exception(code,'offline bound')
                return original(*args,**kwargs)
            with patch.object(self.flow.db,'get_code_chunks_for_hierarchy',side_effect=cutoff):
                model,_,result=self.run_flow()
            self.assertEqual(self.reads(result),[])
            row=self.row(model)
            self.assertEqual(row['held_body_status'],'body_partial')
            self.assertEqual(row['reason'],code)

    def test_canonical_rejection_does_not_promote_held_partial(self):
        self.neutral();original=self.flow.db.get_code_chunks_for_hierarchy
        def corrupted(*args,**kwargs):
            chunks=original(*args,**kwargs)
            return [{**c,'content_hash':'0'*64} for c in chunks] if kwargs.get('limit')==2049 else chunks
        with patch.object(self.flow.db,'get_code_chunks_for_hierarchy',side_effect=corrupted):
            model,_,result=self.run_flow()
        self.assertEqual(len(self.reads(result)),1)
        self.assertFalse(self.reads(result)[0]['committed'])
        self.assertEqual(self.row(model)['reason'],'canonical_rejection')
        self.assertEqual(self.row(model)['held_body_status'],'body_partial')

    def test_final_exclusion_distinguishes_full_stored_body_from_message(self):
        self.neutral(source='class Processor:\n    def dispatch(self):\n        """'+'full source padding '*85+'"""\n        return 417\n',
            base=['filler_0','filler_1','filler_2','filler_3','Processor.dispatch'])
        model,_,result=self.run_flow(limits=replace(AgentLimits(),max_accumulated_evidence_context_bytes=1700),label='stored-full-final-excluded')
        self.assertNotIn('Processor.dispatch',{r['symbol'] for r in final_rows(model)})
        row=self.row(model)
        self.assertEqual(row['held_body_status'],'body_complete')
        self.assertEqual(row['delivery_status'],'blocked_by_budget')
        self.assertEqual(row['reason'],'final_answer_body_selection')
        self.assertNotIn('Processor.dispatch',{r['qualified_name'] for r in result['citations']})

    def test_delivery_budget_rejection_charges_read_without_promoting_body(self):
        source='class Processor:\n    def dispatch(self):\n        """'+('delivery budget padding '*10+'\n        ')*40+'"""\n        return 417\n'
        self.neutral(source=source)
        model,_,result=self.run_flow(limits=replace(AgentLimits(),max_accumulated_evidence_context_bytes=4500),label='read-but-not-committed')
        read=self.reads(result)[0]
        self.assertEqual(read['symbol'],'Processor.dispatch')
        self.assertTrue(read['body_read']);self.assertFalse(read['committed'])
        self.assertEqual(read['status'],'delivery_budget_limit')
        self.assertEqual(result['directed_evidence_audit']['read_bytes'],len(self.target['content'].encode()))
        for stage in ['planner','final']:
            row=self.row(model,stage=stage)
            self.assertEqual(row['held_body_status'],'body_partial')
            self.assertEqual(row['delivery_status'],'body_partial')
            self.assertEqual(row['reason'],'delivery_budget_limit')
            self.assertTrue(row['body_read']);self.assertFalse(row['read_committed'])

    def test_explicit_tool_method_is_completed_after_tool_boundary(self):
        from tests.test_m2_agent import decision
        self.neutral(question='Inspect the stored library.')
        model,_,result=self.run_flow(actions=[decision('continue','lookup_symbol',{'symbol':'Processor.dispatch'})],label='tool-target-completion')
        self.assertEqual(len(model.planner_messages),2)
        self.assertNotEqual(next(r['untrusted_source_excerpt'] for r in model.payload(0)['planner_evidence_view']['rows'] if r['symbol']=='Processor.dispatch'),self.target['content'])
        self.assert_body(model,'Processor.dispatch',self.target['content'])
        self.assertEqual([r['symbol'] for r in self.reads(result)],['Processor.dispatch'])

    def test_early_static_cutoff_still_delivers_known_body_identity(self):
        from app.services.agent_tools import ToolDeadlineExceeded
        self.neutral()
        with patch.object(directed.StaticLocator,'module',side_effect=ToolDeadlineExceeded('tool_timeout','early AST cutoff')):
            model,_,result=self.run_flow(label='early-ast-cutoff')
        self.assertEqual(self.reads(result),[])
        for stage in ['planner','final']:
            row=self.row(model,stage=stage)
            self.assertEqual(row['held_body_status'],'body_partial')
            self.assertEqual(row['binding_status'],'not_checked')
            self.assertEqual(row['reason'],'tool_timeout')
            self.assertEqual(row['chunk_identity'],result['evidence'][0]['chunk_identity'])

    def test_static_inspection_budget_keeps_known_partial_and_specific_reason(self):
        source='class Processor:\n    def dispatch(self):\n        """'+('source padding '*10+'\n        ')*25+'"""\n        return 417\n'+('# unrelated file padding\n'*250)
        self.neutral(source=source)
        model,_,result=self.run_flow(limits=replace(AgentLimits(),max_observation_bytes=1024),label='static-inspection-budget')
        self.assertEqual(self.reads(result),[])
        row=self.row(model)
        self.assertEqual(row['held_body_status'],'body_partial')
        self.assertEqual(row['delivery_status'],'body_partial')
        self.assertEqual(row['reason'],'static_inspection_byte_limit')

    def test_explicit_target_truncation_reports_omission(self):
        extra={'many.py':''.join(f'def target_{i}():\n    return {i}\n' for i in range(9))}
        self.neutral(extra=extra,question='Explain '+', '.join(f'target_{i}' for i in range(9)))
        model,_,result=self.run_flow(label='target-limit-omission')
        self.assertTrue(diagnostics(model)['records_omitted'])
        self.assertLessEqual(len(self.reads(result)),3)

    def test_delivery_report_partial_projection_is_not_full_stored_claim(self):
        self.neutral()
        model,_,result=self.run_flow()
        full=Evidence(**next(e for e in result['evidence'] if e['qualified_name']=='Processor.dispatch'))
        CitationValidator(self.flow.db).validate(full)
        partial=replace(full,excerpt=full.excerpt[:1537]);CitationValidator(self.flow.db).validate(partial)
        ledger=directed.PreparationLedger()
        ledger.record(path=full.path,symbol=full.qualified_name,start_line=full.start_line,end_line=full.end_line,located=True,is_primary=True)
        row=directed.delivery_report(ledger,[partial],8192,'final_answer',held=[full])['rows'][0]
        self.assertEqual(row['held_body_status'],'body_complete')
        self.assertEqual(row['delivery_status'],'body_partial')
        self.assertEqual(row['reason'],'final_answer_body_selection')
        self.assertEqual(row['delivered_coverage']['partial_end_line'],partial.planner_excerpt_coverage()['partial_end_line'])
        # A stale valid flag cannot lend a source proof to a changed identity.
        forged=replace(full,content_hash='0'*64)
        row=directed.delivery_report(ledger,[],8192,'final_answer',held=[forged])['rows'][0]
        self.assertEqual(row['held_body_status'],'not_held')

    def test_diagnostic_identity_merge_priority_and_omission_remain_bounded(self):
        self.neutral();model,_,result=self.run_flow()
        full=Evidence(**result['evidence'][0]);CitationValidator(self.flow.db).validate(full)
        ledger=directed.PreparationLedger()
        for i in range(32): ledger.record(expression='unresolved_'+str(i),source_path='other.py',call_line=i,located=False)
        for reason in ['explicit_entry','static_call']:
            ledger.record(path=full.path,symbol=full.qualified_name,start_line=full.start_line,end_line=full.end_line,located=True,is_primary=True,reason=reason)
        self.assertEqual(len(ledger.checks),32)
        self.assertEqual(sum(r.get('symbol')==full.qualified_name for r in ledger.checks),1)
        report=directed.delivery_report(ledger,[full],1400,'planner',held=[full])
        self.assertEqual(report['rows'][0]['symbol'],full.qualified_name)
        self.assertTrue(report['records_omitted']);self.assertFalse(report['is_evidence'])
        self.assertLessEqual(len(json.dumps(report,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()),1400)

    def test_candidate_disabled_and_default_agent_rag_capacities_unchanged(self):
        self.neutral(base=['Processor.dispatch','filler_0','filler_1','filler_2','filler_3','Processor.__init__'])
        for mode,capacity in [('agent',8),('rag',5)]:
            model,events,result=self.run_flow(enabled=False,mode=mode,label='off-'+mode)
            self.assertNotIn('directed_evidence_audit',result)
            self.assertEqual(len(result['evidence']),5)
            self.assertEqual(events[0]['policy']['remaining_capacity'],capacity-5)
            self.assertIsNone(diagnostics(model))
            self.assertNotEqual(next(r['untrusted_source_excerpt'] for r in final_rows(model) if r['symbol']=='Processor.dispatch'),self.target['content'])
            self.assertIn('Prompt version: m1-v3.',model.final_messages[0][0]['content'])

    def test_frozen_a03_q3_necessary_bodies_remain_delivered(self):
        data=json.loads((FIXTURE.parent/'main_method_regression_click.json').read_text(encoding='utf-8'))
        sources={**data['sources'],'src/click/core.py':json.loads(FIXTURE.read_text(encoding='utf-8'))['sources']['src/click/core.py']}
        for qid,required in [('A03','open_stream'),('Q3','_posixify')]:
            case=data['cases'][qid]
            self.fixture(sources,case['base'],case['question'],data['revision'])
            model,_,result=self.run_flow(label='regression-'+qid)
            self.assert_body(model,required,self.flow.chunks[required]['content'])
            self.assertIn(required,[r['symbol'] for r in self.reads(result)])
            self.assertNotIn('Command.__init__',[r['symbol'] for r in self.reads(result)])
            self.assertLessEqual(len(self.reads(result)),3)
            row=self.row(model,required)
            self.assertEqual(row['delivery_status'],'body_complete')


if __name__=='__main__': unittest.main()
