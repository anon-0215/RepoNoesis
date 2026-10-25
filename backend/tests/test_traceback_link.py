"""Static neutral traceback inputs only; no inspected source is executed."""
import ast
import json
import unittest
from unittest.mock import patch
from app.services import traceback_link as link
from app.services.error_locator import extract_text
from app.services.exploration_sources import record

TEXT='''def caller():
    return run()
def run():
    if True:
        raise ValueError("shared failure")
def other():
    raise ValueError("shared failure")
'''

def data(files=None):
    files=files or {'pkg/core.py':TEXT}
    return {'files':{p:record(p,t,'saved_snapshot') for p,t in files.items()},
            'items':[x for p,t in files.items() for x in extract_text(t,path=p).items]}

def run(text,files=None):
    d=data(files);candidates=[{k:v for k,v in x.items() if k!='pattern'} for x in d['items']]
    return link.associate(d,link.parse(text),candidates),candidates

def tb(frames,terminal='ValueError: shared failure'):
    return 'Traceback (most recent call last):\n'+frames+'\n'+terminal

class TracebackLinkTests(unittest.TestCase):
    def test_windows_posix_and_relative_frames_use_source_ranges_not_name_only(self):
        text=tb('  File "C:\\demo\\pkg\\core.py", line 2, in caller\n    return run()\n  File "/tmp/pkg/core.py", line 5, in run\n    raise ValueError("shared failure")\n    ~~~~^^^^^^^^\n  File "pkg/core.py", line 5, in run')
        result,c=run(text)
        self.assertTrue(result['complete']);self.assertEqual([f['status'] for f in result['frames']],['MATCHED']*3)
        self.assertEqual(result['frames'][0]['matches'][0]['relation'],'FUNCTION_RANGE')
        self.assertEqual(c[0]['trace_frames'],[2,3]);self.assertEqual(c[1]['trace_frames'],[])
        self.assertEqual(result['frames'][2]['matches'][0]['source_check'],'NOT_SUPPLIED')

    def test_duplicate_suffixes_and_basename_are_never_auto_selected(self):
        files={'pkg/core.py':TEXT,'nested/pkg/core.py':TEXT,'other/core.py':TEXT}
        for path,count in [('/root/nested/pkg/core.py',2),('core.py',3)]:
            result,c=run(tb(f'  File "{path}", line 5, in run'),files)
            self.assertEqual(result['frames'][0]['status'],'AMBIGUOUS');self.assertEqual(result['frames'][0]['matches_total'],count)
            self.assertTrue(all(not x['trace_frames'] for x in c))
        result,_=run(tb('  File "wrong/sub/core.py", line 5, in run'),files);self.assertEqual(result['frames'][0]['status'],'MISSING')
        result,_=run(tb('  File "core.py", line 5, in run'));self.assertEqual(result['frames'][0]['status'],'AMBIGUOUS')

    def test_recursion_keeps_duplicate_frames_in_call_order(self):
        result,c=run(tb('\n'.join('  File "pkg/core.py", line 5, in run' for _ in range(4))))
        self.assertEqual([f['index'] for f in result['frames']],[1,2,3,4]);self.assertEqual(c[0]['trace_frames'],[1,2,3,4])

    def test_input_paths_are_never_read_and_traversal_and_illegal_lines_rejected(self):
        for path,line,status in [('pkg/../core.py','5','INVALID'),('C:\\x\\..\\pkg\\core.py','5','INVALID'),('/secret/passwords.py','5','MISSING'),('pkg/core.py','0','INVALID'),('pkg/core.py','-3','INVALID'),('pkg/core.py','99999999999','INVALID'),('pkg/core.py','999','OUT_OF_RANGE'),('pkg/core.py','nan','INVALID')]:
            with self.subTest(path=path,line=line),patch('builtins.open',side_effect=AssertionError('unexpected path read')):
                result,c=run(tb(f'  File "{path}", line {line}, in run'))
                self.assertEqual(result['frames'][0]['status'],status);self.assertTrue(all(not x['trace_frames'] for x in c))
                if status=='INVALID':self.assertFalse(result['complete'])

    def test_source_and_function_conflicts_do_not_shift_or_associate(self):
        result,c=run(tb('  File "pkg/core.py", line 5, in caller\n    return run()'))
        loc=result['frames'][0]['matches'][0];self.assertEqual(loc['status'],'CONFLICT');self.assertEqual(loc['line'],5);self.assertTrue(loc['readable']);self.assertEqual(loc['source_check'],'DIFFERS');self.assertEqual(loc['function_check'],'DIFFERS');self.assertNotIn('scope',loc);self.assertTrue(all(not x['trace_frames'] for x in c))

    def test_partial_format_and_complex_chains_are_explicit(self):
        parsed=link.parse('Unknown stack\n  at mod.fn(file.py:4)\nValueError: shared failure');self.assertEqual(parsed['status'],'UNSUPPORTED');self.assertIn('未完成堆栈关联',parsed['issues'][0]);self.assertEqual(parsed['error_text'],'ValueError: shared failure')
        for text in ['Traceback (most recent call last):\n  File "pkg/core.py", line 5, in run','  File "pkg/core.py", line 5, in run\nValueError: shared failure','Traceback (most recent call last):\n File "pkg/core.py" line 5 in run\nValueError: shared failure']:
            self.assertEqual(link.parse(text)['status'],'PARTIAL')
        for extra in ['During handling of the above exception, another exception occurred:','The above exception was the direct cause of the following exception:','ExceptionGroup: many (2 sub-exceptions)','  [Previous line repeated 3 more times]']:
            parsed=link.parse(tb('  File "pkg/core.py", line 5, in run\n'+extra));self.assertEqual(parsed['status'],'UNSUPPORTED');self.assertFalse(parsed['frames']);self.assertIn('未完成堆栈关联',parsed['issues'][0])

    def test_limits_are_visible_and_all_original_candidates_remain(self):
        result,c=run(tb('\n'.join('  File "pkg/core.py", line 5, in run' for _ in range(40))))
        self.assertEqual(result['status'],'LIMITED');self.assertEqual(result['frames_detected'],40);self.assertEqual(len(result['frames']),32);self.assertFalse(result['complete']);self.assertEqual(len(c),2)
        result,_=run(tb('  File "core.py", line 5, in run'),{f'p{i}/core.py':TEXT for i in range(12)})
        self.assertEqual(result['frames'][0]['matches_total'],12);self.assertEqual(len(result['frames'][0]['matches']),5);self.assertTrue(result['frames'][0]['issues'])

    def test_single_message_and_unicode_physical_lines(self):
        self.assertEqual(link.parse('ValueError: shared failure')['status'],'NOT_PROVIDED')
        for newline in ['\n','\r\n','\r']:
            content='marker="a\u2028\u2029\u0085\v\f\x1c\x1d\x1eb"'+newline+TEXT.replace('\n',newline)
            result,_=run(tb('  File "pkg/core.py", line 6, in run\n    raise ValueError("shared failure")').replace('\n',newline),{'pkg/core.py':content})
            self.assertEqual(result['frames'][0]['status'],'MATCHED');self.assertIn('raise',result['frames'][0]['matches'][0]['source_preview'])

    def test_parse_failure_and_lambda_do_not_guess_scope(self):
        d={'files':{'pkg/core.py':record('pkg/core.py','def ???','saved_snapshot')},'items':[]}
        result=link.associate(d,link.parse(tb('  File "pkg/core.py", line 1, in run')),[]);self.assertEqual(result['frames'][0]['status'],'UNVERIFIED')
        result=link.associate(d,link.parse(tb('  File "pkg/core.py", line 1, in run\n    different()')),[]);self.assertEqual(result['frames'][0]['status'],'CONFLICT')
        result,_=run(tb('  File "pkg/core.py", line 2, in <lambda>'),{'pkg/core.py':'def run():\n    return lambda: 1\n'});self.assertEqual(result['frames'][0]['status'],'UNVERIFIED')

    def test_definition_header_does_not_claim_the_defined_function_is_executing(self):
        result,c=run(tb('  File "pkg/core.py", line 3, in <module>\n    def run():'))
        loc=result['frames'][0]['matches'][0]
        self.assertEqual(loc['status'],'UNVERIFIED');self.assertNotIn('scope',loc)
        self.assertTrue(all(not item['trace_frames'] for item in c))

    def test_long_path_and_source_preview_are_bounded(self):
        self.assertEqual(link.path_matches('a'*1025,[])[0],'INVALID')
        content='def run():\n    return "'+'a'*1000+'"\n';result,_=run(tb('  File "pkg/core.py", line 2, in run'),{'pkg/core.py':content});loc=result['frames'][0]['matches'][0];self.assertEqual(len(loc['source_preview']),400);self.assertTrue(loc['source_preview_truncated'])
