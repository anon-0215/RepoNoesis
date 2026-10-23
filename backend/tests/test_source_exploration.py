"""Neutral fixtures only. Target code and author tests are never executed."""
import asyncio
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlencode
from fastapi import FastAPI
from app.database import Database, SourceFileError
from app.exploration_api import create_router
from app.services import source_exploration as service, exploration_sources as source

LIBRARY='''def render(value, flag=False):
    if flag:
        raise ValueError(f"Invalid source input: {value!r}")
    return value

def other():
    raise ValueError("shared message")
def another():
    raise RuntimeError("shared message")
'''
TESTS='''import pytest
from package import render
@pytest.mark.parametrize("value, expected", [("one", "one"), ("two", "two")])
def test_values(value, expected):
    assert render(value) == expected
'''


class ExplorationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.runtime=self.root/'runtime'
        self.db=Database(self.root/'new.sqlite')
        self.files={'package/__init__.py':'from .core import render\n','package/core.py':LIBRARY,'tests/test_values.py':TESTS}
        self.p=self.create('r1',self.files)

    def create(self,revision,files):
        p=self.db.create_project({'repo_url':'fixture','owner':'local','repo':'neutral',
            'default_branch':'main','repository_revision':revision})
        self.db.save_analysis(p,{'files':[],'modules':[]},[{'path':p,'content':t,'size':len(t.encode())} for p,t in files.items()],[],[])
        return p

    def load(self,p=None,rev='r1'):
        return service.analysis(self.db,self.runtime,p or self.p,rev)

    def test_two_repositories_and_versions_cannot_share_candidates_or_pair_ids(self):
        one=self.load(); p2=self.create('r1',self.files);two=self.load(p2)
        self.assertNotEqual(one['items'][0]['id'],two['items'][0]['id'])
        self.assertNotEqual(one['tests']['comparisons'][0]['id'],two['tests']['comparisons'][0]['id'])
        p3=self.create('r2',self.files); three=self.load(p3,'r2')
        self.assertNotEqual(one['items'][0]['id'],three['items'][0]['id'])
        with self.assertRaises(SourceFileError): self.load(self.p,'r2')
        with self.db.connect() as c: c.execute("UPDATE workspace_revisions SET activation_status='superseded' WHERE project_id=?",(self.p,))
        with self.assertRaises(SourceFileError): self.load()

    def test_content_change_invalidates_cache(self):
        before=self.load()
        with self.db.connect() as c:c.execute('UPDATE repo_files SET content=? WHERE project_id=? AND path=?',(LIBRARY.replace('shared message','new message'),self.p,'package/core.py'))
        after=self.load()
        self.assertNotEqual(before['fingerprint'],after['fingerprint'])
        self.assertEqual(service.error_query(after,'shared message')['total'],0)

    def test_error_multiple_none_formatted_and_real_guard(self):
        data=self.load()
        self.assertEqual(service.error_query(data,'shared message')['status'],'MULTIPLE_CANDIDATES')
        self.assertEqual(service.error_query(data,'missing')['status'],'NO_MATCH')
        r=service.error_query(data,"ValueError: Invalid source input: 'x'")
        self.assertEqual(r['status'],'TEMPLATE_CANDIDATE')
        self.assertEqual(r['candidates'][0]['controls'][0]['source'],'if flag:')
        self.assertEqual(r['candidates'][0]['line'],3)

    def test_error_response_size_includes_traceback_and_metadata(self):
        data=self.load();data['coverage']['oversized_diagnostic']='x'*250001
        with self.assertRaises(SourceFileError) as caught:service.error_query(data,'shared message')
        self.assertEqual(caught.exception.code,'exploration_result_too_large')

    def test_missing_unsupported_zero_pairs_and_missing_helpers_are_distinct(self):
        p=self.create('r1',{'package/core.py':LIBRARY})
        self.assertEqual(service.comparisons(self.load(p))['status'],'MATERIALS_MISSING')
        p=self.create('r1',{'tests/test_x.py':'def test_x(fixture):\n    assert fixture == 1\n'})
        data=service.comparisons(self.load(p));self.assertEqual(data['status'],'UNSUPPORTED')
        self.assertIn('fixture',data['unsupported'][0]['reason'])
        p=self.create('r1',{**self.files,'tests/test_values.py':'from package import render\ndef test_x():\n    assert render("one") == "one"\n'})
        self.assertEqual(service.comparisons(self.load(p))['status'],'ZERO_COMPARISONS')
        p=self.create('r1',{'tests/test_values.py':TESTS})
        r=service.comparisons(self.load(p));self.assertEqual(r['status'],'UNSUPPORTED')
        self.assertTrue(r['unsupported'])

    def test_parse_failure_continues_and_reports_test_parse_failure(self):
        p=self.create('r1',{**self.files,'broken.py':'def ???'})
        data=self.load(p)
        self.assertEqual(len(data['parse_errors']),1)
        self.assertTrue(data['items']);self.assertTrue(data['tests']['comparisons'])
        p=self.create('r1',{'tests/test_x.py':'def ???'})
        # The physical test file exists even though parsing failed.
        r=service.comparisons(self.load(p))
        self.assertEqual(r['status'],'PARSE_FAILED')

    def test_pagination_and_detail_have_no_file_bodies(self):
        data=self.load(); result=service.comparisons(data,limit=1)
        self.assertEqual(len(result['pairs']),1)
        self.assertNotIn('scenarios',result['pairs'][0])
        detail=service.comparison(data,result['pairs'][0]['id'])['pair']
        for scene in detail['scenarios']:
            for ref in scene['assertions']+scene['implementation']:
                record=data['files'][ref['source']['path']]
                self.assertEqual(record['content_sha256'],ref['source']['sha256'])
                self.assertGreaterEqual(ref['source']['start'],1)
        self.assertNotIn('"lines":',json.dumps(detail))
        self.assertNotIn('"content":',json.dumps(detail))
        self.assertEqual(service.comparisons(data,query='nothing')['total'],0)
        with self.assertRaises(SourceFileError):service.comparison(data,'missing')

    def test_size_limit_and_no_silent_truncation(self):
        p=self.create('r1',{'huge.py':'x'*(source.MAX_FILE_BYTES+1),'safe.py':LIBRARY})
        data=self.load(p)
        self.assertNotIn('huge.py',data['files'])
        self.assertEqual(data['coverage']['skipped_count'],1)
        self.assertEqual(data['files']['safe.py']['content'],LIBRARY)
        # Actual shorter-than-recorded material must be explicit, never parsed as complete.
        with self.db.connect() as conn:
            conn.execute('UPDATE repo_files SET size=9999 WHERE project_id=? AND path=?',(p,'safe.py'))
        data=self.load(p)
        self.assertNotIn('safe.py',data['files'])
        self.assertTrue(any(x['reason']=='saved_text_shorter_than_recorded_size' for x in data['coverage']['skipped']))

    def test_colon_literal_and_prefix_both_match(self):
        p=self.create('r1',{'x.py':'raise ValueError("status: denied")\n'})
        data=self.load(p)
        self.assertEqual(service.error_query(data,'status: denied')['total'],1)
        self.assertEqual(service.error_query(data,'ValueError: status: denied')['total'],1)

    def test_adjacent_format_slots_are_bounded_and_equivalent(self):
        import ast
        from app.services.error_locator import message_pattern,matches
        pattern=message_pattern(ast.parse('f"Invalid code: {a}{b}{c}{d} fixed"',mode='eval').body)
        self.assertFalse(matches(pattern,'Invalid code: '+'x'*2048+' wrong'))
        self.assertTrue(matches(pattern,'Invalid code: '+'x'*2048+' fixed'))
        self.assertFalse(matches(pattern,'Invalid code: '+'x'*2049+' fixed'))

    def test_parameter_rows_cannot_bypass_global_scene_work_budget(self):
        test='import pytest\nfrom package import render\n@pytest.mark.parametrize("value", '+repr(list(range(128)))+')\ndef test_mass(value):\n'+''.join('    assert render(value) == 1\n' for _ in range(17))
        p=self.create('r1',{**self.files,'tests/test_values.py':test})
        data=self.load(p);inventory=data['tests']['inventory']
        self.assertLessEqual(inventory['scenarios'],2048)
        self.assertTrue(inventory['work_budget_reached'])
        self.assertLessEqual(len(inventory['pairing_limit']['discarded']),100)

    def test_large_literal_is_reported_instead_of_a_huge_unopenable_pair(self):
        test='from package import render\ndef test_a():\n    assert render('+repr('a'*140000)+') == 1\ndef test_b():\n    assert render('+repr('b'*140000)+') == 1\n'
        p=self.create('r1',{**self.files,'tests/test_values.py':test})
        result=service.comparisons(self.load(p))
        self.assertEqual(result['status'],'UNSUPPORTED')
        self.assertIn('8192',result['unsupported'][0]['reason'])
        self.assertLess(len(json.dumps(result)),10000)

    def git(self,root,*args):
        return subprocess.check_output(['git','-C',str(root),*args],stderr=subprocess.DEVNULL).decode().strip()

    def test_supplement_reads_exact_git_objects_not_worktree_and_keeps_qa_snapshot(self):
        repo=self.root/'repo';repo.mkdir()
        for rel,text in self.files.items():
            f=repo/rel;f.parent.mkdir(exist_ok=True,parents=True);f.write_text(text,encoding='utf-8',newline='\n')
        self.git(repo,'init');self.git(repo,'add','.')
        self.git(repo,'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-m','static fixture')
        revision=self.git(repo,'rev-parse','HEAD')
        # A local Git replace ref must not spoof the exact commit's file body.
        old_blob=self.git(repo,'rev-parse',revision+':tests/test_values.py')
        replacement=repo/'replacement.txt';replacement.write_text('raise RuntimeError("REPLACED")',encoding='utf-8')
        new_blob=self.git(repo,'hash-object','-w',str(replacement))
        self.git(repo,'replace',old_blob,new_blob)
        p=self.create(revision,{'package/core.py':LIBRARY.replace('\n','\r\n')})
        (repo/'tests/test_values.py').write_text('raise RuntimeError("WRONG VERSION")',encoding='utf-8')
        before=self.db.get_bundle(p)
        data=source.supplement(self.db,self.runtime,p,revision,str(repo))
        self.assertEqual(data['files']['tests/test_values.py']['content'],TESTS)
        self.assertEqual(data['files']['package/core.py']['content'],LIBRARY.replace('\n','\r\n'))
        self.assertEqual(len(data['coverage']['supplement_origin']['eol_equivalent']),1)
        self.assertEqual(self.db.get_bundle(p),before)
        self.assertEqual(source.read_material_file(self.db,self.runtime,p,revision,'tests/test_values.py')['content'],TESTS)
        self.assertTrue(self.load(p,revision)['tests']['comparisons'])
        # An incomplete existing row remains reserved for the existing reader.
        # A/B must not cite a Git replacement the reader would not return.
        with self.db.connect() as conn:
            conn.execute('UPDATE repo_files SET content=?,size=? WHERE project_id=? AND path=?',
                         ('# partial snapshot\n',len(LIBRARY),p,'package/core.py'))
        partial=source.read_sources(self.db,self.runtime,p,revision)
        self.assertNotIn('package/core.py',partial['files'])
        self.assertTrue(any(x['reason']=='saved_path_unavailable_not_shadowed' for x in partial['coverage']['skipped']))
        self.assertEqual(service.error_query(self.load(p,revision),'shared message')['total'],0)
        with self.assertRaises(SourceFileError):source.supplement(self.db,self.runtime,p,'0'*40,str(repo))
        destination=source.material_path(self.runtime,data['identity']);blob=json.loads(destination.read_text(encoding='utf-8'))
        blob['files']['tests/test_values.py']['content']='changed';destination.write_text(json.dumps(blob),encoding='utf-8')
        with self.assertRaises(SourceFileError):source.read_sources(self.db,self.runtime,p,revision)

    def test_duplicate_saved_paths_are_not_static_evidence(self):
        with self.db.connect() as conn:
            conn.execute('INSERT INTO repo_files(project_id,path,content,size) VALUES (?,?,?,?)',
                         (self.p,'package/core.py',LIBRARY,len(LIBRARY)))
        data=self.load()
        self.assertNotIn('package/core.py',data['files'])
        self.assertEqual(service.error_query(data,'shared message')['total'],0)

    def test_api_schema_input_boundaries_and_identity(self):
        app=FastAPI();app.include_router(create_router(lambda:self.db,self.runtime))
        async def call(route,body=None,query=None):
            messages=[]
            async def send(m):messages.append(m)
            async def receive():return {'type':'http.request','body':json.dumps(body).encode() if body else b'','more_body':False}
            await app({'type':'http','http_version':'1.1','method':'POST' if body else 'GET','scheme':'http',
                'path':f'/api/projects/{self.p}/exploration'+route,'query_string':urlencode(query or {}).encode(),
                'headers':[(b'content-type',b'application/json')],'server':('test',80),'client':('test',1),'root_path':''},receive,send)
            return next(m['status'] for m in messages if m['type']=='http.response.start'),json.loads(b''.join(m.get('body',b'') for m in messages if m['type']=='http.response.body'))
        status,data=asyncio.run(call('/errors',{'revision':'r1','error':'shared message'}))
        self.assertEqual(status,200);self.assertEqual(data['identity']['project_id'],self.p)
        status,trace=asyncio.run(call('/errors',{'revision':'r1','error':'Traceback (most recent call last):\n  File "package/core.py", line 3, in render\nValueError: Invalid source input: x'}))
        self.assertEqual(status,200);self.assertEqual(trace['traceback']['frames'][0]['status'],'MATCHED')
        self.assertEqual(trace['candidates'][0]['trace_frames'],[1])
        self.assertEqual(asyncio.run(call('/errors',{'revision':'r2','error':'Traceback (most recent call last):\n  File "package/core.py", line 3, in render\nValueError: Invalid source input: x'}))[0],409)
        self.assertEqual(asyncio.run(call('/errors',{'revision':'r1','error':'x'*4097}))[0],422)
        self.assertEqual(asyncio.run(call('/comparisons',query={'revision':'r2'}))[0],409)
        self.assertEqual(asyncio.run(call('/comparisons',query={'revision':'r1','limit':51}))[0],422)


if __name__=='__main__':unittest.main()
