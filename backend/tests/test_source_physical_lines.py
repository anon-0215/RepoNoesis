"""Static neutral fixtures; analyzed source is never executed."""
import ast,hashlib,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from app.source_lines import source_lines
from app.database import Database,CODE_CHUNKER_VERSION,LEGACY_CODE_CHUNKER_VERSION
from app.services.code_chunker import extract_python_code_chunks
from app.services.error_locator import Extractor
from app.services.evidence import EvidenceBuilder,CitationValidator
from app.services import source_exploration
from app.services.agent_contracts import AgentLimits,CancellationToken,ToolCall
from app.services.agent_tools import build_tool_context,build_m2_tool_registry,EvidenceStore
from tests.m1_helpers import disabled_embedding_service
from tests import test_source_file_api as source_api

SEPARATORS=['\v','\f','\x1c','\x1d','\x1e','\x85','\u2028','\u2029']
def fixture(sep='\u2028',newline='\n',trailing=True):
 lines=['def before():','    return 0',f'marker = "left{sep}right"',f'# note {sep} still same comment','def after(value, fail=False):','    try:','        if fail:','            raise ValueError("neutral bad")','        return value','    except KeyError:','        return None']
 return newline.join(lines)+(newline if trailing else '')

class PhysicalSourceLinesTests(unittest.TestCase):
 def test_ast_chunk_and_local_guard_use_physical_lines(self):
  for sep in SEPARATORS:
   for newline in ['\n','\r\n','\r']:
    for trailing in [True,False]:
     with self.subTest(separator=repr(sep),newline=repr(newline),trailing=trailing):
      text=fixture(sep,newline,trailing);lines=source_lines(text,keepends=True);tree=ast.parse(text)
      self.assertEqual(len(lines),11);self.assertEqual(''.join(lines),text)
      chunks=extract_python_code_chunks('neutral.py',text,'rev').chunks;chunk=next(c for c in chunks if c.symbol_name=='after')
      self.assertEqual((chunk.start_line,chunk.end_line),(5,11));self.assertEqual(chunk.content,''.join(lines[4:11]))
      self.assertEqual(chunk.content_hash,hashlib.sha256(chunk.content.encode()).hexdigest())
      ext=Extractor(text,'neutral','neutral.py','rev',hashlib.sha256(text.encode()).hexdigest());ext.walk(tree)
      item=ext.items[0];self.assertEqual(item['line'],8);self.assertIn('raise ValueError',lines[item['line']-1]);self.assertEqual(item['controls'][0]['source'],'try:')
 def test_empty_and_final_line_rules(self):
  for text,expected in [('',[]),('\n',['']),('a\r\nb\r\n',['a','b']),('a\rb\n\n',['a','b','']),('x\u2028',['x\u2028'])]:self.assertEqual(source_lines(text),expected)
 def make_db(self):
  temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup);root=Path(temporary.name);db=Database(root/'physical.sqlite')
  text=fixture();tests='marker="a\u2028b"\nimport pytest\nfrom neutral import after\n@pytest.mark.parametrize("value, expected", [("one", "one"), ("two", "two")])\ndef test_values(value, expected):\n    assert after(value) == expected\n'
  files={'neutral.py':text,'tests/test_values.py':tests};p=db.create_project({'repo_url':'fixture','owner':'local','repo':'physical','default_branch':'main','repository_revision':'rev'})
  chunks=[c.to_dict() for path,t in files.items() for c in extract_python_code_chunks(path,t,'rev').chunks]
  db.save_analysis(p,{'files':[],'modules':[]},[{'path':path,'content':t,'size':len(t.encode()),'language':'Python'} for path,t in files.items()],[],chunks)
  return root,db,p,text,tests
 def test_saved_ast_evidence_tool_and_full_file_api_agree(self):
  root,db,p,text,tests=self.make_db();original_hash=hashlib.sha256(text.encode()).hexdigest();bundle=db.get_bundle(p)
  chunk=next(c for c in db.get_code_chunks(p) if c['symbol_name']=='after');evidence=EvidenceBuilder(10000).build_from_code_chunks([chunk],bundle['project'])[0]
  self.assertEqual(CitationValidator(db).validate(evidence).validation_status,'valid');self.assertEqual(evidence.excerpt,chunk['content'])
  limits=AgentLimits();ctx=build_tool_context(request_id='physical',bundle=bundle,database=db,embedding_service=disabled_embedding_service(),evidence_store=EvidenceStore(),limits=limits,cancellation=CancellationToken(),deadline_monotonic=time.monotonic()+30)
  call=ToolCall(call_id='C1',step_id='S1',tool_name='read_source',tool_version='1',parameters={'path':'neutral.py','start_line':8,'end_line':8},timeout_ms=1000,budget={'max_results':20,'max_bytes':65536})
  observation=build_m2_tool_registry(limits).execute(ctx,call);self.assertEqual(observation.status,'succeeded',observation.error);self.assertEqual(observation.structured_results['content'],source_lines(text,True)[7])
  import app.main as main
  with patch.object(main,'db',db):
   self.main=main
   response=source_api.SourceFileApiTests.http_get(self,f'/api/projects/{p}/source-file',{'revision':'rev','path':'neutral.py'})
  body=response.json();self.assertEqual(response.status_code,200);self.assertEqual(body['content'],text);self.assertEqual(body['total_lines'],11);self.assertEqual(body['content_sha256'],original_hash);self.assertIn('raise ValueError',source_lines(body['content'])[7])
  data=source_exploration.analysis(db,root/'materials',p,'rev');result=source_exploration.error_query(data,'neutral bad');self.assertEqual(result['candidates'][0]['line'],8)
  pairs=data['tests']['comparisons'];self.assertTrue(pairs)
  for scene in pairs[0]['scenarios']:
   for entry in scene['assertions']:self.assertIn('assert after(value)',source_lines(tests)[entry['source']['start']-1])
 def test_old_corrupt_chunk_is_rejected_without_rewriting_source(self):
  root,db,p,text,_=self.make_db();chunk=next(c for c in db.get_code_chunks(p) if c['symbol_name']=='after')
  wrong=''.join(text.splitlines(keepends=True)[chunk['start_line']-1:chunk['end_line']]);self.assertNotEqual(wrong,chunk['content'])
  with db.connect() as conn:conn.execute('UPDATE code_chunks SET content=?,content_hash=? WHERE id=?',(wrong,hashlib.sha256(wrong.encode()).hexdigest(),chunk['id']))
  old=next(c for c in db.get_code_chunks(p) if c['id']==chunk['id']);e=EvidenceBuilder().build_from_code_chunks([old],db.get_bundle(p)['project'])[0]
  validated=CitationValidator(db).validate(e);self.assertEqual(validated.validation_status,'invalid');self.assertIn('stored source no longer matches',validated.invalid_reason)
  with db.connect() as conn:self.assertEqual(conn.execute('SELECT content FROM repo_files WHERE project_id=? AND path=?',(p,'neutral.py')).fetchone()[0],text)
 def test_new_version_marker_does_not_relabel_old_unknown_chunks(self):
  root,db,p,text,_=self.make_db()
  with db.connect() as conn:
   self.assertEqual(conn.execute('SELECT chunker_version FROM workspace_revisions WHERE project_id=?',(p,)).fetchone()[0],CODE_CHUNKER_VERSION)
   conn.execute("UPDATE workspace_revisions SET chunker_version='' WHERE project_id=?",(p,))
  before=db.get_code_chunks(p);db=Database(db.path)
  with db.connect() as conn:self.assertEqual(conn.execute('SELECT chunker_version FROM workspace_revisions WHERE project_id=?',(p,)).fetchone()[0],LEGACY_CODE_CHUNKER_VERSION)
  self.assertEqual(db.get_code_chunks(p),before)
