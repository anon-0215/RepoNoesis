"""Launcher regressions using owned ephemeral ports and inert loopback fixtures."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import os,json,shutil,socket,subprocess,sys,tempfile,threading,time,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PS=Path(os.environ.get('SystemRoot','C:/Windows'))/'System32/WindowsPowerShell/v1.0/powershell.exe'

def free_port():
 with socket.socket() as sock:sock.bind(('127.0.0.1',0));return sock.getsockname()[1]

@contextmanager
def unrelated(port):
 class Handler(BaseHTTPRequestHandler):
  def log_message(self,*a):pass
  def do_GET(self):
   data=b'unrelated old service';self.send_response(200);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('X-RepoNoesis-Startup-Id',os.environ.get('REPONOESIS_STARTUP_ID',''));self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
 server=ThreadingHTTPServer(('127.0.0.1',port),Handler);thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.02});thread.start()
 try:yield server
 finally:server.shutdown();server.server_close();thread.join(2)

FAKE=r'''import os,sys,json,time
from pathlib import Path
from http.server import BaseHTTPRequestHandler,HTTPServer
role='backend' if '--port' not in sys.argv else 'frontend'
Path(role+'-started.pid').write_text(str(os.getpid()))
if os.environ.get('FIXTURE_EXIT')==role:raise SystemExit(42)
time.sleep(float(os.environ.get('FIXTURE_DELAY','0.1')))
port=int(os.environ.get('BACKEND_PORT','8000')) if role=='backend' else int(sys.argv[sys.argv.index('--port')+1])
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*a):pass
 def do_GET(self):
  identity='wrong-instance' if os.environ.get('FIXTURE_BAD_ID')==role else os.environ.get('REPONOESIS_STARTUP_ID','')
  data=json.dumps({'fixture':role,'startup_instance_id':identity}).encode();self.send_response(200);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('X-RepoNoesis-Startup-Id',identity);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
server=HTTPServer(('127.0.0.1',port),Handler)
if os.environ.get('FIXTURE_EXIT_AFTER_HEALTH')=='1' and role=='backend':server.handle_request();server.server_close()
else:server.serve_forever()
'''

@unittest.skipUnless(os.name=='nt','Windows launcher')
class StartupPortIdentityTests(unittest.TestCase):
 def fixture(self,ports,**overrides):
  temp=tempfile.TemporaryDirectory(prefix='owned-launcher-',dir=os.environ.get('REPONOESIS_STARTUP_TEST_TMP'));self.addCleanup(temp.cleanup);root=Path(temp.name)
  for p in ['scripts','backend/app','frontend/node_modules/vite/bin']:(root/p).mkdir(parents=True,exist_ok=True)
  script=(ROOT/'scripts/start_local.ps1').read_text(encoding='utf-8-sig')
  supports_ports='[int]$BackendPort' in script
  # Baseline uses fixed literals. Substitute only test-owned ports, preserving logic.
  if not supports_ports:script=script.replace('8000',str(ports[0])).replace('5173',str(ports[1]))
  (root/'scripts/start_local.ps1').write_text(script,encoding='utf-8-sig');shutil.copy2(ROOT/'scripts/resolve_runtime.ps1',root/'scripts/resolve_runtime.ps1')
  (root/'backend/app/__init__.py').touch();(root/'backend/app/run_server.py').write_text(FAKE,encoding='utf-8');(root/'frontend/node_modules/vite/bin/vite.js').write_text(FAKE,encoding='utf-8')
  env=os.environ.copy();env.pop('PYTHONPATH',None);env.update({'REPONOESIS_PYTHON':sys.executable,'REPONOESIS_NODE':sys.executable,'BACKEND_PORT':str(ports[0]),'REPONOESIS_SKIP_ENV_FILE':'1',**overrides})
  cmd=[str(PS),'-NoProfile','-ExecutionPolicy','Bypass','-File',str(root/'scripts/start_local.ps1'),'-Headless','-HealthTimeoutSeconds','8']
  if supports_ports:cmd+=['-BackendPort',str(ports[0]),'-FrontendPort',str(ports[1])]
  def clean():
   for role in ['backend','frontend']:
    marker=root/role/(role+'-started.pid')
    if marker.exists():
     result=subprocess.run(['taskkill','/PID',marker.read_text(),'/T','/F'],capture_output=True)
     # Windows taskkill returns before cwd/pipe handles are fully released.
     import ctypes
     kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.OpenProcess.restype=ctypes.c_void_p;kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong];kernel.CloseHandle.argtypes=[ctypes.c_void_p]
     handle=kernel.OpenProcess(0x00100000,False,int(marker.read_text()))
     if handle:
      try:self.assertEqual(kernel.WaitForSingleObject(handle,3000),0,'owned fixture child still alive: '+result.stderr.decode(errors='replace'))
      finally:kernel.CloseHandle(handle)
  self.addCleanup(clean)
  return root,env,cmd
 def launch(self,ports,**env):
  root,env,cmd=self.fixture(ports,**env)
  with (root/'launcher.log').open('w',encoding='utf-8') as log:
   result=subprocess.run(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,timeout=25,creationflags=subprocess.CREATE_NO_WINDOW)
  text=(root/'launcher.log').read_text(encoding='utf-8',errors='replace');print(json.dumps({'case':self._testMethodName,'returncode':result.returncode,'ports':ports,'output':text}))
  return root,result,text
 def test_unrelated_backend_200_is_not_accepted(self):
  ports=(free_port(),free_port())
  with unrelated(ports[0]):
   root,result,text=self.launch(ports,FIXTURE_DELAY='0.8');self.assertNotEqual(result.returncode,0,text);self.assertNotIn('backend and frontend are ready',text.lower())
   import urllib.request
   self.assertEqual(urllib.request.urlopen(f'http://127.0.0.1:{ports[0]}').read(),b'unrelated old service')
 def test_occupied_frontend_is_not_accepted(self):
  ports=(free_port(),free_port())
  with unrelated(ports[1]):
   root,result,text=self.launch(ports,FIXTURE_DELAY='0.8');self.assertNotEqual(result.returncode,0,text);self.assertNotIn('backend and frontend are ready',text.lower())
 def test_immediate_child_failure_never_reports_ready(self):
  root,result,text=self.launch((free_port(),free_port()),FIXTURE_EXIT='backend');self.assertNotEqual(result.returncode,0,text);self.assertFalse((root/'frontend/frontend-started.pid').exists());self.assertNotIn('backend and frontend are ready',text.lower())
 def assert_child_stopped(self,root,role):
  import ctypes
  marker=root/role/(role+'-started.pid');self.assertTrue(marker.exists())
  kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.OpenProcess.restype=ctypes.c_void_p;kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong];kernel.CloseHandle.argtypes=[ctypes.c_void_p]
  handle=kernel.OpenProcess(0x00100000,False,int(marker.read_text()))
  if handle:
   try:self.assertEqual(kernel.WaitForSingleObject(handle,0),0,'launcher left its own '+role+' alive')
   finally:kernel.CloseHandle(handle)
 def test_frontend_failure_reaps_started_backend(self):
  root,result,text=self.launch((free_port(),free_port()),FIXTURE_EXIT='frontend');self.assertNotEqual(result.returncode,0,text);self.assert_child_stopped(root,'backend');self.assert_child_stopped(root,'frontend')
 def test_frontend_wrong_instance_200_is_rejected_and_reaped(self):
  root,result,text=self.launch((free_port(),free_port()),FIXTURE_BAD_ID='frontend');self.assertNotEqual(result.returncode,0,text);self.assertNotIn('backend and frontend are ready',text.lower());self.assert_child_stopped(root,'backend');self.assert_child_stopped(root,'frontend')
 def test_normal_start_and_owned_exit(self):
  root,result,text=self.launch((free_port(),free_port()));self.assertEqual(result.returncode,0,text);self.assertIn('backend and frontend are ready',text.lower())
 def test_child_exiting_after_200_is_not_accepted(self):
  root,result,text=self.launch((free_port(),free_port()),FIXTURE_EXIT_AFTER_HEALTH='1');self.assertNotEqual(result.returncode,0,text);self.assertNotIn('backend and frontend are ready',text.lower())
 def test_port_taken_after_precheck_is_not_accepted(self):
  ports=(free_port(),free_port());root,env,cmd=self.fixture(ports,FIXTURE_DELAY='2');stop=threading.Event();ready=threading.Event()
  def race():
   marker=root/'backend/backend-started.pid'
   while not marker.exists() and not stop.wait(.01):pass
   if stop.is_set():return
   with unrelated(ports[0]):ready.set();stop.wait(20)
  thread=threading.Thread(target=race);thread.start()
  try:
   with (root/'launcher.log').open('w',encoding='utf-8') as log:result=subprocess.run(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,timeout=25,creationflags=subprocess.CREATE_NO_WINDOW)
   text=(root/'launcher.log').read_text(encoding='utf-8',errors='replace');self.assertTrue(ready.is_set(),text);self.assertNotEqual(result.returncode,0,text);self.assertNotIn('backend and frontend are ready',text.lower())
  finally:stop.set();thread.join(3)
