import sys
from pathlib import Path
import tempfile
import threading
import types
import asyncio
import json
import unittest
from unittest.mock import Mock,patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import desktop_host as host
from settings_lifecycle import SettingsRestartGate, install
from fastapi import FastAPI,HTTPException


def post(app, path, headers, body):
    async def request():
        messages=[]
        async def receive():return {'type':'http.request','body':json.dumps(body).encode(),'more_body':False}
        async def send(message):messages.append(message)
        await app({'type':'http','method':'POST','path':path,'raw_path':path.encode(),'root_path':'',
            'query_string':b'','headers':[(k.lower().encode(),v.encode()) for k,v in {'content-type':'application/json',**headers}.items()],
            'scheme':'http','http_version':'1.1','server':('127.0.0.1',18771),'client':('127.0.0.1',1)},receive,send)
        return messages[0]['status']
    return asyncio.run(request())


class RestartTests(unittest.TestCase):
    def test_actual_worker_blocks_restart_until_finished_including_detached_stream_worker(self):
        gate=SettingsRestartGate();entered=threading.Event();finish=threading.Event()
        def work():entered.set();finish.wait(3)
        worker=threading.Thread(target=gate.wrap(work));worker.start()
        try:
            self.assertTrue(entered.wait(1))
            self.assertFalse(gate.acquire('a'))  # HTTP stream can be gone; core worker still counted.
        finally:finish.set();worker.join(3)
        self.assertEqual(gate.active,0)
        self.assertTrue(gate.acquire('a'))
        with self.assertRaises(HTTPException) as error:gate.wrap(lambda:None)()
        self.assertEqual(error.exception.status_code,503)
        gate.release('a');self.assertEqual(gate.wrap(lambda:42)(),42)

    def test_exception_releases_worker_and_restart_endpoint_checks_instance(self):
        app=FastAPI();main=types.SimpleNamespace(_execute_ask=Mock(return_value='done'))
        gate=install(app,main,'owned-instance')
        url='/api/desktop/settings-restart';headers={'X-RepoNoesis-Instance':'owned-instance'}
        body={'action':'acquire','lease_id':'a'*32}
        self.assertEqual(post(app,url,{}, body),409)
        self.assertEqual(post(app,url,headers, body),200)
        with self.assertRaises(HTTPException):main._execute_ask()
        self.assertEqual(post(app,url,headers, {'action':'release','lease_id':'a'*32}),200)
        self.assertEqual(main._execute_ask(),'done')
        def fail():raise RuntimeError('fixture')
        with self.assertRaises(RuntimeError):gate.wrap(fail)()
        self.assertEqual(gate.active,0)

    def test_busy_save_writes_nothing_and_does_not_restart_or_touch_key(self):
        with tempfile.TemporaryDirectory() as folder:
            c=host.Controller(Path(folder),{})
            c.store.save({'api_key':'fixture-only','hyde_enabled':False})
            before=c.store.path.read_bytes();c.child=Mock();c.child.pid=123;c.child.poll.return_value=None
            with patch.object(c,'_settings_restart',return_value=False):
                result=c.save_settings({'hyde_enabled':True})
            self.assertFalse(result['ok']);self.assertIn('未保存',result['message'])
            self.assertEqual(before,c.store.path.read_bytes());self.assertFalse(c.reload_event.is_set())

    def test_safe_save_leases_before_write_and_failure_releases_lease(self):
        with tempfile.TemporaryDirectory() as folder:
            c=host.Controller(Path(folder),{});c.store.save({'hyde_enabled':False})
            before=c.store.path.read_bytes();c.child=Mock();c.child.pid=123;c.child.poll.return_value=None
            with patch.object(c,'_settings_restart',return_value=True) as lease,patch.object(host,'atomic_json',side_effect=OSError()):
                result=c.save_settings({'hyde_enabled':True})
                self.assertEqual([c.args[0] for c in lease.call_args_list], ['acquire','release'])
                self.assertEqual(lease.call_args_list[0].args[1],lease.call_args_list[1].args[1])
            self.assertFalse(result['ok']);self.assertEqual(before,c.store.path.read_bytes())
            self.assertFalse(c.reload_event.is_set())
            with patch.object(c,'_settings_restart',return_value=True) as lease:
                result=c.save_settings({'hyde_enabled':True})
            self.assertTrue(result['ok']);self.assertTrue(result['settings']['hyde_enabled'])
            self.assertTrue(c.reload_event.is_set());self.assertEqual(c.state['phase'],'restarting')
            self.assertEqual([c.args[0] for c in lease.call_args_list],['acquire','acquire'])

    def test_lost_acquire_response_is_compensated_without_changing_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            c=host.Controller(Path(folder),{});c.store.save({'hyde_enabled':False})
            before=c.store.path.read_bytes();c.child=Mock();c.child.poll.return_value=None
            gate=SettingsRestartGate()
            def transport(action,lease):
                if action=='acquire':
                    self.assertTrue(gate.acquire(lease));return False  # Server succeeded, response lost.
                gate.release(lease);return True
            with patch.object(c,'_settings_restart',side_effect=transport):
                result=c.save_settings({'hyde_enabled':True})
            self.assertFalse(result['ok']);self.assertEqual(before,c.store.path.read_bytes())
            self.assertTrue(gate.acquire('other'))

    def test_lost_release_expires_and_different_owner_cannot_release(self):
        now=[0];gate=SettingsRestartGate(clock=lambda:now[0])
        self.assertTrue(gate.acquire('a'));gate.release('other')
        self.assertFalse(gate.acquire('b'))
        now[0]=31
        self.assertEqual(gate.wrap(lambda:42)(),42)
        self.assertTrue(gate.acquire('b'))

    def test_expired_lease_after_write_restores_settings_and_does_not_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            c=host.Controller(Path(folder),{});c.store.save({'hyde_enabled':False})
            before=c.store.load();c.child=Mock();c.child.poll.return_value=None
            with patch.object(c,'_settings_restart',side_effect=[True,False,True]):
                result=c.save_settings({'hyde_enabled':True})
            self.assertFalse(result['ok']);self.assertIn('原设置已恢复',result['message'])
            self.assertEqual(c.store.load(),before);self.assertFalse(c.reload_event.is_set())

    def test_failed_restore_after_successful_write_does_not_claim_unsaved(self):
        with tempfile.TemporaryDirectory() as folder:
            c=host.Controller(Path(folder),{});c.store.save({'hyde_enabled':False})
            c.child=Mock();c.child.poll.return_value=None
            original=host.atomic_json;writes=[]
            def write(path,value):
                writes.append(path)
                if len(writes)==2:raise OSError('fixture-only')
                return original(path,value)
            with patch.object(c,'_settings_restart',side_effect=[True,False,True]),patch.object(host,'atomic_json',side_effect=write):
                result=c.save_settings({'hyde_enabled':True})
            self.assertFalse(result['ok']);self.assertIn('可能已保存但未应用',result['message'])
            self.assertTrue(c.store.load()['hyde_enabled']);self.assertFalse(c.reload_event.is_set())


if __name__=='__main__':unittest.main()
