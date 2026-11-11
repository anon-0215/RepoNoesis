"""Small application assembly only; never copies a Python runtime or opens UI."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build


class ApplicationBuildTests(unittest.TestCase):
    def test_frontend_flags_are_exact_and_only_apply_to_build_children(self):
        import build_frontend
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {
            'VITE_API_BASE_URL': 'https://old.invalid', 'VITE_DESKTOP': 'false'}), \
                patch.object(build_frontend.subprocess, 'run') as run:
            build_frontend.main(['--output', temporary, '--node', 'selected-node'])
            self.assertEqual(run.call_count, 2)
            for call in run.call_args_list:
                self.assertEqual(call.kwargs['env']['VITE_API_BASE_URL'], '')
                self.assertEqual(call.kwargs['env']['VITE_DESKTOP'], 'true')
                self.assertEqual(call.args[0][0], 'selected-node')
            self.assertEqual(os.environ['VITE_API_BASE_URL'], 'https://old.invalid')
            self.assertEqual(os.environ['VITE_DESKTOP'], 'false')

    def test_build_entry_with_minimal_runtime_uses_complete_application_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'app'
            for name in ('python.exe', 'pythonw.exe', 'python312.dll', 'Scripts/conda-unpack-script.py',
                         'Lib/site-packages/torch/__init__.py', 'Lib/site-packages/webview/__init__.py',
                         'Lib/site-packages/pythonnet/__init__.py', 'Lib/site-packages/clr_loader/__init__.py'):
                path = target / 'runtime' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'fake runtime fixture - never execute')
            frontend = Path(temporary) / 'dist'
            frontend.mkdir()
            (frontend / 'index.html').write_text('desktop fixture', encoding='utf-8')
            build.main(['--output', str(target), '--tools', temporary, '--reuse-runtime',
                        '--frontend-dist', str(frontend)])
            self.assertTrue((target / 'desktop/desktop_host.py').is_file())
            self.assertTrue((target / 'desktop/shell.html').is_file())
            self.assertEqual(json.loads((target / 'BUILD.json').read_text())['conda_pack'], 'reused')

    def test_initial_build_copies_every_file_used_by_window_entry(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'new-app'
            frontend = Path(temporary) / 'desktop-dist'
            frontend.mkdir()
            (frontend / 'index.html').write_text('built desktop UI', encoding='utf-8')
            build.copy_application(HERE.parent, target, frontend_dist=frontend)
            for name in ('desktop_host.py', 'gui_settings.py', 'shell.html', 'launcher.py', 'server.py', 'acceptance.py'):
                self.assertEqual((target / 'desktop' / name).read_bytes(), (HERE / name).read_bytes())
            self.assertIn('desktop_host.py', (target / 'Start.cmd').read_text())
            self.assertTrue((target / '源鉴.vbs').is_file())
            self.assertEqual((target / 'frontend/index.html').read_text(), 'built desktop UI')
            self.assertTrue((target / 'backend/app/main.py').is_file())
            self.assertFalse((target / '.env').exists())
            self.assertFalse((target / 'user-data').exists())
            self.assertEqual(json.loads((target / 'desktop-config.json').read_text())['hyde_enabled'], False)

    def test_refresh_requires_explicit_replace_and_preserves_config_and_runtime(self):
        import refresh_app
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'app'
            (target / 'runtime').mkdir(parents=True)
            runtime = target / 'runtime/python.exe'
            runtime.write_bytes(b'untouched fake runtime')
            config = target / 'desktop-config.json'
            config.write_text('{"hyde_enabled":true,"custom":"keep"}', encoding='utf-8')
            before = config.read_bytes()
            with self.assertRaises(SystemExit), patch.object(build, 'copy_application') as copy:
                refresh_app.main(['--output', str(target)])
            copy.assert_not_called()
            frontend = Path(temporary) / 'dist'
            frontend.mkdir()
            (frontend / 'index.html').write_text('desktop', encoding='utf-8')
            refresh_app.main(['--output', str(target), '--frontend-dist', str(frontend), '--replace-app'])
            self.assertEqual(config.read_bytes(), before)
            self.assertEqual(runtime.read_bytes(), b'untouched fake runtime')
            self.assertTrue((target / 'desktop/desktop_host.py').is_file())

    def test_missing_frontend_fails_before_creating_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'new-app'
            with self.assertRaises(RuntimeError):
                build.copy_application(HERE.parent, target, frontend_dist=Path(temporary) / 'missing')
            self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
