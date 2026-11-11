import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

package = Path(__file__).resolve().parents[1]
if __name__ == '__main__' and '--package' in sys.argv:
    index = sys.argv.index('--package')
    package = Path(sys.argv[index + 1]).resolve()
    del sys.argv[index:index + 2]
print('Desktop host under test:', package)
spec = importlib.util.spec_from_file_location("packaged_desktop_host", package / "desktop/desktop_host.py")
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)


class DesktopHostTests(unittest.TestCase):
    def test_graceful_stop_closes_only_owned_child_and_job(self):
        with tempfile.TemporaryDirectory() as folder:
            controller = host.Controller(Path(folder), {})
            child, job = Mock(), Mock()
            child.pid = 123
            child.returncode = 0
            child.poll.return_value = None
            controller.child, controller.job = child, job
            controller._stop_backend()
            child.stdin.write.assert_called_once_with(b'stop\n')
            child.stdin.flush.assert_called_once()
            child.stdin.close.assert_called_once()
            job.close.assert_called_once()
            self.assertIsNone(controller.child)
            self.assertIsNone(controller.job)
            self.assertTrue((Path(folder) / 'logs/desktop-closeout.json').is_file())

    def test_stop_timeout_still_closes_owned_job(self):
        with tempfile.TemporaryDirectory() as folder:
            controller = host.Controller(Path(folder), {})
            child, job = Mock(), Mock()
            child.pid, child.returncode = 123, 1
            child.poll.return_value = None
            child.wait.side_effect = [host.subprocess.TimeoutExpired('fake backend', 12), 1]
            controller.child, controller.job = child, job
            controller._stop_backend()
            job.close.assert_called_once()
            self.assertEqual(child.wait.call_count, 2)

    def test_unchanged_save_does_not_restart_model_again(self):
        with tempfile.TemporaryDirectory() as folder:
            controller = host.Controller(Path(folder), {})
            settings = {"model_directory": "missing-model", "git_directory": "missing-git"}
            self.assertTrue(controller.save_settings(settings)["changed"])
            self.assertTrue(controller.reload_event.is_set())
            controller.reload_event.clear()
            self.assertFalse(controller.save_settings(settings)["changed"])
            self.assertFalse(controller.reload_event.is_set())

    def test_single_instance_lock_is_exclusive_and_released(self):
        with tempfile.TemporaryDirectory() as folder:
            first = host.InstanceGuard(Path(folder))
            try:
                with self.assertRaises(RuntimeError):
                    host.InstanceGuard(Path(folder))
            finally:
                first.close()
            second = host.InstanceGuard(Path(folder))
            second.close()

    def test_bridge_does_not_expose_store_or_runtime_secret(self):
        with tempfile.TemporaryDirectory() as folder:
            controller = host.Controller(Path(folder), {})
            controller.store.save({"api_key": "test-only-placeholder"})
            api = host.WindowApi(controller)
            public = [name for name in dir(api) if not name.startswith("_")]
            self.assertEqual(sorted(public), sorted(["get_status", "get_settings", "save_settings", "choose_folder", "open_data_directory", "retry"]))
            self.assertTrue(api.get_settings()["key_saved"])
            self.assertNotIn("api_key", api.get_settings())
            self.assertNotIn("api_key_dpapi", api.get_settings())


if __name__ == "__main__":
    unittest.main()
