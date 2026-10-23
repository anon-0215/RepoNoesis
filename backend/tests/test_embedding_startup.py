import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from app.services.embedding_service import (
    EmbeddingService, EmbeddingModelLoadError, SemanticCapacityUnavailable,
)
from tests.test_embedding_service import FakeEmbeddingBackend, _settings


class EmbeddingStartupTests(unittest.TestCase):
    def wait_for(self, predicate, timeout=2):
        deadline = time.monotonic() + timeout
        wake = threading.Event()
        while not predicate():
            remaining = deadline - time.monotonic()
            self.assertGreater(remaining, 0, 'cleanup condition did not complete')
            wake.wait(min(.01, remaining))

    def wait_for_idle(self, service, *, unloaded=False):
        self.wait_for(lambda: service.runtime_status()['active_tasks'] == 0
                      and (not unloaded or service._backend is None))

    def service(self, backend):
        service = EmbeddingService(_settings(), backend_factory=lambda: backend,
                                   cuda_available=lambda: False)
        def cleanup():
            service.close()
            self.wait_for_idle(service, unloaded=True)
        self.addCleanup(cleanup)
        return service

    def test_startup_is_single_flight_and_ready_after_encoding(self):
        entered, release = threading.Event(), threading.Event()
        class Backend(FakeEmbeddingBackend):
            def encode(self, *args, **kwargs):
                entered.set()
                release.wait(2)
                return super().encode(*args, **kwargs)
        backend = Backend()
        service = self.service(backend)
        future = service.start_preparation()
        try:
            self.assertTrue(entered.wait(1))
            self.assertIs(service.start_preparation(), future)
            self.assertFalse(service.runtime_status()['ready'])
            self.assertEqual(service.model_load_state, 'loading')
            self.assertEqual(service.runtime_status()['tasks'][0]['phase'], 'query_encode')
        finally:
            release.set()
        future.result(2)
        self.wait_for_idle(service)
        self.assertTrue(service.runtime_status()['ready'])
        self.assertEqual(backend.load_calls, 1)
        self.assertEqual(len(backend.encode_calls), 1)

    def test_failed_preparation_keeps_exception_type_and_does_not_retry(self):
        service = self.service(FakeEmbeddingBackend(fail=True))
        future = service.start_preparation()
        with self.assertRaises(Exception):
            future.result(2)
        self.wait_for_idle(service)
        self.assertIs(service.start_preparation(), future)
        status = service.runtime_status()
        self.assertFalse(status['ready'])
        self.assertEqual(status['model_state'], 'failed')
        self.assertEqual(status['last_task']['exception_type'], 'EmbeddingEncodeError')
        self.assertEqual(status['last_task']['cause_type'], 'RuntimeError')
        self.assertEqual(status['last_task']['failed_phase'], 'query_encode')
        self.assertNotIn('backend exploded', repr(status))

    def test_timeout_tasks_remain_visible_until_slots_recover(self):
        service = self.service(FakeEmbeddingBackend())
        release = threading.Event()
        first = service.submit_semantic_work(lambda: release.wait(2))
        second = service.submit_semantic_work(lambda: release.wait(2))
        try:
            with self.assertRaises(SemanticCapacityUnavailable):
                service.submit_semantic_work(lambda: None)
            self.assertEqual(service.runtime_status()['active_tasks'], 2)
            self.assertFalse(first.done())
        finally:
            release.set()
        first.result(2); second.result(2)
        self.wait_for_idle(service)
        self.assertEqual(service.runtime_status()['active_tasks'], 0)
        self.assertEqual(service.submit_semantic_work(lambda: 1).result(1), 1)

    def test_close_during_load_is_nonblocking_and_skips_encode(self):
        entered, release = threading.Event(), threading.Event()
        class Backend(FakeEmbeddingBackend):
            def load_model(self, *args):
                entered.set(); release.wait(2)
                super().load_model(*args)
        backend = Backend(); service = self.service(backend)
        future = service.start_preparation()
        try:
            self.assertTrue(entered.wait(1))
            started = time.monotonic(); service.close()
            self.assertLess(time.monotonic()-started, .2)
        finally:
            release.set()
        future.result(2)
        self.wait_for_idle(service, unloaded=True)
        self.assertEqual(backend.encode_calls, [])
        self.assertIsNone(service._backend)
        with self.assertRaises(SemanticCapacityUnavailable):
            service.submit_semantic_work(lambda: None)

    def test_lifespan_owns_preparation_and_shutdown(self):
        from app import main
        service = self.service(FakeEmbeddingBackend())
        async def run():
            with patch.object(main, 'db', new=MagicMock()), patch.object(main, 'learning_service', new=MagicMock()), \
                 patch.object(main, 'WorkspaceUpdateService'), \
                 patch.object(main, 'LearningContinuityService'), \
                 patch.object(main, 'embedding_service', service):
                async with main.app_lifespan(main.app):
                    self.assertTrue(service.runtime_status()['ready'])
                    self.assertTrue(main.health()['embedding_ready'])
                self.assertTrue(service.runtime_status()['closed'])
        asyncio.run(run())

    def test_lifespan_wait_is_bounded_without_cancelling_native_work(self):
        from app import main
        entered, release = threading.Event(), threading.Event()
        class Backend(FakeEmbeddingBackend):
            def load_model(self, *args):
                entered.set(); release.wait(2); super().load_model(*args)
        service = self.service(Backend())
        async def run():
            with patch.object(main, 'db', new=MagicMock()), patch.object(main, 'learning_service', new=MagicMock()), \
                 patch.object(main, 'WorkspaceUpdateService'), \
                 patch.object(main, 'LearningContinuityService'), \
                 patch.object(main, 'embedding_service', service), \
                 patch.object(main, 'get_embedding_startup_timeout_seconds', return_value=.05):
                try:
                    started = time.monotonic()
                    async with main.app_lifespan(main.app):
                        self.assertLess(time.monotonic()-started, .3)
                        self.assertTrue(entered.is_set())
                        self.assertEqual(service.runtime_status()['active_tasks'], 1)
                        self.assertFalse(service._preparation_future.cancelled())
                        self.assertFalse(main.health()['embedding_ready'])
                        release.set()
                        await asyncio.wrap_future(service._preparation_future)
                        self.wait_for_idle(service)
                        self.assertTrue(main.health()['embedding_ready'])
                finally:
                    release.set()
        asyncio.run(run())

    def test_immediate_submission_failure_degrades_and_cleans_up(self):
        from app import main
        for exception_type in (OSError, RuntimeError, EmbeddingModelLoadError):
            with self.subTest(exception_type=exception_type.__name__):
                service = self.service(FakeEmbeddingBackend())
                async def run():
                    with patch.object(main, 'db', new=MagicMock()), patch.object(main, 'learning_service', new=MagicMock()), \
                         patch.object(main, 'WorkspaceUpdateService'), \
                         patch.object(main, 'LearningContinuityService'), \
                         patch.object(main, 'embedding_service', service), \
                         patch.object(service, 'submit_semantic_work',
                                      side_effect=exception_type('PRIVATE-SUBMISSION-DETAIL')), \
                         self.assertLogs(main.logger, level='WARNING') as logs:
                        async with main.app_lifespan(main.app):
                            status = service.runtime_status()
                            self.assertFalse(status['ready'])
                            self.assertEqual(status['model_state'], 'failed')
                            self.assertEqual(status['preparation']['failed_phase'], 'submission')
                            self.assertEqual(status['preparation']['exception_type'], exception_type.__name__)
                            self.assertNotIn('PRIVATE-SUBMISSION-DETAIL', repr(status))
                        self.assertTrue(service.runtime_status()['closed'])
                        self.wait_for_idle(service, unloaded=True)
                    self.assertIn(exception_type.__name__, ' '.join(logs.output))
                    self.assertNotIn('PRIVATE-SUBMISSION-DETAIL', ' '.join(logs.output))
                asyncio.run(run())

    def test_plain_preparation_failure_degrades_and_cleans_up(self):
        from app import main
        for exception_type in (OSError, RuntimeError):
            with self.subTest(exception_type=exception_type.__name__):
                service = self.service(FakeEmbeddingBackend())
                async def run():
                    with patch.object(main, 'db', new=MagicMock()), patch.object(main, 'learning_service', new=MagicMock()), \
                         patch.object(main, 'WorkspaceUpdateService'), \
                         patch.object(main, 'LearningContinuityService'), \
                         patch.object(main, 'embedding_service', service), \
                         patch.object(service, 'load_model',
                                      side_effect=exception_type('PRIVATE-PREPARATION-DETAIL')), \
                         self.assertLogs(main.logger, level='WARNING') as logs:
                        async with main.app_lifespan(main.app):
                            self.wait_for_idle(service)
                            status = service.runtime_status()
                            self.assertFalse(status['ready'])
                            self.assertEqual(status['model_state'], 'failed')
                            self.assertEqual(status['last_task']['failed_phase'], 'model_identity')
                            self.assertEqual(status['last_task']['exception_type'], exception_type.__name__)
                            self.assertNotIn('PRIVATE-PREPARATION-DETAIL', repr(status))
                        self.assertTrue(service.runtime_status()['closed'])
                        self.wait_for_idle(service, unloaded=True)
                    self.assertIn(exception_type.__name__, ' '.join(logs.output))
                    self.assertNotIn('PRIVATE-PREPARATION-DETAIL', ' '.join(logs.output))
                asyncio.run(run())

    def test_device_detection_exception_is_normalized_with_cause(self):
        for exception_type in (OSError, RuntimeError):
            with self.subTest(exception_type=exception_type.__name__):
                service = self.service(FakeEmbeddingBackend())
                with patch.object(service, '_cuda_available',
                                  side_effect=exception_type('PRIVATE-DEVICE-DETAIL')):
                    with self.assertRaises(EmbeddingModelLoadError) as raised:
                        service.load_model()
                self.assertIsInstance(raised.exception.__cause__, exception_type)
                self.assertEqual(service.model_load_state, 'failed')
                self.assertNotIn('PRIVATE-DEVICE-DETAIL', str(raised.exception))

    def test_device_failure_leaves_health_readable_without_reprobing(self):
        from app import main
        for exception_type in (OSError, RuntimeError):
            with self.subTest(exception_type=exception_type.__name__):
                service = self.service(FakeEmbeddingBackend())
                async def run():
                    with patch.object(main, 'db', new=MagicMock()), patch.object(main, 'learning_service', new=MagicMock()), \
                         patch.object(main, 'WorkspaceUpdateService'), \
                         patch.object(main, 'LearningContinuityService'), \
                         patch.object(main, 'embedding_service', service), \
                         patch.object(service, '_cuda_available',
                                      side_effect=exception_type('PRIVATE-DEVICE-DETAIL')) as detect:
                        async with main.app_lifespan(main.app):
                            self.wait_for_idle(service)
                            health = main.health()
                            self.assertTrue(health['ok'])
                            self.assertFalse(health['embedding_ready'])
                            self.assertEqual(health['embedding_device'], 'auto')
                            self.assertEqual(health['embedding_runtime']['last_task']['cause_type'],
                                             exception_type.__name__)
                            self.assertNotIn('PRIVATE-DEVICE-DETAIL', repr(health))
                            detect.assert_called_once()
                asyncio.run(run())

    def test_cancellation_and_exit_signals_propagate_and_close(self):
        from app import main
        for exception in (asyncio.CancelledError(), KeyboardInterrupt(), SystemExit(7)):
            with self.subTest(exception_type=type(exception).__name__):
                service = self.service(FakeEmbeddingBackend())
                async def run():
                    with patch.object(main, 'db', new=MagicMock()), patch.object(main, 'learning_service', new=MagicMock()), \
                         patch.object(main, 'WorkspaceUpdateService'), \
                         patch.object(main, 'LearningContinuityService'), \
                         patch.object(main, 'embedding_service', service), \
                         patch.object(service, 'start_preparation', side_effect=exception):
                        with self.assertRaises(type(exception)):
                            async with main.app_lifespan(main.app):
                                self.fail('signal was swallowed')
                        self.assertTrue(service.runtime_status()['closed'])
                asyncio.run(run())

    def test_database_failure_is_not_disguised_as_preparation_failure(self):
        from app import main
        service = self.service(FakeEmbeddingBackend())
        async def run():
            with patch.object(main, '_initialize_runtime_value', side_effect=OSError('database failed')), \
                 patch.object(main, 'embedding_service', service), \
                 patch.object(service, 'start_preparation') as prepare:
                with self.assertRaisesRegex(OSError, 'database failed'):
                    async with main.app_lifespan(main.app):
                        self.fail('database startup failure was swallowed')
                prepare.assert_not_called()
        asyncio.run(run())

    def test_executor_submission_failure_releases_slot(self):
        service = self.service(FakeEmbeddingBackend())
        with patch('app.services.embedding_service.ThreadPoolExecutor') as executor:
            executor.return_value.submit.side_effect = RuntimeError('PRIVATE-EXECUTOR-DETAIL')
            with self.assertRaises(RuntimeError):
                service.start_preparation()
        status = service.runtime_status()
        self.assertEqual(status['active_tasks'], 0)
        self.assertEqual(service._semantic_slots._value, 2)
        self.assertEqual(status['model_state'], 'failed')
        self.assertEqual(status['preparation']['failed_phase'], 'submission')
        self.assertNotIn('PRIVATE-EXECUTOR-DETAIL', repr(status))

    def test_waits_for_cleanup_when_result_precedes_done_callback(self):
        entered, release_work = threading.Event(), threading.Event()
        callback_entered, release_callback = threading.Event(), threading.Event()
        class Backend(FakeEmbeddingBackend):
            def encode(self, *args, **kwargs):
                entered.set(); release_work.wait(2)
                return super().encode(*args, **kwargs)
        class Executor(ThreadPoolExecutor):
            def submit(self, *args, **kwargs):
                future = super().submit(*args, **kwargs)
                def pause(_done):
                    callback_entered.set(); release_callback.wait(2)
                future.add_done_callback(pause)
                return future
        service = self.service(Backend())
        service._semantic_executor = Executor(max_workers=2)
        future = service.start_preparation()
        try:
            self.assertTrue(entered.wait(1))
            release_work.set()
            future.result(2)
            self.assertTrue(callback_entered.wait(1))
            self.assertEqual(service.runtime_status()['active_tasks'], 1)
        finally:
            release_work.set(); release_callback.set()
        self.wait_for_idle(service)
        self.assertEqual(service.runtime_status()['last_task']['status'], 'completed')

    def test_cancelling_startup_wait_closes_without_cancelling_native_work(self):
        from app import main
        entered, release = threading.Event(), threading.Event()
        class Backend(FakeEmbeddingBackend):
            def load_model(self, *args):
                entered.set(); release.wait(2); super().load_model(*args)
        backend = Backend(); service = self.service(backend)
        async def lifespan():
            async with main.app_lifespan(main.app):
                self.fail('startup should still be waiting')
        async def run():
            with patch.object(main, 'db', new=MagicMock()), patch.object(main, 'learning_service', new=MagicMock()), \
                 patch.object(main, 'WorkspaceUpdateService'), \
                 patch.object(main, 'LearningContinuityService'), \
                 patch.object(main, 'embedding_service', service):
                task = asyncio.create_task(lifespan())
                try:
                    self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                    self.assertTrue(service.runtime_status()['closed'])
                    self.assertFalse(service._preparation_future.cancelled())
                    self.assertEqual(service.runtime_status()['active_tasks'], 1)
                finally:
                    release.set()
                await asyncio.wrap_future(service._preparation_future)
                self.wait_for_idle(service, unloaded=True)
                self.assertEqual(backend.encode_calls, [])
        asyncio.run(run())

    def test_late_failure_after_startup_timeout_is_recorded_and_cleaned(self):
        from app import main
        entered, release = threading.Event(), threading.Event()
        service = self.service(FakeEmbeddingBackend())
        def failed_detection():
            entered.set(); release.wait(2)
            raise OSError('PRIVATE-LATE-DETAIL')
        async def run():
            with patch.object(main, 'db', new=MagicMock()), patch.object(main, 'learning_service', new=MagicMock()), \
                 patch.object(main, 'WorkspaceUpdateService'), \
                 patch.object(main, 'LearningContinuityService'), \
                 patch.object(main, 'embedding_service', service), \
                 patch.object(service, '_cuda_available', side_effect=failed_detection), \
                 patch.object(main, 'get_embedding_startup_timeout_seconds', return_value=.05):
                try:
                    async with main.app_lifespan(main.app):
                        self.assertTrue(entered.is_set())
                        self.assertFalse(service._preparation_future.done())
                        release.set()
                        with self.assertRaises(EmbeddingModelLoadError):
                            await asyncio.wrap_future(service._preparation_future)
                        self.wait_for_idle(service)
                        health = main.health()
                        self.assertEqual(health['embedding_runtime']['model_state'], 'failed')
                        self.assertEqual(health['embedding_runtime']['last_task']['cause_type'], 'OSError')
                        self.assertNotIn('PRIVATE-LATE-DETAIL', repr(health))
                finally:
                    release.set()
            self.wait_for_idle(service, unloaded=True)
        asyncio.run(run())

    def test_unknown_programming_failure_is_not_downgraded(self):
        from app import main
        service = self.service(FakeEmbeddingBackend())
        async def run():
            with patch.object(main, 'db', new=MagicMock()), patch.object(main, 'learning_service', new=MagicMock()), \
                 patch.object(main, 'WorkspaceUpdateService'), \
                 patch.object(main, 'LearningContinuityService'), \
                 patch.object(main, 'embedding_service', service), \
                 patch.object(service, 'start_preparation', side_effect=AssertionError('bug')):
                with self.assertRaises(AssertionError):
                    async with main.app_lifespan(main.app):
                        self.fail('programming failure was swallowed')
                self.assertTrue(service.runtime_status()['closed'])
        asyncio.run(run())

    def test_failed_lifespan_serves_explicit_degradation(self):
        from app import main
        service = self.service(FakeEmbeddingBackend(fail=True))
        async def run():
            with patch.object(main, 'db', new=MagicMock()), patch.object(main, 'learning_service', new=MagicMock()), \
                 patch.object(main, 'WorkspaceUpdateService'), \
                 patch.object(main, 'LearningContinuityService'), \
                 patch.object(main, 'embedding_service', service):
                async with main.app_lifespan(main.app):
                    self.assertFalse(main.health()['embedding_ready'])
                    self.assertEqual(main.health()['embedding_runtime']['model_state'], 'failed')
        asyncio.run(run())
