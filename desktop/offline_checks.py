"""Focused source checks with temporary storage and no external network/model."""
import argparse
import os
from pathlib import Path
import socket
import sqlite3
import sys
import tempfile
import unittest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, help='Explicitly check this built backend instead of source')
    parser.add_argument('--without-timing', action='store_true')
    parser.add_argument('--transport-only', action='store_true')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    product = args.package.resolve() if args.package else root
    # The test suite belongs to source; the selected app path is reported explicitly.
    sys.path[:0] = [str(product / 'backend'), str(root / 'backend'), str(root / 'desktop')]
    with tempfile.TemporaryDirectory(prefix='reponoesis-desktop-tests-') as temporary:
        data = Path(temporary).resolve()
        for name in tuple(os.environ):
            if name.startswith(('LLM_', 'EMBEDDING_', 'GITHUB_', 'AGENT_', 'REPONOESIS_', 'RETRIEVAL_')):
                os.environ.pop(name)
        os.environ.update(GITLEARN_DB=str(data / 'guard.sqlite'), RUNTIME_DATA_DIR=str(data / 'runtime'),
                          EMBEDDING_ENABLED='false', REPONOESIS_SKIP_ENV_FILE='true')
        original_connect = sqlite3.connect
        original_socket_connect = socket.socket.connect

        def guarded_database(database, *a, **kw):
            if str(database) != ':memory:':
                resolved = Path(database).resolve()
                if not resolved.is_relative_to(Path(tempfile.gettempdir()).resolve()):
                    raise AssertionError('Non-temporary database access denied')
            return original_connect(database, *a, **kw)

        def guarded_network(sock, address):
            if isinstance(address, tuple) and address[0] not in ('127.0.0.1', 'localhost', '::1'):
                raise AssertionError('External network denied')
            return original_socket_connect(sock, address)

        sqlite3.connect = guarded_database
        socket.socket.connect = guarded_network
        try:
            import app
            expected = (product / 'backend/app/__init__.py').resolve()
            if Path(app.__file__).resolve() != expected:
                raise RuntimeError('Selected application was not imported')
            print('Backend under test:', Path(app.__file__).resolve(), flush=True)
            names = ['test_launcher', 'tests.test_source_file_api', 'tests.test_embedding_startup',
                     'tests.test_product_provider_integration', 'tests.test_provider_absolute_deadline',
                     'tests.test_repository_import', 'tests.test_source_exploration']
            if args.without_timing:
                names.remove('tests.test_provider_absolute_deadline')
            if args.transport_only:
                names = ['tests.test_provider_absolute_deadline']
            suite = unittest.defaultTestLoader.loadTestsFromNames(names)
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            return 0 if result.wasSuccessful() else 1
        finally:
            sqlite3.connect = original_connect
            socket.socket.connect = original_socket_connect


if __name__ == '__main__':
    sys.exit(main())
