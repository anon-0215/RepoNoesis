from __future__ import annotations

import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from app import config
from app.database import Database


@contextmanager
def working_directory(path: Path):
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


class DatabasePathConfigTests(unittest.TestCase):
    def test_relative_environment_path_is_independent_of_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repository"
            backend = root / "backend"
            other = Path(directory) / "other"
            backend.mkdir(parents=True)
            other.mkdir()
            expected = (root / "backend" / "data" / "test.sqlite").resolve()

            with (
                patch.object(config, "PROJECT_ROOT", root),
                patch.dict(
                    os.environ,
                    {"GITLEARN_DB": "backend/data/test.sqlite"},
                    clear=True,
                ),
            ):
                resolved = []
                selected = []
                for cwd in (root, backend, other):
                    with working_directory(cwd):
                        resolved.append(config.get_database_path())
                        selected.append(Database().path)

            self.assertEqual(resolved, [expected, expected, expected])
            self.assertEqual(selected, [expected, expected, expected])
            self.assertTrue(expected.is_file())
            self.assertFalse((backend / "backend/data/test.sqlite").exists())
            self.assertFalse((other / "backend/data/test.sqlite").exists())

    def test_absolute_environment_path_is_independent_of_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repository"
            backend = root / "backend"
            other = Path(directory) / "other"
            target = Path(directory) / "database" / "absolute.sqlite"
            backend.mkdir(parents=True)
            other.mkdir()

            with (
                patch.object(config, "PROJECT_ROOT", root),
                patch.dict(os.environ, {"GITLEARN_DB": str(target)}, clear=True),
            ):
                resolved = []
                selected = []
                for cwd in (root, backend, other):
                    with working_directory(cwd):
                        resolved.append(config.get_database_path())
                        selected.append(Database().path)

            expected = target.resolve()
            self.assertEqual(resolved, [expected, expected, expected])
            self.assertEqual(selected, [expected, expected, expected])
            self.assertTrue(expected.is_file())

    def test_config_status_and_database_use_the_same_environment_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repository"
            backend = root / "backend"
            other = Path(directory) / "other"
            backend.mkdir(parents=True)
            other.mkdir()

            with (
                patch.object(config, "PROJECT_ROOT", root),
                patch.object(config, "BACKEND_ROOT", backend),
                patch.dict(
                    os.environ,
                    {"GITLEARN_DB": "backend/data/product.sqlite"},
                    clear=True,
                ),
                working_directory(other),
            ):
                status = config.get_product_config_status()
                database = Database()

            self.assertTrue(database.path.is_absolute())
            self.assertEqual(Path(status["runtime"]["database"]), database.path)
            self.assertEqual(database.path, (root / "backend/data/product.sqlite").resolve())

    def test_empty_or_unset_environment_path_uses_normalized_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repository"
            backend = root / "backend"
            backend.mkdir(parents=True)
            expected = (backend / "data" / "gitlearn.sqlite").resolve()

            with (
                patch.object(config, "PROJECT_ROOT", root),
                patch.object(config, "BACKEND_ROOT", backend),
            ):
                for environment in ({}, {"GITLEARN_DB": "   "}):
                    with patch.dict(os.environ, environment, clear=True):
                        self.assertEqual(config.get_database_path(), expected)

    def test_explicit_database_paths_keep_existing_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cwd = root / "cwd"
            cwd.mkdir()
            environment_target = root / "environment.sqlite"
            absolute_target = root / "absolute.sqlite"

            with (
                patch.dict(
                    os.environ,
                    {"GITLEARN_DB": str(environment_target)},
                    clear=True,
                ),
                working_directory(cwd),
            ):
                relative = Database("relative.sqlite")
                absolute = Database(absolute_target)

            self.assertEqual(relative.path, Path("relative.sqlite"))
            self.assertTrue((cwd / "relative.sqlite").is_file())
            self.assertEqual(absolute.path, absolute_target)
            self.assertTrue(absolute_target.is_file())
            self.assertFalse(environment_target.exists())


if __name__ == "__main__":
    unittest.main()
