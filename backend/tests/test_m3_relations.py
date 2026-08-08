from __future__ import annotations

import sqlite3
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.database import Database, SCHEMA_VERSION
from app.services.code_chunker import extract_python_code_chunks_from_files
from app.services.relation_analysis import index_project_relations
from tests.m3_helpers import call_chain_sources, make_relation_project


class M3SchemaAndRelationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.directory.name) / "m3.sqlite")

    def tearDown(self):
        self.directory.cleanup()

    def test_fresh_schema_is_v10_with_relation_tables_and_indexes(self):
        with self.db.connect() as conn:
            version = conn.execute(
                "SELECT version FROM schema_versions WHERE key = 'database'"
            ).fetchone()["version"]
            tables = {
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            indexes = {
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                )
            }
        self.assertEqual(SCHEMA_VERSION, 10)
        self.assertEqual(version, 10)
        self.assertTrue(
            {"relation_nodes", "code_relations", "relation_index_runs"}.issubset(
                tables
            )
        )
        self.assertTrue(
            {
                "idx_code_relations_revision_source",
                "idx_code_relations_revision_target",
                "idx_code_relations_revision_type",
                "idx_code_relations_revision_symbol",
            }.issubset(indexes)
        )

    def test_v4_version_migrates_idempotently_and_preserves_m1_data(self):
        project_id, _bundle = make_relation_project(
            self.db, {"app.py": "def main():\n    return 1\n"}
        )
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE schema_versions SET version = 4 WHERE key = 'database'"
            )
        reopened = Database(self.db.path)
        reopened_again = Database(self.db.path)
        self.assertEqual(reopened.get_project(project_id)["repo"], "reponoesis-m3-fixture")
        self.assertEqual(len(reopened_again.get_code_chunks(project_id)), 1)
        with reopened.connect() as conn:
            self.assertEqual(
                conn.execute(
                    "SELECT version FROM schema_versions WHERE key='database'"
                ).fetchone()["version"],
                SCHEMA_VERSION,
            )

    def test_migration_failure_does_not_report_a_false_v5_version(self):
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE schema_versions SET version = 4 WHERE key = 'database'"
            )
        with patch.object(
            Database, "_migrate_schema", side_effect=RuntimeError("forced migration failure")
        ):
            with self.assertRaises(RuntimeError):
                Database(self.db.path)
        raw = sqlite3.connect(self.db.path)
        try:
            version = raw.execute(
                "SELECT version FROM schema_versions WHERE key='database'"
            ).fetchone()[0]
        finally:
            raw.close()
        self.assertEqual(version, 4)

    def test_import_call_reference_and_defines_are_conservative(self):
        sources = {
            "pkg/__init__.py": "",
            "pkg/utils.py": (
                "def helper(value):\n    return value\n\n"
                "class Worker:\n"
                "    def first(self):\n        return self.second()\n"
                "    def second(self):\n        return 2\n"
            ),
            "pkg/service.py": (
                "import os\n"
                "import pkg.missing\n"
                "import pkg.utils as utils\n"
                "from .utils import helper as do_help\n\n"
                "def local(value):\n    return value\n\n"
                "def process(value):\n"
                "    local(value)\n"
                "    do_help(value)\n"
                "    return utils.helper(value)\n\n"
                "def shadow(do_help):\n    return do_help()\n\n"
                "def dynamic(obj):\n    return obj.run()\n"
            ),
        }
        project_id, _bundle = make_relation_project(self.db, sources)
        edges = self.db.get_relations(project_id, "revision-m3")
        imports = [item for item in edges if item["relation_type"] == "imports"]
        calls = [item for item in edges if item["relation_type"] == "calls"]
        defines = [item for item in edges if item["relation_type"] == "defines"]

        self.assertTrue(
            any(
                item["raw_target_name"] == "os"
                and item["resolution_status"] == "external"
                for item in imports
            )
        )
        self.assertTrue(
            any(
                item["raw_target_name"] == "pkg.missing"
                and item["resolution_status"] == "unresolved"
                for item in imports
            )
        )
        self.assertTrue(
            any(
                item["target_symbol"] == "helper"
                and item["resolution_rule"] in {"import_alias", "module_alias"}
                for item in calls
            )
        )
        self.assertTrue(
            any(
                item["raw_target_name"] == "do_help"
                and item["resolution_status"] == "unresolved"
                and item["resolution_rule"] == "local_or_parameter_shadowing"
                for item in calls
            )
        )
        self.assertTrue(
            any(
                item["raw_target_name"] == "obj.run"
                and item["resolution_status"] == "unresolved"
                and item["resolution_rule"] == "dynamic_attribute"
                for item in calls
            )
        )
        self.assertGreaterEqual(len(defines), 7)

    def test_self_method_and_cross_file_call_edges_are_resolved(self):
        project_id, _bundle = make_relation_project(
            self.db,
            {
                "worker.py": (
                    "class Worker:\n"
                    "    def first(self):\n        return self.second()\n"
                    "    def second(self):\n        return 2\n"
                ),
                "caller.py": (
                    "from worker import Worker\n\n"
                    "def call(worker):\n"
                    "    return Worker.second(worker)\n"
                ),
            },
        )
        calls = self.db.get_relations(
            project_id, "revision-m3", relation_types=["calls"]
        )
        self.assertTrue(
            any(
                item["target_symbol"] == "Worker.second"
                and item["resolution_rule"] == "self_method"
                for item in calls
            )
        )
        self.assertTrue(
            any(
                item["target_symbol"] == "Worker.second"
                and item["resolution_rule"] == "class_qualified"
                for item in calls
            )
        )

    def test_repeat_index_is_stable_and_does_not_duplicate_edges(self):
        project_id, _bundle = make_relation_project(self.db, call_chain_sources())
        first = self.db.get_relations(project_id, "revision-m3")
        first_ids = [item["edge_id"] for item in first]
        index_project_relations(self.db, project_id)
        second_ids = [
            item["edge_id"]
            for item in self.db.get_relations(project_id, "revision-m3")
        ]
        self.assertEqual(first_ids, second_ids)
        self.assertEqual(len(second_ids), len(set(second_ids)))

    def test_direct_code_chunk_replacement_invalidates_relation_index(self):
        project_id, _bundle = make_relation_project(self.db, call_chain_sources())
        chunks = self.db.get_code_chunks(project_id, path="pkg/a.py")
        self.assertTrue(self.db.get_relations(project_id, "revision-m3"))
        self.db.replace_code_chunks_for_file(
            project_id,
            "pkg/a.py",
            chunks,
            repository_revision="revision-m3",
        )
        self.assertEqual(self.db.get_relations(project_id, "revision-m3"), [])
        self.assertIsNone(
            self.db.get_relation_index_status(project_id, "revision-m3")
        )

    def test_reindex_content_change_and_deleted_symbol_leave_no_ghost_edges(self):
        project_id, _bundle = make_relation_project(self.db, call_chain_sources())
        old_edge_ids = {
            item["edge_id"]
            for item in self.db.get_relations(project_id, "revision-m3")
        }
        changed = {
            "pkg/__init__.py": "",
            "pkg/a.py": (
                "from .b import b\n\n"
                "def a():\n"
                "    # content identity changed\n"
                "    return b()\n"
            ),
            "pkg/b.py": "def b():\n    return 2\n",
        }
        self._replace_snapshot(project_id, changed)
        self.assertEqual(
            self.db.get_relations(project_id, "revision-m3"), []
        )
        index_project_relations(self.db, project_id)
        changed_edges = self.db.get_relations(project_id, "revision-m3")
        self.assertTrue(changed_edges)
        self.assertTrue(
            old_edge_ids.isdisjoint(item["edge_id"] for item in changed_edges)
        )

        without_b = {
            "pkg/__init__.py": "",
            "pkg/a.py": "from .b import b\n\ndef a():\n    return b()\n",
        }
        self._replace_snapshot(project_id, without_b)
        index_project_relations(self.db, project_id)
        nodes = self.db.get_relation_nodes(project_id, "revision-m3")
        self.assertNotIn("b", [item["qualified_name"] for item in nodes])
        self.assertFalse(
            any(
                item["target_symbol"] == "b"
                and item["resolution_status"] == "resolved"
                for item in self.db.get_relations(project_id, "revision-m3")
            )
        )

    def test_parser_failure_is_partial_and_never_executes_source(self):
        marker = Path(self.directory.name) / "executed"
        project_id, _bundle = make_relation_project(
            self.db,
            {
                "ok.py": (
                    "def safe():\n"
                    f"    open({str(marker)!r}, 'w').write('bad')\n"
                ),
                "broken.py": "def broken(:\n    pass\n",
            },
        )
        status = self.db.get_relation_index_status(project_id, "revision-m3")
        self.assertEqual(status["status"], "partial")
        self.assertEqual(status["failed_files"], 1)
        self.assertFalse(marker.exists())

    def test_relation_replace_rolls_back_on_invalid_edge(self):
        project_id, _bundle = make_relation_project(self.db, call_chain_sources())
        before = self.db.get_relations(project_id, "revision-m3")
        nodes = self.db.get_relation_nodes(project_id, "revision-m3")
        invalid = dict(before[0])
        invalid["edge_id"] = "R" + "0" * 64
        invalid["source_node_id"] = "N" + "f" * 64
        with self.assertRaises(ValueError):
            self.db.replace_relation_index(
                project_id,
                "revision-m3",
                nodes,
                [invalid],
                status="complete",
                parsed_files=1,
                failed_files=0,
                unsupported_files=0,
                warnings=[],
            )
        self.assertEqual(
            [item["edge_id"] for item in before],
            [
                item["edge_id"]
                for item in self.db.get_relations(project_id, "revision-m3")
            ],
        )

    def _replace_snapshot(self, project_id, sources):
        files = [
            {
                "path": path,
                "extension": Path(path).suffix,
                "language": "Python",
                "size": len(content.encode("utf-8")),
                "content": content,
                "summary": path,
                "importance": 1,
                "is_core": True,
                "imports": [],
                "exports": [],
                "symbols": [],
            }
            for path, content in sorted(sources.items())
        ]
        chunks = extract_python_code_chunks_from_files(files, "revision-m3")
        self.db.save_analysis(
            project_id,
            {
                "primary_language": "Python",
                "frameworks": [],
                "files": files,
                "modules": [],
                "overview": "updated",
            },
            files,
            [],
            [item.to_dict() for item in chunks.chunks],
        )


if __name__ == "__main__":
    unittest.main()
