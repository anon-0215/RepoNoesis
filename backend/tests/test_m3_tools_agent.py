from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import time
import unittest

from app.database import Database
from app.services.agent_contracts import AgentLimits, CancellationToken, ToolCall
from app.services.agent_core import run_bounded_agent
from app.services.agent_tools import (
    EvidenceStore,
    build_m2_tool_registry,
    build_tool_context,
)
from tests.m1_helpers import disabled_embedding_service
from tests.m3_helpers import call_chain_sources, make_relation_project
from tests.test_m2_agent import NoLlm, ScriptedPlanner, decision


class M3ToolAndAgentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.directory.name) / "tools-agent.sqlite")
        self.project_id, self.bundle = make_relation_project(
            self.db, call_chain_sources()
        )
        self.limits = AgentLimits()
        self.store = EvidenceStore()
        self.context = build_tool_context(
            request_id="request-m3",
            bundle=self.bundle,
            database=self.db,
            embedding_service=disabled_embedding_service(),
            evidence_store=self.store,
            limits=self.limits,
            cancellation=CancellationToken(),
            deadline_monotonic=time.monotonic() + 60,
        )
        self.registry = build_m2_tool_registry(self.limits)

    def tearDown(self):
        self.directory.cleanup()

    def _call(self, name, parameters, *, context=None):
        call = ToolCall(
            "C1",
            "S1",
            name,
            "1",
            parameters,
            15_000,
            {"max_results": 128, "max_bytes": 65_536},
        )
        return self.registry.execute(context or self.context, call)

    def test_expand_evidence_seed_registers_supporting_evidence_chains_and_metrics(self):
        search = self._call("search_code", {"query": "def a", "top_k": 1})
        self.assertEqual(search.status, "succeeded")
        self.assertEqual(search.structured_results["evidence"][0]["evidence_id"], "E1")
        expanded = self._call(
            "expand_relations",
            {
                "seed_evidence_ids": ["E1"],
                "relation_types": ["calls"],
                "direction": "outbound",
                "max_depth": 2,
                "per_node_limit": 20,
            },
        )
        self.assertEqual(expanded.status, "succeeded")
        self.assertEqual(expanded.structured_results["analysis_mode"], "relation_expanded")
        self.assertGreaterEqual(expanded.metrics["node_count"], 3)
        self.assertGreaterEqual(expanded.metrics["edge_count"], 2)
        self.assertGreaterEqual(expanded.metrics["path_count"], 2)
        self.assertGreaterEqual(expanded.metrics["evidence_count"], 3)
        self.assertTrue(expanded.structured_results["evidence_chains"])
        self.assertNotIn("content", str(expanded.structured_results))

    def test_expand_symbol_seed_and_both_seed_types(self):
        lookup = self._call("lookup_symbol", {"symbol": "a"})
        node_id = lookup.structured_results[0]["relation_node_id"]
        self.assertTrue(node_id.startswith("N"))
        search = self._call("search_code", {"query": "def a", "top_k": 1})
        expanded = self._call(
            "expand_relations",
            {
                "seed_evidence_ids": ["E1"],
                "seed_symbol_ids": [node_id],
                "relation_types": ["calls"],
            },
        )
        self.assertEqual(expanded.status, "succeeded")
        self.assertEqual(expanded.metrics["seed_count"], 1)

    def test_expand_rejects_no_seed_forgery_identity_unknown_type_and_direction(self):
        for parameters in (
            {},
            {"seed_evidence_ids": ["E999"]},
            {"seed_symbol_ids": ["N" + "0" * 64]},
            {"seed_symbol_ids": ["N" + "0" * 64], "relation_types": ["owns"]},
            {"seed_symbol_ids": ["N" + "0" * 64], "direction": "sideways"},
            {
                "seed_symbol_ids": ["N" + "0" * 64],
                "project_id": self.project_id,
            },
        ):
            with self.subTest(parameters=parameters):
                observation = self._call("expand_relations", parameters)
                self.assertIn(observation.status, {"rejected", "failed"})

    def test_server_depth_and_per_node_limits_cannot_be_raised(self):
        search = self._call("search_code", {"query": "def a", "top_k": 1})
        self.assertEqual(search.status, "succeeded")
        rejected = self._call(
            "expand_relations",
            {
                "seed_evidence_ids": ["E1"],
                "max_depth": 100,
                "per_node_limit": 100,
            },
        )
        self.assertEqual(rejected.status, "rejected")
        small = replace(
            self.limits,
            max_relation_depth=1,
            max_relation_neighbors_per_node=1,
        )
        small_context = replace(self.context, limits=small)
        registry = build_m2_tool_registry(small)
        call = ToolCall(
            "C",
            "S",
            "expand_relations",
            "1",
            {"seed_evidence_ids": ["E1"], "max_depth": 2},
            15_000,
            {},
        )
        observation = registry.execute(small_context, call)
        self.assertLessEqual(
            max(
                (
                    item["path_depth"]
                    for item in observation.structured_results["paths"]
                ),
                default=0,
            ),
            1,
        )

    def test_no_relation_index_degrades_to_retrieval_only(self):
        _project_id, bundle = make_relation_project(
            self.db,
            {"plain.py": "def plain():\n    return 1\n"},
            index_relations=False,
        )
        context = build_tool_context(
            request_id="no-index",
            bundle=bundle,
            database=self.db,
            embedding_service=disabled_embedding_service(),
            evidence_store=EvidenceStore(),
            limits=self.limits,
            cancellation=CancellationToken(),
            deadline_monotonic=time.monotonic() + 60,
        )
        search_call = ToolCall(
            "C",
            "S",
            "search_code",
            "1",
            {"query": "plain"},
            15_000,
            {},
        )
        self.registry.execute(context, search_call)
        observation = self._call(
            "expand_relations", {"seed_evidence_ids": ["E1"]}, context=context
        )
        self.assertEqual(observation.status, "succeeded")
        self.assertEqual(
            observation.structured_results["analysis_mode"], "retrieval_only"
        )

    def test_agent_search_expand_answer_returns_valid_cross_file_chain(self):
        planner = ScriptedPlanner(
            [
                decision("continue", "search_code", {"query": "def a", "top_k": 1}),
                decision(
                    "continue",
                    "expand_relations",
                    {
                        "seed_evidence_ids": ["E1"],
                        "relation_types": ["calls"],
                        "max_depth": 2,
                    },
                ),
                decision("answer"),
            ]
        )
        result = run_bounded_agent(
            "trace a to its called definitions",
            self.bundle,
            NoLlm(),
            self.db,
            disabled_embedding_service(),
            planner=planner,
        )
        self.assertEqual(result["agent_status"], "completed")
        self.assertEqual(result["analysis_mode"], "relation_expanded")
        self.assertTrue(result["evidence_chains"])
        self.assertGreaterEqual(len({item["path"] for item in result["evidence"]}), 3)
        self.assertTrue(
            all(item["validation_status"] == "valid" for item in result["evidence"])
        )
        self.assertIn("静态关系", result["answer"])
        self.assertIn("`calls`", result["answer"])

    def test_agent_lookup_expand_read_answer_and_second_expand_are_bounded(self):
        node_a = next(
            item["node_id"]
            for item in self.db.get_relation_nodes(
                self.project_id, "revision-m3", qualified_name="a"
            )
        )
        planner = ScriptedPlanner(
            [
                decision("continue", "lookup_symbol", {"symbol": "a"}),
                decision(
                    "continue",
                    "expand_relations",
                    {
                        "seed_symbol_ids": [node_a],
                        "relation_types": ["calls"],
                        "max_depth": 1,
                    },
                ),
                decision(
                    "continue",
                    "read_source",
                    {"path": "pkg/b.py", "start_line": 1, "end_line": 4},
                ),
                decision("answer"),
            ]
        )
        result = run_bounded_agent(
            "inspect a",
            self.bundle,
            NoLlm(),
            self.db,
            disabled_embedding_service(),
            planner=planner,
        )
        self.assertEqual(
            [item["action"] for item in result["agent_trace"]],
            ["lookup_symbol", "expand_relations", "read_source", "answer"],
        )
        self.assertLessEqual(result["budget_usage"]["steps_used"], 5)
        self.assertEqual(result["analysis_mode"], "relation_expanded")

    def test_agent_search_expand_then_expand_new_seed_reaches_second_hop(self):
        planner = ScriptedPlanner(
            [
                decision("continue", "search_code", {"query": "def a", "top_k": 1}),
                decision(
                    "continue",
                    "expand_relations",
                    {
                        "seed_evidence_ids": ["E1"],
                        "relation_types": ["calls"],
                        "max_depth": 1,
                    },
                ),
                decision(
                    "continue",
                    "expand_relations",
                    {
                        "seed_evidence_ids": ["E2"],
                        "relation_types": ["calls"],
                        "max_depth": 1,
                    },
                ),
                decision("answer"),
            ]
        )
        result = run_bounded_agent(
            "trace a in two bounded expansions",
            self.bundle,
            NoLlm(),
            self.db,
            disabled_embedding_service(),
            planner=planner,
        )
        self.assertEqual(
            [item["action"] for item in result["agent_trace"]],
            [
                "search_code",
                "expand_relations",
                "expand_relations",
                "answer",
            ],
        )
        self.assertEqual(
            {"pkg/a.py", "pkg/b.py", "pkg/c.py"},
            {item["path"] for item in result["evidence"]},
        )
        self.assertGreaterEqual(len(result["evidence_chains"]), 2)

    def test_deleted_relation_before_final_answer_removes_chain(self):
        class DeletingPlanner(ScriptedPlanner):
            def decide(inner_self, state, *, repair_hint=None):
                if inner_self.calls == 2:
                    with self.db.connect() as conn:
                        conn.execute("DELETE FROM code_relations")
                return super().decide(state, repair_hint=repair_hint)

        planner = DeletingPlanner(
            [
                decision("continue", "search_code", {"query": "def a", "top_k": 1}),
                decision(
                    "continue",
                    "expand_relations",
                    {"seed_evidence_ids": ["E1"], "relation_types": ["calls"]},
                ),
                decision("answer"),
            ]
        )
        result = run_bounded_agent(
            "trace a",
            self.bundle,
            NoLlm(),
            self.db,
            disabled_embedding_service(),
            planner=planner,
        )
        self.assertEqual(result["analysis_mode"], "retrieval_only")
        self.assertEqual(result["evidence_chains"], [])
        self.assertTrue(any("chain" in item.lower() for item in result["warnings"]))


if __name__ == "__main__":
    unittest.main()
