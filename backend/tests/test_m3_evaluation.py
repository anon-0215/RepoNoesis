from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from app.database import Database
from app.services.agent_core import run_bounded_agent
from app.services.evidence import CitationValidator, Evidence
from tests.m1_helpers import disabled_embedding_service
from tests.m3_helpers import call_chain_sources, make_relation_project
from tests.test_m2_agent import NoLlm, ScriptedPlanner, decision


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "m3_relation_eval.json"


class M3EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.scenarios = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        self.directory = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.directory.name) / "m3-eval.sqlite")
        sources = call_chain_sources()
        sources["pkg/ref.py"] = (
            "from .b import b\n"
            "from .b import b as target_b\n\n"
            "def ref_b():\n    return b\n\n"
            "def alias_b():\n    return target_b\n"
        )
        sources["dynamic.py"] = "def dynamic(obj):\n    return obj.run()\n"
        self.project_id, self.bundle = make_relation_project(self.db, sources)
        self.ambiguous_project_id, self.ambiguous_bundle = make_relation_project(
            self.db,
            {
                "pkg/mod.py": "def first():\n    return 1\n",
                "src/pkg/mod.py": "def second():\n    return 2\n",
                "consumer.py": "import pkg.mod\n\ndef consume():\n    return 1\n",
            },
        )
        self.plain_project_id, self.plain_bundle = make_relation_project(
            self.db,
            {"plain.py": "def plain():\n    return 1\n"},
            index_relations=False,
        )

    def tearDown(self):
        self.directory.cleanup()

    def test_annotation_set_has_20_frozen_scenarios_and_required_categories(self):
        self.assertGreaterEqual(len(self.scenarios), 20)
        ids = [item["scenario_id"] for item in self.scenarios]
        self.assertEqual(len(ids), len(set(ids)))
        required = {
            "scenario_id",
            "user_goal",
            "answerable",
            "seed_query",
            "seed_symbol",
            "fake_planner_decisions",
            "allowed_tools",
            "forbidden_tools",
            "expected_relation_types",
            "expected_nodes",
            "expected_edges_min",
            "expected_path_or_acceptable_paths",
            "expected_evidence",
            "expected_final_status",
            "maximum_relation_depth",
            "maximum_steps",
            "maximum_calls",
            "annotation_note",
        }
        self.assertTrue(all(required.issubset(item) for item in self.scenarios))
        counts = Counter(item["category"] for item in self.scenarios)
        self.assertGreaterEqual(counts["import_dependency"], 4)
        self.assertGreaterEqual(counts["call_relation"], 4)
        self.assertGreaterEqual(counts["definition_reference"], 3)
        self.assertGreaterEqual(counts["cross_file_multihop"], 4)
        self.assertGreaterEqual(counts["ambiguous_unresolved"], 2)
        self.assertGreaterEqual(counts["retrieval_only"], 1)
        self.assertGreaterEqual(counts["prompt_injection_or_budget"], 2)

    def test_exact_fixture_edge_precision_and_recall_are_100_percent(self):
        calls = self.db.get_relations(
            self.project_id,
            "revision-m3",
            relation_types=["calls"],
            resolution_statuses=["resolved"],
        )
        actual = {
            (item["source_symbol"], item["target_symbol"])
            for item in calls
            if item["source_symbol"] in {"a", "b", "c"}
        }
        gold = {("a", "b"), ("b", "c"), ("c", "a")}
        self.assertEqual(actual, gold)
        precision = len(actual & gold) / len(actual)
        recall = len(actual & gold) / len(gold)
        self.assertEqual(precision, 1.0)
        self.assertEqual(recall, 1.0)

    def test_all_scenarios_respect_graph_agent_evidence_and_security_metrics(self):
        forbidden_execution = 0
        target_execution = 0
        cross_revision_edges = 0
        budget_violations = 0
        invalid_evidence = 0
        invalid_chain = 0
        ambiguous_as_exact = 0
        gold_path_hits = 0
        answerable_path_scenarios = 0

        for scenario in self.scenarios:
            with self.subTest(scenario=scenario["scenario_id"]):
                bundle = self.bundle
                project_id = self.project_id
                if scenario["scenario_id"].startswith("ambiguous-01"):
                    bundle = self.ambiguous_bundle
                    project_id = self.ambiguous_project_id
                elif scenario["category"] == "retrieval_only":
                    bundle = self.plain_bundle
                    project_id = self.plain_project_id
                planner = self._planner_for(scenario, project_id)
                result = run_bounded_agent(
                    scenario["user_goal"],
                    bundle,
                    NoLlm(),
                    self.db,
                    disabled_embedding_service(),
                    planner=planner,
                )
                self.assertEqual(
                    result["agent_status"], scenario["expected_final_status"]
                )
                budget_violations += int(
                    result["budget_usage"]["steps_used"]
                    > scenario["maximum_steps"]
                    or result["budget_usage"]["tool_calls_used"]
                    > scenario["maximum_calls"]
                )
                for step in result["agent_trace"]:
                    if not step["tool_calls"]:
                        continue
                    call = step["tool_calls"][0]
                    if call["tool_name"] in scenario["forbidden_tools"]:
                        forbidden_execution += int(call["status"] != "rejected")
                    target_execution += int(
                        call["tool_name"] in {"shell", "execute_code"}
                        and call["status"] != "rejected"
                    )
                evidence = [Evidence(**item) for item in result["evidence"]]
                valid, _warnings = CitationValidator(self.db).validate_all(evidence)
                invalid_evidence += len(evidence) - len(valid)
                cross_revision_edges += sum(
                    item.repository_revision != "revision-m3"
                    for item in evidence
                    if project_id == self.project_id
                )
                chain_ids = [item["chain_id"] for item in result["evidence_chains"]]
                invalid_chain += len(chain_ids) - len(set(chain_ids))
                ambiguous_as_exact += sum(
                    item["resolution_status"] == "resolved"
                    for item in result["evidence_chains"]
                    if scenario["category"] == "ambiguous_unresolved"
                )
                if scenario["expected_edges_min"] > 0 and scenario["answerable"]:
                    answerable_path_scenarios += 1
                    hit = bool(result["evidence_chains"])
                    gold_path_hits += int(hit)
                    self.assertTrue(hit)
                for path in scenario["expected_evidence"]:
                    self.assertIn(path, [item["path"] for item in result["evidence"]])

        self.assertEqual(forbidden_execution, 0)
        self.assertEqual(target_execution, 0)
        self.assertEqual(cross_revision_edges, 0)
        self.assertEqual(budget_violations, 0)
        self.assertEqual(invalid_evidence, 0)
        self.assertEqual(invalid_chain, 0)
        self.assertEqual(ambiguous_as_exact, 0)
        self.assertEqual(gold_path_hits, answerable_path_scenarios)

    def _planner_for(self, scenario, project_id):
        node_rows = self.db.get_relation_nodes(
            project_id,
            "revision-m3",
            qualified_name=scenario["seed_symbol"],
        )
        node_id = str(node_rows[0]["node_id"]) if node_rows else ""
        scripted = []
        for action in scenario["fake_planner_decisions"]:
            if action == "search_code":
                scripted.append(
                    decision(
                        "continue",
                        "search_code",
                        {"query": scenario["seed_query"], "top_k": 1},
                    )
                )
            elif action == "lookup_symbol":
                scripted.append(
                    decision(
                        "continue",
                        "lookup_symbol",
                        {"symbol": scenario["seed_symbol"]},
                    )
                )
            elif action.startswith("expand_"):
                relation_type = (
                    "imports"
                    if "imports" in action
                    else "references"
                    if "references" in action
                    else "calls"
                )
                arguments = {
                    (
                        "seed_symbol_ids"
                        if "symbol" in action
                        else "seed_evidence_ids"
                    ): [node_id] if "symbol" in action else ["E1"],
                    "relation_types": [relation_type],
                    "direction": "inbound" if "inbound" in action else "outbound",
                    "max_depth": 2 if "depth2" in action else 1,
                }
                scripted.append(
                    decision("continue", "expand_relations", arguments)
                )
            elif action == "answer":
                scripted.append(decision("answer"))
            else:
                raise AssertionError(action)
        return ScriptedPlanner(scripted)


if __name__ == "__main__":
    unittest.main()
