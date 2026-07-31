from __future__ import annotations

import importlib
from dataclasses import asdict
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from app.database import Database
from app.services.agent_contracts import AgentLimits, CancellationToken, ToolCall
from app.services.agent_core import run_bounded_agent
from app.services.agent_tools import EvidenceStore, build_m2_tool_registry, build_tool_context
from app.services.code_chunker import extract_python_code_chunks
from app.services.hierarchy_normalization import (
    HIERARCHY_MODE_NORMALIZE_V1,
    HIERARCHY_NORMALIZATION_VERSION,
    HierarchyResolver,
)
from app.services.lexical_retriever import LexicalSearchResult
from app.services.evidence import CitationValidator
from app.services.learning_service import LearningService
from app.services.relation_graph import RelationValidator
from app.services.retrieval_v2 import retrieve_code
from tests.m1_helpers import disabled_embedding_service
from tests.test_m1_ask import NoLlm
from tests.test_m2_agent import ScriptedPlanner, decision


REVISION = "revision-hierarchy-integration"


class HierarchyIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.directory.name) / "integration.sqlite")
        self.project_id = self.database.create_project(
            {
                "repo_url": "https://github.com/demo/hierarchy-integration",
                "owner": "demo",
                "repo": "hierarchy-integration",
                "default_branch": "main",
                "repository_revision": REVISION,
            }
        )
        self.source = (
            "def outer():\n"
            "    def inner():\n"
            "        return 1\n"
            "    return inner()\n"
        )
        chunks = extract_python_code_chunks("src/app.py", self.source, REVISION).chunks
        files = [
            {
                "path": "src/app.py",
                "extension": ".py",
                "language": "Python",
                "size": len(self.source.encode()),
                "content": self.source,
                "summary": "nested fixture",
                "importance": 100,
                "is_core": True,
                "imports": [],
                "exports": [],
                "symbols": ["outer", "outer.inner"],
            }
        ]
        self.database.save_analysis(
            self.project_id,
            {
                "primary_language": "Python",
                "frameworks": [],
                "files": files,
                "modules": [],
                "overview": "hierarchy fixture",
            },
            files,
            [],
            [item.to_dict() for item in chunks],
        )
        self.bundle = self.database.get_bundle(self.project_id)
        assert self.bundle is not None
        self.chunks = {
            item["qualified_name"]: item
            for item in self.database.get_code_chunks(self.project_id)
        }

    def tearDown(self):
        self.directory.cleanup()

    def _lexical_inner(self, *_args, **_kwargs):
        item = self.chunks["outer.inner"]
        return [
            LexicalSearchResult(
                project_id=self.project_id,
                repository_revision=REVISION,
                code_chunk_id=int(item["id"]),
                language=item["language"],
                path=item["path"],
                chunk_type=item["chunk_type"],
                symbol_name=item["symbol_name"],
                qualified_name=item["qualified_name"],
                start_line=int(item["start_line"]),
                end_line=int(item["end_line"]),
                content=item["content"],
                content_hash=item["content_hash"],
                lexical_score=10.0,
                lexical_rank=1,
            )
        ]

    def test_request_schema_defaults_off_and_strictly_rejects_invalid_modes_and_v1_pair(self):
        main = importlib.import_module("app.main")
        self.assertEqual(main.AskRequest(question="inner").hierarchy_mode, "off")
        self.assertEqual(
            main.AskRequest(
                question="inner",
                retrieval_version="v2",
                hierarchy_mode="normalize_v1",
            ).hierarchy_mode,
            "normalize_v1",
        )
        for invalid in ("unknown", "", "   ", "NORMALIZE_V1", 1, {"mode": "normalize_v1"}):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                main.AskRequest(
                    question="inner",
                    retrieval_version="v2",
                    hierarchy_mode=invalid,
                )
        with self.assertRaises(ValidationError):
            main.AskRequest(
                question="inner",
                retrieval_version="v1",
                hierarchy_mode="normalize_v1",
            )

    def test_context_binds_hierarchy_per_request_and_planner_cannot_override_it(self):
        common = {
            "bundle": self.bundle,
            "database": self.database,
            "embedding_service": disabled_embedding_service(),
            "limits": AgentLimits(),
            "cancellation": CancellationToken(),
            "deadline_monotonic": time.monotonic() + 60,
            "retrieval_version": "v2",
        }
        off = build_tool_context(
            request_id="off",
            evidence_store=EvidenceStore(),
            hierarchy_mode="off",
            **common,
        )
        normalized = build_tool_context(
            request_id="normalized",
            evidence_store=EvidenceStore(),
            hierarchy_mode="normalize_v1",
            **common,
        )
        self.assertEqual((off.hierarchy_mode, normalized.hierarchy_mode), ("off", "normalize_v1"))

        call = ToolCall(
            "C1",
            "S1",
            "search_code",
            "1",
            {"query": "inner", "hierarchy_mode": "off"},
            15_000,
            {},
        )
        observation = build_m2_tool_registry(AgentLimits()).execute(normalized, call)
        self.assertEqual(observation.status, "rejected")

    def test_plain_v2_never_calls_resolver_and_freezes_phase2_output(self):
        with (
            patch("app.services.lexical_retriever.LexicalRetriever.search", side_effect=self._lexical_inner),
            patch("app.services.symbol_retriever.SymbolRetriever.search", return_value=[]),
            patch.object(HierarchyResolver, "resolve", side_effect=AssertionError("resolver called")),
        ):
            omitted = retrieve_code(
                self.database,
                disabled_embedding_service(),
                self.project_id,
                "inner",
                retrieval_version="v2",
                evidence_count=2,
            )
            explicit_off = retrieve_code(
                self.database,
                disabled_embedding_service(),
                self.project_id,
                "inner",
                retrieval_version="v2",
                hierarchy_mode="off",
                evidence_count=2,
            )

        self.assertEqual(asdict(omitted), asdict(explicit_off))
        self.assertEqual([item.qualified_name for item in omitted.results], ["outer.inner"])
        self.assertNotIn("hierarchy", omitted.audit)

    def test_off_normalized_off_calls_do_not_share_request_state(self):
        with (
            patch("app.services.lexical_retriever.LexicalRetriever.search", side_effect=self._lexical_inner),
            patch("app.services.symbol_retriever.SymbolRetriever.search", return_value=[]),
        ):
            first_off = retrieve_code(
                self.database,
                disabled_embedding_service(),
                self.project_id,
                "inner",
                retrieval_version="v2",
                hierarchy_mode="off",
                evidence_count=2,
            )
            normalized = retrieve_code(
                self.database,
                disabled_embedding_service(),
                self.project_id,
                "inner",
                retrieval_version="v2",
                hierarchy_mode="normalize_v1",
                evidence_count=2,
            )
            second_off = retrieve_code(
                self.database,
                disabled_embedding_service(),
                self.project_id,
                "inner",
                retrieval_version="v2",
                hierarchy_mode="off",
                evidence_count=2,
            )

        self.assertEqual(asdict(first_off), asdict(second_off))
        self.assertEqual(len(first_off.results), 1)
        self.assertEqual(len(normalized.results), 2)

    def test_v2_hierarchy_adds_real_parent_and_keeps_fusion_and_normalization_versions_separate(self):
        with (
            patch("app.services.lexical_retriever.LexicalRetriever.search", side_effect=self._lexical_inner),
            patch("app.services.symbol_retriever.SymbolRetriever.search", return_value=[]),
        ):
            outcome = retrieve_code(
                self.database,
                disabled_embedding_service(),
                self.project_id,
                "inner",
                retrieval_version="v2",
                hierarchy_mode="normalize_v1",
                evidence_count=2,
            )

        self.assertEqual([item.qualified_name for item in outcome.results], ["outer.inner", "outer"])
        parent = outcome.results[1]
        self.assertEqual(parent.retrieval_sources, ["hierarchy"])
        self.assertEqual(parent.code_chunk_id, self.chunks["outer"]["id"])
        self.assertEqual(parent.content, self.chunks["outer"]["content"])
        self.assertEqual(outcome.audit["fusion_version"], "weighted_rrf_v2@1")
        self.assertEqual(
            outcome.audit["hierarchy"]["normalization_version"],
            HIERARCHY_NORMALIZATION_VERSION,
        )

    def test_run_and_http_route_propagate_mode_into_existing_evidence_validation_and_learning(self):
        main = importlib.import_module("app.main")
        with (
            patch("app.services.lexical_retriever.LexicalRetriever.search", side_effect=self._lexical_inner),
            patch("app.services.symbol_retriever.SymbolRetriever.search", return_value=[]),
        ):
            result = run_bounded_agent(
                "inner",
                self.bundle,
                NoLlm(),
                self.database,
                disabled_embedding_service(),
                retrieval_version="v2",
                hierarchy_mode=HIERARCHY_MODE_NORMALIZE_V1,
                evidence_count=2,
            )
        self.assertEqual({item["qualified_name"] for item in result["evidence"]}, {"outer", "outer.inner"})
        self.assertTrue(all(item["validation_status"] == "valid" for item in result["evidence"]))
        self.assertEqual(result["relation_schema_version"], 1)
        self.assertEqual(result["learning_schema_version"], 1)
        self.assertLessEqual(result["budget_usage"]["tool_calls_used"], 8)

        with (
            patch.object(main, "db", self.database),
            patch.object(main, "learning_service", LearningService(self.database)),
            patch.object(main, "llm", NoLlm()),
            patch.object(main, "embedding_service", disabled_embedding_service()),
            patch("app.services.lexical_retriever.LexicalRetriever.search", side_effect=self._lexical_inner),
            patch("app.services.symbol_retriever.SymbolRetriever.search", return_value=[]),
        ):
            route_result = main.ask_project(
                self.project_id,
                main.AskRequest(
                    question="inner",
                    retrieval_version="v2",
                    hierarchy_mode="normalize_v1",
                    evidence_count=2,
                ),
            )
            validated = main.AskResponse.model_validate(route_result)
        self.assertEqual(validated.evidence_schema_version, 1)
        self.assertTrue(validated.evidence)

    def test_bounded_agent_still_executes_citation_and_relation_validators(self):
        planner = ScriptedPlanner(
            [
                decision("continue", "search_code", {"query": "inner", "top_k": 2}),
                decision("answer"),
            ]
        )
        original_citations = CitationValidator.validate_all
        original_relations = RelationValidator.validate_chains
        with (
            patch("app.services.lexical_retriever.LexicalRetriever.search", side_effect=self._lexical_inner),
            patch("app.services.symbol_retriever.SymbolRetriever.search", return_value=[]),
            patch.object(
                CitationValidator,
                "validate_all",
                autospec=True,
                side_effect=original_citations,
            ) as citation_validate,
            patch.object(
                RelationValidator,
                "validate_chains",
                autospec=True,
                side_effect=original_relations,
            ) as relation_validate,
        ):
            result = run_bounded_agent(
                "inner",
                self.bundle,
                NoLlm(),
                self.database,
                disabled_embedding_service(),
                retrieval_version="v2",
                hierarchy_mode="normalize_v1",
                evidence_count=2,
                planner=planner,
            )

        self.assertTrue(result["evidence"])
        self.assertGreaterEqual(citation_validate.call_count, 2)
        self.assertGreaterEqual(relation_validate.call_count, 2)

    def test_invalid_direct_pair_fails_before_any_retrieval(self):
        with patch(
            "app.services.agent_tools.retrieve_code",
            side_effect=AssertionError("retrieval partially executed"),
        ):
            with self.assertRaises(ValueError):
                run_bounded_agent(
                    "inner",
                    self.bundle,
                    NoLlm(),
                    self.database,
                    disabled_embedding_service(),
                    retrieval_version="v1",
                    hierarchy_mode="normalize_v1",
                )


if __name__ == "__main__":
    unittest.main()
