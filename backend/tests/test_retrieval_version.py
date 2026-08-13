from __future__ import annotations

import importlib
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
from app.services.embedding_indexer import EmbeddingIndexer
from app.services.learning_service import LearningService
from app.services.retrieval_v2 import V2_FUSION_VERSION
from tests.m1_helpers import disabled_embedding_service, make_project
from tests.test_m1_ask import NoLlm, enabled_embedding_service


class RetrievalVersionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.directory.name) / "retrieval-version.sqlite")
        self.project_id, self.bundle = make_project(
            self.database,
            [
                (
                    "src/auth.py",
                    "authenticate_user",
                    "def authenticate_user(password):\n    return verify_password(password)\n",
                ),
                ("src/helper.py", "verify_password", "def verify_password(value):\n    return bool(value)\n"),
            ],
        )

    def tearDown(self):
        self.directory.cleanup()

    def test_request_defaults_to_v1_and_rejects_unknown_case_blank_and_whitespace(self):
        main = importlib.import_module("app.main")
        self.assertEqual(main.AskRequest(question="auth").retrieval_version, "v1")
        self.assertEqual(
            main.AskRequest(question="auth", retrieval_version="v1").retrieval_version,
            "v1",
        )
        self.assertEqual(
            main.AskRequest(question="auth", retrieval_version="v2").retrieval_version,
            "v2",
        )
        for invalid in ("v3", "V1", "V2", "", "   "):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                main.AskRequest(question="auth", retrieval_version=invalid)

    def test_context_binds_versions_per_request_without_global_state(self):
        common = {
            "bundle": self.bundle,
            "database": self.database,
            "embedding_service": disabled_embedding_service(),
            "evidence_store": EvidenceStore(),
            "limits": AgentLimits(),
            "cancellation": CancellationToken(),
            "deadline_monotonic": time.monotonic() + 60,
        }
        v1 = build_tool_context(request_id="v1-a", retrieval_version="v1", **common)
        v2 = build_tool_context(
            request_id="v2",
            retrieval_version="v2",
            **{**common, "evidence_store": EvidenceStore()},
        )
        v1_again = build_tool_context(
            request_id="v1-b",
            **{**common, "evidence_store": EvidenceStore()},
        )
        self.assertEqual((v1.retrieval_version, v2.retrieval_version, v1_again.retrieval_version), ("v1", "v2", "v1"))

    def test_v1_does_not_invoke_phase1_analysis_or_symbol_source(self):
        with (
            patch("app.services.retrieval_v2.QueryAnalyzer.analyze", side_effect=AssertionError("v2 analyzer called")),
            patch("app.services.retrieval_v2.SymbolRetriever.search", side_effect=AssertionError("v2 symbol called")),
        ):
            result = run_bounded_agent(
                "authenticate_user",
                self.bundle,
                NoLlm(),
                self.database,
                disabled_embedding_service(),
            )
        self.assertTrue(result["evidence"])
        self.assertEqual(result["retrieval_mode"], "lexical")
        self.assertTrue(
            all(
                item["retrieval_strategy_version"] == "weighted-rrf-v1"
                for item in result["evidence"]
            )
        )

    def test_v2_runs_three_sources_and_enters_existing_evidence_validation_chain(self):
        service = enabled_embedding_service()
        EmbeddingIndexer(self.database, service).index_project(self.project_id)
        result = run_bounded_agent(
            "Where is `authenticate_user` defined?",
            self.database.get_bundle(self.project_id),
            NoLlm(),
            self.database,
            service,
            retrieval_version="v2",
        )

        self.assertTrue(result["evidence"])
        auth = next(
            item for item in result["evidence"] if item["qualified_name"] == "authenticate_user"
        )
        self.assertEqual(auth["validation_status"], "valid")
        self.assertEqual(auth["retrieval_strategy_version"], V2_FUSION_VERSION)
        self.assertEqual(auth["retrieval_sources"], ["dense", "lexical", "symbol"])
        self.assertEqual(result["relation_schema_version"], 1)
        self.assertEqual(result["learning_schema_version"], 1)
        self.assertLessEqual(result["budget_usage"]["tool_calls_used"], 8)

    def test_server_context_not_planner_arguments_selects_v2(self):
        service = enabled_embedding_service()
        EmbeddingIndexer(self.database, service).index_project(self.project_id)
        limits = AgentLimits()
        context = build_tool_context(
            request_id="request-v2",
            bundle=self.database.get_bundle(self.project_id),
            database=self.database,
            embedding_service=service,
            evidence_store=EvidenceStore(),
            limits=limits,
            cancellation=CancellationToken(),
            deadline_monotonic=time.monotonic() + 60,
            retrieval_version="v2",
        )
        registry = build_m2_tool_registry(limits)
        forged = ToolCall(
            "C1",
            "S1",
            "search_code",
            "1",
            {"query": "authenticate_user", "retrieval_version": "v1"},
            15_000,
            {},
        )
        self.assertEqual(registry.execute(context, forged).status, "rejected")

        valid = ToolCall(
            "C2",
            "S2",
            "search_code",
            "1",
            {"query": "Where is `authenticate_user` defined?"},
            15_000,
            {},
        )
        observation = registry.execute(context, valid)
        self.assertEqual(observation.status, "succeeded")
        self.assertEqual(observation.structured_results["retrieval_version"], "v2")
        self.assertEqual(
            observation.structured_results["retrieval_audit"]["fusion_version"],
            V2_FUSION_VERSION,
        )
        self.assertEqual(
            set(observation.structured_results["retrieval_audit"]["sources"]),
            {"dense", "lexical", "symbol"},
        )

    def test_http_route_propagates_v2_and_preserves_public_response_shape_and_learning(self):
        main = importlib.import_module("app.main")
        service = enabled_embedding_service()
        EmbeddingIndexer(self.database, service).index_project(self.project_id)
        learning_service = LearningService(self.database)
        with (
            patch.object(main, "db", self.database),
            patch.object(main, "learning_service", learning_service),
            patch.object(main, "llm", NoLlm()),
            patch.object(main, "embedding_service", service),
        ):
            result = main.ask_project(
                self.project_id,
                main.AskRequest(
                    question="Where is `authenticate_user` defined?",
                    retrieval_version="v2",
                ),
            )
            validated = main.AskResponse.model_validate(result)
        self.assertEqual(validated.evidence_schema_version, 1)
        self.assertEqual(validated.agent_schema_version, 1)
        self.assertEqual(validated.relation_schema_version, 1)
        self.assertEqual(validated.learning_schema_version, 1)
        self.assertTrue(validated.evidence)
        self.assertEqual(
            validated.evidence[0].retrieval_strategy_version,
            V2_FUSION_VERSION,
        )

    def test_unknown_direct_version_fails_before_retrieval(self):
        with self.assertRaises(ValueError):
            run_bounded_agent(
                "authenticate_user",
                self.bundle,
                NoLlm(),
                self.database,
                disabled_embedding_service(),
                retrieval_version="v3",
            )


if __name__ == "__main__":
    unittest.main()
