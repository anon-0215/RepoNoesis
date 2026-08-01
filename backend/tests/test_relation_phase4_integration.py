from __future__ import annotations

from dataclasses import asdict
import copy
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
from app.services.evidence import CitationValidator
from app.services.lexical_retriever import LexicalSearchResult
from app.services.learning_service import LearningService
from app.services.relation_graph import RelationValidator
from app.services.relation_retrieval import RelationRetrievalExpander
from app.services.retrieval_v2 import retrieve_code
from tests.m1_helpers import disabled_embedding_service
from tests.m3_helpers import call_chain_sources, make_relation_project
from tests.test_m1_ask import NoLlm


class RelationPhase4IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.directory.name) / "phase4.sqlite")
        self.project_id, self.bundle = make_relation_project(
            self.database, call_chain_sources()
        )
        self.chunks = {
            item["qualified_name"]: item
            for item in self.database.get_code_chunks(self.project_id)
        }

    def tearDown(self):
        self.directory.cleanup()

    def _lexical_a(self, *_args, **_kwargs):
        item = self.chunks["a"]
        return [
            LexicalSearchResult(
                project_id=self.project_id,
                repository_revision="revision-m3",
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

    def test_request_defaults_off_and_strictly_validates_combinations(self):
        main = importlib.import_module("app.main")
        self.assertEqual(main.AskRequest(question="a").relation_mode, "off")
        self.assertEqual(
            main.AskRequest(
                question="a", retrieval_version="v2", relation_mode="expand_v1"
            ).relation_mode,
            "expand_v1",
        )
        for invalid in ("", "   ", "EXPAND_V1", 1, True, [], {}, None):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                main.AskRequest(
                    question="a", retrieval_version="v2", relation_mode=invalid
                )
        with self.assertRaises(ValidationError):
            main.AskRequest(
                question="a", retrieval_version="v1", relation_mode="expand_v1"
            )

    def test_relation_off_never_calls_expander_and_freezes_plain_v2(self):
        with (
            patch("app.services.lexical_retriever.LexicalRetriever.search", side_effect=self._lexical_a),
            patch("app.services.symbol_retriever.SymbolRetriever.search", return_value=[]),
            patch.object(RelationRetrievalExpander, "expand", side_effect=AssertionError("called")),
        ):
            omitted = retrieve_code(
                self.database,
                disabled_embedding_service(),
                self.project_id,
                "a",
                retrieval_version="v2",
                evidence_count=3,
            )
            explicit = retrieve_code(
                self.database,
                disabled_embedding_service(),
                self.project_id,
                "a",
                retrieval_version="v2",
                relation_mode="off",
                evidence_count=3,
            )
        self.assertEqual(asdict(omitted), asdict(explicit))
        self.assertNotIn("relation", omitted.audit)

    def test_v2_relation_and_hierarchy_relation_call_phase4_after_frozen_stage(self):
        with (
            patch("app.services.lexical_retriever.LexicalRetriever.search", side_effect=self._lexical_a),
            patch("app.services.symbol_retriever.SymbolRetriever.search", return_value=[]),
        ):
            relation = retrieve_code(
                self.database,
                disabled_embedding_service(),
                self.project_id,
                "a",
                retrieval_version="v2",
                relation_mode="expand_v1",
                evidence_count=3,
            )
            combined = retrieve_code(
                self.database,
                disabled_embedding_service(),
                self.project_id,
                "a",
                retrieval_version="v2",
                hierarchy_mode="normalize_v1",
                relation_mode="expand_v1",
                evidence_count=3,
            )
        self.assertIn("relation", relation.audit)
        self.assertIn("relation", combined.audit)
        self.assertIn("hierarchy", combined.audit)
        self.assertTrue(any(item.retrieval_sources == ["relation"] for item in relation.results))

    def test_context_fingerprint_and_tool_schema_are_server_controlled(self):
        context = build_tool_context(
            request_id="phase4",
            bundle=self.bundle,
            database=self.database,
            embedding_service=disabled_embedding_service(),
            evidence_store=EvidenceStore(),
            limits=AgentLimits(),
            cancellation=CancellationToken(),
            deadline_monotonic=time.monotonic() + 60,
            retrieval_version="v2",
            relation_mode="expand_v1",
        )
        registry = build_m2_tool_registry(AgentLimits())
        forged = ToolCall(
            "C1",
            "S1",
            "search_code",
            "1",
            {"query": "a", "relation_mode": "off"},
            15_000,
            {},
        )
        self.assertEqual(registry.execute(context, forged).status, "rejected")
        m3_schema = registry.get("expand_relations").input_model.model_json_schema()
        self.assertNotIn("relation_mode", m3_schema["properties"])
        duplicate = ToolCall(
            "C2",
            "S2",
            "expand_relations",
            "1",
            {"seed_evidence_ids": ["E1"], "relation_types": ["calls"]},
            15_000,
            {},
        )
        self.assertEqual(registry.execute(context, duplicate).status, "failed")

    def test_agent_and_http_route_validate_relation_evidence_and_keep_learning(self):
        original_citations = CitationValidator.validate_all
        original_relations = RelationValidator.validate_chains
        with (
            patch("app.services.lexical_retriever.LexicalRetriever.search", side_effect=self._lexical_a),
            patch("app.services.symbol_retriever.SymbolRetriever.search", return_value=[]),
            patch.object(CitationValidator, "validate_all", autospec=True, side_effect=original_citations) as citations,
            patch.object(RelationValidator, "validate_chains", autospec=True, side_effect=original_relations) as relations,
        ):
            result = run_bounded_agent(
                "a",
                self.bundle,
                NoLlm(),
                self.database,
                disabled_embedding_service(),
                retrieval_version="v2",
                relation_mode="expand_v1",
                evidence_count=3,
            )
        self.assertTrue(any(item["retrieval_sources"] == ["relation"] for item in result["evidence"]))
        self.assertTrue(all(item["validation_status"] == "valid" for item in result["evidence"]))
        self.assertGreaterEqual(citations.call_count, 2)
        self.assertGreaterEqual(relations.call_count, 2)
        self.assertTrue(result["evidence_chains"])
        self.assertEqual(result["learning_schema_version"], 1)

        main = importlib.import_module("app.main")
        with (
            patch.object(main, "db", self.database),
            patch.object(main, "learning_service", LearningService(self.database)),
            patch.object(main, "llm", NoLlm()),
            patch.object(main, "embedding_service", disabled_embedding_service()),
            patch("app.services.lexical_retriever.LexicalRetriever.search", side_effect=self._lexical_a),
            patch("app.services.symbol_retriever.SymbolRetriever.search", return_value=[]),
        ):
            route = main.ask_project(
                self.project_id,
                main.AskRequest(
                    question="a",
                    retrieval_version="v2",
                    relation_mode="expand_v1",
                    evidence_count=3,
                ),
            )
        validated = main.AskResponse.model_validate(route)
        self.assertEqual(validated.evidence_schema_version, 1)
        self.assertEqual(validated.relation_schema_version, 1)
        self.assertTrue(validated.evidence)

    def test_phase4_chain_rejects_swapped_direction_type_and_evidence_binding(self):
        context = build_tool_context(
            request_id="chain-validation",
            bundle=self.bundle,
            database=self.database,
            embedding_service=disabled_embedding_service(),
            evidence_store=EvidenceStore(),
            limits=AgentLimits(),
            cancellation=CancellationToken(),
            deadline_monotonic=time.monotonic() + 60,
            retrieval_version="v2",
            relation_mode="expand_v1",
        )
        registry = build_m2_tool_registry(AgentLimits())
        with (
            patch("app.services.lexical_retriever.LexicalRetriever.search", side_effect=self._lexical_a),
            patch("app.services.symbol_retriever.SymbolRetriever.search", return_value=[]),
        ):
            observation = registry.execute(
                context,
                ToolCall(
                    "C1", "S1", "search_code", "1",
                    {"query": "a", "top_k": 3}, 15_000, {},
                ),
            )
        self.assertEqual(observation.status, "succeeded")
        evidence, _ = CitationValidator(self.database).validate_all(
            context.evidence_store.all(context.request_id)
        )
        evidence_by_chunk = {item.code_chunk_id: item.evidence_id for item in evidence}
        chain = context.chain_store.all(context.request_id)[0]
        validator = RelationValidator(self.database)
        valid, warnings = validator.validate_chains(
            owner_id=context.request_id,
            project_id=context.project_id,
            repository_revision=context.repository_revision,
            chains=[chain],
            valid_evidence_ids={item.evidence_id for item in evidence},
            evidence_by_chunk_id=evidence_by_chunk,
        )
        self.assertEqual((len(valid), warnings), (1, []))

        wrong_direction = copy.deepcopy(chain)
        wrong_direction.ordered_directions = ["incoming"]
        wrong_type = copy.deepcopy(chain)
        wrong_type.relation_types = ["imports"]
        wrong_evidence = copy.deepcopy(chain)
        wrong_evidence.supporting_evidence_ids = list(chain.seed_evidence_ids)
        for forged in (wrong_direction, wrong_type, wrong_evidence):
            with self.subTest(forged=forged):
                valid, warnings = validator.validate_chains(
                    owner_id=context.request_id,
                    project_id=context.project_id,
                    repository_revision=context.repository_revision,
                    chains=[forged],
                    valid_evidence_ids={item.evidence_id for item in evidence},
                    evidence_by_chunk_id=evidence_by_chunk,
                )
                self.assertEqual(valid, [])
                self.assertTrue(warnings)

    def test_interleaved_modes_do_not_share_request_state(self):
        with (
            patch("app.services.lexical_retriever.LexicalRetriever.search", side_effect=self._lexical_a),
            patch("app.services.symbol_retriever.SymbolRetriever.search", return_value=[]),
        ):
            first_off = retrieve_code(
                self.database, disabled_embedding_service(), self.project_id, "a",
                retrieval_version="v2", relation_mode="off", evidence_count=3,
            )
            enabled = retrieve_code(
                self.database, disabled_embedding_service(), self.project_id, "a",
                retrieval_version="v2", relation_mode="expand_v1", evidence_count=3,
            )
            second_off = retrieve_code(
                self.database, disabled_embedding_service(), self.project_id, "a",
                retrieval_version="v2", relation_mode="off", evidence_count=3,
            )
        self.assertEqual(asdict(first_off), asdict(second_off))
        self.assertNotEqual(asdict(first_off), asdict(enabled))


if __name__ == "__main__":
    unittest.main()
