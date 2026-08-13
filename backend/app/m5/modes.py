from __future__ import annotations

from dataclasses import replace
from typing import Any

from app.database import Database
from app.m5.contracts import EXPERIMENT_MODES, Scenario
from app.services.agent_contracts import AgentLimits, PlannerDecision
from app.services.agent_core import Planner, run_bounded_agent
from app.services.agent_tools import ToolRegistry, build_m2_tool_registry
from app.services.embedding_service import EmbeddingService
from app.services.evidence import EvidenceBuilder
from app.services.hybrid_retriever import HybridRetriever, HybridSearchResult
from app.services.lexical_retriever import LexicalRetriever
from app.services.qa_agent import answer_from_evidence, answer_question
from app.services.semantic_retriever import SemanticRetriever


MODE_TOOLS: dict[str, tuple[str, ...]] = {
    "fixed_lexical_rag": (),
    "fixed_dense_rag": (),
    "m1_hybrid_rag": (),
    "m2_bounded_agent": ("search_code", "lookup_symbol", "read_source", "validate_evidence"),
    "m3_relation_agent": (
        "search_code", "lookup_symbol", "read_source", "validate_evidence", "expand_relations"
    ),
    "m4_profiled_agent": (
        "get_learning_context", "search_code", "lookup_symbol", "read_source",
        "validate_evidence", "expand_relations",
    ),
    "m4_adaptive_sequence": (),
}


class ModeExecutionError(RuntimeError):
    pass


class DeterministicBenchmarkPlanner(Planner):
    """Offline planner that exercises the same Agent Core without pretending to be live."""

    def __init__(self, mode: str, scenario: Scenario) -> None:
        self.mode = mode
        self.scenario = scenario

    def decide(
        self,
        state: dict[str, Any],
        *,
        repair_hint: str | None = None,
    ) -> tuple[dict[str, Any], int]:
        observations = state.get("observations", [])
        tools = [str(item.get("tool")) for item in observations]
        known_evidence = list(state.get("known_evidence_ids", []))
        if self.mode == "m4_profiled_agent" and "get_learning_context" not in tools:
            decision = {
                "status": "continue",
                "action": "get_learning_context",
                "arguments": {},
                "decision_summary": "Read server-bound learning context once.",
            }
        elif "search_code" not in tools:
            decision = {
                "status": "continue",
                "action": "search_code",
                "arguments": {"query": self.scenario.question, "top_k": 5},
                "decision_summary": "Search the fixed repository snapshot.",
            }
        elif (
            self.mode in {"m3_relation_agent", "m4_profiled_agent"}
            and self.scenario.category in {"relation", "impact"}
            and "expand_relations" not in tools
            and known_evidence
        ):
            decision = {
                "status": "continue",
                "action": "expand_relations",
                "arguments": {
                    "seed_evidence_ids": known_evidence[:4],
                    "relation_types": ["imports", "calls", "references", "defines"],
                    "direction": "both",
                    "max_depth": 2,
                    "per_node_limit": 12,
                },
                "decision_summary": "Expand bounded static relations for relation-sensitive gold.",
            }
        else:
            decision = {
                "status": "insufficient_evidence" if self.scenario.unanswerable else "answer",
                "action": None,
                "arguments": {},
                "decision_summary": "Stop after bounded evidence collection.",
            }
        return decision, max(1, len(str(decision)) // 4)


def execute_mode(
    mode: str,
    scenario: Scenario,
    bundle: dict[str, Any],
    database: Database,
    embedding_service: EmbeddingService,
    llm: Any,
    *,
    limits: AgentLimits,
    deterministic_planner: bool,
    learning_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if mode not in EXPERIMENT_MODES:
        raise ModeExecutionError(f"unknown experiment mode: {mode}")
    if mode == "m4_adaptive_sequence":
        raise ModeExecutionError("adaptive sequences use the isolated learning sequence runner")
    if mode == "fixed_lexical_rag":
        result = _fixed_lexical(scenario, bundle, database, llm)
    elif mode == "fixed_dense_rag":
        result = _fixed_dense(scenario, bundle, database, embedding_service, llm)
    elif mode == "m1_hybrid_rag":
        result = answer_question(
            scenario.question,
            bundle,
            llm,
            database,
            embedding_service,
            evidence_count=5,
        )
    else:
        registry = _restricted_registry(limits, MODE_TOOLS[mode])
        planner = DeterministicBenchmarkPlanner(mode, scenario) if deterministic_planner else None
        result = run_bounded_agent(
            scenario.question,
            bundle,
            llm,
            database,
            embedding_service,
            evidence_count=5,
            limits=replace(
                limits,
                max_agent_steps=min(limits.max_agent_steps, scenario.maximum_steps),
                max_tool_calls=min(limits.max_tool_calls, scenario.maximum_tool_calls),
            ),
            planner=planner,
            registry=registry,
            learning_context=learning_context if mode == "m4_profiled_agent" else None,
        )
    return {
        **result,
        "experiment_mode": mode,
        "mode_control_source": "trusted_benchmark_config",
        "allowed_tools": list(MODE_TOOLS[mode]),
        "citation_validator_enabled": True,
        "relation_validator_enabled": mode in {"m3_relation_agent", "m4_profiled_agent"},
        "learning_context_is_repository_evidence": False,
    }


def _fixed_lexical(
    scenario: Scenario,
    bundle: dict[str, Any],
    database: Database,
    llm: Any,
) -> dict[str, Any]:
    project = bundle["project"]
    results = LexicalRetriever(database).search(project["id"], scenario.question, top_k=5)
    hybrid = [
        HybridSearchResult(
            project_id=item.project_id,
            repository_revision=item.repository_revision,
            code_chunk_id=item.code_chunk_id,
            language=item.language,
            path=item.path,
            chunk_type=item.chunk_type,
            symbol_name=item.symbol_name,
            qualified_name=item.qualified_name,
            start_line=item.start_line,
            end_line=item.end_line,
            content=item.content,
            content_hash=item.content_hash,
            retrieval_sources=["lexical"],
            lexical_score=item.lexical_score,
            lexical_rank=item.lexical_rank,
            fusion_score=item.lexical_score,
            fusion_rank=index,
        )
        for index, item in enumerate(results, start=1)
    ]
    evidence = EvidenceBuilder().build(hybrid, project)
    return answer_from_evidence(
        scenario.question,
        evidence,
        llm,
        database,
        retrieval_mode="lexical",
        max_answer_tokens=1_600,
        answer_timeout_seconds=60.0,
    )


def _fixed_dense(
    scenario: Scenario,
    bundle: dict[str, Any],
    database: Database,
    embedding_service: EmbeddingService,
    llm: Any,
) -> dict[str, Any]:
    if not embedding_service.settings.enabled:
        raise ModeExecutionError("fixed_dense_rag requires an enabled embedding provider")
    project = bundle["project"]
    outcome = SemanticRetriever(database, embedding_service).search(
        project["id"], scenario.question, top_k=5, local_files_only=True
    )
    hybrid = [
        HybridSearchResult(
            project_id=item.project_id,
            repository_revision=item.repository_revision,
            code_chunk_id=item.code_chunk_id,
            language=item.language,
            path=item.path,
            chunk_type=item.chunk_type,
            symbol_name=item.symbol_name,
            qualified_name=item.qualified_name,
            start_line=item.start_line,
            end_line=item.end_line,
            content=item.content,
            content_hash=item.content_hash,
            retrieval_sources=["semantic"],
            semantic_score=item.semantic_score,
            semantic_rank=index,
            fusion_score=item.semantic_score,
            fusion_rank=index,
        )
        for index, item in enumerate(outcome.results, start=1)
    ]
    evidence = EvidenceBuilder().build(hybrid, project)
    return answer_from_evidence(
        scenario.question,
        evidence,
        llm,
        database,
        retrieval_mode="dense",
        warnings=outcome.warnings,
        max_answer_tokens=1_600,
        answer_timeout_seconds=60.0,
    )


def _restricted_registry(limits: AgentLimits, allowed: tuple[str, ...]) -> ToolRegistry:
    full = build_m2_tool_registry(limits)
    restricted = ToolRegistry()
    for name in allowed:
        restricted.register(full.get(name))
    return restricted
