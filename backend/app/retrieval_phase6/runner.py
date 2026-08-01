from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from app.database import Database
from app.m5.contracts import Scenario
from app.retrieval_phase5.contracts import FROZEN_PATHS
from app.retrieval_phase5.runner import CountingEmbeddingService, Phase5Harness
from app.services.embedding_service import EmbeddingService


class Phase6Harness:
    def __init__(
        self,
        *,
        database: Database,
        embedding_service: EmbeddingService | CountingEmbeddingService,
        projects_by_repo: dict[str, str],
        scenarios_by_repo: dict[str, Iterable[Scenario]],
        strata_by_query: dict[str, str],
        formal: bool,
    ) -> None:
        if set(projects_by_repo) != set(scenarios_by_repo):
            raise ValueError("project and scenario repository sets must match")
        self.database = database
        self.embedding_service = (
            embedding_service
            if isinstance(embedding_service, CountingEmbeddingService)
            else CountingEmbeddingService(embedding_service)
        )
        self.projects_by_repo = dict(projects_by_repo)
        self.scenarios_by_repo = {
            repo: tuple(sorted(values, key=lambda item: item.scenario_id))
            for repo, values in scenarios_by_repo.items()
        }
        query_ids = [
            item.scenario_id
            for repo in sorted(self.scenarios_by_repo)
            for item in self.scenarios_by_repo[repo]
        ]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("query identities must be globally unique across repositories")
        if set(query_ids) != set(strata_by_query):
            raise ValueError("every query must have exactly one frozen primary stratum")
        for repo, scenarios in self.scenarios_by_repo.items():
            if any(item.repo_id != repo for item in scenarios):
                raise ValueError("scenario repository attribution differs from its matrix repository")
        self.strata_by_query = dict(strata_by_query)
        self.harnesses = {
            repo: Phase5Harness(
                database=database,
                embedding_service=self.embedding_service,
                project_id=self.projects_by_repo[repo],
                scenarios=self.scenarios_by_repo[repo],
                formal=formal,
            )
            for repo in sorted(self.scenarios_by_repo)
        }

    def run_matrix(
        self,
        *,
        repo_order: list[str],
        path_order: list[str],
        reverse_queries: bool = False,
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        if len(repo_order) != len(set(repo_order)) or set(repo_order) != set(self.harnesses):
            raise ValueError("repository order must cover each frozen repository exactly once")
        expected_paths = {item.path_id for item in FROZEN_PATHS}
        if len(path_order) != len(set(path_order)) or set(path_order) != expected_paths:
            raise ValueError("path order must cover paths A through E exactly once")
        output: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for repo in repo_order:
            scenarios = self.scenarios_by_repo[repo]
            scenario_order = [item.scenario_id for item in (reversed(scenarios) if reverse_queries else scenarios)]
            matrix = self.harnesses[repo].run_matrix(
                path_order=path_order,
                scenario_order=scenario_order,
            )
            output[repo] = {
                path: [
                    {
                        **item,
                        "repository_id": repo,
                        "primary_stratum": self.strata_by_query[item["query_id"]],
                    }
                    for item in records
                ]
                for path, records in matrix.items()
            }
        return output


def phase6_determinism_summary(
    first: dict[str, dict[str, list[dict[str, Any]]]],
    second: dict[str, dict[str, list[dict[str, Any]]]],
    subset: dict[str, dict[str, list[dict[str, Any]]]] | None = None,
) -> dict[str, Any]:
    mismatches: list[dict[str, str]] = []
    if set(first) != set(second):
        raise ValueError("determinism matrices cover different repositories")
    for repo in sorted(first):
        if set(first[repo]) != set(second[repo]):
            raise ValueError("determinism matrices cover different paths")
        for path in sorted(first[repo]):
            first_by_id = {item["query_id"]: item for item in first[repo][path]}
            second_by_id = {item["query_id"]: item for item in second[repo][path]}
            if set(first_by_id) != set(second_by_id):
                raise ValueError("determinism matrices cover different query sets")
            for query_id in sorted(first_by_id):
                if _rank_identity(first_by_id[query_id]) != _rank_identity(second_by_id[query_id]):
                    mismatches.append({"repository_id": repo, "path_id": path, "query_id": query_id, "comparison": "reordered"})
    if subset:
        for repo in sorted(subset):
            for path in sorted(subset[repo]):
                baseline = {item["query_id"]: item for item in first[repo][path]}
                for item in subset[repo][path]:
                    if item["query_id"] not in baseline or _rank_identity(item) != _rank_identity(baseline[item["query_id"]]):
                        mismatches.append({"repository_id": repo, "path_id": path, "query_id": item["query_id"], "comparison": "repeated_subset"})
    return {
        "passed": not mismatches,
        "mismatches": mismatches,
        "normal_result_identity": matrix_identity(first),
        "reordered_result_identity": matrix_identity(second),
        "subset_result_identity": matrix_identity(subset) if subset else None,
        "rank_identity_gold_match_stable": not mismatches,
    }


def matrix_identity(matrix: dict[str, dict[str, list[dict[str, Any]]]] | None) -> str | None:
    if matrix is None:
        return None
    value = {
        repo: {
            path: {
                item["query_id"]: _rank_identity(item)
                for item in sorted(records, key=lambda record: record["query_id"])
            }
            for path, records in sorted(paths.items())
        }
        for repo, paths in sorted(matrix.items())
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _rank_identity(record: dict[str, Any]) -> list[tuple[Any, ...]]:
    return [
        (
            int(item.get("rank", 0)),
            str(item.get("chunk_identity", "")),
            bool(item.get("gold_match")),
        )
        for item in record.get("candidates", [])
    ]
