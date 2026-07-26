from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from app.m5.dataset import repository_content_fingerprint


REVISION_BY_REPO = {
    "repo-a": "a" * 40,
    "repo-b": "b" * 40,
    "repo-c": "c" * 40,
}
SOURCE = "def target():\n    return 1\n"
SOURCE_HASH = hashlib.sha256(SOURCE.encode("utf-8")).hexdigest()


def make_dataset(root: Path) -> tuple[Path, Path]:
    dataset = root / "dataset"
    repositories_root = root / "repositories"
    dataset.mkdir()
    repositories_root.mkdir()
    repositories = []
    for repo_id, revision in REVISION_BY_REPO.items():
        checkout = repositories_root / repo_id
        checkout.mkdir()
        (checkout / "app.py").write_text(SOURCE, encoding="utf-8")
        fingerprint = repository_content_fingerprint(checkout, ["app.py"])
        repositories.append(
            {
                "repo_id": repo_id,
                "display_name": repo_id,
                "source_url": f"https://example.invalid/{repo_id}",
                "license": "MIT",
                "language": "python",
                "exact_commit_sha": revision,
                "default_branch": "main",
                "acquisition_method": "local_existing",
                "checkout_name": repo_id,
                "content_fingerprint": f"sha256:{fingerprint}",
                "analysis_configuration": {
                    "include_globs": ["**/*.py"],
                    "maximum_file_bytes": 1000,
                    "maximum_files": 10,
                },
                "excluded_paths": [],
                "annotation_status": "agent_curated_pending_human_review",
            }
        )
    categories = ["locate"] * 9 + ["explain"] * 9 + ["relation"] * 9 + ["impact"] * 6 + ["unanswerable"] * 3
    scenarios = []
    for index, category in enumerate(categories):
        repo_id = list(REVISION_BY_REPO)[index % 3]
        unanswerable = category == "unanswerable"
        scenarios.append(
            {
                "scenario_id": f"scenario-{index:02d}",
                "dataset_version": "fixture-v1",
                "repo_id": repo_id,
                "repository_revision": REVISION_BY_REPO[repo_id],
                "language": "python",
                "question": f"Question {index} for target",
                "category": category,
                "difficulty": "hard" if category in {"relation", "impact", "unanswerable"} else "easy",
                "expected_target_type": "none" if unanswerable else ("relation" if category == "relation" else "symbol"),
                "expected_files": [] if unanswerable else ["app.py"],
                "expected_symbols": [] if unanswerable else ["target"],
                "expected_source_spans": [] if unanswerable else [{
                    "path": "app.py", "qualified_symbol": "target", "start_line": 1,
                    "end_line": 2, "content_hash": SOURCE_HASH,
                }],
                "expected_content_hashes": [] if unanswerable else [SOURCE_HASH],
                "expected_relation_edges": [] if category != "relation" else [{
                    "relation_type": "calls", "source_path": "app.py", "source_symbol": "target",
                    "target_path": "app.py", "target_symbol": "target",
                }],
                "expected_key_points": ["target"],
                "unanswerable": unanswerable,
                "allowed_evidence_scope": {"paths": [] if unanswerable else ["app.py"], "repository_only": True},
                "maximum_steps": 5,
                "maximum_tool_calls": 8,
                "annotation_provenance": "agent_assisted_developer_curation",
                "annotation_status": "agent_curated_pending_human_review",
                "annotation_note": "offline fixture pending review",
            }
        )
    sequences = [
        {
            "sequence_id": f"sequence-{index}", "dataset_version": "fixture-v1",
            "repo_id": repo_id, "repository_revision": REVISION_BY_REPO[repo_id],
            "target_path": "app.py", "target_symbol": "target",
            "steps": [{
                "step_id": "attempt-1", "task_type": "explain_symbol", "answer_text": "pass",
                "expected_verdict": "pass", "expected_state": "demonstrated",
                "expected_adaptation": "verified_pass_advance",
            }],
            "annotation_status": "agent_curated_pending_human_review",
        }
        for index, repo_id in enumerate(["repo-a", "repo-a", "repo-b", "repo-b", "repo-c", "repo-c"], 1)
    ]
    write_json(dataset / "manifest.json", {
        "benchmark_schema_version": 1, "metric_schema_version": 1,
        "dataset_version": "fixture-v1", "title": "fixture",
        "repositories_file": "repositories.json", "scenarios_file": "scenarios.jsonl",
        "sequences_file": "sequences.jsonl", "annotation_status": "agent_curated_pending_human_review",
        "minimum_scenarios": 36,
    })
    write_json(dataset / "repositories.json", repositories)
    write_jsonl(dataset / "scenarios.jsonl", scenarios)
    write_jsonl(dataset / "sequences.jsonl", sequences)
    return dataset, repositories_root


def mocked_git():
    def run(root: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", "HEAD"):
            return REVISION_BY_REPO[root.name]
        if arguments == ("ls-tree", "-r", "--name-only", "HEAD"):
            return "app.py"
        raise AssertionError(arguments)
    return patch("app.m5.dataset._git", side_effect=run)


def mutate_json(path: Path, callback: Any) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    callback(value)
    write_json(path, value)


def mutate_first_jsonl(path: Path, callback: Any) -> None:
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    callback(values[0])
    write_jsonl(path, values)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, values: list[Any]) -> None:
    path.write_text("".join(json.dumps(item) + "\n" for item in values), encoding="utf-8")
