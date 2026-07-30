from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from pydantic import ValidationError

from app.m5.contracts import (
    AdaptiveSequence,
    DatasetManifest,
    RepositorySpec,
    Scenario,
)
from app.m5.live_dense_protocol import (
    DEFAULT_LIVE_DENSE_PROTOCOL_PATH,
    LiveDenseProtocolError,
    load_live_dense_protocol,
    validate_protocol_repository_coverage,
)


@dataclass
class ValidationReport:
    valid: bool
    dataset_version: str = ""
    repository_count: int = 0
    scenario_count: int = 0
    sequence_count: int = 0
    category_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "dataset_version": self.dataset_version,
            "repository_count": self.repository_count,
            "scenario_count": self.scenario_count,
            "sequence_count": self.sequence_count,
            "category_counts": dict(self.category_counts),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class LoadedDataset:
    manifest: DatasetManifest
    repositories: list[RepositorySpec]
    scenarios: list[Scenario]
    sequences: list[AdaptiveSequence]
    directory: Path


class DatasetValidationError(ValueError):
    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        super().__init__("benchmark dataset validation failed: " + "; ".join(report.errors[:5]))


class BenchmarkDatasetValidator:
    def __init__(
        self,
        dataset_directory: Path,
        repository_root: Path,
        live_dense_protocol_path: Path = DEFAULT_LIVE_DENSE_PROTOCOL_PATH,
    ) -> None:
        self.dataset_directory = dataset_directory.resolve()
        self.repository_root = repository_root.resolve()
        self.live_dense_protocol_path = live_dense_protocol_path.resolve()

    def validate(self) -> ValidationReport:
        report = ValidationReport(valid=False)
        try:
            dataset = self._load()
        except (
            OSError,
            json.JSONDecodeError,
            ValidationError,
            LiveDenseProtocolError,
            ValueError,
        ) as exc:
            report.errors.append(f"dataset load failed: {type(exc).__name__}: {exc}")
            return report
        report.dataset_version = dataset.manifest.dataset_version
        report.repository_count = len(dataset.repositories)
        report.scenario_count = len(dataset.scenarios)
        report.sequence_count = len(dataset.sequences)
        report.category_counts = dict(Counter(item.category for item in dataset.scenarios))
        self._validate_identities(dataset, report)
        repository_indexes: dict[str, _RepositoryIndex] = {}
        for repository in dataset.repositories:
            try:
                repository_indexes[repository.repo_id] = self._validate_repository(repository)
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                report.errors.append(f"repository {repository.repo_id}: {exc}")
        for scenario in dataset.scenarios:
            index = repository_indexes.get(scenario.repo_id)
            if index is not None:
                self._validate_scenario(scenario, index, report)
        for sequence in dataset.sequences:
            index = repository_indexes.get(sequence.repo_id)
            if index is not None:
                self._validate_sequence(sequence, index, report)
        self._validate_dataset_shape(dataset, report)
        report.valid = not report.errors
        annotations = [*dataset.scenarios, *dataset.sequences]
        if all(item.annotation_status == "agent_curated_pending_human_review" for item in annotations):
            report.warnings.append("All annotations are pending human review.")
        return report

    def load_validated(self) -> LoadedDataset:
        report = self.validate()
        if not report.valid:
            raise DatasetValidationError(report)
        return self._load()

    def _load(self) -> LoadedDataset:
        manifest = DatasetManifest.model_validate(_read_json(self.dataset_directory / "manifest.json"))
        repositories_raw = _read_json(self.dataset_directory / manifest.repositories_file)
        if not isinstance(repositories_raw, list):
            raise ValueError("repositories.json must contain a JSON array")
        repositories = [RepositorySpec.model_validate(item) for item in repositories_raw]
        protocol = load_live_dense_protocol(self.live_dense_protocol_path)
        validate_protocol_repository_coverage(protocol, repositories)
        scenarios = [Scenario.model_validate(item) for item in _read_jsonl(self.dataset_directory / manifest.scenarios_file)]
        sequences = [AdaptiveSequence.model_validate(item) for item in _read_jsonl(self.dataset_directory / manifest.sequences_file)]
        return LoadedDataset(manifest, repositories, scenarios, sequences, self.dataset_directory)

    def _validate_identities(self, dataset: LoadedDataset, report: ValidationReport) -> None:
        _require_unique([item.repo_id for item in dataset.repositories], "repo_id", report)
        _require_unique([item.scenario_id for item in dataset.scenarios], "scenario_id", report)
        _require_unique([item.sequence_id for item in dataset.sequences], "sequence_id", report)
        known = {item.repo_id: item for item in dataset.repositories}
        questions: set[tuple[str, str]] = set()
        for scenario in dataset.scenarios:
            repository = known.get(scenario.repo_id)
            if repository is None:
                report.errors.append(f"{scenario.scenario_id}: unknown repo_id {scenario.repo_id}")
                continue
            if scenario.dataset_version != dataset.manifest.dataset_version:
                report.errors.append(f"{scenario.scenario_id}: dataset version mismatch")
            if scenario.repository_revision != repository.exact_commit_sha:
                report.errors.append(f"{scenario.scenario_id}: repository revision cross-contamination")
            key = (scenario.repo_id, " ".join(scenario.question.casefold().split()))
            if key in questions:
                report.errors.append(f"{scenario.scenario_id}: duplicate normalized question")
            questions.add(key)
        for sequence in dataset.sequences:
            repository = known.get(sequence.repo_id)
            if repository is None or sequence.repository_revision != repository.exact_commit_sha:
                report.errors.append(f"{sequence.sequence_id}: sequence repository/revision mismatch")
            if sequence.dataset_version != dataset.manifest.dataset_version:
                report.errors.append(f"{sequence.sequence_id}: sequence dataset version mismatch")
        if dataset.manifest.annotation_status == "human_reviewed" and any(
            item.annotation_status != "human_reviewed" for item in [*dataset.scenarios, *dataset.sequences]
        ):
            report.errors.append("human-reviewed manifest contains annotations without completed review")

    def _validate_repository(self, spec: RepositorySpec) -> "_RepositoryIndex":
        root = _safe_checkout(self.repository_root, spec.checkout_name)
        if not root.is_dir():
            raise ValueError("checkout directory does not exist")
        head = _git(root, "rev-parse", "HEAD")
        if head != spec.exact_commit_sha:
            raise ValueError(f"HEAD mismatch: expected {spec.exact_commit_sha}, got {head}")
        files = _git(root, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
        index = _index_repository(root, files, spec)
        fingerprint = repository_content_fingerprint(root, files, spec.excluded_paths)
        if spec.content_fingerprint != f"sha256:{fingerprint}":
            raise ValueError("content fingerprint mismatch")
        return index

    def _validate_scenario(
        self,
        scenario: Scenario,
        index: "_RepositoryIndex",
        report: ValidationReport,
    ) -> None:
        prefix = scenario.scenario_id
        paths = [*scenario.expected_files, *scenario.allowed_evidence_scope.paths]
        paths.extend(span.path for span in scenario.expected_source_spans)
        for path in paths:
            error = _path_error(path)
            if error:
                report.errors.append(f"{prefix}: {error}: {path}")
            elif path not in index.lines:
                report.errors.append(f"{prefix}: expected file is missing: {path}")
        for symbol in scenario.expected_symbols:
            if symbol not in index.symbols:
                report.errors.append(f"{prefix}: expected symbol is missing: {symbol}")
        span_hashes: list[str] = []
        for span in scenario.expected_source_spans:
            lines = index.lines.get(span.path)
            if lines is None:
                continue
            if span.end_line > len(lines):
                report.errors.append(f"{prefix}: source span exceeds file: {span.path}")
                continue
            source = "".join(lines[span.start_line - 1 : span.end_line])
            digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
            span_hashes.append(digest)
            if digest != span.content_hash:
                report.errors.append(f"{prefix}: source span content hash mismatch")
            if not _span_matches_symbol(index, span, allow_subspan=scenario.category == "relation"):
                report.errors.append(f"{prefix}: source span does not match AST symbol identity")
        if sorted(set(scenario.expected_content_hashes)) != sorted(set(span_hashes)):
            report.errors.append(f"{prefix}: expected_content_hashes do not match source spans")
        for edge in scenario.expected_relation_edges:
            if edge.source_path not in index.lines:
                report.errors.append(f"{prefix}: relation source path is missing")
            if edge.source_symbol not in index.symbols:
                report.errors.append(f"{prefix}: relation source symbol is missing")
            if edge.target_path and edge.target_path not in index.lines:
                report.errors.append(f"{prefix}: relation target path is missing")
            if edge.target_symbol not in index.symbols:
                report.errors.append(f"{prefix}: relation target symbol is missing")
            if edge.relation_type == "calls" and not _call_edge_exists(index, edge.source_path, edge.source_symbol, edge.target_symbol):
                report.errors.append(f"{prefix}: declared call relation is not present in source")
        if sum(len(value) for value in scenario.expected_key_points) > 4_000:
            report.errors.append(f"{prefix}: expected key points exceed byte budget")

    def _validate_sequence(
        self,
        sequence: AdaptiveSequence,
        index: "_RepositoryIndex",
        report: ValidationReport,
    ) -> None:
        prefix = sequence.sequence_id
        if sequence.target_path not in index.lines:
            report.errors.append(f"{prefix}: sequence target file is missing")
        if sequence.target_symbol not in index.symbols:
            report.errors.append(f"{prefix}: sequence target symbol is missing")
        for step in sequence.steps:
            step_prefix = f"{prefix}/{step.step_id}"
            if "controlled benchmark answer" in step.answer_text.casefold():
                report.errors.append(f"{step_prefix}: placeholder answer_text is forbidden")
            if sum(len(value) for value in step.expected_key_points) > 4_000:
                report.errors.append(f"{step_prefix}: expected key points exceed byte budget")
            for span in step.expected_source_spans:
                lines = index.lines.get(span.path)
                if lines is None:
                    report.errors.append(f"{step_prefix}: expected source file is missing")
                    continue
                if span.end_line > len(lines):
                    report.errors.append(f"{step_prefix}: source span exceeds file")
                    continue
                source = "".join(lines[span.start_line - 1 : span.end_line])
                digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
                if digest != span.content_hash:
                    report.errors.append(f"{step_prefix}: source span content hash mismatch")
                if not _span_matches_symbol(
                    index,
                    span,
                    allow_subspan=step.task_type == "trace_static_relation",
                ):
                    report.errors.append(f"{step_prefix}: source span does not match AST symbol identity")
            for edge in step.expected_relation_edges:
                if edge.source_path not in index.lines or edge.target_path not in index.lines:
                    report.errors.append(f"{step_prefix}: relation file is missing")
                if edge.source_symbol not in index.symbols or edge.target_symbol not in index.symbols:
                    report.errors.append(f"{step_prefix}: relation symbol is missing")
                if edge.relation_type == "calls" and not _call_edge_exists(
                    index, edge.source_path, edge.source_symbol, edge.target_symbol
                ):
                    report.errors.append(f"{step_prefix}: declared call relation is not present in source")

    @staticmethod
    def _validate_dataset_shape(dataset: LoadedDataset, report: ValidationReport) -> None:
        if len(dataset.scenarios) < dataset.manifest.minimum_scenarios:
            report.errors.append("dataset has fewer scenarios than manifest minimum")
        minimums = {"locate": 9, "explain": 9, "relation": 9, "impact": 6, "unanswerable": 3}
        counts = Counter(item.category for item in dataset.scenarios)
        for category, minimum in minimums.items():
            if counts[category] < minimum:
                report.errors.append(f"dataset requires at least {minimum} {category} scenarios")
        per_repo = Counter(item.repo_id for item in dataset.scenarios)
        for repository in dataset.repositories:
            if per_repo[repository.repo_id] < 10:
                report.errors.append(f"repository {repository.repo_id} has fewer than 10 scenarios")
        if len(dataset.repositories) < 3:
            report.errors.append("pilot dataset requires at least three repositories")
        if len(dataset.sequences) < 6:
            report.errors.append("pilot dataset requires at least six adaptive sequences")


@dataclass(frozen=True)
class _RepositoryIndex:
    lines: dict[str, list[str]]
    symbols: dict[str, list[tuple[str, int, int]]]
    calls: dict[tuple[str, str], set[str]]


def repository_content_fingerprint(
    root: Path,
    tracked_files: Iterable[str] | None = None,
    excluded_paths: Iterable[str] = (),
) -> str:
    files = list(tracked_files) if tracked_files is not None else _git(
        root, "ls-tree", "-r", "--name-only", "HEAD"
    ).splitlines()
    excluded = tuple(value.rstrip("/") + "/" for value in excluded_paths)
    digest = hashlib.sha256()
    for relative in sorted(files):
        if not relative.endswith(".py") or any(relative.startswith(prefix) for prefix in excluded):
            continue
        path = _safe_file(root, relative)
        if path.is_symlink() or not path.is_file():
            continue
        content = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def _index_repository(root: Path, files: list[str], spec: RepositorySpec) -> _RepositoryIndex:
    lines: dict[str, list[str]] = {}
    symbols: dict[str, list[tuple[str, int, int]]] = {}
    calls: dict[tuple[str, str], set[str]] = {}
    excluded = tuple(value.rstrip("/") + "/" for value in spec.excluded_paths)
    selected = [
        path for path in files
        if path.endswith(".py") and not any(path.startswith(prefix) for prefix in excluded)
    ]
    if len(selected) > spec.analysis_configuration.maximum_files:
        raise ValueError("repository exceeds configured Python file limit")
    for relative in selected:
        error = _path_error(relative)
        if error:
            raise ValueError(error)
        path = _safe_file(root, relative)
        if path.is_symlink():
            raise ValueError(f"tracked Python symlink is not allowed: {relative}")
        if path.stat().st_size > spec.analysis_configuration.maximum_file_bytes:
            continue
        source = path.read_text(encoding="utf-8")
        source_lines = source.splitlines(keepends=True)
        lines[relative] = source_lines
        try:
            tree = ast.parse(source, filename=relative)
        except SyntaxError:
            continue
        stack: list[str] = []

        class Visitor(ast.NodeVisitor):
            def _visit(self, node: ast.AST) -> None:
                name = str(getattr(node, "name"))
                qualified = ".".join([*stack, name])
                start = min(
                    [int(getattr(node, "lineno")), *[int(item.lineno) for item in getattr(node, "decorator_list", [])]]
                )
                end = int(getattr(node, "end_lineno"))
                symbols.setdefault(qualified, []).append((relative, start, end))
                symbols.setdefault(name, []).append((relative, start, end))
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    calls[(relative, qualified)] = _collect_direct_calls(node.body)
                stack.append(name)
                self.generic_visit(node)
                stack.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self._visit(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self._visit(node)

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                self._visit(node)

        Visitor().visit(tree)
    return _RepositoryIndex(lines, symbols, calls)


def _collect_direct_calls(body: list[ast.stmt]) -> set[str]:
    names: set[str] = set()

    class CallVisitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return None

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return None

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return None

    visitor = CallVisitor()
    for statement in body:
        visitor.visit(statement)
    return names


def _call_edge_exists(
    index: _RepositoryIndex,
    source_path: str,
    source_symbol: str,
    target_symbol: str,
) -> bool:
    called_name = target_symbol.rsplit(".", 1)[-1]
    return called_name in index.calls.get((source_path, source_symbol), set())


def _span_matches_symbol(
    index: _RepositoryIndex,
    span: SourceSpan,
    *,
    allow_subspan: bool,
) -> bool:
    identities = index.symbols.get(span.qualified_symbol, [])
    if not allow_subspan:
        return (span.path, span.start_line, span.end_line) in identities
    return any(
        path == span.path and start <= span.start_line <= span.end_line <= end
        for path, start, end in identities
    )


def _safe_checkout(repository_root: Path, checkout_name: str) -> Path:
    candidate = (repository_root / checkout_name).resolve()
    if candidate == repository_root or repository_root not in candidate.parents:
        raise ValueError("checkout path escapes repository root")
    return candidate


def _safe_file(root: Path, relative: str) -> Path:
    candidate = (root / PurePosixPath(relative)).resolve()
    if root not in candidate.parents:
        raise ValueError(f"repository path escapes checkout: {relative}")
    return candidate


def _path_error(value: str) -> str | None:
    if not value or "\\" in value or value.startswith(("/", "~")):
        return "path is not normalized relative POSIX form"
    path = PurePosixPath(value)
    if any(part in {"", ".", ".."} for part in path.parts):
        return "path traversal or non-normalized segment"
    if re.match(r"^[A-Za-z]:", value):
        return "absolute Windows path is forbidden"
    return None


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=30,
    )
    return completed.stdout.strip()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[Any]:
    values: list[Any] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            values.append(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path.name}:{line_number}") from exc
    return values


def _require_unique(values: list[str], label: str, report: ValidationReport) -> None:
    counts = Counter(values)
    for value, count in counts.items():
        if count > 1:
            report.errors.append(f"duplicate {label}: {value}")
