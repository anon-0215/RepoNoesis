from __future__ import annotations

import ast
import hashlib
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any


PYTHON_LANGUAGE = "python"


@dataclass(frozen=True)
class CodeChunk:
    repository_revision: str
    language: str
    path: str
    chunk_type: str
    symbol_name: str
    qualified_name: str
    parent_symbol: str
    start_line: int
    end_line: int
    content: str
    content_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CodeChunkWarning:
    path: str
    message: str
    line: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CodeChunkExtractionResult:
    chunks: list[CodeChunk]
    warnings: list[CodeChunkWarning]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "warnings": [warning.to_dict() for warning in self.warnings],
        }


@dataclass(frozen=True)
class _SymbolContext:
    name: str
    qualified_name: str
    kind: str


def extract_python_code_chunks(
    path: str,
    source: str,
    repository_revision: str | None = None,
) -> CodeChunkExtractionResult:
    normalized_path = _normalize_repo_path(path)
    try:
        tree = ast.parse(source, filename=normalized_path)
    except SyntaxError as exc:
        return CodeChunkExtractionResult(
            chunks=[],
            warnings=[
                CodeChunkWarning(
                    path=normalized_path,
                    message=f"Python syntax error: {exc.msg}",
                    line=exc.lineno,
                )
            ],
        )

    visitor = _ChunkVisitor(
        path=normalized_path,
        source=source,
        repository_revision=repository_revision or "",
    )
    visitor.visit(tree)
    return CodeChunkExtractionResult(chunks=visitor.chunks, warnings=visitor.warnings)


def extract_python_code_chunks_from_files(
    files: list[Any],
    repository_revision: str | None = None,
) -> CodeChunkExtractionResult:
    chunks: list[CodeChunk] = []
    warnings: list[CodeChunkWarning] = []
    for file in files:
        path = _file_value(file, "path", "")
        if PurePosixPath(_normalize_repo_path(path)).suffix.lower() != ".py":
            continue
        content = _file_value(file, "content", None)
        if not isinstance(content, str):
            warnings.append(
                CodeChunkWarning(
                    path=_normalize_repo_path(path),
                    message="Python file content is unavailable; skipped code chunk extraction.",
                )
            )
            continue
        result = extract_python_code_chunks(path, content, repository_revision)
        chunks.extend(result.chunks)
        warnings.extend(result.warnings)
    return CodeChunkExtractionResult(chunks=chunks, warnings=warnings)


class _ChunkVisitor(ast.NodeVisitor):
    def __init__(self, path: str, source: str, repository_revision: str) -> None:
        self.path = path
        self.repository_revision = repository_revision
        self.lines = source.splitlines(keepends=True)
        self.chunks: list[CodeChunk] = []
        self.warnings: list[CodeChunkWarning] = []
        self._symbol_stack: list[_SymbolContext] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_callable(node, async_function=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_callable(node, async_function=True)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qualified_name = self._qualified_name(node.name)
        self._record_chunk(node, "class", node.name, qualified_name)
        self._symbol_stack.append(_SymbolContext(node.name, qualified_name, "class"))
        self.generic_visit(node)
        self._symbol_stack.pop()

    def _visit_callable(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        async_function: bool,
    ) -> None:
        qualified_name = self._qualified_name(node.name)
        inside_class = bool(self._symbol_stack and self._symbol_stack[-1].kind == "class")
        if inside_class:
            chunk_type = "async_method" if async_function else "method"
            context_kind = "method"
        else:
            chunk_type = "async_function" if async_function else "function"
            context_kind = "function"
        self._record_chunk(node, chunk_type, node.name, qualified_name)
        self._symbol_stack.append(_SymbolContext(node.name, qualified_name, context_kind))
        self.generic_visit(node)
        self._symbol_stack.pop()

    def _record_chunk(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        chunk_type: str,
        symbol_name: str,
        qualified_name: str,
    ) -> None:
        start_line = self._start_line(node)
        end_line = self._end_line(node)
        if start_line < 1 or end_line < start_line or end_line > len(self.lines):
            self.warnings.append(
                CodeChunkWarning(
                    path=self.path,
                    message=(
                        f"Invalid AST line range for {qualified_name}: "
                        f"{start_line}-{end_line}."
                    ),
                    line=getattr(node, "lineno", None),
                )
            )
            return
        content = "".join(self.lines[start_line - 1 : end_line])
        self.chunks.append(
            CodeChunk(
                repository_revision=self.repository_revision,
                language=PYTHON_LANGUAGE,
                path=self.path,
                chunk_type=chunk_type,
                symbol_name=symbol_name,
                qualified_name=qualified_name,
                parent_symbol=self._parent_symbol(),
                start_line=start_line,
                end_line=end_line,
                content=content,
                content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            )
        )

    def _qualified_name(self, symbol_name: str) -> str:
        parents = [context.name for context in self._symbol_stack]
        return ".".join([*parents, symbol_name])

    def _parent_symbol(self) -> str:
        return self._symbol_stack[-1].qualified_name if self._symbol_stack else ""

    @staticmethod
    def _start_line(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> int:
        line_numbers = [node.lineno]
        line_numbers.extend(decorator.lineno for decorator in node.decorator_list)
        return min(line_numbers)

    @staticmethod
    def _end_line(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> int:
        end_line = getattr(node, "end_lineno", None)
        if end_line is None:
            raise RuntimeError(
                "Python AST end_lineno is required for safe code chunk extraction. "
                "Run RepoNoesis with Python 3.8 or newer."
            )
        return int(end_line)


def _normalize_repo_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("/")


def _file_value(file: Any, name: str, default: Any) -> Any:
    if isinstance(file, dict):
        return file.get(name, default)
    return getattr(file, name, default)
