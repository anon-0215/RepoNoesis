from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from app.database import Database


RELATION_SCHEMA_VERSION = 1
RELATION_TYPES = frozenset({"imports", "calls", "references", "defines"})
RESOLUTION_STATUSES = frozenset(
    {"resolved", "ambiguous", "unresolved", "external", "unsupported"}
)


@dataclass(frozen=True)
class RelationNode:
    node_id: str
    project_id: str
    repository_revision: str
    language: str
    node_type: str
    path: str
    code_chunk_id: int | None
    symbol_name: str
    qualified_name: str
    start_line: int
    end_line: int
    content_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RelationEdge:
    edge_id: str
    project_id: str
    repository_revision: str
    relation_type: str
    source_node_id: str
    source_path: str
    source_chunk_id: int | None
    source_symbol: str
    source_start_line: int
    source_end_line: int
    target_node_id: str | None
    target_path: str | None
    target_chunk_id: int | None
    target_symbol: str | None
    target_start_line: int | None
    target_end_line: int | None
    raw_target_name: str
    resolution_status: str
    resolution_rule: str
    language: str
    source_content_hash: str
    target_content_hash: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RelationIndexResult:
    nodes: list[RelationNode]
    edges: list[RelationEdge]
    parsed_files: int
    failed_files: int
    unsupported_files: int
    warnings: list[str]

    @property
    def status(self) -> str:
        return "complete" if self.failed_files == 0 else "partial"


@dataclass(frozen=True)
class _ModuleTarget:
    path: str
    module: str


@dataclass
class _Scope:
    qualified_name: str
    kind: str
    parent: _Scope | None
    definitions: dict[str, list[RelationNode]]
    shadowed: set[str]
    class_name: str = ""


class PythonRelationIndexer:
    """Extract conservative static relations without importing or executing source."""

    def build(
        self,
        *,
        project_id: str,
        repository_revision: str,
        files: list[dict[str, Any]],
        code_chunks: list[dict[str, Any]],
    ) -> RelationIndexResult:
        python_files = sorted(
            (
                file
                for file in files
                if PurePosixPath(_normalize_path(str(file.get("path", "")))).suffix.lower()
                == ".py"
            ),
            key=lambda item: _normalize_path(str(item.get("path", ""))),
        )
        chunks = sorted(
            (
                chunk
                for chunk in code_chunks
                if str(chunk.get("language", "")).casefold() == "python"
                and str(chunk.get("repository_revision", "")) == repository_revision
            ),
            key=lambda item: (
                _normalize_path(str(item.get("path", ""))),
                int(item.get("start_line", 0)),
                str(item.get("qualified_name", "")),
                int(item.get("id", 0)),
            ),
        )
        nodes, file_nodes, chunk_nodes = self._build_nodes(
            project_id, repository_revision, files, chunks
        )
        module_map = _build_module_map(python_files)
        chunks_by_path: dict[str, list[dict[str, Any]]] = {}
        for chunk in chunks:
            chunks_by_path.setdefault(_normalize_path(str(chunk["path"])), []).append(chunk)

        edges: list[RelationEdge] = []
        warnings: list[str] = []
        failed_files = 0
        for file in python_files:
            path = _normalize_path(str(file.get("path", "")))
            content = file.get("content")
            if not isinstance(content, str):
                failed_files += 1
                warnings.append(f"{path}: Python source content unavailable.")
                continue
            try:
                tree = ast.parse(content, filename=path)
            except SyntaxError as exc:
                failed_files += 1
                warnings.append(
                    f"{path}:{exc.lineno or 0}: Python syntax error; relations skipped."
                )
                continue
            resolver = _FileResolver(
                project_id=project_id,
                repository_revision=repository_revision,
                path=path,
                source=content,
                tree=tree,
                file_node=file_nodes[path],
                file_nodes=file_nodes,
                chunks=chunks_by_path.get(path, []),
                chunk_nodes=chunk_nodes,
                module_map=module_map,
                all_chunks=chunks,
            )
            edges.extend(resolver.resolve())

        for chunk in chunks:
            node = chunk_nodes[int(chunk["id"])]
            parent = _definition_parent(
                file_nodes[node.path],
                node,
                chunks_by_path.get(node.path, []),
                chunk_nodes,
            )
            edges.append(
                _make_edge(
                    relation_type="defines",
                    source=parent,
                    target=node,
                    source_line=node.start_line,
                    raw_target_name=node.qualified_name,
                    status="resolved",
                    rule=(
                        "class_defines_method"
                        if parent.node_type in {"class", "function", "method"}
                        else "file_defines_symbol"
                    ),
                )
            )

        unique_nodes = {node.node_id: node for node in nodes}
        unique_edges = {edge.edge_id: edge for edge in edges}
        return RelationIndexResult(
            nodes=sorted(
                unique_nodes.values(),
                key=lambda item: (
                    item.path,
                    item.start_line,
                    item.qualified_name,
                    item.node_id,
                ),
            ),
            edges=sorted(
                unique_edges.values(),
                key=lambda item: (
                    item.source_path,
                    item.source_start_line,
                    item.relation_type,
                    item.raw_target_name,
                    item.target_path or "",
                    item.target_start_line or 0,
                    item.edge_id,
                ),
            ),
            parsed_files=len(python_files) - failed_files,
            failed_files=failed_files,
            unsupported_files=max(0, len(files) - len(python_files)),
            warnings=warnings,
        )

    @staticmethod
    def _build_nodes(
        project_id: str,
        repository_revision: str,
        files: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
    ) -> tuple[
        list[RelationNode],
        dict[str, RelationNode],
        dict[int, RelationNode],
    ]:
        nodes: list[RelationNode] = []
        file_nodes: dict[str, RelationNode] = {}
        for file in sorted(files, key=lambda item: str(item.get("path", ""))):
            path = _normalize_path(str(file.get("path", "")))
            content = str(file.get("content", ""))
            language = str(file.get("language", "")).casefold()
            if language == "python":
                language = "python"
            node = _make_node(
                project_id=project_id,
                revision=repository_revision,
                language=language or "text",
                node_type="file",
                path=path,
                code_chunk_id=None,
                symbol_name=PurePosixPath(path).name,
                qualified_name=_module_name(path),
                start_line=1,
                end_line=max(1, len(content.splitlines())),
                content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            )
            nodes.append(node)
            file_nodes[path] = node

        chunk_nodes: dict[int, RelationNode] = {}
        for chunk in chunks:
            chunk_id = int(chunk["id"])
            node = _make_node(
                project_id=project_id,
                revision=repository_revision,
                language="python",
                node_type=str(chunk["chunk_type"]),
                path=_normalize_path(str(chunk["path"])),
                code_chunk_id=chunk_id,
                symbol_name=str(chunk["symbol_name"]),
                qualified_name=str(chunk["qualified_name"]),
                start_line=int(chunk["start_line"]),
                end_line=int(chunk["end_line"]),
                content_hash=str(chunk["content_hash"]),
            )
            nodes.append(node)
            chunk_nodes[chunk_id] = node
        return nodes, file_nodes, chunk_nodes


def index_project_relations(
    database: Database,
    project_id: str,
) -> RelationIndexResult:
    """Index the persisted SQLite snapshot in a separate, replace-all transaction."""
    bundle = database.get_bundle(project_id)
    if bundle is None:
        raise ValueError("project does not exist")
    chunks = bundle.get("code_chunks", [])
    revisions = {str(item.get("repository_revision", "")) for item in chunks}
    project_revision = str(
        (bundle.get("project") or {}).get("repository_revision", "")
    )
    if project_revision:
        revisions.add(project_revision)
    revisions.discard("")
    if len(revisions) != 1:
        raise ValueError("project must have exactly one repository revision")
    revision = next(iter(revisions))
    result = PythonRelationIndexer().build(
        project_id=project_id,
        repository_revision=revision,
        files=bundle.get("files", []),
        code_chunks=chunks,
    )
    database.replace_relation_index(
        project_id,
        revision,
        [item.to_dict() for item in result.nodes],
        [item.to_dict() for item in result.edges],
        status=result.status,
        parsed_files=result.parsed_files,
        failed_files=result.failed_files,
        unsupported_files=result.unsupported_files,
        warnings=result.warnings,
    )
    return result


class _FileResolver(ast.NodeVisitor):
    def __init__(
        self,
        *,
        project_id: str,
        repository_revision: str,
        path: str,
        source: str,
        tree: ast.Module,
        file_node: RelationNode,
        file_nodes: dict[str, RelationNode],
        chunks: list[dict[str, Any]],
        chunk_nodes: dict[int, RelationNode],
        module_map: dict[str, list[_ModuleTarget]],
        all_chunks: list[dict[str, Any]],
    ) -> None:
        self.project_id = project_id
        self.repository_revision = repository_revision
        self.path = path
        self.source = source
        self.tree = tree
        self.file_node = file_node
        self.file_nodes = file_nodes
        self.chunks = chunks
        self.chunk_nodes = chunk_nodes
        self.module_map = module_map
        self.all_chunks = all_chunks
        self.edges: list[RelationEdge] = []
        self.import_bindings: dict[str, list[RelationNode]] = {}
        self.module_bindings: dict[str, list[str]] = {}
        self.scope = _Scope("", "module", None, {}, set())
        self._scope_stack: list[_Scope] = [self.scope]
        self._build_scope_definitions()

    def resolve(self) -> list[RelationEdge]:
        self.visit(self.tree)
        return self.edges

    def _build_scope_definitions(self) -> None:
        scopes: dict[str, _Scope] = {"": self.scope}
        for chunk in sorted(self.chunks, key=lambda item: int(item["start_line"])):
            qualified = str(chunk["qualified_name"])
            parent_name = str(chunk.get("parent_symbol", ""))
            parent_scope = scopes.get(parent_name, self.scope)
            node = self.chunk_nodes[int(chunk["id"])]
            parent_scope.definitions.setdefault(node.symbol_name, []).append(node)
            scopes[qualified] = _Scope(
                qualified_name=qualified,
                kind=str(chunk["chunk_type"]),
                parent=parent_scope,
                definitions={},
                shadowed=set(),
                class_name=(
                    qualified
                    if str(chunk["chunk_type"]) == "class"
                    else parent_scope.class_name
                ),
            )
        self._scopes = scopes

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            targets, status, rule = self._resolve_module(alias.name)
            binding = alias.asname or alias.name.split(".", 1)[0]
            self.module_bindings[binding] = [target.path for target in targets]
            self._record_candidates(
                "imports", node.lineno, alias.name, targets, status, rule
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module_name, module_status, module_rule = self._absolute_from_module(node)
        module_targets, resolved_status, resolved_rule = self._resolve_module(module_name)
        if module_status != "resolved":
            resolved_status, resolved_rule = module_status, module_rule
        for alias in node.names:
            raw = f"{'.' * node.level}{node.module or ''}:{alias.name}"
            symbol_targets: list[RelationNode] = []
            submodule_targets: list[_ModuleTarget] = []
            for target in module_targets:
                symbol_targets.extend(self._symbols_in_path(target.path, alias.name))
                submodule_targets.extend(
                    self.module_map.get(
                        ".".join(value for value in (target.module, alias.name) if value),
                        [],
                    )
                )
            targets: list[RelationNode | _ModuleTarget]
            if symbol_targets:
                targets = _dedupe_nodes(symbol_targets)
                status = "resolved" if len(targets) == 1 else "ambiguous"
                rule = "relative_import" if node.level else "explicit_import"
            elif submodule_targets:
                targets = _dedupe_modules(submodule_targets)
                status = "resolved" if len(targets) == 1 else "ambiguous"
                rule = "relative_import" if node.level else "explicit_import"
            else:
                targets = module_targets
                status = resolved_status
                rule = resolved_rule
            binding = alias.asname or alias.name
            relation_nodes = [self._as_node(target) for target in targets]
            if symbol_targets:
                self.import_bindings[binding] = relation_nodes
            elif submodule_targets:
                self.module_bindings[binding] = [target.path for target in submodule_targets]
            self._record_nodes("imports", node.lineno, raw, relation_nodes, status, rule)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_callable(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_callable(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scope(node)

    def _visit_callable(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        scope = self._scope_for_node(node)
        if scope is None:
            self.generic_visit(node)
            return
        scope.shadowed.update(_argument_names(node.args))
        scope.shadowed.update(_assigned_names(node))
        self._scope_stack.append(scope)
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)
        for statement in node.body:
            self.visit(statement)
        self._scope_stack.pop()

    def _visit_scope(self, node: ast.ClassDef) -> None:
        scope = self._scope_for_node(node)
        if scope is None:
            self.generic_visit(node)
            return
        self._scope_stack.append(scope)
        for base in node.bases:
            self.visit(base)
        for statement in node.body:
            self.visit(statement)
        self._scope_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        raw = _safe_unparse(node.func)
        targets, status, rule = self._resolve_expression(node.func)
        self._record_nodes("calls", node.lineno, raw, targets, status, rule)
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Name(self, node: ast.Name) -> None:
        if not isinstance(node.ctx, ast.Load):
            return
        targets, status, rule = self._resolve_name(node.id)
        if targets or status in {"ambiguous", "unresolved"}:
            self._record_nodes(
                "references", node.lineno, node.id, targets, status, rule
            )

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if not isinstance(node.ctx, ast.Load):
            return
        targets, status, rule = self._resolve_expression(node)
        if targets or status in {"ambiguous", "unresolved"}:
            self._record_nodes(
                "references",
                node.lineno,
                _safe_unparse(node),
                targets,
                status,
                rule,
            )
        self.visit(node.value)

    def _resolve_expression(
        self, expression: ast.expr
    ) -> tuple[list[RelationNode], str, str]:
        if isinstance(expression, ast.Name):
            return self._resolve_name(expression.id)
        if isinstance(expression, ast.Attribute):
            if isinstance(expression.value, ast.Name):
                base = expression.value.id
                if base in self.module_bindings:
                    targets: list[RelationNode] = []
                    for path in self.module_bindings[base]:
                        targets.extend(self._symbols_in_path(path, expression.attr))
                    targets = _dedupe_nodes(targets)
                    return _candidate_status(targets, "module_alias")
                if base in {"self", "cls"} and self._current_scope().class_name:
                    targets = self._symbols_by_qualified(
                        f"{self._current_scope().class_name}.{expression.attr}"
                    )
                    return _candidate_status(targets, "self_method")
                class_targets, class_status, _rule = self._resolve_name(base)
                class_nodes = [
                    item for item in class_targets if item.node_type == "class"
                ]
                if len(class_nodes) == 1:
                    targets = self._symbols_by_qualified(
                        f"{class_nodes[0].qualified_name}.{expression.attr}",
                        path=class_nodes[0].path,
                    )
                    return _candidate_status(targets, "class_qualified")
                if class_status == "ambiguous":
                    return [], "ambiguous", "class_qualified_ambiguous"
            return [], "unresolved", "dynamic_attribute"
        return [], "unresolved", "dynamic_call"

    def _resolve_name(self, name: str) -> tuple[list[RelationNode], str, str]:
        scope: _Scope | None = self._current_scope()
        while scope is not None:
            targets = scope.definitions.get(name, [])
            if targets:
                rule = "same_local_scope" if scope.kind != "module" else "same_module"
                return _candidate_status(targets, rule)
            if name in scope.shadowed:
                return [], "unresolved", "local_or_parameter_shadowing"
            scope = scope.parent
        if name in self.import_bindings:
            return _candidate_status(self.import_bindings[name], "import_alias")
        if name in self.module_bindings:
            return [], "unresolved", "module_object_not_callable"
        if name in _BUILTINS:
            return [], "external", "python_builtin"
        return [], "unresolved", "name_not_bound"

    def _resolve_module(
        self, module: str
    ) -> tuple[list[_ModuleTarget], str, str]:
        targets = _dedupe_modules(self.module_map.get(module, []))
        if len(targets) == 1:
            return targets, "resolved", "explicit_import"
        if len(targets) > 1:
            return targets, "ambiguous", "multiple_internal_modules"
        first = module.split(".", 1)[0]
        internal_prefix = any(
            name == first or name.startswith(first + ".") for name in self.module_map
        )
        return (
            [],
            "unresolved" if internal_prefix else "external",
            "internal_module_missing" if internal_prefix else "external_dependency",
        )

    def _absolute_from_module(self, node: ast.ImportFrom) -> tuple[str, str, str]:
        module = node.module or ""
        if node.level == 0:
            return module, "resolved", "explicit_import"
        current = _module_name(self.path)
        package_parts = current.split(".") if current else []
        if PurePosixPath(self.path).name != "__init__.py":
            package_parts = package_parts[:-1]
        ascend = node.level - 1
        if ascend > len(package_parts):
            return module, "unresolved", "relative_import_out_of_bounds"
        base = package_parts[: len(package_parts) - ascend]
        absolute = ".".join([*base, *([module] if module else [])])
        return absolute, "resolved", "relative_import"

    def _symbols_in_path(self, path: str, symbol: str) -> list[RelationNode]:
        return [
            self.chunk_nodes[int(chunk["id"])]
            for chunk in self.all_chunks
            if _normalize_path(str(chunk["path"])) == path
            and (
                str(chunk["symbol_name"]) == symbol
                or str(chunk["qualified_name"]) == symbol
            )
        ]

    def _symbols_by_qualified(
        self, qualified_name: str, *, path: str | None = None
    ) -> list[RelationNode]:
        return [
            self.chunk_nodes[int(chunk["id"])]
            for chunk in self.all_chunks
            if str(chunk["qualified_name"]) == qualified_name
            and (path is None or _normalize_path(str(chunk["path"])) == path)
        ]

    def _scope_for_node(self, node: ast.AST) -> _Scope | None:
        current = self._current_scope().qualified_name
        name = str(getattr(node, "name", ""))
        qualified = ".".join(value for value in (current, name) if value)
        return self._scopes.get(qualified)

    def _current_scope(self) -> _Scope:
        return self._scope_stack[-1]

    def _source_node(self, line: int) -> RelationNode:
        candidates = [
            self.chunk_nodes[int(chunk["id"])]
            for chunk in self.chunks
            if int(chunk["start_line"]) <= line <= int(chunk["end_line"])
        ]
        if not candidates:
            return self.file_node
        return min(
            candidates,
            key=lambda item: (
                item.end_line - item.start_line,
                -item.start_line,
                item.node_id,
            ),
        )

    def _as_node(self, target: RelationNode | _ModuleTarget) -> RelationNode:
        if isinstance(target, RelationNode):
            return target
        return self._file_node_for_path(target.path)

    def _file_node_for_path(self, path: str) -> RelationNode:
        return self.file_nodes[path]

    def _record_candidates(
        self,
        relation_type: str,
        line: int,
        raw: str,
        targets: list[_ModuleTarget],
        status: str,
        rule: str,
    ) -> None:
        self._record_nodes(
            relation_type,
            line,
            raw,
            [self._as_node(target) for target in targets],
            status,
            rule,
        )

    def _record_nodes(
        self,
        relation_type: str,
        line: int,
        raw: str,
        targets: list[RelationNode],
        status: str,
        rule: str,
    ) -> None:
        source = self._source_node(line)
        if status == "resolved" and len(targets) != 1:
            status = "unresolved" if not targets else "ambiguous"
        if status == "ambiguous":
            # Preserve each candidate edge, explicitly marked ambiguous.
            if targets:
                for target in targets:
                    self.edges.append(
                        _make_edge(
                            relation_type,
                            source,
                            target,
                            line,
                            raw,
                            "ambiguous",
                            rule,
                        )
                    )
                return
        target = targets[0] if len(targets) == 1 else None
        self.edges.append(
            _make_edge(
                relation_type,
                source,
                target,
                line,
                raw,
                status,
                rule,
            )
        )


def _definition_parent(
    file_node: RelationNode,
    node: RelationNode,
    chunks: list[dict[str, Any]],
    chunk_nodes: dict[int, RelationNode],
) -> RelationNode:
    parent_name = next(
        (
            str(chunk.get("parent_symbol", ""))
            for chunk in chunks
            if int(chunk["id"]) == node.code_chunk_id
        ),
        "",
    )
    if not parent_name:
        return file_node
    parents = [
        chunk_nodes[int(chunk["id"])]
        for chunk in chunks
        if str(chunk["qualified_name"]) == parent_name
    ]
    return parents[0] if len(parents) == 1 else file_node


def _make_node(
    *,
    project_id: str,
    revision: str,
    language: str,
    node_type: str,
    path: str,
    code_chunk_id: int | None,
    symbol_name: str,
    qualified_name: str,
    start_line: int,
    end_line: int,
    content_hash: str,
) -> RelationNode:
    identity = {
        "project_id": project_id,
        "revision": revision,
        "language": language,
        "node_type": node_type,
        "path": path,
        "qualified_name": qualified_name,
        "start_line": start_line,
        "end_line": end_line,
        "content_hash": content_hash,
    }
    node_id = "N" + hashlib.sha256(_canonical(identity)).hexdigest()
    return RelationNode(
        node_id=node_id,
        project_id=project_id,
        repository_revision=revision,
        language=language,
        node_type=node_type,
        path=path,
        code_chunk_id=code_chunk_id,
        symbol_name=symbol_name,
        qualified_name=qualified_name,
        start_line=start_line,
        end_line=end_line,
        content_hash=content_hash,
    )


def _make_edge(
    relation_type: str,
    source: RelationNode,
    target: RelationNode | None,
    source_line: int,
    raw_target_name: str,
    status: str,
    rule: str,
) -> RelationEdge:
    raw = raw_target_name[:500]
    identity = {
        "project_id": source.project_id,
        "revision": source.repository_revision,
        "relation_type": relation_type,
        "source_node_id": source.node_id,
        "source_line": source_line,
        "target_node_id": target.node_id if target else None,
        "raw_target_name": raw,
        "resolution_status": status,
        "resolution_rule": rule,
    }
    edge_id = "R" + hashlib.sha256(_canonical(identity)).hexdigest()
    return RelationEdge(
        edge_id=edge_id,
        project_id=source.project_id,
        repository_revision=source.repository_revision,
        relation_type=relation_type,
        source_node_id=source.node_id,
        source_path=source.path,
        source_chunk_id=source.code_chunk_id,
        source_symbol=source.qualified_name,
        source_start_line=source_line,
        source_end_line=source_line,
        target_node_id=target.node_id if target else None,
        target_path=target.path if target else None,
        target_chunk_id=target.code_chunk_id if target else None,
        target_symbol=target.qualified_name if target else None,
        target_start_line=target.start_line if target else None,
        target_end_line=target.end_line if target else None,
        raw_target_name=raw,
        resolution_status=status,
        resolution_rule=rule,
        language="python",
        source_content_hash=source.content_hash,
        target_content_hash=target.content_hash if target else None,
    )


def _build_module_map(files: list[dict[str, Any]]) -> dict[str, list[_ModuleTarget]]:
    result: dict[str, list[_ModuleTarget]] = {}
    for file in files:
        path = _normalize_path(str(file.get("path", "")))
        full = _module_name(path)
        aliases = {full}
        parts = full.split(".") if full else []
        if parts and parts[0] in {"src", "lib"}:
            aliases.add(".".join(parts[1:]))
        for alias in aliases:
            if alias:
                result.setdefault(alias, []).append(_ModuleTarget(path, alias))
    for name in result:
        result[name] = _dedupe_modules(result[name])
    return result


def _module_name(path: str) -> str:
    pure = PurePosixPath(_normalize_path(path))
    parts = list(pure.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("/")


def _candidate_status(
    targets: list[RelationNode], rule: str
) -> tuple[list[RelationNode], str, str]:
    unique = _dedupe_nodes(targets)
    if len(unique) == 1:
        return unique, "resolved", rule
    if len(unique) > 1:
        return unique, "ambiguous", f"{rule}_multiple_candidates"
    return [], "unresolved", f"{rule}_target_missing"


def _dedupe_nodes(values: Iterable[RelationNode]) -> list[RelationNode]:
    return sorted(
        {value.node_id: value for value in values}.values(),
        key=lambda item: (item.path, item.start_line, item.qualified_name, item.node_id),
    )


def _dedupe_modules(values: Iterable[_ModuleTarget]) -> list[_ModuleTarget]:
    return sorted(
        {(value.path, value.module): value for value in values}.values(),
        key=lambda item: (item.path, item.module),
    )


def _argument_names(arguments: ast.arguments) -> set[str]:
    values = [
        *arguments.posonlyargs,
        *arguments.args,
        *arguments.kwonlyargs,
    ]
    result = {value.arg for value in values}
    if arguments.vararg:
        result.add(arguments.vararg.arg)
    if arguments.kwarg:
        result.add(arguments.kwarg.arg)
    return result


def _assigned_names(node: ast.AST) -> set[str]:
    collector = _AssignmentCollector()
    for statement in getattr(node, "body", []):
        collector.visit(statement)
    return collector.names


class _AssignmentCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.names.add(node.id)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


def _safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)[:500]
    except Exception:
        return type(node).__name__


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


_BUILTINS = frozenset(
    {
        "abs",
        "all",
        "any",
        "bool",
        "bytes",
        "dict",
        "enumerate",
        "float",
        "getattr",
        "hasattr",
        "int",
        "isinstance",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "object",
        "open",
        "print",
        "range",
        "repr",
        "set",
        "sorted",
        "str",
        "sum",
        "super",
        "tuple",
        "type",
        "zip",
        "__import__",
        "eval",
        "exec",
    }
)
