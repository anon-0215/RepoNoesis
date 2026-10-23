"""Opt-in mechanical source preparation. Static locators are never Evidence.

Only lexical module definitions and explicit imports are followed. This locates
source definitions, not runtime dispatch. No analyzed module is imported.
"""
from __future__ import annotations

from app.source_lines import source_lines as physical_source_lines

import ast
from dataclasses import dataclass, field, replace
import json
from pathlib import PurePosixPath
import re
import time
from typing import Any, Callable

from app.config import get_env_value
from app.database import SOURCE_FILE_MAX_BYTES
from app.services.candidate_evidence import CandidateEvidencePool
from app.services.evidence import CitationValidator, Evidence

DELIVERY_GUIDANCE = (
    ' source_delivery_diagnostics is bounded untrusted metadata, not Evidence or citation '
    'authority. Held fields describe validated stored coverage; delivery fields describe only '
    'this request. candidate_definitions are locators: annotations suggest source, not receiver '
    'types or exhaustive targets. Valid source/citations do not prove call binding. Unresolved '
    'is not absent; complete bodies or omitted rows do not imply sufficient evidence. Knowledge/'
    'docstrings/call clues cannot supply missing implementation. Separate source facts, stated-condition '
    'inferences and unknowns; unrelated gaps must not suppress supported parts.'
)


def enabled() -> bool:
    return get_env_value('AGENT_DIRECTED_EVIDENCE_ENABLED', '0').strip().lower() in {'1', 'true'}


def _local_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        names.update(a.arg for a in ast.walk(node.args) if isinstance(a, ast.arg))

    class Bindings(ast.NodeVisitor):
        def visit_Name(self, value):
            if isinstance(value.ctx, (ast.Store, ast.Del)):
                names.add(value.id)

        def visit_Attribute(self, value):
            if isinstance(value.ctx, (ast.Store, ast.Del)):
                base = value.value
                while isinstance(base, (ast.Attribute, ast.Subscript)):
                    base = base.value
                if isinstance(base, ast.Name):
                    names.add(base.id)
            self.generic_visit(value)

        def visit_Subscript(self, value):
            if isinstance(value.ctx, (ast.Store, ast.Del)):
                base = value.value
                while isinstance(base, (ast.Attribute, ast.Subscript)):
                    base = base.value
                if isinstance(base, ast.Name):
                    names.add(base.id)
            self.generic_visit(value)

        def visit_FunctionDef(self, value):
            names.add(value.name)

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_ClassDef = visit_FunctionDef

        def visit_Import(self, value):
            names.update(a.asname or a.name.split('.')[0] for a in value.names)

        def visit_ImportFrom(self, value):
            names.update(a.asname or a.name for a in value.names)

        def visit_Global(self, value):
            names.update(value.names)

        visit_Nonlocal = visit_Global

        def visit_Call(self, value):
            if isinstance(value.func, ast.Name):
                if value.func.id in {'exec', 'eval', 'globals', 'locals'}:
                    names.add('*')
                elif value.func.id in {'setattr', 'delattr'} and value.args and isinstance(value.args[0], ast.Name):
                    names.add(value.args[0].id)
            self.generic_visit(value)

        def visit_MatchAs(self, value):
            if value.name:
                names.add(value.name)
            self.generic_visit(value)

        visit_MatchStar = visit_MatchAs

        def visit_MatchMapping(self, value):
            if value.rest:
                names.add(value.rest)
            self.generic_visit(value)

        def visit_ExceptHandler(self, value):
            if value.name:
                names.add(value.name)
            self.generic_visit(value)

    visitor = Bindings()
    for child in getattr(node, 'body', []):
        visitor.visit(child)
    return names


class StaticLocator:
    """Bounded AST syntax inspection of already loaded source snapshots."""
    def __init__(self, files: list[dict[str, Any]], check: Callable[[], None], byte_limit: int, authority=None):
        self.files = {str(f['path']): str(f.get('content', '')) for f in files if str(f['path']).endswith('.py')}
        self.check = check
        self.byte_limit = byte_limit
        self.inspected_bytes = 0
        self.cache: dict[str, tuple[ast.Module, dict[str, list[ast.AST]]] | None] = {}
        self.authority = authority
        self.inspection_rejections: dict[str, str] = {}

    def module(self, path):
        self.check()
        if path in self.cache:
            return self.cache[path]
        source = self.files.get(path)
        self.cache[path] = None
        if source is None:
            return None
        size = len(source.encode('utf-8'))
        if size > SOURCE_FILE_MAX_BYTES or self.inspected_bytes + size > self.byte_limit:
            self.inspection_rejections[path] = 'static_inspection_byte_limit'
            return None
        self.inspected_bytes += size
        if self.authority is not None and self.authority(path) != source:
            self.inspection_rejections[path] = 'binding_snapshot_mismatch'
            return None
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError, RecursionError):
            return None
        bindings: dict[str, list[ast.AST]] = {}
        def bind(name, node):
            bindings.setdefault(name, []).append(node)
        for statement in tree.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bind(statement.name, statement)
            elif isinstance(statement, (ast.Import, ast.ImportFrom)):
                for alias in statement.names:
                    bind(alias.asname or (alias.name.split('.')[0] if isinstance(statement, ast.Import) else alias.name), statement)
            else:
                # Conditional definitions, writes, deletion and explicit patches
                # poison the binding rather than selecting a same-named target.
                for node in ast.walk(statement):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                        bind(node.id, statement)
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        bind(node.name, statement)
                    elif isinstance(node, (ast.Import, ast.ImportFrom)):
                        for alias in node.names:
                            bind(alias.asname or alias.name.split('.')[0], statement)
                    elif isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)) and isinstance(node.value, ast.Name):
                        bind(node.value.id, statement)
                    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {'setattr', 'delattr'} and node.args and isinstance(node.args[0], ast.Name):
                        bind(node.args[0].id, statement)
                for name in _local_names(ast.Module(body=[statement], type_ignores=[])):
                    if name not in bindings or statement not in bindings[name]:
                        bind(name, statement)
        # Visible mutations in another function also invalidate a global target;
        # this is conservative source inspection, not an interprocedural proof.
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                for name in node.names:
                    bind(name, node)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {'exec', 'eval'}:
                bind('*', node)
            elif isinstance(node, ast.Subscript) and isinstance(node.ctx, (ast.Store, ast.Del)) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id in {'globals', 'locals'}:
                bind('*', node)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Call) and isinstance(node.func.value.func, ast.Name) and node.func.value.func.id in {'globals', 'locals'}:
                bind('*', node)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {'setattr', 'delattr'} and node.args and isinstance(node.args[0], ast.Name) and node.args[0].id in bindings:
                bind(node.args[0].id, node)
            elif isinstance(node, ast.Subscript) and isinstance(node.ctx, (ast.Store, ast.Del)):
                base = node.value
                while isinstance(base, (ast.Attribute, ast.Subscript)):
                    base = base.value
                if isinstance(base, ast.Name) and base.id in bindings:
                    bind(base.id, node)
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
                base = node.value
                while isinstance(base, (ast.Attribute, ast.Subscript)):
                    base = base.value
                if isinstance(base, ast.Name) and base.id in bindings:
                    # Parameter names are local; other roots can patch a global.
                    # Conservatively reject even if a local spelling collides.
                    bind(base.id, node)
        self.check()
        self.cache[path] = (tree, bindings)
        return self.cache[path]

    def import_path(self, path, module, level=0):
        if level:
            parent = PurePosixPath(path).parent
            for _ in range(level - 1):
                if str(parent) == '.':
                    return None
                parent = parent.parent
            stem = str(parent / (module or '').replace('.', '/')).rstrip('/')
            matches = [p for p in (stem + '.py', stem + '/__init__.py') if p in self.files]
        else:
            stem = (module or '').replace('.', '/')
            matches = [p for p in self.files if p in {stem + '.py', stem + '/__init__.py'} or p.endswith('/' + stem + '.py') or p.endswith('/' + stem + '/__init__.py')]
        return matches[0] if len(matches) == 1 else None

    def definition(self, path, name):
        module = self.module(path)
        values = module[1].get(name, []) if module and '*' not in module[1] else []
        if len(values) != 1:
            return None
        node = values[0]
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) or node.decorator_list:
            return None
        if isinstance(node, ast.ClassDef) and (node.bases or node.keywords):
            return None
        return path, node

    def declaration(self, path, qualified_name):
        """One unconditional declaration, or proven typing overload stubs + one body.

        Source declaration identity only: no inheritance or runtime dispatch proof.
        Unknown decorators and writes retain ambiguity, including in class scope.
        """
        module = self.module(path)
        if module is None or '*' in module[1]:
            return None, self.inspection_rejections.get(path, 'binding_unavailable')

        def values(scope, name):
            if scope is module[0]:
                return module[1].get(name, [])
            result = []
            for statement in scope.body:
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if statement.name == name:
                        result.append(statement)
                else:
                    names = _local_names(ast.Module(body=[statement], type_ignores=[]))
                    if name in names or '*' in names:
                        result.append(statement)
            return result

        def overload(node, scope):
            if len(node.decorator_list) != 1:
                return False
            decorator = node.decorator_list[0]
            base = decorator.id if isinstance(decorator, ast.Name) else (
                decorator.value.id if isinstance(decorator, ast.Attribute) and
                isinstance(decorator.value, ast.Name) and decorator.attr == 'overload' else None)
            if base is None or (scope is not module[0] and values(scope, base)):
                return False
            imports = module[1].get(base, [])
            if len(imports) != 1:
                return False
            binding = imports[0]
            if isinstance(decorator, ast.Name) and isinstance(binding, ast.ImportFrom):
                return binding.level == 0 and binding.module == 'typing' and any(
                    a.name == 'overload' and (a.asname or a.name) == base for a in binding.names)
            if isinstance(decorator, ast.Attribute) and isinstance(binding, ast.Import):
                return any(a.name == 'typing' and (a.asname or a.name) == base for a in binding.names)
            return False

        scope = module[0]
        parts = qualified_name.split('.')
        for name in parts[:-1]:
            bindings = values(scope, name)
            if len(bindings) != 1 or not isinstance(bindings[0], ast.ClassDef) or bindings[0].decorator_list:
                return None, 'conditional_or_rebound_scope'
            scope = bindings[0]
        bindings = values(scope, parts[-1])
        if not bindings or any(not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for n in bindings):
            return None, 'conditional_or_rebound_definition'
        if len(bindings) == 1:
            node = bindings[0]
            if node.decorator_list:
                return None, 'unknown_decorator'
            return node, 'unique_declaration'
        functions = [n for n in bindings if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        stubs = [n for n in functions if overload(n, scope) and len(n.body) == 1 and
            (isinstance(n.body[0], ast.Pass) or isinstance(n.body[0], ast.Expr) and
             isinstance(n.body[0].value, ast.Constant) and n.body[0].value.value is Ellipsis)]
        bodies = [n for n in functions if not n.decorator_list]
        if (len(functions) == len(bindings) and len(stubs) == len(bindings) - 1 and len(bodies) == 1
                and all(n.lineno < bodies[0].lineno for n in stubs)):
            return bodies[0], 'overload_unique_implementation'
        return None, 'ambiguous_declaration'

    def resolve(self, path, expression, shadowed, *, declaration_only=False):
        module = self.module(path)
        if module is None or '*' in module[1] or '*' in shadowed:
            return None
        base = expression.id if isinstance(expression, ast.Name) else expression.value.id if isinstance(expression, ast.Attribute) and isinstance(expression.value, ast.Name) else None
        if base is None or base in shadowed:
            return None
        values = module[1].get(base, [])
        if len(values) != 1:
            return None
        binding = values[0]
        def lookup(target_path, name):
            if not declaration_only:
                return self.definition(target_path, name)
            node, _ = self.declaration(target_path, name)
            return (target_path, node) if node is not None else None
        if isinstance(expression, ast.Name):
            if isinstance(binding, ast.ImportFrom):
                aliases = [a for a in binding.names if (a.asname or a.name) == base and a.name != '*']
                target = self.import_path(path, binding.module, binding.level)
                return lookup(target, aliases[0].name) if target and len(aliases) == 1 else None
            return lookup(path, base)
        if isinstance(binding, ast.Import):
            aliases = [a for a in binding.names if (a.asname or a.name.split('.')[0]) == base and (a.asname or '.' not in a.name)]
            target = self.import_path(path, aliases[0].name) if len(aliases) == 1 else None
            return lookup(target, expression.attr) if target else None
        return None

    def annotation_candidates(self, item, expression, chunks, limit):
        """Direct source declarations suggested by a parameter annotation.

        No receiver binding is inferred, even for one candidate. Do not follow
        bases/MRO, dynamic annotations or unindexed external definitions.
        """
        module = self.module(item.path)
        if module is None:
            return [], False
        try:
            call = ast.parse(expression, mode='eval').body
        except (SyntaxError, ValueError, RecursionError):
            return [], False
        if not isinstance(call, ast.Attribute) or not isinstance(call.value, ast.Name):
            return [], False
        function = next((n for n in ast.walk(module[0]) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and (n.lineno, n.end_lineno) == (item.start_line, item.end_line)), None)
        if function is None:
            return [], False
        parameter = next((a for a in ast.walk(function.args) if isinstance(a, ast.arg) and a.arg == call.value.id), None)
        if parameter is None or parameter.annotation is None:
            return [], False
        owner = self.resolve(item.path, parameter.annotation, set(), declaration_only=True)
        if owner is None or not isinstance(owner[1], ast.ClassDef):
            return [], False
        path, node = owner
        name = node.name + '.' + call.attr
        # Only this owner's direct declarations. Never derive inherited targets.
        matches = [c for c in chunks if c['path'] == path and c['qualified_name'] == name
            and node.lineno < c['start_line'] <= c['end_line'] <= node.end_lineno]
        implementation, status = self.declaration(path, name)
        if implementation is not None:
            matches = [c for c in matches if (c['start_line'], c['end_line']) ==
                (implementation.lineno, implementation.end_lineno)]
        matches.sort(key=lambda c: (c['start_line'], c['end_line'], c['id']))
        return [{**{k: c[k] for k in ['path', 'start_line', 'end_line',
                'content_hash', 'repository_revision']}, 'symbol': c['qualified_name'],
            'discovery_basis': 'parameter_annotation', 'declaration_status': status,
            'runtime_binding_proven': False} for c in matches[:limit]], len(matches) > limit

    def data_member_declaration(self, path, qualified_name, member):
        """Locate an explicitly named, directly assigned class member's owner.

        This retrieves the declaration body, never resolves inherited attributes,
        descriptors or runtime receiver types. Bases do not lend member evidence.
        """
        node, status = self.declaration(path, qualified_name)
        if not isinstance(node, ast.ClassDef) or node.keywords:
            return None, status if node is None else 'dynamic_class_namespace'
        writes = []
        for statement in node.body:
            names = _local_names(ast.Module(body=[statement], type_ignores=[]))
            if member in names or '*' in names:
                writes.append(statement)
        if len(writes) == 1:
            statement = writes[0]
            target = (statement.targets[0] if isinstance(statement, ast.Assign) and len(statement.targets) == 1
                      else statement.target if isinstance(statement, ast.AnnAssign) and statement.value is not None else None)
            if isinstance(target, ast.Name) and target.id == member:
                return node, 'direct_data_member_declaration'
        return None, 'data_member_not_statically_confirmed'

    def state_access_candidates(self, item, question, chunks, limit):
        """Rank a narrow snapshot/restore source shape, never prove dispatch.

        An annotation alone is insufficient. Require an explicitly asked state
        slot, the same receiver/key, and a simple read/write pair whose saved
        result controls an asked-slot write and is passed back. This is bounded
        local syntax ranking, not conditional execution or binding analysis.
        """
        module = self.module(item.path)
        if module is None or item.planner_excerpt_coverage().get('truncated') is not False:
            return []
        function = next((n for n in ast.walk(module[0]) if isinstance(n, ast.FunctionDef)
            and (n.lineno, n.end_lineno) == (item.start_line, item.end_line)), None)
        if function is None:
            return []
        nodes = list(ast.walk(function))
        if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda,
                             ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
                             ast.For, ast.AsyncFor, ast.While, ast.Match))
               and n is not function for n in nodes):
            return []
        # No branch execution or alias/type inference. Any visible rebinding or
        # namespace mutation is enough to decline automatic acquisition.
        changed = {n.id for n in nodes if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))}
        for n in nodes:
            if isinstance(n, ast.Attribute) and isinstance(n.ctx, (ast.Store, ast.Del)):
                root = n.value
                while isinstance(root, (ast.Attribute, ast.Subscript)):
                    root = root.value
                if isinstance(root, ast.Name):
                    changed.add(root.id)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                if n.func.id in {'exec', 'eval', 'globals', 'locals'}:
                    return []
                if n.func.id in {'setattr', 'delattr'} and n.args:
                    root = n.args[0]
                    while isinstance(root, (ast.Attribute, ast.Subscript)):
                        root = root.value
                    if isinstance(root, ast.Name):
                        changed.add(root.id)
            if isinstance(n, (ast.Global, ast.Nonlocal, ast.Import, ast.ImportFrom)):
                return []
            if isinstance(n, (ast.MatchAs, ast.MatchStar, ast.ExceptHandler)) and n.name:
                changed.add(n.name)
            if isinstance(n, ast.MatchMapping) and n.rest:
                changed.add(n.rest)
        question = re.sub(r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'", ' ', question)
        spelled = set(re.findall(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+', question))
        syntax = lambda n: ast.dump(n, include_attributes=False)
        anchors = {(n.value.value.id, syntax(n.slice)) for n in nodes if isinstance(n, ast.Subscript)
            and isinstance(n.value, ast.Attribute) and isinstance(n.value.value, ast.Name)
            and ast.unparse(n.value) in spelled and n.value.value.id not in changed}
        if not anchors:
            return []

        # Trace only unique, earlier local assignments, at most eight steps.
        # Keep the complete spelled field here: another map on the same receiver
        # and key must not lend relevance to the field the user actually asked.
        stores = {}
        definitions = {}
        for n in nodes:
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                stores[n.id] = stores.get(n.id, 0) + 1
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                definitions[n.targets[0].id] = n
        if len(stores) > 128:
            return []
        relevant = {anchor: set() for anchor in anchors}
        class AskedWrites(ast.NodeVisitor):
            conditions = ()
            def visit_If(visitor, node):
                previous = visitor.conditions
                visitor.conditions = (*previous, node.test)
                for child in [*node.body, *node.orelse]:
                    visitor.visit(child)
                visitor.conditions = previous
            def visit_Subscript(visitor, node):
                if (isinstance(node.ctx, ast.Store) and isinstance(node.value, ast.Attribute)
                        and isinstance(node.value.value, ast.Name) and ast.unparse(node.value) in spelled):
                    anchor = (node.value.value.id, syntax(node.slice))
                    if anchor in relevant:
                        for condition in visitor.conditions:
                            frontier = {(n.id, condition.lineno) for n in ast.walk(condition)
                                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
                            visited = set()
                            for _ in range(8):
                                next_names = set()
                                for name, use_line in frontier - visited:
                                    visited.add((name, use_line))
                                    definition = definitions.get(name)
                                    if stores.get(name) != 1 or definition is None or definition.end_lineno >= use_line:
                                        continue
                                    relevant[anchor].add(name)
                                    next_names.update((n.id, definition.lineno) for n in ast.walk(definition.value)
                                        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load))
                                frontier = next_names
                visitor.generic_visit(node)
        AskedWrites().visit(function)

        def accessor(node):
            if not isinstance(node, ast.FunctionDef) or node.decorator_list:
                return None
            args = node.args
            positional = args.posonlyargs + args.args
            if args.vararg or args.kwarg or args.kwonlyargs or args.defaults or len(positional) not in {2, 3}:
                return None
            receiver, key = positional[0].arg, positional[1].arg
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                body = body[1:]
            if len(body) != 1:
                return None
            statement = body[0]
            if len(positional) == 2 and isinstance(statement, ast.Return):
                call = statement.value
                if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == 'get'
                        and len(call.args) == 1 and not call.keywords and isinstance(call.args[0], ast.Name)
                        and call.args[0].id == key):
                    storage = call.func.value
                    kind = 'read'
                else:
                    return None
            elif len(positional) == 3 and isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target = statement.targets[0]
                if (not isinstance(target, ast.Subscript) or not isinstance(target.slice, ast.Name)
                        or target.slice.id != key or not isinstance(statement.value, ast.Name)
                        or statement.value.id != positional[2].arg):
                    return None
                storage, kind = target.value, 'write'
            else:
                return None
            if isinstance(storage, ast.Attribute) and isinstance(storage.value, ast.Name) and storage.value.id == receiver:
                return kind, storage.attr
            return None

        calls = []
        for call in sorted((n for n in nodes if isinstance(n, ast.Call)), key=lambda n: (n.lineno, n.col_offset)):
            self.check()
            if (not isinstance(call.func, ast.Attribute) or not isinstance(call.func.value, ast.Name)
                    or not call.args or call.keywords or any(isinstance(a, ast.Starred) for a in call.args)):
                continue
            receiver = call.func.value.id
            if (receiver, syntax(call.args[0])) not in anchors:
                continue
            key_nodes = list(ast.walk(call.args[0]))
            if (any(isinstance(n, (ast.Call, ast.Subscript, ast.NamedExpr)) for n in key_nodes)
                    or any(isinstance(n, ast.Name) and n.id in changed for n in key_nodes)):
                continue
            candidates, omitted = self.annotation_candidates(item, ast.unparse(call.func), chunks, 2)
            if omitted or len(candidates) != 1:
                continue
            candidate = candidates[0]
            node, status = self.declaration(candidate['path'], candidate['symbol'])
            owner, _ = self.declaration(candidate['path'], candidate['symbol'].rsplit('.', 1)[0])
            if not isinstance(owner, ast.ClassDef) or owner.bases or owner.keywords:
                continue
            shape = accessor(node)
            if shape is None or len(call.args) != (1 if shape[0] == 'read' else 2):
                continue
            if status not in {'unique_declaration', 'overload_unique_implementation'}:
                continue
            calls.append((call, candidate, shape, receiver, syntax(call.args[0])))
        selected = {}
        for call, candidate, shape, receiver, key in calls:
            if shape[0] != 'read':
                continue
            assignment = next((n for n in nodes if isinstance(n, ast.Assign) and n.value is call
                and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)), None)
            if assignment is None:
                continue
            saved = assignment.targets[0].id
            if sum(isinstance(n, ast.Name) and n.id == saved and isinstance(n.ctx, (ast.Store, ast.Del)) for n in nodes) != 1:
                continue
            if saved not in relevant[(receiver, key)]:
                continue
            writers = [(other, c) for other, c, s, r, k in calls if s == ('write', shape[1]) and r == receiver and k == key
                and c['path'] == candidate['path'] and c['symbol'].rsplit('.', 1)[0] == candidate['symbol'].rsplit('.', 1)[0]
                and isinstance(other.args[1], ast.Name) and other.args[1].id == saved
                and other.lineno > assignment.end_lineno]
            if not writers:
                continue
            for site, c in [(call, candidate), *writers]:
                identity = (c['path'], c['symbol'], c['start_line'], c['end_line'])
                selected.setdefault(identity, (c, site.lineno))
        return list(selected.values())[:limit]

    def calls(self, item: Evidence):
        module = self.module(item.path)
        coverage = item.planner_excerpt_coverage()
        if module is None or coverage.get('status') != 'known':
            return []
        end = coverage.get('complete_end_line') or 0
        results = []
        locator = self
        class Calls(ast.NodeVisitor):
            def __init__(self):
                self.scopes = []
            def visit_FunctionDef(self, node):
                self.scopes.append(_local_names(node))
                for child in node.body:
                    self.visit(child)
                self.scopes.pop()
            visit_AsyncFunctionDef = visit_FunctionDef
            def visit_ClassDef(self, node):
                # Class body execution has its own namespace. Only inspect
                # method bodies, where ordinary lexical rules apply.
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.visit(child)
            def visit_Lambda(self, node):
                # Lambda and comprehension scope/binding are outside this prototype.
                pass
            def visit_ListComp(self, node): pass
            visit_SetComp = visit_ListComp
            visit_DictComp = visit_ListComp
            visit_GeneratorExp = visit_ListComp
            def visit_Call(self, node):
                if item.start_line <= node.lineno <= node.end_lineno <= end:
                    shadowed = set().union(*self.scopes) if self.scopes else set()
                    target = locator.resolve(item.path, node.func, shadowed)
                    results.append((node.lineno, ast.unparse(node.func)[:160], target))
                self.generic_visit(node)
        Calls().visit(module[0])
        return sorted(results, key=lambda r: r[0])


@dataclass
class PreparationLedger:
    events: list[dict[str, Any]] = field(default_factory=list)
    attempted: set[tuple[Any, ...]] = field(default_factory=set)
    expanded: set[tuple[Any, ...]] = field(default_factory=set)
    depths: dict[str, int] = field(default_factory=dict)
    read_count: int = 0
    read_bytes: int = 0
    read_lines: int = 0
    locator: StaticLocator | None = None
    binding_validation_reads: int = 0
    binding_validation_bytes: int = 0
    checks: list[dict[str, Any]] = field(default_factory=list)
    check_keys: set[tuple[Any, ...]] = field(default_factory=set)
    checks_omitted: bool = False
    requested_targets: dict[tuple[Any, ...], dict[str, Any]] = field(default_factory=dict)
    selection_status: str = 'no_reliable_entry'

    def record(self, **row):
        key = (('body', row.get('path'), row.get('symbol'), row.get('start_line'), row.get('end_line'))
               if row.get('located') else ('hint', row.get('source_path'), row.get('call_line'), row.get('expression')))
        if key in self.check_keys:
            existing = next(c for c in self.checks if (
                ('body', c.get('path'), c.get('symbol'), c.get('start_line'), c.get('end_line'))
                if c.get('located') else ('hint', c.get('source_path'), c.get('call_line'), c.get('expression'))) == key)
            # Preserve the strongest relevance and binding annotation; call-site
            # repetitions do not consume diagnostic rows or replace primary state.
            for name, value in row.items():
                if name not in existing or name in {'chunk_identity', 'content_hash', 'code_chunk_id'} or (
                        name == 'binding_status' and (existing.get(name) == 'not_checked' or
                            value in {'unique_declaration', 'overload_unique_implementation'})):
                    existing[name] = value
            existing['is_primary'] = existing.get('is_primary', False) or row.get('is_primary', False)
            return
        if len(self.checks) >= 32:
            self.checks_omitted = True
            if row.get('is_primary'):
                removable = next((c for c in reversed(self.checks) if not c.get('is_primary')), None)
                if removable is not None:
                    self.checks.remove(removable)
                    self.check_keys = {('body', c.get('path'), c.get('symbol'), c.get('start_line'), c.get('end_line'))
                        if c.get('located') else ('hint', c.get('source_path'), c.get('call_line'), c.get('expression')) for c in self.checks}
                else:
                    return
            else:
                return
        self.check_keys.add(key)
        self.checks.append(row)


def _chunk_key(chunk):
    return (chunk['path'], chunk['qualified_name'], chunk['start_line'], chunk['end_line'])


def _validated_body(item):
    proof = item._excerpt_coverage
    return (item.validation_status == 'valid' and proof is not None and
        proof.source_identity == item._planner_source_identity() and
        item.planner_excerpt_coverage().get('status') == 'known')


def _named_targets(ledger, chunks, text, path=None, language=None, *, explicit_requested=False, held=()):
    """Exact indexed declarations or one direct module import, never a name guess."""
    selected = {}
    # Quoted data and path components (including assumed environment values)
    # are not symbol requests. Backticks remain available for code identifiers.
    text = re.sub(r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'", ' ', text)
    tokens = list(dict.fromkeys(m.group() for m in re.finditer(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*', text)
        if (m.start() == 0 or text[m.start() - 1] not in '/\\')
        and (m.end() == len(text) or text[m.end()] not in '/\\')))
    if len(tokens) > 128:
        ledger.checks_omitted = True
    tokens = tokens[:128]
    # Known validated identities survive a later AST inspection budget/cutoff.
    # These are coverage records, not permission to bind or complete all namesakes.
    for item in held:
        module = str(PurePosixPath(item.path).with_suffix('')).replace('/', '.')
        if module.endswith('.__init__'):
            module = module[:-9]
        if (_validated_body(item) and (path is None or item.path == path)
                and (language is None or item.language.casefold() == language.casefold()) and any(
                    token == item.qualified_name or ('.' not in token and token == item.symbol_name) or
                    ('.' + module + '.' + item.qualified_name).endswith('.' + token) for token in tokens)):
            ledger.record(path=item.path, symbol=item.qualified_name, start_line=item.start_line, end_line=item.end_line,
                located=True, is_primary=item.chunk_type != 'class', binding_status='not_checked',
                reason='explicit_held_body', chunk_identity=item.chunk_identity)
    for token in tokens:
        candidates = []
        data_member = False
        for c in chunks:
            if path is not None and c['path'] != path:
                continue
            if language is not None and c.get('language', '').casefold() != language.casefold():
                continue
            module = str(PurePosixPath(c['path']).with_suffix('')).replace('/', '.')
            if module.endswith('.__init__'):
                module = module[:-9]
            if token == c['qualified_name'] or ('.' not in token and token == c['symbol_name']) or ('.' + module + '.' + c['qualified_name']).endswith('.' + token):
                candidates.append(c)
        # Data members are not separate indexed chunks. A qualified explicit
        # member can request its unique owner's declaration, without turning a
        # bare class mention into construction or inferring a receiver binding.
        if not candidates and '.' in token:
            prefix, member = token.rsplit('.', 1)
            for c in chunks:
                if c.get('chunk_type') != 'class' or (path is not None and c['path'] != path):
                    continue
                if language is not None and c.get('language', '').casefold() != language.casefold():
                    continue
                module = str(PurePosixPath(c['path']).with_suffix('')).replace('/', '.')
                if module.endswith('.__init__'):
                    module = module[:-9]
                if prefix == c['qualified_name'] or ('.' + module + '.' + c['qualified_name']).endswith('.' + prefix):
                    candidates.append(c)
            data_member = bool(candidates)
        # Public module exports can be a single explicit ImportFrom. Inspect
        # only that named module, under the same authority/byte/time bounds.
        if not candidates and '.' in token:
            prefix, name = token.rsplit('.', 1)
            module_path = ledger.locator.import_path('', prefix)
            if module_path and any(c['symbol_name'] == name for c in chunks):
                target = ledger.locator.resolve(module_path, ast.Name(id=name), set())
                if target:
                    p, node = target
                    candidates = [c for c in chunks if c['path'] == p and c['symbol_name'] == node.name
                        and c['start_line'] == node.lineno and c['end_line'] == node.end_lineno
                        and (path is None or c['path'] == path)
                        and (language is None or c.get('language', '').casefold() == language.casefold())]
        constructor_words = r'(?:constructor|construction|instantiate|instantiation|initialization|构造|初始化)'
        construction = (re.search(r'(?<![\w.])' + re.escape(token) + r'\s*\(', text) is not None or
            re.search(r'(?<![\w.])' + re.escape(token) + r'\s*(?:的\s*)?' + constructor_words +
                r'|\b' + constructor_words + r'\s+(?:of\s+)?' + re.escape(token) + r'(?![\w.])', text, re.I) is not None)
        if candidates and all(c.get('chunk_type') == 'class' for c in candidates) and not construction and not data_member:
            continue
        scopes = {(c['path'], c['qualified_name']) for c in candidates}
        implementation, binding = None, 'ambiguous_explicit_name'
        if len(scopes) == 1:
            p, name = next(iter(scopes))
            node, binding = (ledger.locator.data_member_declaration(p, name, token.rsplit('.', 1)[1])
                if data_member else ledger.locator.declaration(p, name))
            implementation = next((c for c in candidates if node is not None and
                (c['start_line'], c['end_line']) == (node.lineno, node.end_lineno)), None)
        for item in held:
            if _validated_body(item) and any(_chunk_key(c) == (item.path, item.qualified_name, item.start_line, item.end_line) for c in candidates):
                ledger.record(path=item.path, symbol=item.qualified_name, start_line=item.start_line, end_line=item.end_line,
                    located=True, is_primary=item.chunk_type != 'class',
                    binding_status=binding, reason='explicit_held_body',
                    chunk_identity=item.chunk_identity, code_chunk_id=item.code_chunk_id, content_hash=item.content_hash)
        if implementation is not None:
            is_class = implementation.get('chunk_type') == 'class'
            if not is_class or construction or data_member:
                key = _chunk_key(implementation)
                # A member request must not erase a separate explicit Class()
                # request, nor the latter erase declaration-only relevance.
                previous = selected.get(key, {})
                selected[key] = {**implementation,
                    '_construction': previous.get('_construction', False) or (is_class and construction and not data_member),
                    '_data_member': previous.get('_data_member', False) or data_member,
                    '_binding_status': binding}
        elif len(candidates) > 0:
            ledger.record(expression=token[:160], reason=binding if binding in {
                'static_inspection_byte_limit', 'binding_snapshot_mismatch', 'data_member_not_statically_confirmed',
                'dynamic_class_namespace'} else 'ambiguous_explicit_name', located=False)
        elif explicit_requested:
            ledger.record(expression=token[:160], source_path=path,
                reason='explicit_target_not_statically_located', located=False)
    return list(selected.values())


def delivery_report(ledger, selected: list[Evidence], max_bytes: int, stage: str, *, held=None):
    """Project only checked targets against bodies in this exact message.

    This is diagnostic metadata, not Evidence. Omission and static resolution
    are deliberately different from source absence or whole-answer sufficiency.
    """
    if ledger is None:
        return None
    rows = []
    for check in sorted(ledger.checks, key=lambda c: not c.get('is_primary', False)):
        row = {k: v for k, v in check.items() if k not in {'located', 'is_primary', 'content_hash', 'code_chunk_id'}}
        if not check.get('located'):
            row['delivery_status'] = 'not_statically_located'
        else:
            def coverage_of(items):
                intervals, partials = [], []
                for item in items:
                    if not _validated_body(item):
                        continue
                    coverage = item.planner_excerpt_coverage()
                    if item.path == check['path'] and coverage.get('status') == 'known':
                        end = coverage.get('complete_end_line') or 0
                        if end >= check['start_line'] and item.start_line <= check['end_line']:
                            intervals.append((max(item.start_line, check['start_line']), min(end, check['end_line'])))
                        partial = coverage.get('partial_end_line')
                        if partial is not None and check['start_line'] <= partial <= check['end_line']:
                            partials.append(partial)
                end = check['start_line'] - 1
                for a, b in sorted(intervals):
                    if a <= end + 1:
                        end = max(end, b)
                return {'complete_end_line': end if end >= check['start_line'] else None,
                    'partial_end_line': max(partials) if partials else None}, bool(intervals or partials)
            delivered, visible = coverage_of(selected)
            stored, stored_visible = coverage_of(held if held is not None else selected)
            row.update(held_body_status=('body_complete' if (stored['complete_end_line'] or 0) >= check['end_line']
                else 'body_partial' if stored_visible else 'not_held'), held_coverage=stored,
                delivered_coverage=delivered)
            event = next((e for e in reversed(ledger.events) if (e.get('path'), e.get('symbol'), e.get('start_line'), e.get('end_line')) ==
                          (check['path'], check['symbol'], check['start_line'], check['end_line'])), {})
            reason = event.get('status', check.get('reason', 'not_acquired'))
            if not event and check.get('binding_status') in {
                    'static_inspection_byte_limit', 'binding_snapshot_mismatch', 'binding_unavailable'}:
                reason = check['binding_status']
            if not event and ledger.events and ledger.events[-1]['status'] in {'tool_timeout', 'deadline_exceeded', 'cancelled'}:
                reason = ledger.events[-1]['status']
            if (delivered['complete_end_line'] or 0) >= check['end_line']:
                status = 'body_complete'
            elif visible:
                status = 'body_partial'
            elif reason in {'capacity_limit', 'read_count_limit', 'line_limit', 'read_budget_limit',
                            'delivery_budget_limit', 'depth_limit', 'tool_timeout', 'deadline_exceeded',
                            'static_inspection_byte_limit', 'target_limit', 'cancelled'}:
                status = 'blocked_by_budget'
            elif event.get('evidence_id') or stored_visible:
                status, reason = 'blocked_by_budget', stage + '_body_selection'
            else:
                status = 'located_not_delivered'
            if visible and stored_visible and stored != delivered:
                row['acquisition_reason'] = reason
                reason = stage + '_body_selection'
            row.update(delivery_status=status, reason=reason)
            if event.get('body_read'):
                row.update(body_read=True, read_committed=event.get('committed', False))
        rows.append(row)
    base = {'task': 'directed_source_delivery', 'version': 3, 'stage': stage,
        'selection_status': ledger.selection_status, 'rows': [],
        'may_be_incomplete': True, 'records_omitted': ledger.checks_omitted,
        'is_evidence': False, 'runtime_binding_proven': False}
    # Whole rows only, no truncated exact identities. Source bodies have priority.
    limit = max(0, min(8192, max_bytes))
    def size(value):
        return len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8'))
    if size(base) > limit:
        return None
    for row in rows:
        trial = {**base, 'rows': base['rows'] + [row]}
        if size(trial) > limit:
            base['records_omitted'] = True
            break
        base = trial
    return base if size(base) <= limit else None


def prepare(state, fits: Callable[[list[Evidence], Evidence], bool], origin_call_id=None, tool_request=None):
    """Prepare source on copies; commit only authority- and budget-safe changes."""
    if not enabled():
        return
    from app.services.agent_tools import _chunk_as_read_candidate, ToolDeadlineExceeded, ToolCancelled
    context = state.context
    ledger = getattr(state, 'directed_evidence_ledger', None)
    if ledger is None:
        ledger = PreparationLedger()
        state.directed_evidence_ledger = ledger
        # Reuse the repository's stored-file cap and existing search window as
        # a separate metadata scan bound; these bytes never become Evidence.
        def authority(path):
            anchors = [c for c in context.bundle.get('code_chunks', []) if c.get('path') == path
                       and c.get('repository_revision') == context.repository_revision]
            if not anchors:
                return None
            source = context.database.get_evidence_source(context.project_id, int(anchors[0]['id']), path)
            ledger.binding_validation_reads += 1
            if source is None or source.get('repository_revision') != context.repository_revision:
                return None
            text = str(source['file_content'])
            ledger.binding_validation_bytes += len(text.encode('utf-8'))
            return text
        ledger.locator = StaticLocator(context.bundle.get('files', []), context.check_active,
            min(SOURCE_FILE_MAX_BYTES, context.limits.max_observation_bytes * 4), authority)
    limits = context.limits
    max_reads = min(3, limits.max_same_tool_calls)
    max_candidates = min(8, limits.max_search_results)
    max_depth = min(2, limits.max_relation_depth)
    queue = []
    deadline = min(context.effective_deadline_monotonic, time.monotonic() + limits.default_tool_timeout_ms / 1000)

    def check():
        context.check_active()
        if time.monotonic() >= deadline:
            raise ToolDeadlineExceeded('tool_timeout', 'directed source preparation cutoff')

    def acquire(path, node, depth, source_id, qualified_name=None, *, construction=False):
        check()
        # Only an explicit construction request or confirmed Class() edge can
        # redirect a large class to its constructor. Ownership is insufficient.
        symbol = qualified_name or node.name
        if isinstance(node, ast.ClassDef) and node.end_lineno - node.lineno + 1 > limits.max_source_read_lines and construction:
            if node.bases or node.keywords or node.decorator_list:
                return None
            constructor, _ = ledger.locator.declaration(path, symbol + '.__init__')
            if not isinstance(constructor, (ast.FunctionDef, ast.AsyncFunctionDef)):
                ledger.record(expression=symbol + '.__init__', source_path=path,
                    reason='constructor_not_statically_confirmed', located=False)
                return None
            symbol += '.__init__'
            node = constructor
        key = (path, symbol, node.lineno, node.end_lineno)
        ledger.record(path=path, symbol=symbol, start_line=node.lineno, end_line=node.end_lineno,
            source_evidence_id=source_id, depth=depth, located=True)
        if key in ledger.attempted:
            return None
        event = {'path': path, 'symbol': symbol, 'start_line': node.lineno, 'end_line': node.end_lineno,
            'source_evidence_id': source_id, 'depth': depth, 'origin_call_id': origin_call_id,
            'status': 'located', 'body_read': False, 'committed': False, 'read_count': 0,
            'planner_delivery': 'not_confirmed', 'final_inclusion': 'not_confirmed'}
        ledger.events.append(event)
        if len(ledger.attempted) >= max_candidates:
            event['status'] = 'target_limit'
            return None
        ledger.attempted.add(key)
        started = time.monotonic()
        try:
            held = context.evidence_store.all(state.request_id)
            existing = [e for e in held if (e.path, e.qualified_name, e.start_line, e.end_line) == key]
            if len(existing) == 1 and existing[0].planner_excerpt_coverage().get('truncated') is False:
                event.update(status='already_complete', evidence_id=existing[0].evidence_id)
                return existing[0]
            if (ledger.read_count + state.tool_counts.get('read_source', 0) >= max_reads
                    or ledger.read_count + state.tool_call_count >= limits.max_tool_calls):
                event['status'] = 'read_count_limit'
                return None
            if not existing and len(held) >= context.evidence_store.capacity:
                event['status'] = 'capacity_limit'
                return None
            # Only locator metadata from the stored AST index crosses here.
            chunks = context.database.get_code_chunks_for_hierarchy(context.project_id, context.repository_revision, path, limit=2049)
            matching = [c for c in chunks if (c['qualified_name'], c['start_line'], c['end_line']) == (symbol, node.lineno, node.end_lineno)]
            if len(chunks) >= 2049 or len(matching) != 1:
                event['status'] = 'ambiguous_or_missing_chunk'
                return None
            lines = node.end_lineno - node.lineno + 1
            if lines > limits.max_source_read_lines:
                event['status'] = 'line_limit'
                return None
            predicted_size = len(str(matching[0]['content']).encode('utf-8'))
            if (predicted_size > limits.max_source_read_bytes
                    or ledger.read_bytes + predicted_size > limits.max_source_read_bytes * max_reads
                    or ledger.read_lines + lines > limits.max_source_read_lines * max_reads):
                event['status'] = 'read_budget_limit'
                return None
            # Canonical promotion owns rereading; guessed binding is not promoted.
            check()
            ledger.read_count += 1
            event['read_count'] = 1
            pool = CandidateEvidencePool(capacity=context.evidence_store.capacity)
            candidates = pool.normalize_retrieval_results([_chunk_as_read_candidate(matching[0], rank=1)], provenance='planner_read_source')
            promotion = pool.promote(candidates, database=context.database, project=context.bundle['project'],
                project_id=context.project_id, repository_revision=context.repository_revision,
                retrieval_strategy_version='weighted-rrf-v1', held_chunk_identities=set(), accepted_limit=1, new_evidence_limit=1)
            event['canonical_rejections'] = promotion.rejection_counts
            if len(promotion.evidence) != 1:
                event['status'] = 'canonical_rejection'
                return None
            canonical = promotion.evidence[0]
            source = context.database.get_evidence_source(context.project_id, canonical.code_chunk_id, path)
            if source is None:
                event['status'] = 'authority_missing'
                return None
            content = str(source['chunk_content'])
            if str(source['file_content']) != ledger.locator.files.get(path):
                event['status'] = 'binding_snapshot_changed'
                return None
            size = len(content.encode('utf-8'))
            ledger.read_bytes += size
            ledger.read_lines += lines
            event.update(body_read=True, acquired_bytes=size, acquired_lines=lines)
            if size > limits.max_source_read_bytes or ledger.read_bytes > limits.max_source_read_bytes * max_reads or ledger.read_lines > limits.max_source_read_lines * max_reads:
                event['status'] = 'read_budget_limit'
                return None
            prepared = replace(existing[0] if len(existing) == 1 else canonical, excerpt=content)
            if existing and (len(existing) != 1 or existing[0]._planner_source_identity() != canonical._planner_source_identity()):
                event['status'] = 'held_identity_mismatch'
                return None
            CitationValidator(context.database).validate(prepared)
            if prepared.validation_status != 'valid':
                event['status'] = 'authority_rejection'
                return None
            trial = [prepared if e.chunk_identity == prepared.chunk_identity else e for e in held]
            if not existing:
                # Predict the Store-owned ID without consuming it on rejection.
                prepared.evidence_id = f'E{context.evidence_store._next_id}'
                trial.append(prepared)
            check()
            if not fits(trial, prepared):
                event['status'] = 'delivery_budget_limit'
                return None
            check()
            if existing:
                item = existing[0]
                item.excerpt = prepared.excerpt
                item._excerpt_coverage = prepared._excerpt_coverage
                item.validation_status = prepared.validation_status
                item.invalid_reason = prepared.invalid_reason
            else:
                added = context.evidence_store.add(state.request_id, [prepared])
                if not added:
                    event['status'] = 'store_rejection'
                    return None
                item = added[0]
            state.planner_source_origins[item._planner_source_identity()] = (item.excerpt, origin_call_id)
            ledger.depths[item.chunk_identity] = depth
            event.update(status='committed', committed=True, evidence_id=item.evidence_id,
                chunk_identity=item.chunk_identity, excerpt_coverage=item.planner_excerpt_coverage())
            return item
        except (ToolDeadlineExceeded, ToolCancelled) as exc:
            event['status'] = exc.code
            raise
        finally:
            event['duration_ms'] = int((time.monotonic() - started) * 1000)

    def acquire_candidates(group, source_id, depth):
        """Stage the existing legal range reader; commit only selected identities.

        A group is one/two eligible adjacent declarations, not an expanded
        range search. Read costs include whitespace; held IDs never change.
        """
        from app.services.agent_contracts import ReadSourceInput
        from app.services.agent_tools import EvidenceStore, _read_source
        check()
        held = context.evidence_store.all(state.request_id)
        keys = [(c['path'], c['symbol'], c['start_line'], c['end_line']) for c in group]
        existing = {key: next((e for e in held if (e.path, e.qualified_name, e.start_line, e.end_line) == key), None)
                    for key in keys}
        missing = [(c, key) for c, key in zip(group, keys) if not (existing[key] and _validated_body(existing[key])
                   and existing[key].planner_excerpt_coverage().get('truncated') is False)]
        if not missing or any(key in ledger.attempted for _, key in missing):
            return
        if len(ledger.attempted) + len(missing) > max_candidates:
            return
        events = []
        for c, key in missing:
            ledger.attempted.add(key)
            event = {'path': c['path'], 'symbol': c['symbol'], 'start_line': c['start_line'], 'end_line': c['end_line'],
                'source_evidence_id': source_id, 'depth': depth, 'origin_call_id': origin_call_id,
                'selection_reason': 'question_state_accessor_pair', 'runtime_binding_proven': False,
                'status': 'located', 'body_read': False, 'committed': False, 'read_count': 0}
            ledger.events.append(event)
            events.append(event)

        def status(value):
            for event in events:
                event['status'] = value

        start = min(c['start_line'] for c in group)
        end = max(c['end_line'] for c in group)
        path = group[0]['path']
        lines = end - start + 1
        text = ''.join(physical_source_lines(ledger.locator.files[path], keepends=True)[start-1:end])
        size = len(text.encode('utf-8'))
        for event in events:
            event['read_range'] = {'path': path, 'start_line': start, 'end_line': end}
        if depth > max_depth:
            status('depth_limit')
            return
        if ledger.read_count + state.tool_counts.get('read_source', 0) >= max_reads or ledger.read_count + state.tool_call_count >= limits.max_tool_calls:
            status('read_count_limit')
            return
        if sum(existing[key] is None for _, key in missing) > context.evidence_store.capacity - len(held):
            status('capacity_limit')
            return
        if lines > limits.max_source_read_lines:
            status('line_limit')
            return
        if (size > limits.max_source_read_bytes or ledger.read_bytes + size > limits.max_source_read_bytes * max_reads
                or ledger.read_lines + lines > limits.max_source_read_lines * max_reads):
            status('read_budget_limit')
            return
        # Existing read_source owns mapping, canonical promotion, authoritative
        # rereads and prefix completion. A private Store prevents premature
        # mutations/ID consumption when message selection cannot keep all bodies.
        store = EvidenceStore(capacity=context.evidence_store.capacity)
        store._items = {k: replace(e) for k, e in context.evidence_store._items.items()}
        store._owners = dict(context.evidence_store._owners)
        store._next_id = context.evidence_store._next_id
        staged = replace(context, evidence_store=store, candidate_pool=CandidateEvidencePool(capacity=store.capacity))
        started = time.monotonic()
        try:
            check()
            ledger.read_count += 1
            events[0]['read_count'] = 1
            # Charge the actual requested source, even if canonical admission or
            # downstream delivery fails. No second read is hidden in a group.
            ledger.read_bytes += size
            ledger.read_lines += lines
            result, _, truncated = _read_source(staged, ReadSourceInput(path=path, start_line=start, end_line=end))
            if truncated or result['content'] != text:
                status('binding_snapshot_changed')
                return
            trial = store.all(state.request_id)
            before = {e.chunk_identity for e in held}
            allowed = set(keys)
            if any((e.path, e.qualified_name, e.start_line, e.end_line) not in allowed
                   for e in trial if e.chunk_identity not in before):
                status('canonical_rejection')
                return
            prepared = []
            for (c, key), event in zip(missing, events):
                item = next((e for e in trial if (e.path, e.qualified_name, e.start_line, e.end_line) == key), None)
                if (item is None or not _validated_body(item) or item.planner_excerpt_coverage().get('truncated') is not False
                        or item.repository_revision != c['repository_revision'] or item.content_hash != c['content_hash']
                        or existing[key] and item._planner_source_identity() != existing[key]._planner_source_identity()):
                    status('canonical_rejection')
                    return
                event.update(body_read=True, acquired_bytes=len(item.excerpt.encode('utf-8')),
                             acquired_lines=item.end_line-item.start_line+1)
                prepared.append(item)
            check()
            if not all(fits(trial, item) for item in prepared):
                status('delivery_budget_limit')
                return
            check()
            for ((_, key), item, event) in zip(missing, prepared, events):
                if existing[key]:
                    target = existing[key]
                    target.excerpt = item.excerpt
                    target._excerpt_coverage = item._excerpt_coverage
                    target.validation_status = item.validation_status
                    target.invalid_reason = item.invalid_reason
                else:
                    target = context.evidence_store.add(state.request_id, [item])[0]
                state.planner_source_origins[target._planner_source_identity()] = (target.excerpt, origin_call_id)
                ledger.depths[target.chunk_identity] = depth
                event.update(status='committed', committed=True, evidence_id=target.evidence_id,
                    chunk_identity=target.chunk_identity, excerpt_coverage=target.planner_excerpt_coverage())
        except (ToolDeadlineExceeded, ToolCancelled) as exc:
            status(exc.code)
            raise
        finally:
            for event in events:
                event['duration_ms'] = int((time.monotonic() - started) * 1000)

    try:
        chunks = [c for c in context.bundle.get('code_chunks', [])
                  if c.get('repository_revision') == context.repository_revision and c.get('path', '').endswith('.py')]
        held_items = context.evidence_store.all(state.request_id)
        targets = _named_targets(ledger, chunks, state.user_goal, held=held_items)
        if tool_request is not None and tool_request.execution_started and tool_request.effective_parameters:
            params = tool_request.effective_parameters
            explicit = []
            if tool_request.tool_name in {'lookup_symbol', 'search_code'}:
                name = params.get('symbol') or params.get('query', '')
                if re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*', name or ''):
                    explicit = _named_targets(ledger, chunks, name, params.get('path'), params.get('language'), explicit_requested=True, held=held_items)
            elif tool_request.tool_name == 'read_source':
                explicit = [c for c in chunks if c['path'] == params.get('path') and
                    params.get('start_line', 0) <= c['start_line'] <= c['end_line'] <= params.get('end_line', 0)]
            if len(explicit) > max_candidates:
                ledger.checks_omitted = True
            for c in explicit[:max_candidates]:
                if len(ledger.requested_targets) < max_candidates:
                    ledger.requested_targets[_chunk_key(c)] = c
                elif _chunk_key(c) not in ledger.requested_targets:
                    ledger.checks_omitted = True
        targets = list({_chunk_key(c): c for c in [*targets, *ledger.requested_targets.values()]}.values())
        ledger.selection_status = 'no_reliable_entry'
        # Schedule all held explicit method bodies before missing targets/classes
        # and before any outward expansion. Updating one identity uses no slot.
        held_keys = {(e.path, e.qualified_name, e.start_line, e.end_line) for e in held_items}
        targets.sort(key=lambda c: (c.get('chunk_type') == 'class', _chunk_key(c) not in held_keys))
        if len(targets) > max_candidates:
            ledger.checks_omitted = True
        for c in targets[:max_candidates]:
            check()
            module = ledger.locator.module(c['path'])
            node = next((n for n in ast.walk(module[0]) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and n.lineno == c['start_line'] and n.end_lineno == c['end_line'] and n.name == c['symbol_name']), None) if module else None
            declaration, binding_status = ledger.locator.declaration(c['path'], c['qualified_name'])
            if node is not declaration:
                node = None
            if node is None:
                ledger.record(expression=c['qualified_name'], source_path=c['path'],
                    reason=ledger.locator.inspection_rejections.get(c['path'], 'entry_not_statically_confirmed'), located=False)
                continue
            ledger.selection_status = 'explicit_targets'
            held = next((e for e in context.evidence_store.all(state.request_id) if
                (e.path, e.qualified_name, e.start_line, e.end_line) == _chunk_key(c)), None)
            ledger.record(path=c['path'], symbol=c['qualified_name'], start_line=c['start_line'], end_line=c['end_line'],
                depth=0, located=True, reason='explicit_entry', is_primary=not isinstance(node, ast.ClassDef),
                binding_status=c.get('_binding_status', binding_status))
            if held is None or held.planner_excerpt_coverage().get('truncated'):
                acquired = acquire(c['path'], node, 0, held.evidence_id if held else None,
                    c['qualified_name'], construction=c.get('_construction', False))
                held = acquired or held
            # A data-member declaration is terminal source support, not a seed
            # for traversing every unrelated method of its owning class.
            if held and (not c.get('_data_member') or c.get('_construction')):
                queue.append((held, ledger.depths.get(held.chunk_identity, 0)))
        # Candidate discovery and dispatch proof are different operations. Give
        # Planner bounded definition locators before ordinary outward reads can
        # spend the remaining read quota and slots. Explicit bodies above retain
        # priority. Only the narrow question-linked state shape below may be
        # acquired automatically; candidate bodies are never recursively expanded.
        pending_candidates = False
        automatic = []
        for item, _ in queue:
            if not _validated_body(item):
                continue
            for line, expression, resolved in ledger.locator.calls(item):
                check()
                if resolved is not None:
                    continue
                candidates, omitted = ledger.locator.annotation_candidates(item, expression, chunks,
                    min(2, max_candidates))
                ledger.checks_omitted |= omitted
                if not candidates:
                    continue
                ledger.record(source_evidence_id=item.evidence_id, source_path=item.path, call_line=line,
                    expression=expression, reason='unresolved_static_call', located=False,
                    binding_status='unconfirmed', candidate_definitions=candidates,
                    candidate_definitions_omitted=omitted)
                for candidate in candidates:
                    ledger.record(path=candidate['path'], symbol=candidate['symbol'],
                        start_line=candidate['start_line'], end_line=candidate['end_line'], located=True,
                        binding_status='candidate_definition_binding_unconfirmed', reason='definition_candidate')
                    body = next((e for e in context.evidence_store.all(state.request_id) if
                        (e.path, e.qualified_name, e.start_line, e.end_line) ==
                        (candidate['path'], candidate['symbol'], candidate['start_line'], candidate['end_line'])), None)
                    if body is None or not _validated_body(body) or body.planner_excerpt_coverage().get('truncated'):
                        pending_candidates = True
            for candidate, line in ledger.locator.state_access_candidates(item, state.user_goal, chunks, max_candidates):
                ledger.record(path=candidate['path'], symbol=candidate['symbol'], start_line=candidate['start_line'],
                    end_line=candidate['end_line'], located=True, selection_reason='question_state_accessor_pair')
                automatic.append((candidate, item.evidence_id, ledger.depths.get(item.chunk_identity, 0)+1))
        # Stable first-related-call order; skip complete owned identities. Adjacent
        # eligible definitions may share the existing range reader only when the
        # gap is whitespace. No unselected method is brought into the range.
        seen = set()
        selected = []
        for c, source_id, depth in automatic:
            key = (c['path'], c['symbol'], c['start_line'], c['end_line'])
            body = next((e for e in context.evidence_store.all(state.request_id) if
                (e.path, e.qualified_name, e.start_line, e.end_line) == key), None)
            if key in seen or (body and _validated_body(body) and body.planner_excerpt_coverage().get('truncated') is False):
                continue
            seen.add(key)
            selected.append((c, source_id, depth))
        while selected:
            c, source_id, depth = selected.pop(0)
            group = [c]
            if selected:
                other, other_source, other_depth = selected[0]
                ordered = sorted([c, other], key=lambda value: value['start_line'])
                start, end = ordered[0]['start_line'], ordered[1]['end_line']
                source_lines = physical_source_lines(ledger.locator.files[c['path']], keepends=True)
                gap = ''.join(source_lines[ordered[0]['end_line']:ordered[1]['start_line']-1])
                size = len(''.join(source_lines[start-1:end]).encode('utf-8'))
                slots = context.evidence_store.capacity-len(context.evidence_store.all(state.request_id))
                needed = sum(not any((e.path,e.qualified_name,e.start_line,e.end_line) ==
                    (g['path'],g['symbol'],g['start_line'],g['end_line']) for e in context.evidence_store.all(state.request_id)) for g in [c,other])
                if (other['path'] == c['path'] and other['repository_revision'] == c['repository_revision']
                        and other_source == source_id and other_depth == depth and ordered[0]['end_line'] < ordered[1]['start_line']
                        and not gap.strip() and end-start+1 <= limits.max_source_read_lines
                        and size <= limits.max_source_read_bytes and needed <= slots):
                    group.append(selected.pop(0)[0])
            acquire_candidates(group, source_id, depth)
        if automatic:
            pending_candidates = any(not any(_validated_body(e) and e.planner_excerpt_coverage().get('truncated') is False
                and (e.path,e.qualified_name,e.start_line,e.end_line) == (c['path'],c['symbol'],c['start_line'],c['end_line'])
                for e in context.evidence_store.all(state.request_id)) for check_row in ledger.checks
                for c in check_row.get('candidate_definitions', []))
        while queue and len(ledger.attempted) < max_candidates:
            check()
            item, depth = queue.pop(0)
            if item.validation_status != 'valid' or item.planner_excerpt_coverage().get('status') != 'known':
                continue
            # Complete a known target first; never split it to bypass read limits.
            module = ledger.locator.module(item.path)
            node = next((n for n in ast.walk(module[0]) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.lineno == item.start_line and n.end_lineno == item.end_line), None) if module else None
            if node and item.planner_excerpt_coverage().get('truncated'):
                completed = acquire(item.path, node, depth, item.evidence_id, item.qualified_name)
                if completed and completed.chunk_identity != item.chunk_identity:
                    queue.insert(0, (completed, depth))
            binding = (*item._planner_source_identity(), item.excerpt)
            if binding in ledger.expanded:
                continue
            # Deferred is not completed: a later tool can supply the candidate
            # bodies without spending this read quota. attempted still dedupes
            # actual acquisitions when ordinary expansion resumes.
            if not pending_candidates:
                ledger.expanded.add(binding)
            for line, expression, resolved in ledger.locator.calls(item):
                if resolved is None:
                    ledger.record(source_evidence_id=item.evidence_id, source_path=item.path, call_line=line,
                        expression=expression, reason='unresolved_static_call', located=False)
                    continue
                path, target = resolved
                ledger.record(path=path, symbol=target.name, start_line=target.lineno, end_line=target.end_lineno,
                    source_evidence_id=item.evidence_id, source_path=item.path, call_line=line,
                    expression=expression, depth=depth + 1, located=True,
                    reason='depth_limit' if depth >= max_depth else
                        'deferred_for_candidate_review' if pending_candidates else 'static_call')
                if depth >= max_depth or pending_candidates:
                    continue
                acquired = acquire(path, target, depth + 1, item.evidence_id, construction=isinstance(target, ast.ClassDef))
                if acquired:
                    queue.append((acquired, depth + 1))
        if queue:
            ledger.checks_omitted = True
    except (ToolDeadlineExceeded, ToolCancelled) as exc:
        ledger.events.append({'status': exc.code, 'committed': False})
    # Other errors propagate: collection or implementation failures are not
    # silently converted into successful supplementation.
