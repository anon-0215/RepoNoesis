"""AST-only exception text probe. Does not import or execute inspected code."""
import ast
import hashlib
import re

BOUNDARY = '仅为静态候选和局部结构；不证明本次运行根因、完整执行路径或异常类型运行绑定。外部调用、动态消息和运行状态未知。'

def digest(data):
    return hashlib.sha256(data).hexdigest()

def anchor_strength(text):
    """Count Unicode non-whitespace characters; never normalize matching text."""
    return sum(not char.isspace() for char in text)

def message_pattern(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value:
        return {'mode': 'exact', 'literal': node.value, 'template': node.value}
    if not isinstance(node, ast.JoinedStr):
        return None
    slots = sum(isinstance(x, ast.FormattedValue) for x in node.values)
    literals = [x.value for x in node.values if isinstance(x, ast.Constant) and isinstance(x.value, str)]
    if not 1 <= slots <= 4 or sum(anchor_strength(x) for x in literals) < 8 or max([anchor_strength(x) for x in literals] or [0]) < 6:
        return None
    pieces = []
    display = []
    for x in node.values:
        if isinstance(x, ast.Constant) and isinstance(x.value, str):
            pieces.append(re.escape(x.value))
            display.append(x.value)
        elif isinstance(x, ast.FormattedValue):
            # Bound matching only. No evaluation, formatting or expression resolution.
            pieces.append(r'[\s\S]{0,512}?')
            display.append('{…}')
        else:
            return None
    return {'mode': 'template', 'regex': ''.join(pieces), 'template': ''.join(display),
            'segments': [x.value if isinstance(x, ast.Constant) else None for x in node.values]}


def matches(pattern, text):
    """Same finite-slot semantics without regex backtracking (O(segments * text))."""
    if pattern['mode'] == 'exact':
        return text == pattern['literal']
    reachable = [True] + [False] * len(text)
    for segment in pattern['segments']:
        if segment is None:
            following = []; window = 0
            for i in range(len(reachable)):
                window += reachable[i]
                if i > 512: window -= reachable[i-513]
                following.append(window > 0)
        else:
            size = len(segment)
            following = [False] * len(reachable)
            for i, possible in enumerate(reachable):
                if possible and text.startswith(segment, i):
                    following[i+size] = True
        reachable = following
    return reachable[-1]

class Extractor:
    def __init__(self, text, repo, path, revision, sha256):
        self.text, self.repo, self.path, self.revision, self.sha256 = text, repo, path, revision, sha256
        self.lines = physical_source_lines(text)
        self.items = []
        self.stats = {'raise_total': 0, 'supported': 0, 'unsupported': 0}

    def raw(self, node):
        return ast.get_source_segment(self.text, node) or ''

    def frame(self, node, kind, source=None, line=None):
        return {'kind': kind, 'line': line or node.lineno, 'source': source if source is not None else self.lines[node.lineno - 1].strip()}

    def walk(self, node, symbols=(), controls=()):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Definition-time ancestors do not control execution of a later call.
            for child in node.body:
                self.walk(child, symbols + (node.name,), ())
            return
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                self.walk(child, symbols + (node.name,), ())
            return
        if isinstance(node, ast.If):
            kind = 'elif' if self.lines[node.lineno - 1].lstrip().startswith('elif ') else 'if'
            source = f'{kind} {self.raw(node.test)}:'
            for child in node.body:
                self.walk(child, symbols, controls + (self.frame(node, kind, source),))
            for child in node.orelse:
                self.walk(child, symbols, controls + (self.frame(node, 'else', f'else（对应 {source} 的条件为假分支）'),))
            return
        if isinstance(node, (ast.Try, ast.TryStar)):
            for child in node.body:
                self.walk(child, symbols, controls + (self.frame(node, 'try'),))
            for handler in node.handlers:
                for child in handler.body:
                    self.walk(child, symbols, controls + (self.frame(handler, 'except'),))
            for field, kind in [('orelse', 'try_else'), ('finalbody', 'finally')]:
                for child in getattr(node, field):
                    self.walk(child, symbols, controls + (self.frame(node, kind, kind + '（所属 try 行 ' + str(node.lineno) + '）'),))
            return
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith)):
            kind = type(node).__name__.lower()
            for child in node.body:
                self.walk(child, symbols, controls + (self.frame(node, kind),))
            for child in getattr(node, 'orelse', []):
                self.walk(child, symbols, controls + (self.frame(node, kind + '_else', 'else（循环正常结束分支；所属循环行 ' + str(node.lineno) + '）'),))
            return
        if isinstance(node, ast.Match):
            for case in node.cases:
                source = 'case ' + self.raw(case.pattern)
                if case.guard:
                    source += ' if ' + self.raw(case.guard)
                for child in case.body:
                    self.walk(child, symbols, controls + (self.frame(case.pattern, 'case', source),))
            return
        if isinstance(node, ast.Raise):
            self.stats['raise_total'] += 1
            exc = node.exc
            msg = None
            if isinstance(exc, ast.Call):
                msg = exc.args[0] if exc.args else next((k.value for k in exc.keywords if k.arg == 'message'), None)
            pattern = message_pattern(msg)
            if pattern is None:
                self.stats['unsupported'] += 1
                return
            self.stats['supported'] += 1
            key = f'{self.repo}:{self.revision}:{self.sha256}:{self.path}:{node.lineno}:{node.col_offset}'
            self.items.append({
                'id': digest(key.encode())[:20], 'repo': self.repo, 'revision': self.revision,
                'path': self.path, 'line': node.lineno, 'end_line': node.end_lineno,
                'symbol': '.'.join(symbols) or '<module>', 'exception_expression': self.raw(exc.func),
                'message_source': self.raw(msg), 'raise_source': self.raw(node),
                'controls': list(controls), 'pattern': pattern, 'sha256': self.sha256,
                'boundary': BOUNDARY,
                'unconditional_in_local_scope': not controls,
            })
            return
        for child in ast.iter_child_nodes(node):
            self.walk(child, symbols, controls)

def extract_text(text, repo='fixture', path='example.py', revision='test'):
    ext = Extractor(text, repo, path, revision, digest(text.encode()))
    ext.walk(ast.parse(text))
    return ext

from app.source_lines import source_lines as physical_source_lines
