"""Bounded evidence from project descriptions already stored in the snapshot.

No network, execution, embedding, or code-chunk fabrication is involved.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import PurePosixPath
from typing import Any

from app.source_lines import source_lines
from app.services.evidence import Evidence, CitationValidator

MAX_DOCUMENTS = 32
MAX_DOCUMENT_BYTES = 1_048_576


def _stored_bundle(database: Any, project_id: str) -> dict[str, Any]:
    # One read transaction binds file IDs/content and the project revision.
    with database.connect() as conn:
        conn.execute('PRAGMA query_only = ON')
        conn.execute('BEGIN')
        project = conn.execute('SELECT * FROM projects WHERE id = ?', (project_id,)).fetchone()
        files = conn.execute('SELECT id, path, content FROM repo_files WHERE project_id = ?',
                             (project_id,)).fetchall()
    return {'project': dict(project) if project else {}, 'files': [dict(f) for f in files]}


def is_project_document(path: str) -> bool:
    parts = path.split('/')
    if not path or '\\' in path or ':' in path or any(p in {'', '.', '..'} for p in parts):
        return False
    name = PurePosixPath(path).name.casefold()
    # Saved textual project descriptions only; not arbitrary repository files.
    return name in {'readme', 'readme.md', 'readme.rst', 'readme.txt',
                    'readme.markdown', 'overview.md', 'description.md'}


def document_inventory(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for file in bundle.get('files', []):
        path, content = file.get('path'), file.get('content')
        if (not isinstance(path, str) or not is_project_document(path)
                or not isinstance(content, str) or not content.strip()
                or len(content.encode('utf-8')) > MAX_DOCUMENT_BYTES):
            continue
        rows.append({'path': path, 'total_lines': len(source_lines(content)),
                     'utf8_bytes': len(content.encode('utf-8')), 'source_kind': 'document'})
    return sorted(rows, key=lambda x: (x['path'].count('/'), x['path']))[:MAX_DOCUMENTS]


def planner_document_inventory(bundle: dict[str, Any], budget: int = 2048) -> dict[str, Any] | None:
    inventory = document_inventory(bundle)
    if not inventory:
        return None
    result = {'untrusted_data': True, 'read_tool': 'read_source', 'files': [],
              'omitted_count': len(inventory), 'may_be_incomplete': True}
    for item in inventory:
        trial = {**result, 'files': [*result['files'], item]}
        if len(item['path']) <= 500 and len(json.dumps(trial, ensure_ascii=False).encode('utf-8')) <= budget:
            result['files'].append(item)
    result['omitted_count'] = len(inventory) - len(result['files'])
    result['may_be_incomplete'] = bool(result['omitted_count'] or len(inventory) >= MAX_DOCUMENTS)
    return result if len(json.dumps(result, ensure_ascii=False).encode('utf-8')) <= budget else None


def document_identity(project_id: str, revision: str, file_id: int, path: str,
                      start: int, end: int, digest: str) -> str:
    return '|'.join(map(str, ('document-v1', project_id, revision, file_id, path, start, end, digest)))


def build_document(database: Any, project_id: str, revision: str, path: str,
                   start: int, end: int, *, excerpt_limit: int = 8192) -> Evidence:
    bundle = _stored_bundle(database, project_id)
    project = (bundle or {}).get('project') or {}
    if project.get('id') != project_id or project.get('repository_revision') != revision:
        raise ValueError('document_revision_mismatch')
    if not is_project_document(path):
        raise ValueError('document_path_unsupported')
    files = [f for f in bundle.get('files', []) if f.get('path') == path]
    if len(files) != 1 or not isinstance(files[0].get('content'), str):
        raise ValueError('source_file_not_found')
    file = files[0]
    content = file['content']
    if len(content.encode('utf-8')) > MAX_DOCUMENT_BYTES:
        raise ValueError('document_too_large')
    lines = source_lines(content, keepends=True)
    if type(start) is not int or type(end) is not int or start < 1 or end < start or end > len(lines):
        raise ValueError('source_line_out_of_range')
    text = ''.join(lines[start - 1:end])
    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
    file_id = int(file['id'])
    if file_id < 1:
        raise ValueError('invalid_document_identity')
    item = Evidence(
        evidence_id='E1', project_id=project_id,
        repository_id=f"{project.get('owner', '')}/{project.get('repo', '')}",
        repository_url=str(project.get('repo_url', '')), repository_revision=revision,
        path=path, language='text', code_chunk_id=None,
        chunk_identity=document_identity(project_id, revision, file_id, path, start, end, digest),
        chunk_type='document', symbol_name='', qualified_name=path,
        start_line=start, end_line=end, content_hash=digest, excerpt=text[:excerpt_limit],
        retrieval_sources=['stored_document'], lexical_score=None, lexical_rank=None,
        semantic_score=None, semantic_rank=None, fusion_score=0.0, fusion_rank=1,
        selection_reason='stored_project_document', source_kind='document', document_file_id=file_id,
    )
    return CitationValidator(database).validate(item)


def retrieve_documents(database: Any, project_id: str, revision: str, query: str,
                       *, path: str | None = None, limit: int = 1) -> list[Evidence]:
    """Lexical windows, at most one supplementary document per search.

    Query tokens must occur in the stored text/path. File inventory remains
    available to the Planner even when a cross-language query has no hits.
    """
    bundle = _stored_bundle(database, project_id)
    tokens = set(re.findall(r'[\w]+', query.casefold()))
    tokens = {t for t in tokens if len(t) > 1}
    scored = []
    files = {f['path']: f for f in bundle.get('files', [])}
    for meta in document_inventory(bundle):
        if path and path != meta['path']:
            continue
        file = files[meta['path']]
        lines = source_lines(file['content'], keepends=True)
        for start in range(1, len(lines) + 1, 32):
            end = min(len(lines), start + 31)
            haystack = (meta['path'] + '\n' + ''.join(lines[start - 1:end])).casefold()
            score = sum(1 for token in tokens if token in haystack)
            if score:
                scored.append((-score, meta['path'].count('/'), meta['path'], start, end))
    result = []
    for _, _, candidate_path, start, end in sorted(scored)[:max(0, min(limit, 2))]:
        try:
            item = build_document(database, project_id, revision, candidate_path, start, end,
                                  excerpt_limit=2000)
        except ValueError:
            # A missing/changed/ambiguous document must not disable code Q&A.
            continue
        if item.validation_status == 'valid':
            item.evidence_id = f'E{len(result) + 1}'
            result.append(item)
    return result


def validate_document(database: Any, item: Evidence) -> str | None:
    if (item.code_chunk_id is not None or item.chunk_type != 'document'
            or item.language != 'text' or item.symbol_name != '' or item.qualified_name != item.path
            or type(item.document_file_id) is not int or item.document_file_id < 1
            or not is_project_document(item.path)):
        return 'invalid document identity'
    bundle = _stored_bundle(database, item.project_id)
    project = bundle.get('project') or {}
    if (project.get('id') != item.project_id
            or project.get('repository_revision') != item.repository_revision
            or str(project.get('repo_url', '')) != item.repository_url
            or f"{project.get('owner', '')}/{project.get('repo', '')}" != item.repository_id):
        return 'repository identity or revision mismatch'
    files = [f for f in bundle.get('files', []) if f.get('path') == item.path]
    if (len(files) != 1 or files[0].get('id') != item.document_file_id
            or not isinstance(files[0].get('content'), str)):
        return 'document file identity mismatch'
    text = files[0]['content']
    lines = source_lines(text, keepends=True)
    if (len(text.encode('utf-8')) > MAX_DOCUMENT_BYTES or item.start_line < 1
            or item.end_line < item.start_line or item.end_line > len(lines)):
        return 'invalid document line range'
    content = ''.join(lines[item.start_line - 1:item.end_line])
    digest = hashlib.sha256(content.encode('utf-8')).hexdigest()
    if (digest != item.content_hash or not content.startswith(item.excerpt)
            or not item.excerpt or item.chunk_identity != document_identity(
                item.project_id, item.repository_revision, item.document_file_id,
                item.path, item.start_line, item.end_line, digest)):
        return 'document content or identity mismatch'
    from app.services.evidence import _excerpt_coverage
    item._excerpt_coverage = _excerpt_coverage(content, item.excerpt, item.start_line, item.end_line)
    item._bind_planner_source()
    return None
