"""Bounded source contract for static exploration. Never changes the Q&A index.

All content hashes are SHA-256 of the saved decoded text encoded as UTF-8.
Git blob/raw-byte hashes, when available, are separate provenance fields.
"""
from __future__ import annotations

from app.source_lines import source_lines as physical_source_lines

import hashlib
import io
import json
import re
import subprocess
import tokenize
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from app.database import SourceFileError

MAX_FILES = 500
MAX_FILE_BYTES = 1_048_576
MAX_TOTAL_BYTES = 20_000_000


def sha(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def same_source_text(a: str, b: str) -> bool:
    # Git may check out LF blobs as CRLF on Windows. Compare only that transport
    # difference; keep the saved body and its exact UTF-8 hash unchanged.
    return a == b or a.replace('\r\n','\n') == b.replace('\r\n','\n')


def safe_path(path: str) -> bool:
    return bool(path and '\\' not in path and ':' not in path and '\0' not in path
                and not path.startswith('/') and all(p not in ('', '.', '..') for p in path.split('/')))


def identity(conn: Any, project_id: str, revision: str) -> dict[str, Any]:
    row = conn.execute('''SELECT p.id AS project_id,p.repo,p.source_type,p.source_location,
        p.repository_revision AS revision,p.status,wr.workspace_id,wr.activation_status,
        wr.repository_revision AS linked_revision,w.active_project_id,w.activation_version
        FROM projects p LEFT JOIN workspace_revisions wr ON wr.project_id=p.id
        LEFT JOIN repository_workspaces w ON w.id=wr.workspace_id WHERE p.id=?''', (project_id,)).fetchone()
    if row is None:
        raise SourceFileError('source_project_not_found', 404)
    if not revision or row['revision'] != revision:
        raise SourceFileError('source_revision_mismatch', 409)
    if (row['status'] != 'done' or not row['workspace_id'] or row['active_project_id'] != project_id
            or row['activation_status'] != 'active' or row['linked_revision'] != revision):
        raise SourceFileError('source_snapshot_inactive', 409)
    return {k: row[k] for k in ('project_id','workspace_id','revision','activation_version','repo','source_type','source_location')}


def context(db: Any, project_id: str, revision: str) -> dict[str, Any]:
    with db.connect() as conn:
        conn.execute('PRAGMA query_only=ON')
        return identity(conn, project_id, revision)


def material_path(runtime: Path, ctx: dict[str, Any]) -> Path:
    key = sha(json.dumps([ctx[k] for k in ('workspace_id','project_id','revision')]))
    return runtime / 'source-exploration' / (key + '.json')


def record(path: str, content: str, origin: str, **extra: Any) -> dict[str, Any]:
    return {'path':path, 'content':content, 'content_sha256':sha(content),
            'utf8_bytes':len(content.encode('utf-8')), 'included':True,
            'truncated':False, 'parse_status':'not_checked', 'origin':origin, **extra}


def read_sources(db: Any, runtime: Path, project_id: str, revision: str) -> dict[str, Any]:
    files: dict[str, Any] = {}; skipped = []; total = 0; saved_paths=set()
    with db.connect() as conn:
        conn.execute('PRAGMA query_only=ON'); conn.execute('BEGIN')
        ctx = identity(conn, project_id, revision)
        rows = conn.execute('''SELECT path,content,size FROM repo_files
            WHERE project_id=? AND path LIKE '%.py' ORDER BY path''', (project_id,))
        for row in rows:
            path = row['path']; text = row['content']
            if path in saved_paths:
                previous=files.pop(path,None)
                if previous: total-=previous['utf8_bytes']
                skipped.append({'path':path,'reason':'duplicate_saved_path'}); continue
            saved_paths.add(path)
            size = len(text.encode('utf-8')) if isinstance(text,str) else 0
            if not safe_path(path) or not isinstance(text,str) or size > MAX_FILE_BYTES or len(files)>=MAX_FILES or total+size>MAX_TOTAL_BYTES:
                skipped.append({'path':path,'reason':'path_or_size_limit'}); continue
            if '\ufffd' in text:
                skipped.append({'path':path,'reason':'decoding_uncertain'}); continue
            if isinstance(row['size'],int) and row['size']>size:
                skipped.append({'path':path,'reason':'saved_text_shorter_than_recorded_size',
                                'included':False,'parse_status':'not_parsed','truncation_status':'uncertain'}); continue
            files[path] = record(path,text,'saved_snapshot', original_size=row['size'])
            total += size
    path = material_path(runtime,ctx)
    supplemental = None
    if path.exists():
        try:
            if path.stat().st_size > MAX_TOTAL_BYTES * 6:
                raise ValueError('material too large')
            supplemental = json.loads(path.read_text(encoding='utf-8'))
            if any(supplemental['identity'][k] != ctx[k] for k in ('project_id','workspace_id','revision')):
                raise ValueError('material identity')
            for rel, item in supplemental['files'].items():
                if not safe_path(rel) or not rel.endswith('.py') or sha(item['content']) != item['content_sha256']:
                    raise ValueError('material hash')
                if rel in files:
                    if not same_source_text(files[rel]['content'],item['content']):
                        raise ValueError('saved snapshot differs from Git object')
                    continue
                if rel in saved_paths:
                    # The original reader owns existing paths, including rejected
                    # rows. Never analyze a different body it cannot return.
                    skipped.append({'path':rel,'reason':'saved_path_unavailable_not_shadowed'}); continue
                size = len(item['content'].encode('utf-8'))
                if size>MAX_FILE_BYTES or len(files)>=MAX_FILES or total+size>MAX_TOTAL_BYTES:
                    skipped.append({'path':rel,'reason':'combined_size_limit'}); continue
                files[rel] = item; total += size
            skipped.extend(supplemental['skipped'])
        except (OSError,ValueError,KeyError,TypeError) as exc:
            raise SourceFileError('exploration_material_invalid',409) from exc
    ctx['hash_kind'] = 'sha256(decoded_text.encode(utf-8))'
    fingerprint = sha(json.dumps([(p,f['content_sha256']) for p,f in sorted(files.items())]))
    return {'identity':ctx, 'files':files, 'fingerprint':fingerprint,
            'coverage':{'saved_snapshot_scope':'已保存文件集合；不能据此断言原仓库没有其他测试',
                        'same_revision_inventory':bool(supplemental), 'files':len(files),
                        'utf8_bytes':total, 'skipped':skipped[:100], 'skipped_count':len(skipped),
                        'limits':{'files':MAX_FILES,'file_bytes':MAX_FILE_BYTES,'total_bytes':MAX_TOTAL_BYTES},
                        'supplement_origin': supplemental['origin'] if supplemental else None}}


def git(root: Path, *args: str, limit: int = MAX_TOTAL_BYTES) -> bytes:
    # No target code, hooks, shell, network, checkout or dependency installation.
    with subprocess.Popen(['git','--no-replace-objects','-c',f'safe.directory={root.as_posix()}', '-C',str(root),*args],
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as process:
        import threading
        timer = threading.Timer(30, process.kill); timer.start()
        try:
            assert process.stdout is not None
            data = process.stdout.read(limit+1)
            if len(data)>limit:
                process.kill(); raise SourceFileError('exploration_git_limit',413)
            if process.wait()!=0:
                raise SourceFileError('exploration_revision_unavailable',409)
            return data
        finally:
            timer.cancel()


def supplement(db: Any, runtime: Path, project_id: str, revision: str, local_path: str) -> dict[str, Any]:
    ctx = context(db,project_id,revision)
    if not re.fullmatch(r'[0-9a-fA-F]{40}',revision):
        raise SourceFileError('exploration_exact_revision_required',409)
    try:
        root = Path(local_path).expanduser().resolve(strict=True)
        if not root.is_dir(): raise ValueError()
    except (OSError,ValueError,RuntimeError) as exc:
        raise SourceFileError('exploration_directory_unavailable',422) from exc
    commit = git(root,'rev-parse','--verify',revision+'^{commit}',limit=100).decode().strip()
    if commit.lower()!=revision.lower():
        raise SourceFileError('exploration_revision_unavailable',409)
    tree = git(root,'ls-tree','-rz','-l',revision,limit=2_000_000)
    files = {}; skipped = []; total = 0
    for entry in tree.split(b'\0'):
        if not entry: continue
        meta,rawpath = entry.split(b'\t',1)
        mode,kind,blob,size_raw = meta.split()
        rel = rawpath.decode('utf-8',errors='strict')
        if not rel.endswith('.py'): continue
        size = int(size_raw) if size_raw != b'-' else MAX_FILE_BYTES+1
        if (mode not in (b'100644',b'100755') or kind!=b'blob' or not safe_path(rel)
                or size>MAX_FILE_BYTES or len(files)>=MAX_FILES or total+size>MAX_TOTAL_BYTES):
            skipped.append({'path':rel,'reason':'git_type_or_size_limit'}); continue
        raw = git(root,'cat-file','blob',blob.decode(),limit=MAX_FILE_BYTES)
        actual_blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if actual_blob!=blob.decode():
            raise SourceFileError('exploration_material_invalid',409)
        try:
            encoding = tokenize.detect_encoding(io.BytesIO(raw).readline)[0]
            text = raw.decode(encoding)
        except (SyntaxError,UnicodeError,LookupError):
            skipped.append({'path':rel,'reason':'decoding_failed'}); continue
        size = len(text.encode('utf-8'))
        if total+size>MAX_TOTAL_BYTES:
            skipped.append({'path':rel,'reason':'text_size_limit'}); continue
        files[rel] = record(rel,text,'same_revision_git_object',git_blob=blob.decode(),
                            raw_sha256=hashlib.sha256(raw).hexdigest())
        total += size
    # Overlap must agree, including version-specific text. Never overwrite a snapshot.
    current = read_sources(db,runtime,project_id,revision)
    eol_equivalent=[]
    for rel,item in files.items():
        if rel in current['files']:
            saved=current['files'][rel]
            if not same_source_text(saved['content'],item['content']):
                raise SourceFileError('exploration_snapshot_content_conflict',409)
            if saved['content_sha256']!=item['content_sha256']:
                eol_equivalent.append({'path':rel,'saved_text_sha256':saved['content_sha256'],
                                       'git_decoded_text_sha256':item['content_sha256']})
    if context(db,project_id,revision) != ctx:
        raise SourceFileError('source_snapshot_inactive',409)
    result = {'identity':ctx,'origin':{'kind':'local_git_objects','revision':revision,'directory':str(root),
                                     'overlap_policy':'exact text or CRLF/LF transport equivalence; saved bytes unchanged',
                                     'eol_equivalent':eol_equivalent},
              'files':files,'skipped':skipped}
    dest = material_path(runtime,ctx); dest.parent.mkdir(parents=True,exist_ok=True)
    temporary = dest.with_suffix('.'+uuid.uuid4().hex+'.tmp')
    try:
        temporary.write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
        temporary.replace(dest)
    finally:
        if temporary.exists(): temporary.unlink()
    return read_sources(db,runtime,project_id,revision)


def read_material_file(db: Any, runtime: Path, project_id: str, revision: str, path: str) -> dict[str, Any]:
    data = read_sources(db,runtime,project_id,revision)
    item = data['files'].get(path)
    if not item or item['origin']!='same_revision_git_object':
        raise SourceFileError('source_file_not_found',404)
    ctx=data['identity']
    return {k:ctx[k] for k in ('project_id','workspace_id','revision','activation_version')} | {
        'path':path,'content':item['content'],'content_sha256':item['content_sha256'],
        'total_lines':len(physical_source_lines(item['content'])),'complete':True,'utf8_bytes':item['utf8_bytes']}
