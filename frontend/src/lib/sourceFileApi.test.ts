import { afterEach, describe, expect, it, vi } from 'vitest';
import { getSourceFile } from './api';

const body = { project_id: 'p', revision: 'r', path: '中文 /a.py', workspace_id: 'w', activation_version: 1, content: 'a\r\n', total_lines: 1, complete: true, content_sha256: 'a'.repeat(64), utf8_bytes: 3 };
afterEach(() => vi.unstubAllGlobals());
describe('source file API binding', () => {
  it('encodes exact path and carries AbortSignal', async () => {
    const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify(body)));
    vi.stubGlobal('fetch', fetch); const controller = new AbortController();
    expect(await getSourceFile('p', 'r', body.path, 'w', controller.signal)).toEqual(body);
    const [url, options] = fetch.mock.calls[0]; const parsed = new URL(url);
    expect(parsed.searchParams.get('path')).toBe(body.path); expect(parsed.searchParams.get('revision')).toBe('r');
    expect(options.signal).toBe(controller.signal);
  });
  it.each([{project_id:'other'}, {revision:'old'}, {path:'different'}, {workspace_id:'other'}, {complete:false}, {utf8_bytes:1_048_577}, {utf8_bytes:2}, {total_lines:-1}, {total_lines:2}, {content_sha256:'git-blob'}])('rejects mismatched response %j', async (changes) => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({...body, ...changes}))));
    await expect(getSourceFile('p', 'r', body.path, 'w')).rejects.toThrow('身份或完整性合同不符');
  });
  it.each(['source_file_not_found', 'source_revision_mismatch', 'source_snapshot_inactive', 'source_file_too_large', 'source_project_unavailable'])('projects safe distinguishable error %s', async (code) => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: { code, message: 'PRIVATE SERVER DETAIL', retryable: false } }), { status: 409 })));
    await expect(getSourceFile('p', 'r', body.path, 'w')).rejects.toThrow(/已保存|版本|快照|1 MiB|项目/);
    await expect(getSourceFile('p', 'r', body.path, 'w')).rejects.not.toThrow('PRIVATE SERVER DETAIL');
  });
  it('rejects a null response with a safe contract error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('null')));
    await expect(getSourceFile('p', 'r', body.path, 'w')).rejects.toThrow('身份或完整性合同不符');
  });
});
