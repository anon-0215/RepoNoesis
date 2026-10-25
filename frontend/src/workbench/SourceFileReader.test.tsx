// @vitest-environment jsdom
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { getSourceFile } from '../lib/api';
import type { SourceFile } from '../types';
import { sourceLines } from '../lib/sourceLines';
import { Workbench, type WorkbenchAnswer } from './Workbench';
import { ProjectMapView } from './ProjectMapView';

vi.mock('../lib/api', () => ({ getSourceFile: vi.fn() }));
const read = vi.mocked(getSourceFile);
const revision = 'rev-a';
const citation = { path: 'a.py', qualified_name: 'value', summary: '', snippet: 'excerpt only', start_line: 603, end_line: 605 };
const answer: WorkbenchAnswer = { id: 'a', projectId: 'p', workspaceId: 'w', revision, question: 'question', status: 'success', result: { answer: 'Saved answer', citations: [citation], warnings: [], answer_mode: 'llm_grounded', grounding_status: 'grounded' } };
const content = Array.from({ length: 920 }, (_, i) => `line ${i + 1} <script> & 界`).join('\r\n') + '\r\n';
const file = (path = 'a.py', text = content): SourceFile => ({ project_id: 'p', revision, workspace_id: 'w', activation_version: 1, path, content: text, total_lines: sourceLines(text).length, utf8_bytes: text.length, content_sha256: 'a'.repeat(64), complete: true });
let root: Root; let host: HTMLDivElement;
function view(page = 'map', currentRevision = revision, projectId = 'p', item = answer, generation = 1) {
  return <Workbench contextGeneration={generation} project={{ project: { id: projectId, repo: 'fixture', repo_url: '', owner: 'test', default_branch: 'main', status: 'done', primary_language: 'Python', frameworks: [] }, stats: {}, overview: '', start_commands: [], core_files: [], modules: [] }} workspaces={[]} workspaceId="w" revision={currentRevision} question="draft remains" answers={[item]} error={null} loading={false} libraryLoading={false} projectLoading={false} statusMessage="" activePage={page as 'map' | 'ask'} content={<ProjectMapView projectId={projectId} workspaceId="w" revision={currentRevision} projectMap={{ tree: { type: 'directory', name: 'fixture', path: '', children: ['a.py', 'b.md'].map(path => ({ type: 'file', path, name: path })) }, modules: [], core_files: [], dependency_edges: [] }} error={null} loading={false} onRetry={vi.fn()} />} onQuestionChange={vi.fn()} onSubmit={vi.fn()} onOpenWorkspace={vi.fn()} onRefreshLibrary={vi.fn()} onNavigate={vi.fn()} />;
}
async function render(element = view()) {
  if (!root) { host = document.createElement('div'); document.body.appendChild(host); root = createRoot(host); }
  await act(async () => root.render(element));
}
function button(text: string) { if (text === '片段 1') return host.querySelector<HTMLButtonElement>('.wb-citations button')!; const result = Array.from(host.querySelectorAll<HTMLButtonElement>('button')).find(b => !b.closest('[hidden]') && (b.textContent === text || b.getAttribute('aria-label') === text)); if (!result) throw new Error(`missing button ${text}`); return result; }
async function click(text: string) { await act(async () => button(text).click()); }
async function openMap(path = 'a.py') { await click(`文件 ${path}`); await click('打开源码'); }
beforeEach(() => vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true));
afterEach(async () => { if (root) await act(async () => root.unmount()); host?.remove(); root = undefined as unknown as Root; vi.restoreAllMocks(); read.mockReset(); vi.unstubAllGlobals(); });

describe('snapshot full file reader', () => {
  it('map opens first line and preserves selection, viewport and focus on Escape', async () => {
    read.mockResolvedValue(file()); await render(); await openMap();
    expect(read).toHaveBeenCalledWith('p', revision, 'a.py', 'w', expect.any(AbortSignal));
    expect(host.querySelector('[data-source-line="1"]')).not.toBeNull();
    expect(host.querySelector('.sf-line.is-cited')).toBeNull();
    const world = host.querySelector('.pm-world'); const transform = world?.getAttribute('style');
    await act(async () => document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })));
    await new Promise(resolve => setTimeout(resolve, 5));
    expect(host.querySelector('.sf-reader')).toBeNull();
    expect(host.querySelector('.pm-detail')).not.toBeNull();
    expect(host.querySelector('.pm-world')).toBe(world);
    expect(world?.getAttribute('style')).toBe(transform);
    expect(document.activeElement).toBe(button('打开源码'));
  });

  it('citation keeps excerpt, locates server range after segment render, wraps and copies raw full text', async () => {
    const copy = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: copy } });
    read.mockResolvedValue(file()); await render(view('ask')); await click('片段 1');
    expect(host.querySelector('.wb-drawer code')?.textContent).toBe('excerpt only');
    await click('查看完整文件');
    expect(host.querySelectorAll('.sf-line.is-cited')).toHaveLength(3);
    expect(host.querySelector('.sf-line.is-cited')?.getAttribute('data-source-line')).toBe('603');
    expect(host.querySelector('script')).toBeNull();
    await click('折行：关'); expect(host.querySelector('.sf-lines.is-wrapped')).not.toBeNull();
    await click('复制全文'); expect(copy).toHaveBeenCalledWith(content);
    await click('末段'); expect(host.querySelector('[data-source-line="920"]')?.textContent).toContain('line 920');
    await click('首段'); expect(host.querySelector('[data-source-line="1"]')).not.toBeNull();
    await click('关闭全文阅读器');
    expect(host.querySelector('.wb-drawer code')?.textContent).toBe('excerpt only');
    expect(host.querySelector('textarea')?.value).toBe('draft remains');
    expect(host.textContent).toContain('Saved answer');
  });

  it.each([[0, 2], [3, 2], [1, 9999], [1.5, 2]])('does not clamp invalid range %s–%s', async (start, end) => {
    read.mockResolvedValue(file()); await render(view('ask', revision, 'p', { ...answer, result: { ...answer.result!, citations: [{ ...citation, start_line: start, end_line: end }] } }));
    await click('片段 1'); await click('查看完整文件');
    expect(host.textContent).toContain('无法定位该范围');
    expect(host.querySelector('.sf-line.is-cited')).toBeNull();
  });

  it('rejects unknown answer identity and stale answer revision without a read', async () => {
    await render(view('ask', revision, 'p', { ...answer, projectId: undefined }));
    await click('片段 1'); await click('查看完整文件');
    expect(host.textContent).toContain('缺少可靠'); expect(read).not.toHaveBeenCalled();
    await click('关闭全文阅读器');
    await render(view('ask', revision, 'p', { ...answer, revision: 'old' }));
    await click('片段 1'); await click('查看完整文件');
    expect(host.textContent).toContain('版本不符'); expect(read).not.toHaveBeenCalled();
  });

  it('ignores late file response during fast switching', async () => {
    let resolve!: (file: SourceFile) => void;
    read.mockImplementationOnce(() => new Promise(done => { resolve = done; })).mockResolvedValueOnce(file('b.md', 'current file'));
    await render(); await openMap(); expect(host.textContent).toContain('正在读取');
    await click('文件 b.md'); await click('打开源码');
    await act(async () => resolve(file('a.py', 'late file')));
    expect(host.querySelector('.sf-reader')?.textContent).toContain('current file');
    expect(host.querySelector('.sf-reader')?.textContent).not.toContain('late file');
    expect(read.mock.calls[0][4]?.aborted).toBe(true);
  });

  it.each(['close', 'project', 'revision', 'generation'])('ignores late response after %s', async (action) => {
    let resolve!: (file: SourceFile) => void;
    read.mockImplementation(() => new Promise(done => { resolve = done; }));
    await render(); await openMap();
    if (action === 'close') await click('关闭全文阅读器');
    else await render(view('map', action === 'revision' ? 'rev-b' : revision, action === 'project' ? 'p2' : 'p', answer, action === 'generation' ? 2 : 1));
    await act(async () => resolve(file()));
    expect(host.querySelector('.sf-reader')).toBeNull();
    expect(read.mock.calls[0][4]?.aborted).toBe(true);
  });

  it('shows read failure, retries, and handles empty text', async () => {
    read.mockRejectedValueOnce(new Error('文件超过 1 MiB')).mockResolvedValueOnce(file('a.py', ''));
    await render(); await openMap(); expect(host.textContent).toContain('文件超过 1 MiB');
    await click('重试读取'); expect(host.textContent).toContain('合法的空文件');
    expect(host.textContent).toContain('0 行');
  });

  it('switches to another reading surface without stacking dialogs', async () => {
    read.mockResolvedValue(file());
    await render(view('ask', revision, 'p', { ...answer, result: { ...answer.result!, answer: 'Saved answer\n\n```python\nprint(42)\n```' } }));
    await click('片段 1'); await click('查看完整文件');
    await act(async () => { button('片段 1').click(); });
    expect(host.querySelector('.sf-reader')).toBeNull();
    expect(host.querySelector('.wb-drawer')?.hasAttribute('hidden')).toBe(false);
    await click('查看完整文件');
    await act(async () => { Array.from(host.querySelectorAll<HTMLButtonElement>('.wb-markdown button')).find(b => b.textContent === '放大阅读')!.click(); });
    expect(host.querySelector('.sf-reader')).toBeNull();
    expect(host.querySelectorAll('.wb-code-modal')).toHaveLength(1);
  });

  it('uses Python physical line rules without splitting Unicode characters', () => {
    expect(sourceLines('')).toEqual([]); expect(sourceLines('\n')).toEqual(['']);
    expect(sourceLines('a\r\nb\r\n')).toEqual(['a', 'b']);
    expect(sourceLines('a\rb\n\n')).toEqual(['a', 'b', '']);
    expect(sourceLines('a\vb\fc\x1cd\x85e\u2028f\u2029')).toEqual(['a\vb\fc\x1cd\x85e\u2028f\u2029']);
  });
});
