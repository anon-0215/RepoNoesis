// @vitest-environment jsdom
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
vi.mock('./lib/api', async () => ({
  ...await vi.importActual<typeof import('./lib/api')>('./lib/api'),
  getConfigStatus: vi.fn(), listWorkspaces: vi.fn(), getWorkspace: vi.fn(), getProject: vi.fn(),
  getProjectMap: vi.fn(), getLearningPath: vi.fn(), getReport: vi.fn(), getChatHistory: vi.fn(),
  askProjectStream: vi.fn(), getSourceFile: vi.fn(),
}));
import App from './App';
import { ApiError, getChatHistory, getConfigStatus, getWorkspace, getProject, getProjectMap, getLearningPath,
  getReport, listWorkspaces, askProjectStream, getSourceFile } from './lib/api';
import type { ChatAnswer, ChatHistoryPage, ChatHistoryRecord, ConfigStatus, WorkspaceDetail, WorkspaceSummary } from './types';

const w1 = '11111111-1111-4111-8111-111111111111', w2 = '22222222-2222-4222-8222-222222222222';
const citation = { evidence_id: 'E5', path: 'a.py', qualified_name: 'answer', summary: '', snippet: 'saved snippet', start_line: 1, end_line: 2 };
export function record(id = '9007199254740993', overrides: Partial<ChatHistoryRecord> = {}): ChatHistoryRecord {
  return { record_id: id, project_id: 'p1', repository_revision: 'rev-a', revision_source: 'saved', identity_status: 'reliable',
    question: `saved question ${id}`, answer: 'saved answer [E5]', created_at: '2001-02-03 04:05:06', citations: [citation],
    citations_status: 'available', metadata_status: 'available', snapshot_status: 'active', is_current_snapshot: true,
    full_file_available: true, full_file_unavailable_reason: null,
    recovery_metadata: { format_version: 1, metadata_truncated: false, execution_mode: 'rag', answer_mode: 'llm_grounded',
      grounding_status: 'grounded', warnings: [], execution_summary: { status: 'completed', citation_validation_passed: false, relation_validation_passed: null },
      citation_bindings: [{ evidence_id: 'E5', project_id: 'p1', repository_revision: 'rev-a', path: 'a.py', qualified_name: 'answer', symbol_name: 'answer', start_line: 1, end_line: 2 }] },
    ...overrides };
}
const page = (items: ChatHistoryRecord[] = [], next_cursor: string | null = null, workspace_id = w1, active_project_id = 'p1', repository_revision = 'rev-a'): ChatHistoryPage => ({
  workspace_id, active_project_id, repository_revision, activation_version: 1, items, next_cursor,
});
const answer = (id?: string): ChatAnswer => ({ record_id: id, created_at: '2002-01-01 00:00:00', answer: 'new answer',
  execution_mode: 'agent', answer_mode: 'llm_grounded', grounding_status: 'grounded', warnings: [], citations: [] });
function deferred<T>() { let resolve!: (value: T) => void; let reject!: (error: unknown) => void;
  const promise = new Promise<T>((done, fail) => { resolve = done; reject = fail; }); return { promise, resolve, reject }; }
let host: HTMLDivElement, root: Root, version: string;
const summaries = (): WorkspaceSummary[] => [w1, w2].map((workspace_id, i) => ({ workspace_id, display_name: `project ${i + 1}`,
  source_type: 'local', project_status: 'done', repository_revision: 'rev-a', openable: true, project_id: `p${i + 1}`,
  created_at: '', updated_at: '' }));
async function flush() { for (let i = 0; i < 15; i++) await Promise.resolve(); }
async function mount() { await act(async () => { root.render(<App />); await flush(); }); }
function button(text: string) { const found = Array.from(host.querySelectorAll<HTMLButtonElement>('button')).find(node => node.textContent?.includes(text));
  if (!found) throw new Error(`missing ${text}`); return found; }
async function click(text: string) { await act(async () => { button(text).click(); await flush(); }); }
async function draft(text: string) { await act(async () => { const input = host.querySelector('textarea')!;
  Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')!.set!.call(input, text);
  input.dispatchEvent(new Event('input', { bubbles: true })); await flush(); }); }
async function submit() { await act(async () => { host.querySelector('form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true })); await flush(); }); }
beforeEach(() => {
  vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true); version = 'rev-a';
  vi.mocked(getConfigStatus).mockResolvedValue({} as ConfigStatus);
  vi.mocked(listWorkspaces).mockResolvedValue({ items: summaries(), total: 2, limit: 20, offset: 0 });
  vi.mocked(getWorkspace).mockImplementation(async (workspace_id) => ({ ...summaries().find(item => item.workspace_id === workspace_id)!,
    active_snapshot: { project_id: workspace_id === w1 ? 'p1' : 'p2', repository_revision: version,
      status: 'done', primary_language: 'Python', frameworks: [], updated_at: '' }, latest_update_run: null, learning_continuity: null } as unknown as WorkspaceDetail));
  vi.mocked(getProject).mockImplementation(async id => ({ project: { id, repo: id === 'p1' ? 'project 1' : 'project 2', repo_url: '', owner: 'test', default_branch: 'main', status: 'done', primary_language: 'Python', frameworks: [] }, overview: '', stats: {}, start_commands: [], core_files: [], modules: [] }));
  vi.mocked(getProjectMap).mockRejectedValue(new Error('map unavailable'));
  vi.mocked(getLearningPath).mockResolvedValue({ steps: [] }); vi.mocked(getReport).mockResolvedValue({ markdown: '' });
  vi.mocked(getChatHistory).mockImplementation(async (w, p, r) => page([], null, w, p, r));
  vi.mocked(askProjectStream).mockResolvedValue(answer('88'));
  window.history.replaceState(null, '', `/?workspace=${w1}`);
  host = document.createElement('div'); document.body.appendChild(host); root = createRoot(host);
});
afterEach(async () => { await act(async () => root.unmount()); vi.resetAllMocks(); vi.unstubAllGlobals(); window.localStorage.clear(); host.remove(); });

describe('database history integration', () => {
  it('clears only an accepted submission and preserves a new draft on completion', async () => {
    const stream = deferred<ChatAnswer>(); vi.mocked(askProjectStream).mockReturnValue(stream.promise);
    await mount(); await draft('   '); await submit();
    expect(askProjectStream).not.toHaveBeenCalled();
    expect(host.querySelector('textarea')!.value).toBe('   ');
    await draft('submitted question'); await submit();
    expect(host.querySelector('textarea')!.value).toBe('');
    expect(vi.mocked(askProjectStream).mock.calls[0][2]).toBe('submitted question');
    await draft('next draft'); await submit();
    expect(askProjectStream).toHaveBeenCalledTimes(1);
    expect(host.querySelector('textarea')!.value).toBe('next draft');
    await act(async () => { stream.resolve(answer()); await flush(); });
    expect(host.querySelector('textarea')!.value).toBe('next draft');
    expect(host.querySelector('.wb-question')?.textContent).toContain('submitted question');
  });
  it.each(['project', 'revision'])('rejects stale failure recovery after a %s change', async change => {
    const stream = deferred<ChatAnswer>(); vi.mocked(askProjectStream).mockReturnValue(stream.promise);
    await mount(); await draft('old request'); await submit(); await draft('new draft');
    if (change === 'revision') version = 'rev-b';
    await click(change === 'project' ? 'project 2' : 'project 1');
    await act(async () => { stream.reject(new Error('old failure')); await flush(); });
    expect(host.querySelector('textarea')!.value).toBe('new draft');
    expect(host.textContent).not.toContain('old failure');
    expect(host.textContent).not.toContain('恢复到输入框');
    expect(vi.mocked(askProjectStream).mock.calls[0][6]?.aborted).toBe(true);
  });
  it('does not replace even a whitespace draft when recovering a failed question', async () => {
    vi.mocked(askProjectStream).mockRejectedValue(new Error('fixture failure'));
    await mount(); await draft('recover me'); await submit(); await draft('  ');
    expect(button('恢复到输入框').disabled).toBe(true);
    expect(host.querySelector('textarea')!.value).toBe('  ');
    await draft(''); await click('恢复到输入框');
    expect(host.querySelector('textarea')!.value).toBe('recover me');
  });
  it('keeps the draft when no usable project is selected and submission is rejected', async () => {
    await mount(); await draft('unsent before project failure');
    vi.mocked(getProject).mockRejectedValueOnce(new Error('fixture project unavailable'));
    await click('project 2'); await submit();
    expect(host.querySelector('textarea')!.value).toBe('unsent before project failure');
    expect(host.querySelector<HTMLTextAreaElement>('textarea')!.disabled).toBe(true);
    expect(askProjectStream).not.toHaveBeenCalled();
  });
  it('opens QA without fetching optional map, learning or report data', async () => {
    vi.mocked(getLearningPath).mockRejectedValue(new Error('learning unavailable'));
    vi.mocked(getReport).mockRejectedValue(new Error('report unavailable'));
    await mount();
    expect(host.querySelector('.wb-composer textarea')).not.toBeNull();
    expect(getChatHistory).toHaveBeenCalled();
    expect(getLearningPath).not.toHaveBeenCalled();
    expect(getProjectMap).not.toHaveBeenCalled();
    expect(getReport).not.toHaveBeenCalled();
    expect(host.querySelector('.wb-nav')?.textContent).toBe('源码问答项目地图源码探索');
  });
  it('loads optional pages on demand with separate error and retry states', async () => {
    vi.mocked(getReport).mockRejectedValueOnce(new Error('report unavailable')).mockResolvedValueOnce({ markdown: 'analysis material' });
    await mount(); await click('项目地图');
    expect(host.textContent).toContain('地图接口不可用');
    await click('重试'); expect(getProjectMap).toHaveBeenCalledTimes(2);
    await click('项目分析报告');
    expect(host.textContent).toContain('报告读取失败');
    await click('重试报告');
    expect(host.querySelector<HTMLTextAreaElement>('.report-box')?.value).toBe('analysis material');
  });
  it('rejects late optional responses across workspace changes', async () => {
    const delayed = deferred<{ markdown: string }>();
    vi.mocked(getReport).mockReturnValueOnce(delayed.promise).mockResolvedValue({ markdown: 'project 2 report' });
    await mount(); await click('项目分析报告'); await click('project 2');
    await act(async () => { delayed.resolve({ markdown: 'stale project 1 report' }); await flush(); });
    expect(host.textContent).not.toContain('stale project 1 report');
    await click('项目分析报告');
    expect(host.querySelector<HTMLTextAreaElement>('.report-box')?.value).toBe('project 2 report');
  });
  it('discards a late map and keeps a required project failure visible', async () => {
    const delayed = deferred<Awaited<ReturnType<typeof getProjectMap>>>();
    vi.mocked(getProjectMap).mockReturnValueOnce(delayed.promise).mockRejectedValue(new Error('map unavailable'));
    await mount(); await click('项目地图'); await click('project 2');
    await act(async () => { delayed.resolve({ project_id: 'p1', repository_revision: 'rev-a', tree: { name: 'stale-map', path: '', type: 'directory', children: [] }, modules: [], dependency_edges: [], core_files: [] }); await flush(); });
    await click('项目地图');
    expect(host.textContent).toContain('地图接口不可用');
    expect(host.textContent).not.toContain('stale-map');
    vi.mocked(getProject).mockRejectedValueOnce(new Error('required project unavailable'));
    await click('project 1');
    expect(host.querySelector('[role="alert"]')?.textContent).toContain('required project unavailable');
    expect(host.querySelector<HTMLTextAreaElement>('.wb-composer textarea')?.disabled).toBe(true);
  });
  it('restores on reload and reopening with original time and a stable decimal ID', async () => {
    vi.mocked(getChatHistory).mockResolvedValue(page([record()])); await mount();
    expect(host.querySelector('[data-answer-id="saved:9007199254740993"]')).not.toBeNull();
    expect(host.textContent).toContain('2001-02-03 04:05:06'); expect(host.textContent).toContain('历史记录');
    expect(host.textContent).not.toContain('源码引用已校验');
    await click('project 1'); expect(getChatHistory).toHaveBeenCalledTimes(2);
    await act(async () => root.unmount()); root = createRoot(host); await mount();
    expect(host.textContent).toContain('saved answer'); expect(askProjectStream).not.toHaveBeenCalled();
  });
  it('buffers saved rows before stream terminal and deduplicates by record ID while keeping the live card', async () => {
    const history = deferred<ChatHistoryPage>(), stream = deferred<ChatAnswer>();
    vi.mocked(getChatHistory).mockReturnValue(history.promise); vi.mocked(askProjectStream).mockReturnValue(stream.promise);
    await mount(); await draft('new question'); await submit();
    const uiId = host.querySelector<HTMLElement>('.wb-answer')!.dataset.answerId;
    await act(async () => { history.resolve(page([record('88', { question: 'new question' }), record('77')])); await flush(); });
    expect(host.querySelectorAll('.wb-answer')).toHaveLength(1); expect(host.textContent).not.toContain('saved question 77');
    await act(async () => { stream.resolve(answer('88')); await flush(); });
    expect(host.querySelectorAll('.wb-answer')).toHaveLength(2);
    expect(host.querySelectorAll(`[data-answer-id="${uiId}"]`)).toHaveLength(1);
    expect(host.querySelector('[data-answer-id="saved:88"]')).toBeNull();
    expect(Array.from(host.querySelectorAll('.wb-question p')).map(node => node.textContent)).toEqual(['saved question 77', 'new question']);
  });
  it.each([new ApiError('连接中断，最终状态未知。', 0), new ApiError('请求失败', 500)])('releases buffered history after failure without claiming the saved answer (%s)', async error => {
    const history = deferred<ChatHistoryPage>(), stream = deferred<ChatAnswer>();
    vi.mocked(getChatHistory).mockReturnValue(history.promise); vi.mocked(askProjectStream).mockReturnValue(stream.promise);
    await mount(); await draft('same question'); await submit();
    await act(async () => { history.resolve(page([record('88', { question: 'same question' })])); await flush(); });
    await act(async () => { stream.reject(error); await flush(); });
    expect(host.querySelectorAll('.wb-answer')).toHaveLength(2); expect(host.querySelector('.wb-error')).not.toBeNull();
    expect(host.querySelector('textarea')!.value).toBe('');
    await click('恢复到输入框');
    expect(host.querySelector('textarea')!.value).toBe('same question'); expect(askProjectStream).toHaveBeenCalledTimes(1);
  });
  it('keeps two independent successes with identical questions', async () => {
    vi.mocked(getChatHistory).mockResolvedValue(page([record('2', { question: 'same' }), record('1', { question: 'same' })]));
    await mount(); expect(host.querySelectorAll('.wb-answer')).toHaveLength(2);
  });
  it.each(['project', 'revision'])('discards a late history response after %s switch', async change => {
    const first = deferred<ChatHistoryPage>(); vi.mocked(getChatHistory).mockReturnValueOnce(first.promise);
    await mount(); await draft('draft retained');
    if (change === 'revision') version = 'rev-b';
    await click(change === 'project' ? 'project 2' : 'project 1');
    await act(async () => { first.resolve(page([record()])); await flush(); });
    expect(host.textContent).not.toContain('saved answer'); expect(host.querySelector('textarea')!.value).toBe('draft retained');
    expect(vi.mocked(getChatHistory).mock.calls[0][5]?.aborted).toBe(true);
  });
  it('retries a history failure without clearing the draft or a successful session answer', async () => {
    vi.mocked(getChatHistory).mockRejectedValueOnce(new Error('history unavailable')).mockResolvedValueOnce(page([record('1')]));
    await mount(); await draft('question'); await submit(); await draft('keep draft');
    expect(host.textContent).toContain('new answer'); await click('重试历史加载');
    expect(host.textContent).toContain('new answer'); expect(host.textContent).toContain('saved answer');
    expect(host.querySelector('textarea')!.value).toBe('keep draft');
  });
  it('prepends an earlier page while preserving the visible card, current answers and draft', async () => {
    vi.mocked(getChatHistory).mockResolvedValueOnce(page([record('9')], 'cursor')).mockResolvedValueOnce(page([record('8'), record('7')]));
    await mount(); await draft('keep draft');
    const card = host.querySelector<HTMLElement>('[data-answer-id="saved:9"]')!, scroll = host.querySelector<HTMLElement>('.wb-scroll')!;
    let top = 120; vi.spyOn(card, 'getBoundingClientRect').mockImplementation(() => ({ top, bottom: top + 100 } as DOMRect));
    scroll.scrollTop = 50; scroll.dispatchEvent(new Event('scroll')); top = 320;
    await click('加载更早'); expect(scroll.scrollTop).toBe(250);
    expect(host.querySelector('[data-answer-id="saved:9"]')).toBe(card);
    expect(Array.from(host.querySelectorAll('.wb-answer')).map(node => node.getAttribute('data-answer-id'))).toEqual(['saved:7', 'saved:8', 'saved:9']);
    expect(host.querySelector('textarea')!.value).toBe('keep draft');
    expect(vi.mocked(getChatHistory).mock.calls[1][3]).toBe('cursor');
  });
  it('opens current saved binding and full text with the original answer identity', async () => {
    vi.mocked(getChatHistory).mockResolvedValue(page([record()]));
    vi.mocked(getSourceFile).mockResolvedValue({ project_id: 'p1', revision: 'rev-a', path: 'a.py', workspace_id: w1,
      activation_version: 1, content: 'line 1\nline 2', total_lines: 2, complete: true, content_sha256: 'a'.repeat(64), utf8_bytes: 13 });
    await mount(); await act(async () => { (host.querySelector('.wb-evidence-marker') as HTMLButtonElement).click(); });
    expect(host.querySelector('.wb-drawer code')!.textContent).toBe('saved snippet'); await click('查看完整文件');
    expect(getSourceFile).toHaveBeenCalledWith('p1', 'rev-a', 'a.py', w1, expect.any(AbortSignal));
    expect(host.querySelectorAll('.sf-line.is-cited')).toHaveLength(2);
  });
  it('keeps legacy fragments readable, missing execution details unknown, and old full text disabled', async () => {
    vi.mocked(getChatHistory).mockResolvedValue(page([record('2', { recovery_metadata: null, metadata_status: 'missing', revision_source: 'snapshot_link' }),
      record('1', { project_id: 'old', repository_revision: 'rev-old', recovery_metadata: null, metadata_status: 'missing',
        is_current_snapshot: false, snapshot_status: 'superseded', full_file_available: false, full_file_unavailable_reason: 'history_snapshot_old' })]));
    await mount(); expect(host.querySelector('.wb-evidence-marker')).toBeNull();
    expect(host.textContent).toContain('未提供结构化执行详情');
    await act(async () => { (host.querySelector('.wb-citations button') as HTMLButtonElement).click(); });
    expect(host.querySelector('.wb-drawer code')!.textContent).toBe('saved snippet');
    expect(button('查看完整文件').disabled).toBe(true); expect(host.textContent).toContain('本轮不支持该版本的全文');
    expect(getSourceFile).not.toHaveBeenCalled();
  });
});
