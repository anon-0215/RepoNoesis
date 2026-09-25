// @vitest-environment jsdom
import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import { Workbench, type WorkbenchAnswer } from './Workbench';
import type { AskFailure } from '../types';

const answer = (text: string, symbol: string): WorkbenchAnswer => ({
  question: `question for ${symbol}`,
  result: {
    answer: text,
    citations: [{ path: `src/${symbol}.py`, summary: '', snippet: `def ${symbol}(): pass`, qualified_name: symbol, start_line: 10, end_line: 11 }],
    answer_mode: 'llm_grounded', grounding_status: 'grounded', warnings: []
  },
  clientRequestId: symbol === 'first' ? 1 : 2,
  revision: 'a'.repeat(40)
});

const revision = 'a'.repeat(40);
function boundAnswer(id: string, evidenceId: string, symbol: string, extra = ''): WorkbenchAnswer {
  const citation = { evidence_id: evidenceId, path: `src/${symbol}.py`, summary: '', snippet: `def ${symbol}():\n    return '${id}'`, qualified_name: symbol, start_line: 10, end_line: 11 };
  return { id, status: 'success', question: id, projectId: 'p', workspaceId: 'workspace', revision,
    result: { answer: `回答 [${evidenceId}]${extra}`, citations: [citation], evidence: [{ evidence_id: evidenceId, project_id: 'p', repository_revision: revision, path: citation.path, qualified_name: symbol, start_line: 10, end_line: 11 }], answer_mode: 'llm_grounded', grounding_status: 'grounded', warnings: [], execution_summary: { status: 'completed', evidence_count: 1, citation_count: 1, agent_elapsed_ms: id === 'first' ? 123 : 456 } } };
}
function view(answers: WorkbenchAnswer[], currentRevision = revision) {
  return <Workbench project={{ project: { id: 'p', repo: 'demo', repo_url: '', owner: 'local', default_branch: 'main', status: 'ready', primary_language: 'Python', frameworks: [] }, overview: '', stats: {}, start_commands: [], core_files: [], modules: [] }} workspaces={[]} workspaceId="workspace" revision={currentRevision} question="" answers={answers} error={null} loading={false} libraryLoading={false} projectLoading={false} statusMessage="" activePage="ask" content={null} onQuestionChange={vi.fn()} onSubmit={vi.fn()} onOpenWorkspace={vi.fn()} onRefreshLibrary={vi.fn()} onNavigate={vi.fn()} />;
}

describe('answer-owned evidence and execution summary', () => {
  it('marks a validated lexical fallback as a successful answer', () => {
    const item = boundAnswer('lexical-fallback', 'E1', 'auth');
    item.result!.execution_mode = 'rag';
    item.result!.warnings = ['Semantic retrieval did not complete; continuing with lexical code-chunk candidates.'];
    item.progress = [{ request_id: 'fallback-request', client_request_id: '1', project_id: 'p',
      repository_revision: revision, sequence: 1, type: 'base_completed', status: 'succeeded', new_evidence_count: 1 }];
    item.result!.execution_summary!.base_retrieval = { attempted: true, status: 'succeeded', new_evidence_count: 1,
      retrieval_detail: { semantic_status: 'timed_out', retrieval_source: 'lexical' } };
    const markup = renderToStaticMarkup(view([item]));
    expect(markup).toContain('语义检索未完成，本次基于词法检索证据回答');
    expect(markup).toContain('执行进度 · 请求已结束');
    expect(markup).not.toContain('本次请求失败');
  });
  it('keeps each answer mode and its own progress when the composer mode changes', () => {
    const first = boundAnswer('rag', 'E1', 'first');
    first.result!.execution_mode = 'rag';
    first.progress = [{ request_id: 'r1', client_request_id: '1', project_id: 'p', repository_revision: revision, sequence: 1, type: 'base_completed', status: 'succeeded', new_evidence_count: 1 }];
    const second = boundAnswer('agent', 'E5', 'second');
    second.result!.execution_mode = 'agent';
    second.progress = [{ request_id: 'r2', client_request_id: '2', project_id: 'p', repository_revision: revision, sequence: 1, type: 'tool_completed', status: 'succeeded' }];
    const markup = renderToStaticMarkup(view([first, second]));
    expect(markup).toContain('执行模式：RAG');
    expect(markup).toContain('执行模式：Agent');
    expect(markup).toContain('基础检索完成 · 新增证据 1');
    expect(markup).toContain('补充工具已执行 · 成功');
  });
  it('keeps the failed BASE outcome visible and hides vacuous citation passes', () => {
    const failed: WorkbenchAnswer = {
      id: 'failed', status: 'failure', question: 'question', requestedMode: 'rag',
      revision, projectId: 'p', clientRequestId: 2,
      failure: { diagnostics: { request_id: 'server-2', agent_status: 'insufficient_evidence',
        failure_stage: 'retrieval', failure_reason_code: 'evidence_insufficient',
        evidence_count: 0, elapsed_ms: 22,
        base_retrieval: { attempted: true, status: 'zero_hit', retrieval_hit_count: 0,
          normalized_candidate_count: 0, valid_candidate_count: 0,
          new_evidence_count: 0, rejected_candidate_count: 0 }
      } } as AskFailure,
      progress: [
        { request_id: 'server-2', client_request_id: '2', project_id: 'p', repository_revision: revision, sequence: 1, type: 'base_completed', status: 'zero_hit', new_evidence_count: 0 },
        ...[2, 3, 4].map((sequence) => ({ request_id: 'server-2', client_request_id: '2', project_id: 'p', repository_revision: revision, sequence, type: 'citation_checked', phase: 'before_answer' as const, passed: true }))
      ]
    };
    const markup = renderToStaticMarkup(view([failed]));
    expect(markup).toContain('基础检索未命中 · 新增证据 0');
    expect(markup).not.toContain('证据引用预检完成');
    expect(markup).toContain('请求模式：RAG');
    expect(markup).toContain('执行进度 · 请求失败');
  });
  it('explains a BASE timeout without claiming the repository has no matching source', () => {
    const item: WorkbenchAnswer = {
      id: 'base-timeout', status: 'failure', question: 'question', requestedMode: 'rag',
      revision, projectId: 'p',
      failure: { diagnostics: { request_id: 'request-timeout', agent_status: 'insufficient_evidence',
        failure_stage: 'retrieval', failure_reason_code: 'evidence_insufficient', elapsed_ms: 33327,
        base_retrieval: { attempted: true, status: 'timed_out', retrieval_hit_count: 0,
          normalized_candidate_count: 0, valid_candidate_count: 0,
          new_evidence_count: 0, rejected_candidate_count: 0 }
      } } as AskFailure,
      progress: [{ request_id: 'request-timeout', client_request_id: '2', project_id: 'p',
        repository_revision: revision, sequence: 3, type: 'base_completed', status: 'timed_out' }]
    };
    const markup = renderToStaticMarkup(view([item]));
    expect(markup).toContain('基础检索超时');
    expect(markup).toContain('这不表示源码没有相关内容');
    expect(markup).toContain('请求 ID：request-timeout');
    expect(markup).toContain('失败阶段：retrieval · evidence_insufficient');
    expect(markup).not.toContain('检索命中 0 条');
  });
  it('shows three real citation checkpoints without merging equal outcomes', () => {
    const item = boundAnswer('three-checks', 'E1', 'source');
    item.result!.execution_mode = 'rag';
    item.progress = (['initial_evidence', 'generation_input', 'post_answer_evidence'] as const).map((checkpoint, index) => ({
      request_id: 'server-checks', client_request_id: '1', project_id: 'p',
      repository_revision: revision, sequence: index + 1, type: 'citation_checked',
      checkpoint, passed: true,
    }));
    const markup = renderToStaticMarkup(view([item]));
    expect(markup).toContain('初始证据身份校验完成 · 通过');
    expect(markup).toContain('生成输入证据校验完成 · 通过');
    expect(markup).toContain('回答后证据复核完成 · 通过');
    expect(markup).toContain('执行进度 · 请求已结束');
  });
  it('uses the completed Agent summary to explain its recoverable no-progress stop', () => {
    const item = boundAnswer('agent-recovered', 'E1', 'source');
    item.result!.execution_mode = 'agent';
    item.result!.warnings = ['Agent stopped after consecutive no-progress steps.'];
    item.result!.execution_summary = { status: 'completed', planner_termination_reason: 'no_progress' };
    const markup = renderToStaticMarkup(view([item]));
    expect(markup).toContain('补充检索连续未增加证据，已基于现有证据完成回答。');
    expect(markup).not.toContain('Agent stopped after consecutive no-progress steps.');
  });
  it('keeps duplicate E1 in separate answers, bottom and marker selection, modal code, and histories', async () => {
    const host = document.createElement('div'); document.body.appendChild(host); const root = createRoot(host);
    const first = boundAnswer('first', 'E1', 'first'); const second = boundAnswer('second', 'E1', 'second');
    await act(async () => root.render(view([first])));
    await act(async () => root.render(view([first, second])));
    const cards = host.querySelectorAll<HTMLElement>('.wb-answer');
    await act(async () => (cards[0].querySelector('.wb-evidence-marker') as HTMLButtonElement).click());
    expect(host.querySelector('.wb-drawer .wb-code-pane')?.textContent).toContain("return 'first'");
    await act(async () => (cards[1].querySelector('.wb-citations button') as HTMLButtonElement).click());
    expect(host.querySelector('.wb-drawer .wb-code-pane')?.textContent).toContain("return 'second'");
    expect(cards[0].querySelector('.wb-evidence-marker')?.getAttribute('aria-pressed')).toBe('false');
    await act(async () => (host.querySelector('.wb-drawer .wb-code-toolbar button:last-child') as HTMLButtonElement).click());
    expect(host.querySelector('.wb-code-modal .wb-code-pane')?.textContent).toContain("return 'second'");
    await act(async () => (cards[0].querySelector('.wb-citations button') as HTMLButtonElement).click());
    expect(host.querySelector('.wb-code-modal')).toBeNull();
    expect(host.querySelector('.wb-drawer .wb-code-pane')?.textContent).toContain("return 'first'");
    const summaries = host.querySelectorAll<HTMLElement>('.wb-execution');
    expect(summaries[0].textContent).toContain('123 毫秒'); expect(summaries[1].textContent).toContain('456 毫秒');
    await act(async () => root.render(view([first, second], 'b'.repeat(40))));
    expect(host.querySelector('.wb-drawer')).toBeNull();
    await act(async () => root.unmount()); host.remove();
  });

  it('shows sparse E5, requires identity, and groups multiple fragments without selecting one silently', async () => {
    const host = document.createElement('div'); document.body.appendChild(host); const root = createRoot(host);
    const sparse = boundAnswer('sparse', 'E5', 'part');
    const citation = sparse.result!.citations[0];
    sparse.result!.citations.push({ ...citation, snippet: 'second fragment', start_line: 11, end_line: 11 });
    await act(async () => root.render(view([sparse])));
    expect(host.querySelector('.wb-evidence-marker')?.textContent).toBe('[E5]');
    expect(host.querySelector('.wb-citations')?.textContent).toContain('[E5]');
    await act(async () => (host.querySelector('.wb-evidence-marker') as HTMLButtonElement).click());
    expect(host.querySelector('.wb-drawer .wb-code-pane')).toBeNull();
    await act(async () => (host.querySelectorAll('.wb-fragment-list button')[1] as HTMLButtonElement).click());
    expect(host.querySelector('.wb-drawer .wb-code-pane')?.textContent).toContain('second fragment');
    await act(async () => root.unmount()); host.remove();
  });
});

describe('Workbench response projection', () => {
  it('keeps citations scoped to their own answers and renders answer text without HTML execution', () => {
    const markup = renderToStaticMarkup(<Workbench
      project={{ project: { id: 'p', repo: 'demo', repo_url: 'local://demo', owner: 'local', default_branch: 'main', status: 'ready', primary_language: 'Python', frameworks: [] }, overview: '', stats: {}, start_commands: [], core_files: [], modules: [] }}
      workspaces={[]}
      workspaceId="workspace"
      revision={'a'.repeat(40)}
      question=""
      answers={[answer('<img src=x onerror=alert(1)> [1] is plain answer text', 'first'), answer('another answer', 'second')]}
      error={null}
      loading={false}
      libraryLoading={false}
      projectLoading={false}
      statusMessage=""
      activePage="ask"
      content={null}
      onQuestionChange={vi.fn()}
      onSubmit={vi.fn()}
      onOpenWorkspace={vi.fn()}
      onRefreshLibrary={vi.fn()}
      onNavigate={vi.fn()}
    />);

    expect(markup).toContain('&lt;img src=x onerror=alert(1)&gt;');
    expect(markup).not.toContain('<img');
    expect(markup).toContain('<b>片段 1</b>first');
    expect(markup).toContain('<b>片段 1</b>second');
    expect(markup).toContain('第 1 条回答的引用');
    expect(markup).toContain('第 2 条回答的引用');
  });
});
