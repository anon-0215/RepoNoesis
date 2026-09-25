// @vitest-environment jsdom

import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { renderToStaticMarkup } from 'react-dom/server';
import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('./lib/api', async () => {
  const actual = await vi.importActual<typeof import('./lib/api')>('./lib/api');
  return {
    ...actual,
    analyzeProject: vi.fn(),
    askProjectStream: vi.fn(),
    checkWorkspaceRevision: vi.fn(),
    deleteProject: vi.fn(),
    getConfigStatus: vi.fn(),
    getLearningContinuity: vi.fn(),
    getLearningPath: vi.fn(),
    getProject: vi.fn(),
    getProjectMap: vi.fn(),
    getReport: vi.fn(),
    getWorkspace: vi.fn(),
    getWorkspaceUpdateRun: vi.fn(),
    listWorkspaces: vi.fn(),
    retryLearningContinuity: vi.fn(),
    retryWorkspaceUpdateRun: vi.fn(),
    startWorkspaceRefresh: vi.fn()
  };
});

import App, { AskView } from './App';
import {
  ApiError,
  analyzeProject,
  askProjectStream,
  deleteProject,
  getConfigStatus,
  getLearningPath,
  getProject,
  getProjectMap,
  getReport,
  getWorkspace,
  listWorkspaces
} from './lib/api';
import type { AskFailure, ChatAnswer } from './types';

async function flushPromises() {
  for (let index = 0; index < 12; index += 1) await Promise.resolve();
}

function buttonWithText(root: ParentNode, text: string): HTMLButtonElement {
  const button = Array.from(root.querySelectorAll('button')).find((item) =>
    item.textContent?.includes(text)
  );
  if (!(button instanceof HTMLButtonElement)) throw new Error(`button not found: ${text}`);
  return button;
}

function setInputValue(input: HTMLInputElement | HTMLTextAreaElement, value: string) {
  const prototype = input instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
  const setter = Object.getOwnPropertyDescriptor(prototype, 'value')?.set;
  if (!setter) throw new Error('form control value setter is unavailable');
  setter.call(input, value);
  input.dispatchEvent(new InputEvent('input', { bubbles: true, composed: true, data: value }));
  input.dispatchEvent(new Event('change', { bubbles: true, composed: true }));
}

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  vi.clearAllMocks();
  window.localStorage.clear();
  document.body.replaceChildren();
  window.history.replaceState(null, '', '/');
});

const failure: AskFailure = {
  code: 'tool_timeout',
  message: 'safe projected message',
  retryable: true,
  diagnostics: {
    request_id: 'request-safe-123',
    agent_mode: 'bounded',
    agent_status: 'failed',
    answer_mode: 'not_available',
    failure_stage: 'tool',
    failure_reason_code: 'tool_timeout',
    retrieval_version: 'v1',
    hierarchy_mode: 'off',
    relation_mode: 'off',
    steps_used: 1,
    tool_calls_used: 1,
    planner_logical_calls: 1,
    planner_repair_calls: 0,
    final_answer_attempted: false,
    provider_logical_calls: 1,
    evidence_count: 0,
    citation_count: 0,
    citation_failure_reason_code: null,
    relation_failure_reason_code: null,
    elapsed_ms: 15_001
  }
};

const answer: ChatAnswer = {
  answer: 'existing grounded history',
  citations: [],
  answer_mode: 'llm_grounded',
  grounding_status: 'grounded',
  warnings: []
};

describe('AskView safe server rendering', () => {
  it('renders request_id, keeps existing history, and disables submit while loading', () => {
    const markup = renderToStaticMarkup(
      <AskView
        question="pending question"
        setQuestion={vi.fn()}
        answers={[{ question: 'older question', result: answer }]}
        error={failure}
        onSubmit={vi.fn()}
        loading
      />
    );

    expect(markup).toContain('request-safe-123');
    expect(markup).toContain('tool_timeout');
    expect(markup).toContain('existing grounded history');
    expect(markup).toContain('disabled=""');
    expect(markup).not.toContain('safe projected message');
  });

  it('renders Citation cards only from the server citations array', () => {
    const ordinaryText =
      '普通正文 [E999] backend/app/b.py:30-40 只是说明，不是服务器 Citation。';
    const serverCitation = {
      path: 'backend/app/a.py',
      summary: 'authenticate_user',
      snippet: 'def authenticate_user():',
      qualified_name: 'authenticate_user',
      start_line: 10,
      end_line: 20
    };
    const markup = renderToStaticMarkup(
      <AskView
        question="where"
        setQuestion={vi.fn()}
        answers={[
          {
            question: 'where',
            result: { ...answer, answer: ordinaryText, citations: [serverCitation] }
          }
        ]}
        error={null}
        onSubmit={vi.fn()}
        loading={false}
      />
    );
    const rendered = new DOMParser().parseFromString(markup, 'text/html');
    const answerText = rendered.querySelector('.answer > p:not(.citation-trust-note)');
    const citationCards = rendered.querySelectorAll('.citation-grid details');
    const citationSummary = rendered.querySelector('.citation-grid details summary');
    const citationSnippet = rendered.querySelector('.citation-grid details pre');

    expect(rendered.querySelector('.citation-trust-note')?.textContent).toContain(
      '源码引用已校验'
    );
    expect(answerText?.textContent).toContain('[E999] backend/app/b.py:30-40');
    expect(citationCards).toHaveLength(1);
    expect(citationSummary?.textContent).toBe(
      'backend/app/a.py:10-20 · authenticate_user'
    );
    expect(citationSnippet?.textContent).toBe('def authenticate_user():');
    expect(citationSummary?.textContent).not.toContain('backend/app/b.py');
  });

  it('keeps repeated Citation paths and ranges stable, ordered, and warning-free', () => {
    const citations = [
      {
        path: 'backend/app/shared.py',
        summary: 'first_range',
        snippet: 'def first_range(): pass',
        qualified_name: 'first_range',
        start_line: 10,
        end_line: 20
      },
      {
        path: 'backend/app/shared.py',
        summary: 'second_range',
        snippet: 'def second_range(): pass',
        qualified_name: 'second_range',
        start_line: 30,
        end_line: 40
      },
      {
        path: 'backend/app/shared.py',
        summary: 'second_range_duplicate_item',
        snippet: 'def second_range_duplicate_item(): pass',
        qualified_name: 'second_range',
        start_line: 30,
        end_line: 40
      },
      {
        path: 'backend/app/other.py',
        summary: 'other_path',
        snippet: 'def other_path(): pass',
        qualified_name: 'other_path',
        start_line: 5,
        end_line: 9
      }
    ];
    const ordinaryText =
      '普通正文 backend/app/shared.py:999-1000 不应生成额外 Citation 卡片。';
    const originalConsoleError = console.error;
    const consoleError = vi.spyOn(console, 'error').mockImplementation((...args) => {
      if (!String(args[0]).includes('same key')) originalConsoleError(...args);
    });
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
    const container = document.createElement('div');
    document.body.appendChild(container);
    const root = createRoot(container);
    let duplicateKeyErrors: unknown[][] = [];
    let allConsoleErrors: unknown[][] = [];

    try {
      act(() => {
        root.render(
          <AskView
            question="where"
            setQuestion={vi.fn()}
            answers={[
              {
                question: 'where',
                result: { ...answer, answer: ordinaryText, citations }
              }
            ]}
            error={null}
            onSubmit={vi.fn()}
            loading={false}
          />
        );
      });
      allConsoleErrors = [...consoleError.mock.calls];
      duplicateKeyErrors = allConsoleErrors.filter((args) =>
        String(args[0]).includes('same key')
      );
      const answerText = container.querySelector(
        '.answer > p:not(.citation-trust-note)'
      );
      const citationCards = Array.from(
        container.querySelectorAll('.citation-grid details')
      );

      expect(citationCards).toHaveLength(citations.length);
      expect(
        citationCards.map((card) => card.querySelector('summary')?.textContent)
      ).toEqual([
        'backend/app/shared.py:10-20 · first_range',
        'backend/app/shared.py:30-40 · second_range',
        'backend/app/shared.py:30-40 · second_range',
        'backend/app/other.py:5-9 · other_path'
      ]);
      expect(citationCards.map((card) => card.querySelector('pre')?.textContent)).toEqual(
        citations.map((citation) => citation.snippet)
      );
      expect(container.querySelector('.citation-trust-note')?.textContent).toContain(
        '源码引用已校验'
      );
      expect(answerText?.textContent).toContain('backend/app/shared.py:999-1000');
      expect(answerText?.querySelectorAll('a, details')).toHaveLength(0);
      expect(duplicateKeyErrors).toHaveLength(0);
      expect(allConsoleErrors).toHaveLength(0);
    } finally {
      act(() => root.unmount());
      consoleError.mockRestore();
    }
  });

  it.each([
    ['provider_not_configured', 'provider', 'request-provider-1'],
    ['response_contract_invalid', 'response', 'request-response-1']
  ])('renders the safe %s diagnostic card after loading is released', (code, stage, requestId) => {
    const projected: AskFailure = {
      ...failure,
      code,
      message: 'PRIVATE-SERVER-MESSAGE-MUST-NOT-RENDER',
      retryable: false,
      diagnostics: {
        ...failure.diagnostics,
        request_id: requestId,
        failure_stage: stage,
        failure_reason_code: code,
        evidence_count: 2,
        citation_count: 1
      }
    };

    const markup = renderToStaticMarkup(
      <AskView
        question="retry question"
        setQuestion={vi.fn()}
        answers={[]}
        error={projected}
        onSubmit={vi.fn()}
        loading={false}
      />
    );

    expect(markup).toContain(code);
    expect(markup).toContain(stage);
    expect(markup).toContain(requestId);
    expect(markup).toContain('Evidence 数量：2');
    expect(markup).not.toContain('disabled=""');
    expect(markup).not.toContain('PRIVATE-SERVER-MESSAGE-MUST-NOT-RENDER');
    expect(markup).not.toContain('candidate_answer');
    expect(markup).not.toContain('ValidationError');
  });
});

describe('App ask request gate integration', () => {
  it('uses the shared workbench by default, preserves the draft across pages, and blocks a pending duplicate', async () => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
    const originalUrl = `${window.location.pathname}${window.location.search}${window.location.hash}`;
    const workspaceId = '11111111-1111-4111-8111-111111111111';
    const projectId = 'project-app-integration';
    window.history.replaceState(null, '', `/?workspace=${workspaceId}`);
    window.localStorage.clear();

    vi.mocked(getConfigStatus).mockResolvedValue({
      llm: { ready: true, provider: 'fake', model: 'fake-model', missing: [] },
      embedding: { ready: true, model: 'fake-embedding', device: 'cpu', offline: true, missing: [] }
    } as never);
    vi.mocked(listWorkspaces).mockResolvedValue({
      items: [{
        workspace_id: workspaceId,
        display_name: 'fixture',
        source_type: 'local',
        project_status: 'ready',
        repository_revision: 'fixture-revision',
        openable: true,
        created_at: '2026-08-11T00:00:00Z',
        updated_at: '2026-08-11T00:00:00Z'
      }],
      total: 1,
      limit: 20,
      offset: 0
    } as never);
    vi.mocked(getWorkspace).mockResolvedValue({
      workspace_id: workspaceId,
      display_name: 'fixture',
      source_type: 'local',
      project_status: 'ready',
      repository_revision: 'fixture-revision',
      openable: true,
      created_at: '2026-08-11T00:00:00Z',
      updated_at: '2026-08-11T00:00:00Z',
      active_snapshot: { project_id: projectId, repository_revision: 'fixture-revision' },
      latest_update_run: null,
      learning_continuity: null
    } as never);
    vi.mocked(getProject).mockResolvedValue({
      project: {
        id: projectId,
        repo_url: 'local://fixture',
        owner: 'local',
        repo: 'fixture',
        default_branch: 'main',
        status: 'ready',
        primary_language: 'Python',
        frameworks: []
      },
      overview: 'fixture overview',
      stats: {},
      start_commands: [],
      core_files: [],
      modules: []
    });
    vi.mocked(getProjectMap).mockResolvedValue({
      tree: { name: 'fixture', path: '', type: 'directory', children: [] },
      modules: [],
      dependency_edges: [],
      core_files: []
    });
    vi.mocked(getLearningPath).mockResolvedValue({ steps: [] });
    vi.mocked(getReport).mockResolvedValue({ markdown: '' });

    let rejectFirst!: (reason: unknown) => void;
    const firstRequest = new Promise<ChatAnswer>((_resolve, reject) => {
      rejectFirst = reject;
    });
    vi.mocked(askProjectStream)
      .mockImplementationOnce(() => firstRequest)
      .mockResolvedValueOnce({
        answer: 'second grounded answer',
        citations: [],
        answer_mode: 'llm_grounded',
        grounding_status: 'grounded',
        warnings: []
      });

    const consoleError = vi.spyOn(console, 'error');
    const container = document.createElement('div');
    document.body.appendChild(container);
    let root: Root | null = createRoot(container);
    try {
      await act(async () => {
        root!.render(<App />);
        await flushPromises();
      });

      expect(getWorkspace).toHaveBeenCalledWith(workspaceId);
      expect(container.querySelectorAll('.wb-rail')).toHaveLength(1);
      expect(container.querySelectorAll('.wb-context')).toHaveLength(1);
      expect(container.querySelector('form.ask-form')).not.toBeNull();
      const theme = container.querySelector<HTMLButtonElement>('.wb-theme')!;
      await act(async () => {
        theme.click();
        await Promise.resolve();
      });
      expect(document.documentElement.dataset.theme).toBe('dark');

      let form = container.querySelector<HTMLFormElement>('form.ask-form');
      let input = form?.querySelector<HTMLTextAreaElement>('textarea');
      let submit = form?.querySelector<HTMLButtonElement>('button[type="submit"]');
      expect(form).not.toBeNull();
      expect(input).not.toBeNull();
      expect(submit).not.toBeNull();
      await act(async () => {
        buttonWithText(container, '这个仓库的主要入口在哪里？').click();
        await flushPromises();
      });
      expect(input?.value).toBe('这个仓库的主要入口在哪里？');
      expect(askProjectStream).not.toHaveBeenCalled();

      await act(async () => {
        setInputValue(input!, 'first question');
        await Promise.resolve();
      });
      await act(async () => { buttonWithText(container, '项目概览').click(); await flushPromises(); });
      expect(container.querySelectorAll('.wb-rail')).toHaveLength(1);
      expect(container.querySelectorAll('.wb-context')).toHaveLength(1);
      expect(document.documentElement.dataset.theme).toBe('dark');
      await act(async () => { buttonWithText(container, '源码问答').click(); await flushPromises(); });
      input = container.querySelector<HTMLTextAreaElement>('form.ask-form textarea');
      submit = container.querySelector<HTMLButtonElement>('form.ask-form button[type="submit"]');
      expect(input?.value).toBe('first question');
      await act(async () => {
        submit!.click();
        await Promise.resolve();
      });
      expect(askProjectStream).toHaveBeenCalledTimes(1);

      form = container.querySelector<HTMLFormElement>('form.ask-form');
      submit = form?.querySelector<HTMLButtonElement>('button[type="submit"]');
      expect(submit?.disabled).toBe(true);
      await act(async () => {
        form!.dispatchEvent(new SubmitEvent('submit', { bubbles: true, cancelable: true }));
        await Promise.resolve();
      });
      expect(askProjectStream).toHaveBeenCalledTimes(1);

      await act(async () => {
        rejectFirst(new ApiError('safe failure', 500, failure));
        await flushPromises();
      });
      const errorCard = container.querySelector<HTMLElement>('[role="alert"]');
      form = container.querySelector<HTMLFormElement>('form.ask-form');
      input = form?.querySelector<HTMLTextAreaElement>('textarea');
      submit = form?.querySelector<HTMLButtonElement>('button[type="submit"]');
      expect(errorCard?.textContent).toContain('tool_timeout');
      expect(submit?.disabled).toBe(false);
      expect(container.querySelector('.wb-notification')).toBeNull();
      await act(async () => { buttonWithText(container, '项目概览').click(); await flushPromises(); });
      await act(async () => { buttonWithText(container, '源码问答').click(); await flushPromises(); });
      expect(container.querySelector('.wb-error')?.textContent).toContain('request-safe-123');
      form = container.querySelector<HTMLFormElement>('form.ask-form');
      input = form?.querySelector<HTMLTextAreaElement>('textarea');

      await act(async () => {
        setInputValue(input!, 'second question');
        await Promise.resolve();
      });
      await act(async () => {
        form!.requestSubmit();
        await flushPromises();
      });

      expect(askProjectStream).toHaveBeenCalledTimes(2);
      expect(vi.mocked(askProjectStream).mock.calls[1].slice(0, 4)).toEqual([projectId, 'fixture-revision', 'second question', 'agent']);
      expect(container.textContent).toContain('second grounded answer');
      expect(Array.from(container.querySelectorAll('.wb-answer')).map((card) =>
        card.querySelector('.wb-question p')?.textContent
      )).toEqual(['first question', 'second question']);
      expect(consoleError).not.toHaveBeenCalled();
    } finally {
      if (root) {
        await act(async () => root?.unmount());
        root = null;
      }
      container.remove();
      window.localStorage.clear();
      window.history.replaceState(null, '', originalUrl || '/');
      consoleError.mockRestore();
      vi.unstubAllGlobals();
    }
  });
});

describe('App connection and import status integration', () => {
  it('keeps the shared shell for an empty library and exposes import through project management', async () => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
    vi.mocked(getConfigStatus).mockResolvedValue({ embedding: { ready: true }, llm: { ready: true } } as never);
    vi.mocked(listWorkspaces).mockResolvedValue({ items: [], total: 0, limit: 20, offset: 0 } as never);
    const container = document.createElement('div'); document.body.appendChild(container);
    let root: Root | null = createRoot(container);
    try {
      await act(async () => { root!.render(<App />); await flushPromises(); });
      expect(container.querySelectorAll('.wb-rail')).toHaveLength(1);
      expect(container.querySelectorAll('.wb-context')).toHaveLength(1);
      expect(container.textContent).toContain('选择一个已索引项目');
      expect(container.querySelector<HTMLButtonElement>('form.ask-form button[type="submit"]')?.disabled).toBe(true);
      await act(async () => { buttonWithText(container, '项目管理').click(); await flushPromises(); });
      expect(container.querySelector('form.analyze-form')).not.toBeNull();
    } finally {
      if (root) await act(async () => root?.unmount());
      container.remove();
    }
  });

  it('shows incomplete embedding progress, confirms deletion, and removes the item', async () => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
    vi.mocked(getConfigStatus).mockResolvedValue({
      git_proxy_configured: false,
      llm: { ready: true, provider: 'fake', model: 'fake-model', missing: [] },
      embedding: { ready: true, model: 'fake-embedding', device: 'cpu', offline: true, missing: [] }
    } as never);
    vi.mocked(listWorkspaces).mockResolvedValue({
      items: [{
        workspace_id: 'workspace-incomplete', display_name: 'RepoNoesis', source_type: 'git_url',
        project_status: 'incomplete', repository_revision: 'c'.repeat(40), openable: false,
        project_id: 'project-incomplete', total_chunks: 2850, embedding_count: 24,
        created_at: 'now', updated_at: 'now'
      }], total: 1, limit: 20, offset: 0
    });
    vi.mocked(deleteProject).mockResolvedValue({ deleted: true, cleanup_pending: false, retryable: false });
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    const container = document.createElement('div');
    document.body.appendChild(container);
    let root: Root | null = createRoot(container);
    try {
      await act(async () => { root!.render(<App />); await flushPromises(); });
      await act(async () => { buttonWithText(container, '项目管理').click(); await flushPromises(); });
      expect(container.textContent).toContain('incomplete');
      expect(container.textContent).toContain('向量记录：24 / 2850');
      await act(async () => {
        container.querySelector<HTMLButtonElement>('[aria-label="删除 RepoNoesis"]')!.click();
        await flushPromises();
      });
      expect(window.confirm).toHaveBeenCalledWith(expect.stringContaining('不会修改远程 Git 仓库'));
      expect(deleteProject).toHaveBeenCalledWith('project-incomplete');
      expect(container.querySelector('[aria-label="删除 RepoNoesis"]')).toBeNull();
    } finally {
      if (root) await act(async () => root?.unmount());
      container.remove();
    }
  });

  it('keeps a project visible after delete failure so deletion can be retried', async () => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
    vi.mocked(getConfigStatus).mockResolvedValue({ embedding: { ready: true }, llm: { ready: true } } as never);
    vi.mocked(listWorkspaces).mockResolvedValue({
      items: [{ workspace_id: 'w', display_name: 'Retry Repo', source_type: 'local', project_status: 'failed',
        repository_revision: 'a'.repeat(40), openable: false, project_id: 'p', total_chunks: 1,
        embedding_count: 0, created_at: 'now', updated_at: 'now' }], total: 1, limit: 20, offset: 0
    });
    vi.mocked(deleteProject).mockRejectedValueOnce(new ApiError('项目删除失败，请稍后重试。', 500))
      .mockResolvedValueOnce({ deleted: true, cleanup_pending: false, retryable: false });
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    const container = document.createElement('div'); document.body.appendChild(container);
    let root: Root | null = createRoot(container);
    try {
      await act(async () => { root!.render(<App />); await flushPromises(); });
      await act(async () => { buttonWithText(container, '项目管理').click(); await flushPromises(); });
      const selector = '[aria-label="删除 Retry Repo"]';
      await act(async () => { container.querySelector<HTMLButtonElement>(selector)!.click(); await flushPromises(); });
      expect(container.textContent).toContain('Retry Repo');
      await act(async () => { container.querySelector<HTMLButtonElement>(selector)!.click(); await flushPromises(); });
      expect(deleteProject).toHaveBeenCalledTimes(2);
      expect(container.textContent).not.toContain('Retry Repo');
    } finally {
      if (root) await act(async () => root?.unmount());
      container.remove();
    }
  });

  it('clears an older initialization connection error after a newer successful response', async () => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
    let rejectConfig!: (reason: unknown) => void;
    let resolveLibrary!: (value: never) => void;
    vi.mocked(getConfigStatus).mockImplementation(
      () => new Promise((_resolve, reject) => { rejectConfig = reject; })
    );
    vi.mocked(listWorkspaces).mockImplementation(
      () => new Promise((resolve) => { resolveLibrary = resolve; })
    );

    const container = document.createElement('div');
    document.body.appendChild(container);
    let root: Root | null = createRoot(container);
    try {
      await act(async () => {
        root!.render(<App />);
        await flushPromises();
      });
      await act(async () => { buttonWithText(container, '项目管理').click(); await flushPromises(); });
      await act(async () => {
        rejectConfig(new ApiError('无法连接后端服务。请确认后端已启动后重试。', 0));
        await flushPromises();
      });
      expect(container.querySelector('.status-line')?.textContent).toContain('无法连接后端服务');

      await act(async () => {
        resolveLibrary({ items: [], total: 0, limit: 20, offset: 0 } as never);
        await flushPromises();
      });
      expect(container.querySelector('.status-line')?.textContent || '').not.toContain('无法连接后端服务');
    } finally {
      if (root) await act(async () => root?.unmount());
      container.remove();
    }
  });

  it.each([
    [502, '公开 Git 仓库克隆失败。 可以重试此操作。 请求 ID：request-import-1'],
    [504, '公开 Git 仓库克隆超时。 可以重试此操作。 导入失败；部分临时文件将在稍后清理。 请求 ID：request-import-1']
  ])('keeps an HTTP %s import error online across library success and releases retry', async (status, safeMessage) => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
    vi.mocked(getConfigStatus).mockResolvedValue({
      llm: { ready: true, provider: 'fake', model: 'fake-model', missing: [] },
      embedding: { ready: true, model: 'fake-embedding', device: 'cpu', offline: true, missing: [] }
    } as never);
    vi.mocked(listWorkspaces).mockResolvedValue({ items: [], total: 0, limit: 20, offset: 0 } as never);
    vi.mocked(analyzeProject).mockRejectedValue(
      new ApiError(safeMessage, status)
    );

    const container = document.createElement('div');
    document.body.appendChild(container);
    let root: Root | null = createRoot(container);
    try {
      await act(async () => {
        root!.render(<App />);
        await flushPromises();
      });
      await act(async () => { buttonWithText(container, '项目管理').click(); await flushPromises(); });
      await act(async () => {
        buttonWithText(container, '公开 HTTPS Git').click();
        await Promise.resolve();
      });
      const input = container.querySelector<HTMLInputElement>('#repo-source')!;
      await act(async () => {
        setInputValue(input, 'https://public.example/repository.git');
        await Promise.resolve();
      });
      const form = container.querySelector<HTMLFormElement>('form.analyze-form')!;
      await act(async () => {
        form.requestSubmit();
        await flushPromises();
      });
      expect(container.querySelector('.status-line')?.textContent).toContain('request-import-1');
      expect(container.querySelector('.status-line')?.textContent || '').not.toContain('无法连接后端服务');
      if (status === 504) {
        expect(container.querySelector('.status-line')?.textContent).toContain('部分临时文件将在稍后清理');
      }
      expect(form.querySelector<HTMLButtonElement>('button[type="submit"]')?.disabled).toBe(false);

      await act(async () => {
        buttonWithText(container, '刷新列表').click();
        await flushPromises();
      });
      expect(container.querySelector('.status-line')?.textContent).toContain('request-import-1');

      await act(async () => {
        form.requestSubmit();
        await flushPromises();
      });
      expect(analyzeProject).toHaveBeenCalledTimes(2);
    } finally {
      if (root) await act(async () => root?.unmount());
      container.remove();
    }
  });

  it('clears an import network error after the project library proves recovery', async () => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
    vi.mocked(getConfigStatus).mockResolvedValue({
      git_proxy_configured: false,
      llm: { ready: true, provider: 'fake', model: 'fake-model', missing: [] },
      embedding: { ready: true, model: 'fake-embedding', device: 'cpu', offline: true, missing: [] }
    } as never);
    vi.mocked(listWorkspaces).mockResolvedValue({ items: [], total: 0, limit: 20, offset: 0 } as never);
    vi.mocked(analyzeProject).mockRejectedValue(
      new ApiError('无法连接后端服务。请确认后端已启动后重试。', 0)
    );

    const container = document.createElement('div');
    document.body.appendChild(container);
    let root: Root | null = createRoot(container);
    try {
      await act(async () => {
        root!.render(<App />);
        await flushPromises();
      });
      await act(async () => { buttonWithText(container, '项目管理').click(); await flushPromises(); });
      await act(async () => {
        buttonWithText(container, '公开 HTTPS Git').click();
        await Promise.resolve();
      });
      const input = container.querySelector<HTMLInputElement>('#repo-source')!;
      await act(async () => {
        setInputValue(input, 'https://public.example/repository.git');
        await Promise.resolve();
      });
      const form = container.querySelector<HTMLFormElement>('form.analyze-form')!;
      await act(async () => {
        form.requestSubmit();
        await flushPromises();
      });
      expect(container.querySelector('.status-line')?.textContent).toContain('无法连接后端服务');
      expect(form.querySelector<HTMLButtonElement>('button[type="submit"]')?.disabled).toBe(false);

      await act(async () => {
        buttonWithText(container, '刷新列表').click();
        await flushPromises();
      });
      expect(container.querySelector('.status-line')?.textContent || '').not.toContain('无法连接后端服务');
    } finally {
      if (root) await act(async () => root?.unmount());
      container.remove();
    }
  });
});
