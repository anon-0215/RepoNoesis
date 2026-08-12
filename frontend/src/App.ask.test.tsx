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
    askProject: vi.fn(),
    checkWorkspaceRevision: vi.fn(),
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
  askProject,
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

function setInputValue(input: HTMLInputElement, value: string) {
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
  if (!setter) throw new Error('HTMLInputElement value setter is unavailable');
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
  it('blocks a pending duplicate, releases after failure, and submits again through the DOM', async () => {
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
    vi.mocked(askProject)
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
      const askTab = buttonWithText(container, '源码问答');
      await act(async () => {
        askTab.click();
        await Promise.resolve();
      });

      let form = container.querySelector<HTMLFormElement>('form.ask-form');
      let input = form?.querySelector<HTMLInputElement>('input');
      let submit = form?.querySelector<HTMLButtonElement>('button[type="submit"]');
      expect(form).not.toBeNull();
      expect(input).not.toBeNull();
      expect(submit).not.toBeNull();

      await act(async () => {
        setInputValue(input!, 'first question');
        await Promise.resolve();
      });
      await act(async () => {
        submit!.click();
        await Promise.resolve();
      });
      expect(askProject).toHaveBeenCalledTimes(1);

      form = container.querySelector<HTMLFormElement>('form.ask-form');
      submit = form?.querySelector<HTMLButtonElement>('button[type="submit"]');
      expect(submit?.disabled).toBe(true);
      await act(async () => {
        form!.dispatchEvent(new SubmitEvent('submit', { bubbles: true, cancelable: true }));
        await Promise.resolve();
      });
      expect(askProject).toHaveBeenCalledTimes(1);

      await act(async () => {
        rejectFirst(new ApiError('safe failure', 500, failure));
        await flushPromises();
      });
      const errorCard = container.querySelector<HTMLElement>('[role="alert"]');
      form = container.querySelector<HTMLFormElement>('form.ask-form');
      input = form?.querySelector<HTMLInputElement>('input');
      submit = form?.querySelector<HTMLButtonElement>('button[type="submit"]');
      expect(errorCard?.textContent).toContain('tool_timeout');
      expect(submit?.disabled).toBe(false);

      await act(async () => {
        setInputValue(input!, 'second question');
        await Promise.resolve();
      });
      await act(async () => {
        form!.requestSubmit();
        await flushPromises();
      });

      expect(askProject).toHaveBeenCalledTimes(2);
      expect(vi.mocked(askProject).mock.calls[1]).toEqual([projectId, 'second question']);
      expect(container.textContent).toContain('second grounded answer');
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
      await act(async () => {
        rejectConfig(new ApiError('无法连接后端服务。请确认后端已启动后重试。', 0));
        await flushPromises();
      });
      expect(container.querySelector('.status-line')?.textContent).toContain('无法连接后端服务');

      await act(async () => {
        resolveLibrary({ items: [], total: 0, limit: 20, offset: 0 } as never);
        await flushPromises();
      });
      expect(container.querySelector('.status-line')?.textContent).not.toContain('无法连接后端服务');
    } finally {
      if (root) await act(async () => root?.unmount());
      container.remove();
    }
  });

  it('keeps an import error across an unrelated library success and releases submit for retry', async () => {
    vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
    vi.mocked(getConfigStatus).mockResolvedValue({
      llm: { ready: true, provider: 'fake', model: 'fake-model', missing: [] },
      embedding: { ready: true, model: 'fake-embedding', device: 'cpu', offline: true, missing: [] }
    } as never);
    vi.mocked(listWorkspaces).mockResolvedValue({ items: [], total: 0, limit: 20, offset: 0 } as never);
    vi.mocked(analyzeProject).mockRejectedValue(
      new ApiError('公开 Git 仓库克隆失败。 可以重试此操作。 请求 ID：request-import-1', 502)
    );

    const container = document.createElement('div');
    document.body.appendChild(container);
    let root: Root | null = createRoot(container);
    try {
      await act(async () => {
        root!.render(<App />);
        await flushPromises();
      });
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
});
