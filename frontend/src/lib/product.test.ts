import { describe, expect, it } from 'vitest';

import {
  buildAnalyzePayload,
  citationLabel,
  learningContinuityState,
  nextWorkspaceSearch,
  normalizeWorkspaceId,
  providerSummary,
  selectWorkspaceToRestore,
  workspaceLibraryStatus,
  workspaceUpdateState
} from './product';

describe('local product presentation', () => {
  it('switches between the two explicit source payloads', () => {
    expect(buildAnalyzePayload('local', ' D:\\code\\repo ')).toEqual({
      source_type: 'local',
      source: 'D:\\code\\repo'
    });
    expect(buildAnalyzePayload('git_url', ' https://example.test/repo.git ')).toEqual({
      source_type: 'git_url',
      source: 'https://example.test/repo.git'
    });
  });

  it('shows missing provider settings without credentials', () => {
    const text = providerSummary({
      llm: {
        provider: null,
        model: null,
        base_url_configured: false,
        api_key_configured: false,
        ready: false,
        missing: ['LLM_API_KEY', 'LLM_MODEL']
      },
      embedding: {
        provider: 'local_bge_m3',
        model: 'BAAI/bge-m3',
        device: 'cpu',
        offline: true,
        enabled: true,
        ready: true,
        missing: []
      }
    });
    expect(text).toBe('生成模型未配置：LLM_API_KEY, LLM_MODEL');
  });

  it('renders file symbol and exact line range for evidence', () => {
    expect(
      citationLabel({
        path: 'src/app.py',
        summary: 'answer',
        snippet: 'def answer(): pass',
        qualified_name: 'app.answer',
        start_line: 10,
        end_line: 11
      })
    ).toBe('src/app.py:10-11 · app.answer');
  });

  it('restores a stable workspace id from the URL before local recent state', () => {
    const urlId = '11111111-1111-4111-8111-111111111111';
    const recentId = '22222222-2222-4222-8222-222222222222';
    expect(selectWorkspaceToRestore(`?workspace=${urlId}`, recentId)).toEqual({
      workspaceId: urlId,
      source: 'url'
    });
    expect(selectWorkspaceToRestore('', recentId)).toEqual({
      workspaceId: recentId,
      source: 'recent'
    });
  });

  it('rejects malformed URL and stale local values without treating them as ids', () => {
    expect(normalizeWorkspaceId('not-a-workspace')).toBeNull();
    expect(selectWorkspaceToRestore('?workspace=not-a-workspace', 'also-bad')).toEqual({
      workspaceId: null,
      source: 'invalid_url'
    });
  });

  it('writes only the stable workspace id into the URL query', () => {
    const id = '33333333-3333-4333-8333-333333333333';
    expect(nextWorkspaceSearch('?tab=map&project=secret', id)).toBe(`?tab=map&workspace=${id}`);
  });

  it('keeps project-library loading, error, empty and ready states distinct', () => {
    expect(workspaceLibraryStatus('loading')).toBe('loading');
    expect(workspaceLibraryStatus('error')).toBe('error');
    expect(workspaceLibraryStatus('success', 0)).toBe('empty');
    expect(workspaceLibraryStatus('success', 2)).toBe('ready');
  });

  it('keeps explicit workspace update states distinct', () => {
    const check = {
      workspace_id: 'workspace',
      current_revision: 'a'.repeat(40),
      available_revision: 'b'.repeat(40),
      state: 'update_available' as const
    };
    const run = {
      run_id: 'run', workspace_id: 'workspace', target_revision: 'b'.repeat(40),
      status: 'running' as const, phase: 'chunk_update', result: '' as const,
      stats: {}, error_code: '', error_message: '', retryable: false, retry_count: 0,
      active_project_id: null, created_at: '', started_at: '', finished_at: null, updated_at: ''
    };
    expect(workspaceUpdateState(null, null)).toBe('ready');
    expect(workspaceUpdateState({ ...check, state: 'unchanged' }, null)).toBe('unchanged');
    expect(workspaceUpdateState(check, null)).toBe('update-available');
    expect(workspaceUpdateState(check, run)).toBe('updating');
    expect(workspaceUpdateState(check, { ...run, status: 'failed' })).toBe('failed');
    expect(workspaceUpdateState(check, { ...run, status: 'succeeded', result: 'unchanged' })).toBe('unchanged');
    expect(workspaceUpdateState(check, { ...run, status: 'succeeded', result: 'activated' })).toBe('ready');
  });

  it('keeps persisted learning continuity states distinct', () => {
    const continuity = {
      transition_id: 'transition', workspace_id: 'workspace', activation_version: 2,
      status: 'pending' as const, stats: {
        total: 0, unchanged_exact: 0, renamed_exact: 0, modified: 0, deleted: 0,
        ambiguous: 0, unmapped: 0, incompatible: 0, retained: 0, needs_review: 0,
        history_only: 0, not_inherited: 0
      }, error_code: '', error_message: '', retryable: false, retry_count: 0,
      created_at: '', started_at: null, finished_at: null, updated_at: ''
    };
    expect(learningContinuityState(null)).toBe('not-required');
    expect(learningContinuityState(continuity)).toBe('pending');
    expect(learningContinuityState({ ...continuity, status: 'running' })).toBe('running');
    expect(learningContinuityState({ ...continuity, status: 'succeeded' })).toBe('succeeded');
    expect(learningContinuityState({ ...continuity, status: 'failed' })).toBe('failed');
  });
});
