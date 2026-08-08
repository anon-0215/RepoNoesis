import type { Citation, ConfigStatus, LearningContinuity, WorkspaceRevisionCheck, WorkspaceUpdateRun } from '../types';

export type SourceType = 'local' | 'git_url';

export const RECENT_WORKSPACE_KEY = 'reponoesis.recentWorkspaceId';
export type WorkspaceLibraryStatus = 'loading' | 'error' | 'empty' | 'ready';
export type WorkspaceUpdateUiState = 'ready' | 'unchanged' | 'update-available' | 'updating' | 'failed';
export type LearningContinuityUiState = 'not-required' | 'pending' | 'running' | 'succeeded' | 'failed';

const WORKSPACE_ID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export function buildAnalyzePayload(sourceType: SourceType, source: string) {
  return { source_type: sourceType, source: source.trim() };
}

export function providerSummary(status: ConfigStatus | null): string {
  if (!status) return '正在读取后端配置状态…';
  if (!status.llm.ready) return `生成模型未配置：${status.llm.missing.join(', ')}`;
  return `${status.llm.provider} / ${status.llm.model}`;
}

export function citationLabel(citation: Citation): string {
  const symbol = citation.qualified_name ? ` · ${citation.qualified_name}` : '';
  return `${citation.path}:${citation.start_line}-${citation.end_line}${symbol}`;
}

export function normalizeWorkspaceId(value: string | null | undefined): string | null {
  const candidate = (value ?? '').trim().toLowerCase();
  return WORKSPACE_ID_PATTERN.test(candidate) ? candidate : null;
}

export function selectWorkspaceToRestore(search: string, recentValue: string | null) {
  const params = new URLSearchParams(search);
  if (params.has('workspace')) {
    const workspaceId = normalizeWorkspaceId(params.get('workspace'));
    return workspaceId
      ? { workspaceId, source: 'url' as const }
      : { workspaceId: null, source: 'invalid_url' as const };
  }
  const workspaceId = normalizeWorkspaceId(recentValue);
  return workspaceId
    ? { workspaceId, source: 'recent' as const }
    : { workspaceId: null, source: 'none' as const };
}

export function nextWorkspaceSearch(search: string, workspaceId: string | null): string {
  const params = new URLSearchParams(search);
  params.delete('project');
  if (workspaceId) params.set('workspace', workspaceId);
  else params.delete('workspace');
  const rendered = params.toString();
  return rendered ? `?${rendered}` : '';
}

export function workspaceLibraryStatus(
  event: 'loading' | 'error' | 'success',
  itemCount = 0
): WorkspaceLibraryStatus {
  if (event === 'loading') return 'loading';
  if (event === 'error') return 'error';
  return itemCount > 0 ? 'ready' : 'empty';
}

export function workspaceUpdateState(
  check: WorkspaceRevisionCheck | null,
  run: WorkspaceUpdateRun | null
): WorkspaceUpdateUiState {
  if (run?.status === 'pending' || run?.status === 'running') return 'updating';
  if (run?.status === 'failed') return 'failed';
  if (run?.status === 'succeeded' && run.result === 'unchanged') return 'unchanged';
  if (run?.status === 'succeeded' && run.result === 'activated') return 'ready';
  if (check?.state === 'update_available') return 'update-available';
  if (check?.state === 'unchanged') return 'unchanged';
  return 'ready';
}

export function learningContinuityState(
  continuity: LearningContinuity | null
): LearningContinuityUiState {
  if (!continuity || continuity.status === 'not_required') return 'not-required';
  return continuity.status;
}
