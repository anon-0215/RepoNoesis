import type {
  AskFailure,
  ChatAnswer,
  ConfigStatus,
  LearningStep,
  LearningContinuity,
  ProjectMap,
  ProjectResponse,
  WorkspaceDetail,
  WorkspaceListResponse,
  WorkspaceRevisionCheck,
  WorkspaceUpdateRun
} from '../types';
import { buildAnalyzePayload, type SourceType } from './product';

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000';
const SAFE_NETWORK_ERROR_MESSAGE = '无法连接后端服务。请确认后端已启动后重试。';
const SAFE_STRUCTURED_ASK_MESSAGE = '问答未生成可验证答案，请查看安全诊断卡片。';
const MAX_ASK_COUNTER = 1_000_000;
const MAX_ASK_ELAPSED_MS = 86_400_000;
const ASK_PROVIDER_FAILURE_REASONS = new Set([
  'unsupported_provider',
  'provider_not_configured',
  'provider_authentication_failed',
  'provider_rate_limited',
  'provider_unavailable',
  'provider_output_truncated',
  'provider_empty_content',
  'provider_invalid_response',
  'provider_request_rejected',
  'provider_error'
]);
const ASK_FAILURE_REASONS = new Set([
  'planner_invalid',
  'planner_repair_failed',
  'planner_budget_exhausted',
  'deadline_exceeded',
  'tool_timeout',
  'evidence_insufficient',
  'final_answer_not_attempted',
  'citation_missing',
  'citation_format_invalid',
  'citation_unknown',
  'citation_location_missing',
  'citation_path_mismatch',
  'citation_line_range_mismatch',
  'citation_evidence_binding_failed',
  'relation_validation_failed',
  'response_contract_invalid',
  ...ASK_PROVIDER_FAILURE_REASONS,
  'persistence_failed'
]);
const ASK_AGENT_MODES = new Set(['bounded', 'deterministic_fallback', 'unknown']);
const ASK_AGENT_STATUSES = new Set([
  'completed',
  'degraded',
  'insufficient_evidence',
  'tool_budget_exhausted',
  'final_answer_failed',
  'budget_exhausted',
  'cancelled',
  'failed',
  'unknown'
]);
const ASK_ANSWER_MODES = new Set(['llm_grounded', 'deterministic', 'not_available']);
const ASK_FAILURE_STAGES = new Set([
  'planner',
  'budget',
  'deadline',
  'retrieval',
  'final_answer',
  'citation_validation',
  'relation_validation',
  'response',
  'provider',
  'tool',
  'persistence'
]);
const ASK_RETRIEVAL_VERSIONS = new Set(['v1', 'v2']);
const ASK_HIERARCHY_MODES = new Set(['off', 'normalize_v1']);
const ASK_RELATION_MODES = new Set(['off', 'expand_v1']);
const ASK_CITATION_FAILURE_REASONS = new Set(
  [...ASK_FAILURE_REASONS].filter((code) => code.startsWith('citation_'))
);
const ASK_RELATION_FAILURE_REASONS = new Set(['relation_validation_failed']);
const SAFE_LEGACY_ERROR_MESSAGES: Readonly<Record<string, string>> = Object.freeze({
  embedding_not_configured: '本地 Embedding 服务尚未配置，请完成配置后重试。',
  provider_not_configured: '生成服务尚未配置，请完成配置后重试。',
  unsupported_provider: '当前生成服务配置不受支持，请检查配置后重试。',
  provider_unavailable: '生成服务暂时不可用，请稍后重试。',
  provider_authentication_failed: '生成服务认证失败，请检查配置后重试。',
  provider_rate_limited: '生成服务请求过于频繁，请稍后重试。',
  provider_output_truncated: '生成服务返回内容不完整，请调整请求后重试。',
  provider_empty_content: '生成服务未返回有效内容，请调整请求后重试。',
  provider_invalid_response: '生成服务返回了无效响应，请稍后重试。',
  provider_request_rejected: '生成服务拒绝了当前请求，请检查配置或请求后重试。',
  provider_grounding_failed: '生成结果未通过源码证据校验，未展示未经验证的答案。'
});

function safeHttpErrorMessage(status: number): string {
  return `请求失败（HTTP ${status}），服务端未返回可安全展示的错误详情。`;
}

function safeResponseParseErrorMessage(status: number): string {
  return `响应解析失败（HTTP ${status}），服务端未返回可安全处理的数据。`;
}

export class ApiError extends Error {
  readonly status: number;
  readonly detail: AskFailure | null;

  constructor(message: string, status: number, detail: AskFailure | null = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      headers: {
        'Content-Type': 'application/json',
        ...(options?.headers ?? {})
      },
      ...options
    });
  } catch {
    throw new ApiError(SAFE_NETWORK_ERROR_MESSAGE, 0);
  }
  if (!response.ok) {
    let structured: AskFailure | null = null;
    let safeMessage: string | null = null;
    try {
      const data: unknown = await response.json();
      const detail = getErrorDetail(data);
      structured = projectAskFailure(detail);
      if (structured) {
        safeMessage = SAFE_STRUCTURED_ASK_MESSAGE;
      } else {
        safeMessage = getLegacyErrorMessage(detail);
      }
    } catch {}
    throw new ApiError(
      safeMessage ?? safeHttpErrorMessage(response.status),
      response.status,
      structured
    );
  }
  try {
    return (await response.json()) as T;
  } catch {
    throw new ApiError(safeResponseParseErrorMessage(response.status), response.status, null);
  }
}

export async function analyzeProject(
  sourceType: SourceType,
  source: string
): Promise<{ project_id: string; workspace_id: string; status: string; import_action: string }> {
  return request('/api/projects/analyze', {
    method: 'POST',
    body: JSON.stringify(buildAnalyzePayload(sourceType, source))
  });
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function isBoundedInteger(value: unknown, maximum = MAX_ASK_COUNTER): value is number {
  return (
    typeof value === 'number' &&
    Number.isInteger(value) &&
    value >= 0 &&
    value <= maximum
  );
}

function hasSafeLegacyErrorCode(value: unknown): value is string {
  return (
    typeof value === 'string' &&
    Object.prototype.hasOwnProperty.call(SAFE_LEGACY_ERROR_MESSAGES, value)
  );
}

function projectAskFailure(value: unknown): AskFailure | null {
  if (!isRecord(value) || !isRecord(value.diagnostics)) return null;
  const diagnostics = value.diagnostics;
  if (
    typeof value.code !== 'string' ||
    (!ASK_FAILURE_REASONS.has(value.code) && !hasSafeLegacyErrorCode(value.code)) ||
    typeof value.message !== 'string' ||
    value.message.length === 0 ||
    value.message.length > 512 ||
    typeof value.retryable !== 'boolean' ||
    typeof diagnostics.request_id !== 'string' ||
    !/^[A-Za-z0-9-]{1,64}$/.test(diagnostics.request_id) ||
    typeof diagnostics.agent_mode !== 'string' ||
    !ASK_AGENT_MODES.has(diagnostics.agent_mode) ||
    typeof diagnostics.agent_status !== 'string' ||
    !ASK_AGENT_STATUSES.has(diagnostics.agent_status) ||
    typeof diagnostics.answer_mode !== 'string' ||
    !ASK_ANSWER_MODES.has(diagnostics.answer_mode) ||
    typeof diagnostics.failure_stage !== 'string' ||
    !ASK_FAILURE_STAGES.has(diagnostics.failure_stage) ||
    typeof diagnostics.failure_reason_code !== 'string' ||
    !ASK_FAILURE_REASONS.has(diagnostics.failure_reason_code) ||
    (value.code !== diagnostics.failure_reason_code &&
      !(diagnostics.failure_reason_code === 'provider_error' &&
        typeof value.code === 'string' &&
        ASK_PROVIDER_FAILURE_REASONS.has(value.code))) ||
    typeof diagnostics.retrieval_version !== 'string' ||
    !ASK_RETRIEVAL_VERSIONS.has(diagnostics.retrieval_version) ||
    typeof diagnostics.hierarchy_mode !== 'string' ||
    !ASK_HIERARCHY_MODES.has(diagnostics.hierarchy_mode) ||
    typeof diagnostics.relation_mode !== 'string' ||
    !ASK_RELATION_MODES.has(diagnostics.relation_mode) ||
    !isBoundedInteger(diagnostics.steps_used) ||
    !isBoundedInteger(diagnostics.tool_calls_used) ||
    !isBoundedInteger(diagnostics.planner_logical_calls) ||
    !isBoundedInteger(diagnostics.planner_repair_calls) ||
    typeof diagnostics.final_answer_attempted !== 'boolean' ||
    !isBoundedInteger(diagnostics.provider_logical_calls) ||
    !isBoundedInteger(diagnostics.evidence_count) ||
    !isBoundedInteger(diagnostics.citation_count) ||
    (diagnostics.citation_failure_reason_code !== null &&
      (typeof diagnostics.citation_failure_reason_code !== 'string' ||
        !ASK_CITATION_FAILURE_REASONS.has(diagnostics.citation_failure_reason_code))) ||
    (diagnostics.relation_failure_reason_code !== null &&
      (typeof diagnostics.relation_failure_reason_code !== 'string' ||
        !ASK_RELATION_FAILURE_REASONS.has(diagnostics.relation_failure_reason_code))) ||
    !isBoundedInteger(diagnostics.elapsed_ms, MAX_ASK_ELAPSED_MS)
  ) {
    return null;
  }
  return {
    code: value.code,
    message: SAFE_STRUCTURED_ASK_MESSAGE,
    retryable: value.retryable,
    diagnostics: {
      request_id: diagnostics.request_id,
      agent_mode: diagnostics.agent_mode as AskFailure['diagnostics']['agent_mode'],
      agent_status: diagnostics.agent_status,
      answer_mode: diagnostics.answer_mode as AskFailure['diagnostics']['answer_mode'],
      failure_stage: diagnostics.failure_stage,
      failure_reason_code: diagnostics.failure_reason_code,
      retrieval_version:
        diagnostics.retrieval_version as AskFailure['diagnostics']['retrieval_version'],
      hierarchy_mode: diagnostics.hierarchy_mode as AskFailure['diagnostics']['hierarchy_mode'],
      relation_mode: diagnostics.relation_mode as AskFailure['diagnostics']['relation_mode'],
      steps_used: diagnostics.steps_used,
      tool_calls_used: diagnostics.tool_calls_used,
      planner_logical_calls: diagnostics.planner_logical_calls,
      planner_repair_calls: diagnostics.planner_repair_calls,
      final_answer_attempted: diagnostics.final_answer_attempted,
      provider_logical_calls: diagnostics.provider_logical_calls,
      evidence_count: diagnostics.evidence_count,
      citation_count: diagnostics.citation_count,
      citation_failure_reason_code: diagnostics.citation_failure_reason_code,
      relation_failure_reason_code: diagnostics.relation_failure_reason_code,
      elapsed_ms: diagnostics.elapsed_ms
    }
  };
}

function getErrorDetail(value: unknown): unknown {
  if (!value || typeof value !== 'object') return null;
  return (value as Record<string, unknown>).detail;
}

function getLegacyErrorMessage(value: unknown): string | null {
  if (!isRecord(value) || !hasSafeLegacyErrorCode(value.code)) return null;
  if (value.retryable !== undefined && typeof value.retryable !== 'boolean') return null;
  return SAFE_LEGACY_ERROR_MESSAGES[value.code];
}

export async function listWorkspaces(limit = 20, offset = 0): Promise<WorkspaceListResponse> {
  return request(`/api/workspaces?limit=${limit}&offset=${offset}`);
}

export async function getWorkspace(workspaceId: string): Promise<WorkspaceDetail> {
  return request(`/api/workspaces/${encodeURIComponent(workspaceId)}`);
}

export async function checkWorkspaceRevision(workspaceId: string): Promise<WorkspaceRevisionCheck> {
  return request(`/api/workspaces/${encodeURIComponent(workspaceId)}/revision/check`, {
    method: 'POST'
  });
}

export async function startWorkspaceRefresh(workspaceId: string): Promise<WorkspaceUpdateRun> {
  return request(`/api/workspaces/${encodeURIComponent(workspaceId)}/refresh`, {
    method: 'POST'
  });
}

export async function getWorkspaceUpdateRun(
  workspaceId: string,
  runId: string
): Promise<WorkspaceUpdateRun> {
  return request(
    `/api/workspaces/${encodeURIComponent(workspaceId)}/runs/${encodeURIComponent(runId)}`
  );
}

export async function retryWorkspaceUpdateRun(
  workspaceId: string,
  runId: string
): Promise<WorkspaceUpdateRun> {
  return request(
    `/api/workspaces/${encodeURIComponent(workspaceId)}/runs/${encodeURIComponent(runId)}/retry`,
    { method: 'POST' }
  );
}

export async function getLearningContinuity(workspaceId: string): Promise<LearningContinuity> {
  return request(`/api/workspaces/${encodeURIComponent(workspaceId)}/learning-continuity`);
}

export async function retryLearningContinuity(
  workspaceId: string,
  transitionId: string
): Promise<LearningContinuity> {
  return request(
    `/api/workspaces/${encodeURIComponent(workspaceId)}/learning-continuity/${encodeURIComponent(transitionId)}/retry`,
    { method: 'POST' }
  );
}

export async function getConfigStatus(): Promise<ConfigStatus> {
  return request('/api/config/status');
}

export async function getProject(projectId: string): Promise<ProjectResponse> {
  return request(`/api/projects/${projectId}`);
}

export async function getProjectMap(projectId: string): Promise<ProjectMap> {
  return request(`/api/projects/${projectId}/map`);
}

export async function getLearningPath(projectId: string): Promise<{ steps: LearningStep[] }> {
  return request(`/api/projects/${projectId}/learning-path`);
}

export async function askProject(projectId: string, question: string): Promise<ChatAnswer> {
  return request(`/api/projects/${projectId}/ask`, {
    method: 'POST',
    body: JSON.stringify({ question })
  });
}

export async function getReport(projectId: string): Promise<{ markdown: string }> {
  return request(`/api/projects/${projectId}/report`);
}
