import type { ChatAnswer, ConfigStatus, LearningStep, ProjectMap, ProjectResponse } from '../types';
import { buildAnalyzePayload, type SourceType } from './product';

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000';

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
  } catch (error) {
    if (error instanceof TypeError) {
      throw new Error(`无法连接后端服务 ${API_BASE}。请先启动 backend\\run_backend.bat，再刷新页面重试。`);
    }
    throw error;
  }
  if (!response.ok) {
    const text = await response.text();
    let detail = text;
    try {
      const data = JSON.parse(text);
      detail = typeof data.detail === 'object' ? data.detail.message : (data.detail ?? text);
    } catch {
      detail = text;
    }
    throw new Error(detail || `请求失败：${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function analyzeProject(
  sourceType: SourceType,
  source: string
): Promise<{ project_id: string; status: string; import_action: string }> {
  return request('/api/projects/analyze', {
    method: 'POST',
    body: JSON.stringify(buildAnalyzePayload(sourceType, source))
  });
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
