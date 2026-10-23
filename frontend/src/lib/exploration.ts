import { request } from './api';

export interface ExplorationIdentity { project_id: string; workspace_id: string; revision: string; activation_version: number; repo: string }
export interface SourceRef { path: string; start: number; end: number; symbol?: string; sha256: string }
export interface Entry { text?: string; label?: string; reason?: string; source: SourceRef }
export interface Scene { id: string; title: string; status: string; fields: Record<string,string>; conditions: Entry[]; inputs: Entry[]; expectations: Entry[]; assertions: Entry[]; implementation: Entry[]; unknowns: string[] }
export interface PairSummary { id: string; title: string; target: string; status: string; differences: { field: string; a: string; b: string }[] }
export interface Pair extends PairSummary { common: string[]; scenarios: Scene[]; reading: Entry[]; reasons?: string[] }
export interface Candidate { id: string; path: string; line: number; end_line: number; symbol: string; exception_expression: string; raise_source: string; template: string; match_mode: string; type_name_relation: string; controls: { kind: string; line: number; source: string }[]; sha256: string; trace_frames?: number[] }
export interface TraceLocation { path: string; line: number; sha256: string; status: string; readable?: boolean; source_preview?: string; source_preview_truncated?: boolean; checks: string[]; source_check?: string; function_check?: string; scope?: { symbol: string; start: number; end: number; kind: string }; relation?: string; raise_range?: { start: number; end: number }; candidate_ids: string[] }
export interface TraceFrame { index: number; input_path: string; line: number | null; input_line: string; function: string; status: string; issues: string[]; matches_total: number; matches: TraceLocation[] }
export interface TracebackResult { status: string; complete: boolean; algorithm: string; frames_detected: number; frames: TraceFrame[]; issues: string[]; boundary: string }
export interface Exploration {
  identity: ExplorationIdentity; data_version: string; algorithm: string;
  coverage: { files: number; skipped_count: number; same_revision_inventory: boolean; parse_errors: { path: string; line?: number }[]; tests: { test_files: number; tests: number; scenarios: number; pairs: number; exclusion_reasons: Record<string,number> }; [key: string]: unknown };
  status?: string; pairs?: PairSummary[]; pair?: Pair; candidates?: Candidate[];
  traceback?: TracebackResult;
  total?: number; all_pairs?: number; offset?: number; limit?: number; unsupported?: { title: string; reason: string; source: SourceRef }[];
}
export interface ExplorationContext { projectId: string; workspaceId: string; revision: string }

export async function explore(ctx: ExplorationContext, route = '', options: { query?: Record<string,string>; body?: unknown; signal?: AbortSignal } = {}): Promise<Exploration> {
  const query = new URLSearchParams({ revision: ctx.revision, ...options.query });
  const result = await request<Exploration>(`/api/projects/${encodeURIComponent(ctx.projectId)}/exploration${route}?${query}`, {
    signal: options.signal, ...(options.body ? { method: 'POST', body: JSON.stringify(options.body) } : {})
  });
  if (result.identity?.project_id !== ctx.projectId || result.identity?.workspace_id !== ctx.workspaceId || result.identity?.revision !== ctx.revision || !result.data_version) {
    throw new Error('返回材料与当前仓库或版本不符，请重新打开项目。');
  }
  return result;
}
