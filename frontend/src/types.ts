export interface ProjectInfo {
  id: string;
  repo_url: string;
  owner: string;
  repo: string;
  default_branch: string;
  status: string;
  primary_language: string;
  frameworks: string[];
  error_message?: string;
  source_type?: 'local' | 'git_url' | 'legacy_github';
  source_location?: string;
}

export interface FileSummary {
  path: string;
  extension: string;
  language: string;
  size: number;
  summary: string;
  importance: number;
  is_core: boolean;
  imports: string[];
  exports: string[];
  symbols: string[];
}

export interface ModuleSummary {
  name: string;
  responsibility: string;
  files: string[];
  depends_on: string[];
}

export interface ProjectResponse {
  project: ProjectInfo;
  overview: string;
  stats: {
    file_count?: number;
    core_file_count?: number;
    total_text_bytes?: number;
  };
  start_commands: string[];
  core_files: FileSummary[];
  modules: ModuleSummary[];
}

export interface TreeNode {
  name: string;
  path: string;
  type: 'directory' | 'file';
  importance?: number;
  is_core?: boolean;
  children?: TreeNode[] | null;
}

export interface ProjectMap {
  tree: TreeNode;
  modules: ModuleSummary[];
  dependency_edges: Array<{ from: string; to: string }>;
  core_files: FileSummary[];
}

export interface LearningStep {
  order: number;
  title: string;
  goal: string;
  files: string[];
  tasks: string[];
  quiz: Array<{ question: string; answer: string }>;
}

export interface Citation {
  path: string;
  summary: string;
  snippet: string;
  qualified_name: string;
  start_line: number;
  end_line: number;
}

export interface ChatAnswer {
  answer: string;
  citations: Citation[];
  answer_mode: 'llm_grounded' | 'deterministic';
  grounding_status: 'grounded' | 'insufficient_evidence' | 'degraded';
  warnings: string[];
}

export interface AskFailureDiagnostics {
  request_id: string;
  agent_mode: 'bounded' | 'deterministic_fallback' | 'unknown';
  agent_status: string;
  answer_mode: 'llm_grounded' | 'deterministic' | 'not_available';
  failure_stage: string;
  failure_reason_code: string;
  retrieval_version: 'v1' | 'v2';
  hierarchy_mode: 'off' | 'normalize_v1';
  relation_mode: 'off' | 'expand_v1';
  steps_used: number;
  tool_calls_used: number;
  planner_logical_calls: number;
  planner_repair_calls: number;
  final_answer_attempted: boolean;
  provider_logical_calls: number;
  evidence_count: number;
  citation_count: number;
  citation_failure_reason_code: string | null;
  relation_failure_reason_code: string | null;
  elapsed_ms: number;
}

export interface AskFailure {
  code: string;
  message: string;
  retryable: boolean;
  diagnostics: AskFailureDiagnostics;
}

export interface ConfigStatus {
  git_proxy_configured: boolean;
  llm: {
    provider: string | null;
    model: string | null;
    base_url_configured: boolean;
    api_key_configured: boolean;
    ready: boolean;
    missing: string[];
  };
  embedding: {
    provider: string;
    model: string;
    device: string;
    offline: boolean;
    enabled: boolean;
    ready: boolean;
    missing: string[];
  };
}

export interface WorkspaceSummary {
  workspace_id: string;
  display_name: string;
  source_type: string;
  project_status: string;
  repository_revision: string;
  openable: boolean;
  project_id?: string | null;
  total_chunks?: number;
  embedding_count?: number;
  created_at: string;
  updated_at: string;
}

export interface WorkspaceListResponse {
  items: WorkspaceSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface WorkspaceDetail extends WorkspaceSummary {
  active_snapshot: {
    project_id: string;
    repository_revision: string;
    status: string;
    primary_language: string;
    frameworks: string[];
    updated_at: string;
  };
  latest_update_run: WorkspaceUpdateRun | null;
  learning_continuity: LearningContinuity | null;
}

export interface WorkspaceRevisionCheck {
  workspace_id: string;
  current_revision: string;
  available_revision: string;
  state: 'unchanged' | 'update_available';
}

export interface WorkspaceUpdateRun {
  run_id: string;
  workspace_id: string;
  target_revision: string;
  status: 'pending' | 'running' | 'succeeded' | 'failed';
  phase: string;
  result: '' | 'unchanged' | 'activated';
  stats: Record<string, unknown>;
  error_code: string;
  error_message: string;
  retryable: boolean;
  retry_count: number;
  active_project_id: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  updated_at: string;
}

export interface LearningContinuityStats {
  total: number;
  unchanged_exact: number;
  renamed_exact: number;
  modified: number;
  deleted: number;
  ambiguous: number;
  unmapped: number;
  incompatible: number;
  retained: number;
  needs_review: number;
  history_only: number;
  not_inherited: number;
}

export interface LearningContinuity {
  transition_id: string | null;
  workspace_id: string;
  status: 'not_required' | 'pending' | 'running' | 'succeeded' | 'failed';
  activation_version: number;
  mapping_config_identity?: string;
  source_revision?: string;
  target_revision?: string;
  stats: LearningContinuityStats;
  error_code: string;
  error_message: string;
  retryable: boolean;
  retry_count: number;
  created_at: string | null;
  started_at: string | null;
  finished_at: string | null;
  updated_at: string | null;
}

