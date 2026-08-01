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

export interface ConfigStatus {
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

