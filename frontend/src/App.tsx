import {
  BookOpen,
  Brain,
  Clipboard,
  Download,
  FileCode2,
  FolderOpen,
  GitBranch,
  LayoutDashboard,
  Map,
  RefreshCw,
  Search,
  Send,
  Waypoints
} from 'lucide-react';
import { FormEvent, useEffect, useMemo, useState } from 'react';
import {
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
import {
  citationLabel,
  nextWorkspaceSearch,
  providerSummary,
  RECENT_WORKSPACE_KEY,
  selectWorkspaceToRestore,
  workspaceLibraryStatus,
  type WorkspaceLibraryStatus,
  type SourceType
} from './lib/product';
import type {
  ChatAnswer,
  ConfigStatus,
  LearningStep,
  ProjectMap,
  ProjectResponse,
  TreeNode,
  WorkspaceSummary
} from './types';

type Tab = 'dashboard' | 'map' | 'learning' | 'ask' | 'report';

const tabs: Array<{ id: Tab; label: string; icon: typeof LayoutDashboard }> = [
  { id: 'dashboard', label: '概览', icon: LayoutDashboard },
  { id: 'map', label: '项目地图', icon: Map },
  { id: 'learning', label: '学习路线', icon: BookOpen },
  { id: 'ask', label: '源码问答', icon: Search },
  { id: 'report', label: '报告', icon: Clipboard }
];

export default function App() {
  const [sourceType, setSourceType] = useState<SourceType>('local');
  const [source, setSource] = useState('');
  const [configStatus, setConfigStatus] = useState<ConfigStatus | null>(null);
  const [workspaceId, setWorkspaceId] = useState('');
  const [workspaces, setWorkspaces] = useState<WorkspaceSummary[]>([]);
  const [libraryStatus, setLibraryStatus] = useState<WorkspaceLibraryStatus>('loading');
  const [projectId, setProjectId] = useState('');
  const [project, setProject] = useState<ProjectResponse | null>(null);
  const [projectMap, setProjectMap] = useState<ProjectMap | null>(null);
  const [learningSteps, setLearningSteps] = useState<LearningStep[]>([]);
  const [report, setReport] = useState('');
  const [activeTab, setActiveTab] = useState<Tab>('dashboard');
  const [question, setQuestion] = useState('入口文件在哪？');
  const [answers, setAnswers] = useState<Array<{ question: string; result: ChatAnswer }>>([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');

  const hasProject = Boolean(project && projectId);

  useEffect(() => {
    getConfigStatus().then(setConfigStatus).catch((error) => {
      setMessage(error instanceof Error ? error.message : '无法读取配置状态');
    });
    void initializeWorkspace();
  }, []);

  async function initializeWorkspace() {
    await loadWorkspaceLibrary();
    const restore = selectWorkspaceToRestore(
      window.location.search,
      window.localStorage.getItem(RECENT_WORKSPACE_KEY)
    );
    if (restore.source === 'invalid_url') {
      replaceWorkspaceUrl(null);
      setMessage('链接中的 workspace ID 无效，请从项目库重新选择。');
      return;
    }
    if (restore.workspaceId) {
      await openWorkspace(restore.workspaceId, restore.source);
    }
  }

  async function loadWorkspaceLibrary() {
    setLibraryStatus(workspaceLibraryStatus('loading'));
    try {
      const result = await listWorkspaces();
      setWorkspaces(result.items);
      setLibraryStatus(workspaceLibraryStatus('success', result.items.length));
    } catch (error) {
      setLibraryStatus(workspaceLibraryStatus('error'));
      setMessage(error instanceof Error ? error.message : '无法读取项目库');
    }
  }

  function replaceWorkspaceUrl(nextWorkspaceId: string | null) {
    const search = nextWorkspaceSearch(window.location.search, nextWorkspaceId);
    window.history.replaceState(null, '', `${window.location.pathname}${search}${window.location.hash}`);
  }

  async function loadAll(nextProjectId: string) {
    const [projectData, mapData, learningData, reportData] = await Promise.all([
      getProject(nextProjectId),
      getProjectMap(nextProjectId),
      getLearningPath(nextProjectId),
      getReport(nextProjectId)
    ]);
    setProject(projectData);
    setProjectMap(mapData);
    setLearningSteps(learningData.steps);
    setReport(reportData.markdown);
  }

  async function openWorkspace(nextWorkspaceId: string, source: 'url' | 'recent' | 'selection' = 'selection') {
    setLoading(true);
    setMessage('正在从项目库重新打开持久化项目…');
    try {
      const workspace = await getWorkspace(nextWorkspaceId);
      await loadAll(workspace.active_snapshot.project_id);
      setWorkspaceId(workspace.workspace_id);
      setProjectId(workspace.active_snapshot.project_id);
      setAnswers([]);
      window.localStorage.setItem(RECENT_WORKSPACE_KEY, workspace.workspace_id);
      replaceWorkspaceUrl(workspace.workspace_id);
      setActiveTab('dashboard');
      setMessage('已重新打开持久化项目；未重新分析或调用模型。');
      return true;
    } catch (error) {
      if (window.localStorage.getItem(RECENT_WORKSPACE_KEY) === nextWorkspaceId) {
        window.localStorage.removeItem(RECENT_WORKSPACE_KEY);
      }
      if (source === 'url' || source === 'recent') replaceWorkspaceUrl(null);
      setWorkspaceId('');
      setProjectId('');
      setProject(null);
      setMessage(
        `${error instanceof Error ? error.message : '项目重新打开失败'} 请从项目库选择其他项目或重新分析。`
      );
      return false;
    } finally {
      setLoading(false);
    }
  }

  async function handleAnalyze(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setMessage('正在抓取仓库并生成学习导读...');
    setAnswers([]);
    try {
      const result = await analyzeProject(sourceType, source);
      await loadWorkspaceLibrary();
      const opened = await openWorkspace(result.workspace_id);
      if (opened) {
        setMessage(result.import_action === 'reused' ? '已载入持久化项目和索引。' : '分析和本地索引完成。');
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : '分析失败');
    } finally {
      setLoading(false);
    }
  }

  async function handleAsk(event: FormEvent) {
    event.preventDefault();
    if (!projectId || !question.trim()) return;
    setLoading(true);
    setMessage('正在检索源码片段并生成回答...');
    try {
      const result = await askProject(projectId, question);
      setAnswers((current) => [{ question, result }, ...current]);
      setQuestion('');
      setMessage('回答已生成，注意查看引用文件。');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : '问答失败');
    } finally {
      setLoading(false);
    }
  }

  const activeContent = useMemo(() => {
    if (!hasProject || !project) {
      return <EmptyState />;
    }
    if (activeTab === 'dashboard') {
      return <Dashboard project={project} />;
    }
    if (activeTab === 'map') {
      return <MapView projectMap={projectMap} />;
    }
    if (activeTab === 'learning') {
      return <LearningView steps={learningSteps} />;
    }
    if (activeTab === 'ask') {
      return (
        <AskView
          question={question}
          setQuestion={setQuestion}
          answers={answers}
          onSubmit={handleAsk}
          loading={loading}
        />
      );
    }
    return <ReportView markdown={report} />;
  }, [activeTab, answers, hasProject, learningSteps, loading, project, projectMap, question, report]);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <GitBranch aria-hidden="true" />
          <div>
            <strong>GitLearnAgent</strong>
            <span>开源项目学习导读系统</span>
          </div>
        </div>
        <div className="status-line">{message || providerSummary(configStatus)}</div>
      </header>

      <main className="workspace">
        <aside className="sidebar">
          <section className="workspace-library" aria-label="项目库">
            <div className="workspace-library-title">
              <strong>项目库</strong>
              <button type="button" onClick={() => void loadWorkspaceLibrary()} disabled={libraryStatus === 'loading'}>
                <RefreshCw aria-hidden="true" />
                <span>刷新列表</span>
              </button>
            </div>
            {libraryStatus === 'loading' && <p>正在读取已有项目…</p>}
            {libraryStatus === 'error' && <p className="library-error">项目库读取失败，可重试或使用下方新建分析。</p>}
            {libraryStatus === 'empty' && <p>还没有已分析项目，请使用下方入口建立第一个项目。</p>}
            {libraryStatus === 'ready' && (
              <div className="workspace-list">
                {workspaces.map((item) => (
                  <button
                    type="button"
                    key={item.workspace_id}
                    className={workspaceId === item.workspace_id ? 'active' : ''}
                    disabled={!item.openable || loading}
                    onClick={() => void openWorkspace(item.workspace_id)}
                  >
                    <FolderOpen aria-hidden="true" />
                    <span>
                      <strong>{item.display_name}</strong>
                      <small>{item.openable ? `${item.project_status} · ${item.repository_revision.slice(0, 8) || 'no revision'}` : '关联不可用'}</small>
                    </span>
                  </button>
                ))}
              </div>
            )}
          </section>

          <form className="analyze-form" onSubmit={handleAnalyze}>
            <label htmlFor="repo-source">仓库来源</label>
            <div className="source-toggle" role="group" aria-label="仓库来源">
              <button type="button" className={sourceType === 'local' ? 'active' : ''} onClick={() => setSourceType('local')}>本地目录</button>
              <button type="button" className={sourceType === 'git_url' ? 'active' : ''} onClick={() => setSourceType('git_url')}>公开 HTTPS Git</button>
            </div>
            <input
              id="repo-source"
              value={source}
              onChange={(event) => setSource(event.target.value)}
              placeholder={sourceType === 'local' ? 'D:\\Project\\my-python-repo' : 'https://host/owner/repo.git'}
              required
            />
            <button type="submit" disabled={loading}>
              {loading ? <RefreshCw className="spin" aria-hidden="true" /> : <GitBranch aria-hidden="true" />}
              <span>{loading ? '分析中' : '开始分析'}</span>
            </button>
          </form>

          <div className="provider-card">
            <strong>{providerSummary(configStatus)}</strong>
            <span>BGE-M3：{configStatus ? `${configStatus.embedding.model} / ${configStatus.embedding.device} / ${configStatus.embedding.offline ? 'offline' : 'online'}${configStatus.embedding.ready ? '' : `；缺少 ${configStatus.embedding.missing.join(', ')}`}` : '读取中'}</span>
            <small>API Key 只保存在后端根目录 .env，不会发送到浏览器或写入项目数据库。</small>
          </div>

          <nav className="tabs" aria-label="结果导航">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  className={activeTab === tab.id ? 'active' : ''}
                  onClick={() => setActiveTab(tab.id)}
                  disabled={!hasProject}
                  title={tab.label}
                >
                  <Icon aria-hidden="true" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </nav>
        </aside>

        <section className="content">{activeContent}</section>
      </main>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="empty-state">
      <Waypoints aria-hidden="true" />
      <h1>把仓库变成学习路线</h1>
      <p>系统会先抓取公开仓库，再用静态分析识别入口、依赖、模块和核心文件，最后生成适合初学者的导读结果。</p>
      <div className="empty-grid">
        <span>项目地图</span>
        <span>学习任务</span>
        <span>源码引用</span>
        <span>报告导出</span>
      </div>
    </div>
  );
}

function Dashboard({ project }: { project: ProjectResponse }) {
  return (
    <div className="stack">
      <section className="section-header">
        <div>
          <p>{project.project.repo_url}</p>
          <h1>{project.project.repo}</h1>
        </div>
        <div className="badge-row">
          <span>{project.project.primary_language || 'Unknown'}</span>
          {project.project.frameworks.map((framework) => (
            <span key={framework}>{framework}</span>
          ))}
        </div>
      </section>

      <div className="metrics">
        <Metric label="文本文件" value={project.stats.file_count ?? 0} icon={FileCode2} />
        <Metric label="核心文件" value={project.stats.core_file_count ?? 0} icon={Brain} />
        <Metric label="模块数量" value={project.modules.length} icon={Waypoints} />
      </div>

      <section className="panel">
        <h2>项目概览</h2>
        <p>{project.overview}</p>
        {project.start_commands.length > 0 && (
          <div className="command-list">
            {project.start_commands.map((command) => (
              <code key={command}>{command}</code>
            ))}
          </div>
        )}
      </section>

      <section className="panel">
        <h2>核心文件</h2>
        <div className="file-table">
          {project.core_files.map((file) => (
            <div className="file-row" key={file.path}>
              <strong>{file.path}</strong>
              <span>{file.summary}</span>
              <em>{Math.round(file.importance)}</em>
            </div>
          ))}
        </div>
      </section>

      <section className="module-grid">
        {project.modules.map((module) => (
          <article className="module-card" key={module.name}>
            <h3>{module.name}</h3>
            <p>{module.responsibility}</p>
            <small>{module.files.slice(0, 3).join(' / ')}</small>
          </article>
        ))}
      </section>
    </div>
  );
}

function Metric({ label, value, icon: Icon }: { label: string; value: number; icon: typeof FileCode2 }) {
  return (
    <div className="metric">
      <Icon aria-hidden="true" />
      <div>
        <strong>{value}</strong>
        <span>{label}</span>
      </div>
    </div>
  );
}

function MapView({ projectMap }: { projectMap: ProjectMap | null }) {
  if (!projectMap) return null;
  return (
    <div className="two-column">
      <section className="panel">
        <h2>目录树</h2>
        <Tree node={projectMap.tree} />
      </section>
      <section className="panel">
        <h2>模块关系</h2>
        <div className="module-lanes">
          {projectMap.modules.map((module) => (
            <div className="lane" key={module.name}>
              <div>
                <strong>{module.name}</strong>
                <span>{module.depends_on.length ? `依赖 ${module.depends_on.join(', ')}` : '独立模块'}</span>
              </div>
              <p>{module.responsibility}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

function Tree({ node }: { node: TreeNode }) {
  return (
    <div className={node.type === 'directory' ? 'tree-dir' : 'tree-file'}>
      <span className={node.is_core ? 'core-node' : ''}>{node.name}</span>
      {node.children && (
        <div className="tree-children">
          {node.children.map((child) => (
            <Tree key={child.path} node={child} />
          ))}
        </div>
      )}
    </div>
  );
}

function LearningView({ steps }: { steps: LearningStep[] }) {
  return (
    <div className="learning-list">
      {steps.map((step) => (
        <article className="learning-step" key={step.order}>
          <div className="step-index">{step.order}</div>
          <div>
            <h2>{step.title}</h2>
            <p>{step.goal}</p>
            <div className="pill-group">
              {step.files.map((file) => (
                <span key={file}>{file}</span>
              ))}
            </div>
            <ul>
              {step.tasks.map((task) => (
                <li key={task}>{task}</li>
              ))}
            </ul>
            <div className="quiz-list">
              {step.quiz.map((quiz) => (
                <details key={quiz.question}>
                  <summary>{quiz.question}</summary>
                  <p>{quiz.answer}</p>
                </details>
              ))}
            </div>
          </div>
        </article>
      ))}
    </div>
  );
}

function AskView({
  question,
  setQuestion,
  answers,
  onSubmit,
  loading
}: {
  question: string;
  setQuestion: (value: string) => void;
  answers: Array<{ question: string; result: ChatAnswer }>;
  onSubmit: (event: FormEvent) => void;
  loading: boolean;
}) {
  return (
    <div className="stack">
      <form className="ask-form" onSubmit={onSubmit}>
        <input
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="例如：项目怎么启动？登录逻辑在哪？入口文件在哪？"
        />
        <button type="submit" disabled={loading}>
          <Send aria-hidden="true" />
          <span>提问</span>
        </button>
      </form>
      <div className="answer-list">
        {answers.map((item, index) => (
          <article className="answer" key={`${item.question}-${index}`}>
            <h2>{item.question}</h2>
            <p>{item.result.answer}</p>
            <div className="citation-grid">
              {item.result.citations.map((citation) => (
                <details key={citation.path} open>
                  <summary>{citationLabel(citation)}</summary>
                  <pre>{citation.snippet}</pre>
                </details>
              ))}
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}

function ReportView({ markdown }: { markdown: string }) {
  function download() {
    const blob = new Blob([markdown], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'gitlearnagent-report.md';
    anchor.click();
    URL.revokeObjectURL(url);
  }

  async function copy() {
    await navigator.clipboard.writeText(markdown);
  }

  return (
    <div className="stack">
      <div className="report-actions">
        <button onClick={copy} type="button">
          <Clipboard aria-hidden="true" />
          <span>复制</span>
        </button>
        <button onClick={download} type="button">
          <Download aria-hidden="true" />
          <span>下载</span>
        </button>
      </div>
      <textarea className="report-box" value={markdown} readOnly />
    </div>
  );
}
