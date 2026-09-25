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
  Trash2,
  Waypoints
} from 'lucide-react';
import { FormEvent, useEffect, useMemo, useRef, useState } from 'react';
import { Workbench, type WorkbenchAnswer, type WorkbenchPage } from './workbench/Workbench';
import './workbench/workbench.css';
import './workbench/shell.css';
import './workbench/interaction.css';
import './workbench/ask-experience.css';
import './workbench/reading.css';
import './workbench/reading-fixes.css';
import './workbench/evidence-summary.css';
import './workbench/reading-theme.css';
import {
  ApiError,
  analyzeProject,
  askProjectStream,
  checkWorkspaceRevision,
  deleteProject,
  getConfigStatus,
  getLearningPath,
  getLearningContinuity,
  getProject,
  getProjectMap,
  getReport,
  getWorkspace,
  getWorkspaceUpdateRun,
  listWorkspaces,
  retryWorkspaceUpdateRun,
  retryLearningContinuity,
  startWorkspaceRefresh
} from './lib/api';
import { createRequestGate, type RequestToken } from './lib/requestGate';
import type { ExecutionMode } from './types';
import {
  createConnectionStatusGate,
  type ConnectionProbeToken
} from './lib/connectionStatusGate';
import {
  citationLabel,
  learningContinuityState,
  nextWorkspaceSearch,
  providerSummary,
  RECENT_WORKSPACE_KEY,
  selectWorkspaceToRestore,
  workspaceLibraryStatus,
  workspaceUpdateState,
  type WorkspaceLibraryStatus,
  type SourceType
} from './lib/product';
import type {
  AskFailure,
  ChatAnswer,
  ConfigStatus,
  LearningContinuity,
  LearningStep,
  ProjectMap,
  ProjectResponse,
  WorkspaceSummary,
  WorkspaceDetail,
  WorkspaceRevisionCheck,
  WorkspaceUpdateRun
} from './types';
import { ProjectMapView } from './workbench/ProjectMapView';
import './workbench/project-map.css';

type Tab = WorkbenchPage;

interface LoadedProjectData {
  project: ProjectResponse;
  projectMap: ProjectMap | null;
  mapError: string | null;
  learningSteps: LearningStep[];
  report: string;
}

export default function App() {
  const [sourceType, setSourceType] = useState<SourceType>('local');
  const [source, setSource] = useState('');
  const [configStatus, setConfigStatus] = useState<ConfigStatus | null>(null);
  const [workspaceId, setWorkspaceId] = useState('');
  const [workspaces, setWorkspaces] = useState<WorkspaceSummary[]>([]);
  const [libraryStatus, setLibraryStatus] = useState<WorkspaceLibraryStatus>('loading');
  const [projectId, setProjectId] = useState('');
  const [currentRevision, setCurrentRevision] = useState('');
  const [revisionCheck, setRevisionCheck] = useState<WorkspaceRevisionCheck | null>(null);
  const [updateRun, setUpdateRun] = useState<WorkspaceUpdateRun | null>(null);
  const [continuity, setContinuity] = useState<LearningContinuity | null>(null);
  const [project, setProject] = useState<ProjectResponse | null>(null);
  const [projectMap, setProjectMap] = useState<ProjectMap | null>(null);
  const [mapError, setMapError] = useState<string | null>(null);
  const [learningSteps, setLearningSteps] = useState<LearningStep[]>([]);
  const [report, setReport] = useState('');
  const [activeTab, setActiveTab] = useState<Tab>('ask');
  const [question, setQuestion] = useState('');
  const [answers, setAnswers] = useState<WorkbenchAnswer[]>([]);
  const [executionMode, setExecutionMode] = useState<ExecutionMode>('agent');
  const activeAskConnection = useRef<AbortController | null>(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [workspaceLoading, setWorkspaceLoading] = useState(false);
  const [askLoading, setAskLoading] = useState(false);
  const [askError, setAskError] = useState<AskFailure | null>(null);
  const requestGate = useRef(createRequestGate());
  const connectionGate = useRef(createConnectionStatusGate());
  const [message, setMessage] = useState('');
  const [connectionError, setConnectionError] = useState('');
  const [deletingProjectId, setDeletingProjectId] = useState('');

  useEffect(() => () => { activeAskConnection.current?.abort(); }, [workspaceId, projectId, currentRevision]);

  const hasProject = Boolean(project && projectId);
  const updateState = workspaceUpdateState(revisionCheck, updateRun);
  const continuityState = learningContinuityState(continuity);

  useEffect(() => {
    const probe = beginConnectionProbe();
    getConfigStatus().then((result) => {
      settleConnectionProbe(probe);
      setConfigStatus(result);
    }).catch((error) => {
      settleConnectionProbe(probe, error);
      if (!(error instanceof ApiError && error.status === 0)) {
        setMessage(error instanceof Error ? error.message : '无法读取配置状态');
      }
    });
    void initializeWorkspace();
  }, []);

  useEffect(() => {
    if (!workspaceId || !updateRun || !['pending', 'running'].includes(updateRun.status)) return;
    const timer = window.setTimeout(() => void pollUpdateRun(workspaceId, updateRun.run_id), 1000);
    return () => window.clearTimeout(timer);
  }, [workspaceId, updateRun]);

  useEffect(() => {
    if (!workspaceId || !continuity?.transition_id || !['pending', 'running'].includes(continuity.status)) return;
    const timer = window.setTimeout(() => void pollContinuity(workspaceId), 1000);
    return () => window.clearTimeout(timer);
  }, [workspaceId, continuity]);

  async function initializeWorkspace() {
    const initialGeneration = requestGate.current.getContext().generation;
    const library = await loadWorkspaceLibrary();
    if (requestGate.current.getContext().generation !== initialGeneration) return;
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
    } else {
      const firstOpenable = library?.find((item) => item.openable);
      if (firstOpenable) await openWorkspace(firstOpenable.workspace_id, 'selection');
    }
  }

  async function loadWorkspaceLibrary(token?: RequestToken): Promise<WorkspaceSummary[] | null> {
    if (token && !requestGate.current.isActive(token)) return null;
    setLibraryStatus(workspaceLibraryStatus('loading'));
    const probe = beginConnectionProbe();
    try {
      const result = await listWorkspaces();
      if (token && !requestGate.current.isActive(token)) return null;
      settleConnectionProbe(probe);
      setWorkspaces(result.items);
      setLibraryStatus(workspaceLibraryStatus('success', result.items.length));
      return result.items;
    } catch (error) {
      if (token && !requestGate.current.isActive(token)) return null;
      settleConnectionProbe(probe, error);
      setLibraryStatus(workspaceLibraryStatus('error'));
      if (!(error instanceof ApiError && error.status === 0)) {
        setMessage(error instanceof Error ? error.message : '无法读取项目库');
      }
      return null;
    }
  }

  async function handleDeleteWorkspace(item: WorkspaceSummary) {
    if (!item.project_id || deletingProjectId) return;
    const confirmed = window.confirm(
      '将删除该项目的本地分析、索引和学习记录。\n不会修改远程 Git 仓库。'
    );
    if (!confirmed) return;
    setDeletingProjectId(item.project_id);
    try {
      const result = await deleteProject(item.project_id);
      if (!result.deleted) {
        setMessage('本地 checkout 清理尚未完成，可再次点击删除重试。');
        return;
      }
      setWorkspaces((current) => current.filter((value) => value.workspace_id !== item.workspace_id));
      if (workspaceId === item.workspace_id) {
        setWorkspaceId('');
        setProjectId('');
        setProject(null);
        setProjectMap(null);
        setLearningSteps([]);
        setReport('');
        window.localStorage.removeItem(RECENT_WORKSPACE_KEY);
        replaceWorkspaceUrl(null);
      }
      setMessage('项目已删除；远程 Git 仓库未被修改。');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : '项目删除失败，请重试。');
    } finally {
      setDeletingProjectId('');
    }
  }

  function beginConnectionProbe(): ConnectionProbeToken {
    return connectionGate.current.begin(requestGate.current.getContext().generation);
  }

  function settleConnectionProbe(probe: ConnectionProbeToken, error?: unknown) {
    if (!connectionGate.current.settle(probe)) return;
    if (error instanceof ApiError && error.status === 0) {
      setConnectionError(error.message);
    } else {
      setConnectionError('');
    }
  }

  function replaceWorkspaceUrl(nextWorkspaceId: string | null) {
    const search = nextWorkspaceSearch(window.location.search, nextWorkspaceId);
    window.history.replaceState(null, '', `${window.location.pathname}${search}${window.location.hash}`);
  }

  async function loadAll(nextProjectId: string): Promise<LoadedProjectData> {
    const [projectData, mapData, learningData, reportData] = await Promise.all([
      getProject(nextProjectId),
      getProjectMap(nextProjectId).then((value) => ({ value, error: null })).catch(() => ({ value: null, error: '地图接口不可用，请重试。' })),
      getLearningPath(nextProjectId),
      getReport(nextProjectId)
    ]);
    return {
      project: projectData,
      projectMap: mapData.value,
      mapError: mapData.error,
      learningSteps: learningData.steps,
      report: reportData.markdown
    };
  }

  function applyWorkspace(workspace: WorkspaceDetail, loaded: LoadedProjectData) {
    setWorkspaceId(workspace.workspace_id);
    setProjectId(workspace.active_snapshot.project_id);
    setCurrentRevision(workspace.active_snapshot.repository_revision);
    setRevisionCheck(null);
    setUpdateRun(workspace.latest_update_run);
    setContinuity(workspace.learning_continuity);
    setProject(loaded.project);
    const mapMatches = !loaded.projectMap || ((!loaded.projectMap.project_id || loaded.projectMap.project_id === workspace.active_snapshot.project_id) && (!loaded.projectMap.repository_revision || loaded.projectMap.repository_revision === workspace.active_snapshot.repository_revision));
    setProjectMap(mapMatches ? loaded.projectMap : null);
    setMapError(mapMatches ? loaded.mapError : '地图版本与当前项目不一致，请重试。');
    setLearningSteps(loaded.learningSteps);
    setReport(loaded.report);
    setAnswers([]);
    setAskError(null);
    window.localStorage.setItem(RECENT_WORKSPACE_KEY, workspace.workspace_id);
    replaceWorkspaceUrl(workspace.workspace_id);
  }

  function beginContextChange(operation: string, targetWorkspaceId: string) {
    const token = requestGate.current.beginContextChange(operation, targetWorkspaceId);
    connectionGate.current.changeContext(token.context.generation);
    setConnectionError('');
    setAnalysisLoading(false);
    setWorkspaceLoading(false);
    setAskLoading(false);
    setAnswers([]);
    setAskError(null);
    setProjectMap(null);
    setMapError(null);
    return token;
  }

  async function retryMap() {
    if (!projectId || !currentRevision) return;
    const context = requestGate.current.getContext();
    setMapError(null);
    setProjectMap(null);
    try {
      const result = await getProjectMap(projectId);
      const now = requestGate.current.getContext();
      if (now.generation !== context.generation || now.projectId !== projectId || now.revision !== currentRevision) return;
      if ((result.project_id && result.project_id !== projectId) || (result.repository_revision && result.repository_revision !== currentRevision)) throw new Error('map_context_mismatch');
      setProjectMap(result);
    } catch {
      const now = requestGate.current.getContext();
      if (now.generation === context.generation && now.projectId === projectId && now.revision === currentRevision) setMapError('地图接口不可用或版本不匹配，请重试。');
    }
  }

  async function openWorkspace(
    nextWorkspaceId: string,
    source: 'url' | 'recent' | 'selection' = 'selection',
    successMessage = '已重新打开持久化项目；未重新分析或调用模型。',
    resetActiveTab = true
  ) {
    const token = beginContextChange('context', nextWorkspaceId);
    setWorkspaceLoading(true);
    setMessage('正在从项目库重新打开持久化项目…');
    const probe = beginConnectionProbe();
    try {
      const workspace = await getWorkspace(nextWorkspaceId);
      const loaded = await loadAll(workspace.active_snapshot.project_id);
      if (
        !requestGate.current.commitContext(token, {
          workspaceId: workspace.workspace_id,
          projectId: workspace.active_snapshot.project_id,
          revision: workspace.active_snapshot.repository_revision
        })
      ) {
        return false;
      }
      applyWorkspace(workspace, loaded);
      settleConnectionProbe(probe);
      if (resetActiveTab) setActiveTab('ask');
      setMessage(successMessage);
      return true;
    } catch (error) {
      if (!requestGate.current.isActive(token)) return false;
      settleConnectionProbe(probe, error);
      if (window.localStorage.getItem(RECENT_WORKSPACE_KEY) === nextWorkspaceId) {
        window.localStorage.removeItem(RECENT_WORKSPACE_KEY);
      }
      if (source === 'url' || source === 'recent') replaceWorkspaceUrl(null);
      setWorkspaceId('');
      setProjectId('');
      setCurrentRevision('');
      setRevisionCheck(null);
      setUpdateRun(null);
      setContinuity(null);
      setProject(null);
      setProjectMap(null);
      setLearningSteps([]);
      setReport('');
      requestGate.current.clearContext(token);
      setMessage(
        `${error instanceof Error ? error.message : '项目重新打开失败'} 请从项目库选择其他项目或重新分析。`
      );
      return false;
    } finally {
      if (requestGate.current.finish(token)) setWorkspaceLoading(false);
    }
  }

  async function handleCheckRevision() {
    if (!workspaceId) return;
    const token = requestGate.current.tryEnter('revision-check');
    if (!token) return;
    if (token.context.workspaceId !== workspaceId) {
      requestGate.current.finish(token);
      return;
    }
    setMessage('正在显式检查仓库 revision…');
    try {
      const result = await checkWorkspaceRevision(token.context.workspaceId);
      if (!requestGate.current.isCurrent(token)) return;
      setRevisionCheck(result);
      setMessage(result.state === 'unchanged' ? '当前 snapshot 已是最新 revision。' : '检测到可用的新 revision；确认后才会开始更新。');
    } catch (error) {
      if (!requestGate.current.isCurrent(token)) return;
      setMessage(error instanceof Error ? error.message : 'revision 检查失败');
    } finally {
      requestGate.current.finish(token, true);
    }
  }

  async function handleStartRefresh() {
    if (!workspaceId || revisionCheck?.state !== 'update_available') return;
    if (!window.confirm('确认显式更新到检测到的新 revision？旧 snapshot 会在成功激活前保持可用。')) return;
    const token = requestGate.current.tryEnter('refresh-start');
    if (!token) return;
    if (token.context.workspaceId !== workspaceId) {
      requestGate.current.finish(token);
      return;
    }
    try {
      const run = await startWorkspaceRefresh(token.context.workspaceId);
      if (!requestGate.current.isCurrent(token)) return;
      setUpdateRun(run);
      setMessage(run.result === 'unchanged' ? '仓库 revision 未变化，未创建 snapshot。' : '更新已启动；旧 snapshot 仍可继续使用。');
    } catch (error) {
      if (!requestGate.current.isCurrent(token)) return;
      setMessage(error instanceof Error ? error.message : '更新启动失败');
    } finally {
      requestGate.current.finish(token, true);
    }
  }

  async function handleRetryRefresh() {
    if (!workspaceId || !updateRun?.retryable) return;
    if (!window.confirm('确认重试这次更新？旧 snapshot 在重试期间仍保持可用。')) return;
    const token = requestGate.current.tryEnter('refresh-retry');
    if (!token) return;
    if (token.context.workspaceId !== workspaceId) {
      requestGate.current.finish(token);
      return;
    }
    try {
      const run = await retryWorkspaceUpdateRun(token.context.workspaceId, updateRun.run_id);
      if (!requestGate.current.isCurrent(token)) return;
      setUpdateRun(run);
      setMessage('更新重试已启动。');
    } catch (error) {
      if (!requestGate.current.isCurrent(token)) return;
      setMessage(error instanceof Error ? error.message : '更新重试失败');
    } finally {
      requestGate.current.finish(token, true);
    }
  }

  async function handleRetryContinuity() {
    if (!workspaceId || !continuity?.transition_id || !continuity.retryable) return;
    if (!window.confirm('确认重试学习连续性映射？代码 snapshot 不会回滚，也不会伪造学习作答。')) return;
    const token = requestGate.current.tryEnter('continuity-retry');
    if (!token) return;
    if (token.context.workspaceId !== workspaceId) {
      requestGate.current.finish(token);
      return;
    }
    try {
      const result = await retryLearningContinuity(
        token.context.workspaceId,
        continuity.transition_id
      );
      if (!requestGate.current.isCurrent(token)) return;
      setContinuity(result);
      setMessage('学习连续性重试已启动；普通源码问答仍可使用当前 revision。');
    } catch (error) {
      if (!requestGate.current.isCurrent(token)) return;
      setMessage(error instanceof Error ? error.message : '学习连续性重试失败');
    } finally {
      requestGate.current.finish(token, true);
    }
  }

  async function pollContinuity(nextWorkspaceId: string) {
    if (requestGate.current.getContext().workspaceId !== nextWorkspaceId) return;
    const token = requestGate.current.tryEnter('continuity-poll');
    if (!token) return;
    try {
      const result = await getLearningContinuity(token.context.workspaceId);
      if (!requestGate.current.isCurrent(token)) return;
      setContinuity(result);
      if (result.status === 'failed') {
        setMessage(`${result.error_message || '学习连续性失败'} 当前代码 snapshot 和普通问答仍可用；旧 mastery 未被沿用。`);
      } else if (result.status === 'succeeded') {
        setMessage('学习连续性已安全发布；修改目标已进入需要复习。');
      }
    } catch (error) {
      if (!requestGate.current.isCurrent(token)) return;
      setMessage(error instanceof Error ? error.message : '无法恢复学习连续性状态');
    } finally {
      requestGate.current.finish(token, true);
    }
  }

  async function pollUpdateRun(nextWorkspaceId: string, runId: string) {
    if (requestGate.current.getContext().workspaceId !== nextWorkspaceId) return;
    const token = requestGate.current.tryEnter('update-poll');
    if (!token) return;
    try {
      const run = await getWorkspaceUpdateRun(token.context.workspaceId, runId);
      if (!requestGate.current.isCurrent(token)) return;
      setUpdateRun(run);
      if (run.status === 'failed') {
        setMessage(`${run.error_message || '更新失败'} 旧 snapshot 仍可用。`);
      } else if (run.status === 'succeeded' && run.result === 'activated') {
        const opened = await openWorkspace(
          nextWorkspaceId,
          'selection',
          '新 revision 已完整验证并原子激活。',
          false
        );
        if (opened) void loadWorkspaceLibrary();
      } else if (run.status === 'succeeded') {
        setRevisionCheck(null);
        setMessage('仓库 revision 未变化，未创建 snapshot 或运行 Embedding。');
      }
    } catch (error) {
      if (!requestGate.current.isCurrent(token)) return;
      setMessage(error instanceof Error ? error.message : '无法恢复更新状态');
    } finally {
      requestGate.current.finish(token, true);
    }
  }

  async function handleAnalyze(event: FormEvent) {
    event.preventDefault();
    const token = beginContextChange('context', '');
    setAnalysisLoading(true);
    setMessage('正在抓取仓库并生成学习导读...');
    setAnswers([]);
    setAskError(null);
    const probe = beginConnectionProbe();
    try {
      const result = await analyzeProject(sourceType, source);
      if (!requestGate.current.retargetContext(token, result.workspace_id)) return;
      const workspace = await getWorkspace(result.workspace_id);
      const loaded = await loadAll(workspace.active_snapshot.project_id);
      if (
        !requestGate.current.commitContext(token, {
          workspaceId: workspace.workspace_id,
          projectId: workspace.active_snapshot.project_id,
          revision: workspace.active_snapshot.repository_revision
        })
      ) {
        return;
      }
      applyWorkspace(workspace, loaded);
      settleConnectionProbe(probe);
      setActiveTab('ask');
      await loadWorkspaceLibrary(token);
      if (requestGate.current.isActive(token)) {
        setMessage(result.import_action === 'reused' ? '已载入持久化项目和索引。' : '分析和本地索引完成。');
      }
    } catch (error) {
      if (!requestGate.current.isActive(token)) return;
      settleConnectionProbe(probe, error);
      requestGate.current.restoreContext(token);
      if (error instanceof ApiError && error.status === 0) {
        setMessage('');
      } else {
        setMessage(error instanceof Error ? error.message : '分析失败');
      }
    } finally {
      if (requestGate.current.finish(token)) setAnalysisLoading(false);
    }
  }

  async function handleAsk(event: FormEvent) {
    event.preventDefault();
    if (!projectId || !question.trim()) return;
    const context = requestGate.current.getContext();
    if (!context.projectId || context.projectId !== projectId) return;
    const token = requestGate.current.tryEnter('ask');
    if (!token) return;
    const submittedQuestion = question.trim();
    const submittedMode = executionMode;
    const connection = new AbortController();
    activeAskConnection.current = connection;
    const entryId = `${token.context.workspaceId}:${token.context.projectId}:${token.context.revision}:${token.requestId}`;
    setAskLoading(true);
    setAskError(null);
    setMessage('正在检索源码片段并生成回答...');
    setAnswers((current) => [...current, {
      id: entryId,
      status: 'pending',
      requestedMode: submittedMode,
      progress: [],
      question: submittedQuestion,
      clientRequestId: token.requestId,
      workspaceId: token.context.workspaceId,
      projectId: token.context.projectId,
      revision: token.context.revision
    }]);
    try {
      const result = await askProjectStream(token.context.projectId, token.context.revision, submittedQuestion, submittedMode, String(token.requestId), (progress) => {
        if (!requestGate.current.isCurrent(token)) return;
        setAnswers((current) => current.map((entry) => entry.id === entryId && entry.status === 'pending'
          ? { ...entry, progress: [...(entry.progress || []), progress].slice(-30) }
          : entry));
      }, connection.signal);
      if (!requestGate.current.isCurrent(token)) return;
      setAnswers((current) => current.map((entry) => entry.id === entryId
        ? { ...entry, status: 'success', result }
        : entry));
      setQuestion('');
      setMessage('回答已生成，注意查看引用文件。');
    } catch (error) {
      if (!requestGate.current.isCurrent(token)) return;
      const failure = error instanceof ApiError ? error.detail : null;
      setAnswers((current) => current.map((entry) => entry.id === entryId
        ? {
          ...entry,
          status: 'failure',
          failure: failure ?? undefined,
          errorMessage: error instanceof Error ? error.message : '问答失败，请检查服务后重试。'
        }
        : entry));
      // A structured failure already has a persistent, request-owned card.
      // A second transient banner sends the reader to a vague "diagnostic card".
      setMessage(failure ? '' : error instanceof ApiError ? error.message : '问答失败');
    } finally {
      if (activeAskConnection.current === connection) activeAskConnection.current = null;
      if (requestGate.current.finish(token, true)) setAskLoading(false);
    }
  }

  const activeContent = useMemo(() => {
    if (activeTab === 'manage') return <ManagementView
      workspaces={workspaces} libraryStatus={libraryStatus} workspaceId={workspaceId}
      currentRevision={currentRevision} updateState={updateState} revisionCheck={revisionCheck}
      updateRun={updateRun} continuity={continuity} continuityState={continuityState}
      sourceType={sourceType} source={source} configStatus={configStatus} analysisLoading={analysisLoading}
      workspaceLoading={workspaceLoading} deletingProjectId={deletingProjectId}
      onRefresh={() => void loadWorkspaceLibrary()} onOpenWorkspace={(id) => void openWorkspace(id)}
      onDelete={(item) => void handleDeleteWorkspace(item)} onCheckRevision={() => void handleCheckRevision()}
      onStartRefresh={() => void handleStartRefresh()} onRetryRefresh={() => void handleRetryRefresh()}
      onRetryContinuity={() => void handleRetryContinuity()} onSourceType={setSourceType} onSourceChange={setSource}
      onAnalyze={handleAnalyze}
    />;
    if (!hasProject || !project) return <EmptyState />;
    if (activeTab === 'dashboard') return <Dashboard project={project} />;
    if (activeTab === 'map') return <ProjectMapView key={`${workspaceId}:${projectId}:${currentRevision}`} projectId={projectId} revision={currentRevision} projectMap={projectMap} error={mapError} loading={workspaceLoading} onRetry={() => void retryMap()} />;
    if (activeTab === 'learning') return <LearningView steps={learningSteps} continuity={continuity} />;
    if (activeTab === 'report') return <ReportView markdown={report} />;
    return null;
  }, [activeTab, analysisLoading, configStatus, continuity, continuityState, currentRevision, deletingProjectId, hasProject, libraryStatus, mapError, project, projectId, projectMap, report, revisionCheck, source, sourceType, updateRun, updateState, workspaceId, workspaceLoading, workspaces]);

  return <Workbench project={project} workspaces={workspaces} workspaceId={workspaceId} revision={currentRevision} executionMode={executionMode} onExecutionModeChange={setExecutionMode}
    question={question} answers={answers} error={askError} loading={askLoading}
    libraryLoading={libraryStatus === 'loading'} projectLoading={workspaceLoading || analysisLoading}
    statusMessage={connectionError || message} activePage={activeTab}
    content={activeContent} onQuestionChange={setQuestion} onSubmit={handleAsk}
    onOpenWorkspace={(id) => void openWorkspace(id, 'selection', '已切换项目；回答和引用已清空。', false)}
    onRefreshLibrary={() => void loadWorkspaceLibrary()} onNavigate={setActiveTab} />;
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

function ManagementView({
  workspaces, libraryStatus, workspaceId, currentRevision, updateState, revisionCheck, updateRun,
  continuity, continuityState, sourceType, source, configStatus, analysisLoading, workspaceLoading,
  deletingProjectId, onRefresh, onOpenWorkspace, onDelete, onCheckRevision, onStartRefresh,
  onRetryRefresh, onRetryContinuity, onSourceType, onSourceChange, onAnalyze
}: {
  workspaces: WorkspaceSummary[]; libraryStatus: WorkspaceLibraryStatus; workspaceId: string;
  currentRevision: string; updateState: string; revisionCheck: WorkspaceRevisionCheck | null;
  updateRun: WorkspaceUpdateRun | null; continuity: LearningContinuity | null; continuityState: string;
  sourceType: SourceType; source: string; configStatus: ConfigStatus | null; analysisLoading: boolean;
  workspaceLoading: boolean; deletingProjectId: string; onRefresh: () => void;
  onOpenWorkspace: (id: string) => void; onDelete: (item: WorkspaceSummary) => void;
  onCheckRevision: () => void; onStartRefresh: () => void; onRetryRefresh: () => void;
  onRetryContinuity: () => void; onSourceType: (value: SourceType) => void;
  onSourceChange: (value: string) => void; onAnalyze: (event: FormEvent) => void;
}) {
  return <div className="wb-page wb-management">
    <section className="wb-page-heading"><p>项目管理</p><h1>已索引项目与本地分析</h1><span>来源和 revision 仅展示服务端已有记录；删除失败时会保留项目以便安全重试。</span></section>
    <section className="wb-panel" aria-label="项目库"><div className="wb-panel-head"><h2>项目库</h2><button type="button" onClick={onRefresh} disabled={libraryStatus === 'loading'}><RefreshCw />刷新列表</button></div>
      {libraryStatus === 'loading' && <p>正在读取已有项目…</p>}{libraryStatus === 'error' && <p className="wb-notice">项目库读取失败，可重试。</p>}{libraryStatus === 'empty' && <p>还没有已分析项目，请使用下方入口建立第一个项目。</p>}
      {libraryStatus === 'ready' && <div className="wb-workspace-table">{workspaces.map((item) => <article key={item.workspace_id} className={item.workspace_id === workspaceId ? 'is-current' : ''}><div><strong>{item.display_name}</strong><span>{item.source_type === 'local' ? '本地来源' : item.source_type === 'git_url' ? 'URL 来源' : '来源信息未提供'}</span></div><div><span>状态：{item.project_status}</span><span title={item.repository_revision}>revision：{item.repository_revision ? item.repository_revision.slice(0, 12) : '未提供'}</span><span>向量记录：{item.embedding_count ?? 0} / {item.total_chunks ?? 0}</span></div><div className="wb-row-actions"><button type="button" disabled={!item.openable || workspaceLoading || analysisLoading} onClick={() => onOpenWorkspace(item.workspace_id)}>打开</button><button type="button" className="wb-danger" aria-label={`删除 ${item.display_name}`} disabled={!item.project_id || deletingProjectId === item.project_id || analysisLoading} onClick={() => onDelete(item)}><Trash2 />删除</button></div></article>)}</div>}
    </section>
    {workspaceId && <section className={`wb-panel wb-update ${updateState}`} aria-label="revision 更新"><h2>Revision 更新</h2><p>当前：{currentRevision.slice(0, 12) || 'unknown'}</p>{revisionCheck?.state === 'update_available' && <p>可用：{revisionCheck.available_revision.slice(0, 12)}</p>}{updateState === 'updating' && <p>阶段：{updateRun?.phase}</p>}{updateState === 'failed' && <p className="wb-notice">{updateRun?.error_code || 'update_failed'}</p>}<div className="wb-row-actions"><button type="button" onClick={onCheckRevision} disabled={updateState === 'updating'}>检查更新</button>{updateState === 'update-available' && <button type="button" onClick={onStartRefresh}>确认更新</button>}{updateState === 'failed' && updateRun?.retryable && <button type="button" onClick={onRetryRefresh}>安全重试</button>}</div></section>}
    {workspaceId && continuityState !== 'not-required' && continuity && <section className="wb-panel" aria-label="学习连续性"><h2>跨版本学习连续性</h2><p>状态：{continuityState}</p>{continuityState === 'failed' && <p className="wb-notice">{continuity.error_code || 'continuity_failed'}；旧 mastery 未用于当前 revision。</p>}{continuityState === 'failed' && continuity.retryable && <button type="button" onClick={onRetryContinuity}>重试学习连续性</button>}</section>}
    <section className="wb-panel"><h2>导入项目</h2><form className="analyze-form wb-import-form" onSubmit={onAnalyze}><label htmlFor="repo-source">仓库来源</label><div className="source-toggle" role="group" aria-label="仓库来源"><button type="button" className={sourceType === 'local' ? 'active' : ''} onClick={() => onSourceType('local')}>本地目录</button><button type="button" className={sourceType === 'git_url' ? 'active' : ''} onClick={() => onSourceType('git_url')}>公开 HTTPS Git</button></div><input id="repo-source" value={source} onChange={(event) => onSourceChange(event.target.value)} placeholder={sourceType === 'local' ? 'D:\\Project\\my-python-repo' : 'https://host/owner/repo.git'} required /><button type="submit" disabled={analysisLoading}>{analysisLoading ? <RefreshCw className="spin" /> : <GitBranch />}<span>{analysisLoading ? '分析中' : '开始分析'}</span></button></form></section>
    <section className="wb-panel wb-provider"><strong>{providerSummary(configStatus)}</strong><span>BGE-M3：{configStatus ? `${configStatus.embedding.model} / ${configStatus.embedding.device} / ${configStatus.embedding.offline ? 'offline' : 'online'}${configStatus.embedding.ready ? '' : `；缺少 ${configStatus.embedding.missing.join(', ')}`}` : '读取中'}</span><small>API Key 只保存在后端根目录 .env，不会发送到浏览器或写入项目数据库。</small></section>
  </div>;
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

function LearningView({ steps, continuity }: { steps: LearningStep[]; continuity: LearningContinuity | null }) {
  return (
    <div className="learning-list">
      {continuity && continuity.status !== 'not_required' && (
        <section className="panel continuity-summary">
          <h2>跨版本学习连续性</h2>
          {continuity.status === 'succeeded' ? (
            <p>
              已安全保留 {continuity.stats.retained} 个严格等价目标；
              {continuity.stats.needs_review} 个目标需要复习；
              {continuity.stats.history_only} 个已删除目标仅保留历史；
              {continuity.stats.not_inherited} 个歧义或不兼容目标未继承。
            </p>
          ) : continuity.status === 'failed' ? (
            <p>连续性处理失败。当前代码和源码问答仍可用，但学习入口不会静默沿用旧 mastery。</p>
          ) : (
            <p>连续性尚未完成。请等待或在失败后显式重试；系统不会自动伪造学习完成。</p>
          )}
          {continuity.stats.needs_review > 0 && <strong>请优先复习受 revision 修改影响的目标。</strong>}
        </section>
      )}
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

export function AskView({
  question,
  setQuestion,
  answers,
  error,
  onSubmit,
  loading
}: {
  question: string;
  setQuestion: (value: string) => void;
  answers: WorkbenchAnswer[];
  error: AskFailure | null;
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
        {error && (
          <article className="answer ask-error" role="alert">
            <h2>问答未生成可验证答案</h2>
            <p>失败阶段：{error.diagnostics.failure_stage}</p>
            <p>失败码：{error.diagnostics.failure_reason_code}</p>
            <p>请求 ID：{error.diagnostics.request_id}</p>
            <p>Evidence 数量：{error.diagnostics.evidence_count}</p>
            <p>{error.retryable ? '建议稍后安全重试。' : '当前不建议原样重试，请先核对失败阶段。'}</p>
          </article>
        )}
        {answers.map((item, index) => (
          <article className="answer" key={`${item.question}-${index}`}>
             <h2>{item.question}</h2>
             <p>{item.result?.answer || '回答仍在处理中或未能生成。'}</p>
            <p className="citation-trust-note">
              源码引用已校验；生成解释仍可能有误，请结合引用核对。
            </p>
             <div className="citation-grid">
              {(item.result?.citations || []).map((citation, citationIndex) => (
                <details
                  key={`${citation.path}:${citation.start_line}-${citation.end_line}:${citation.qualified_name}:${citationIndex}`}
                  open
                >
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
    anchor.download = 'reponoesis-report.md';
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
