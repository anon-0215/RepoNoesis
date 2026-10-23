import { useEffect, useLayoutEffect, useRef, useState, type FormEvent, type KeyboardEvent, type ReactNode } from 'react';
import { ChevronRight, Clipboard, Code2, FileCode2, FolderCog, FolderOpen, Map, Menu, Moon, Search, Send, Sun, X } from 'lucide-react';
import type { AskFailure, AskProgressEvent, ChatAnswer, ChatHistoryRecord, ReadableCitation, ExecutionMode, ProjectResponse, WorkspaceSummary } from '../types';
import { citationLabel } from '../lib/product';
import { CodeViewer, type CodeSource } from './CodeViewer';
import { MarkdownAnswer } from './MarkdownAnswer';
import { evidenceBindings, savedCitationBindings, type EvidenceBinding } from './evidenceBindings';
import { SourceExploration } from './SourceExploration';
import { ProjectMaterials } from './ProjectMaterials';
import { useConversationScroll } from './useConversationScroll';
import { ExecutionSummary } from './ExecutionSummary';
import { SourceFileContext, SourceFileReader, type SourceFileSelection } from './SourceFileReader';

export interface WorkbenchAnswer {
  id?: string;
  recordId?: string;
  createdAt?: string;
  history?: ChatHistoryRecord;
  status?: 'pending' | 'success' | 'failure';
  question: string;
  result?: ChatAnswer;
  failure?: AskFailure;
  errorMessage?: string;
  clientRequestId?: number;
  workspaceId?: string;
  projectId?: string;
  revision?: string;
  requestedMode?: ExecutionMode;
  progress?: AskProgressEvent[];
}
export type WorkbenchPage = 'manage' | 'dashboard' | 'map' | 'learning' | 'ask' | 'report' | 'exploration';
interface WorkbenchProps { contextGeneration?: number; history?: { loading: boolean; error: string; loaded: boolean; waiting: boolean; cursor: string | null; retry: () => void; loadEarlier: () => void }; project: ProjectResponse | null; workspaces: WorkspaceSummary[]; workspaceId: string; revision: string; question: string; answers: WorkbenchAnswer[]; error: AskFailure | null; loading: boolean; libraryLoading: boolean; projectLoading: boolean; statusMessage: string; activePage: WorkbenchPage; content: ReactNode; executionMode?: ExecutionMode; onExecutionModeChange?: (mode: ExecutionMode) => void; onQuestionChange: (value: string) => void; onSubmit: (event: FormEvent) => void; onOpenWorkspace: (workspaceId: string) => void; onRefreshLibrary: () => void; onNavigate: (page: WorkbenchPage) => void; }
const THEME_KEY = 'reponoesis.workbench.theme';
const examples = ['这个仓库的主要入口在哪里？', '主要模块分别负责什么？', '请解释一个函数的实现，并给出源码引用。'];
const actionable = (message: string) => /失败|无法|错误|超时|未完成|不可|证据不足/.test(message);
const copyText = (value: string) => void navigator.clipboard?.writeText(value).catch(() => undefined);
const citationRange = (citation: ReadableCitation) => validCitationRange(citation) ? `行 ${citation.start_line}–${citation.end_line}` : '行范围未提供或无效';
const validCitationRange = (citation: ReadableCitation) => Number.isInteger(citation.start_line) && Number.isInteger(citation.end_line) && citation.start_line! > 0 && citation.end_line! >= citation.start_line!;
const validCitationPath = (path?: string | null) => Boolean(path && !/^[\\/]|^[a-zA-Z]:|\0/.test(path) && path.split(/[\\/]/).every(part => part && part !== '.' && part !== '..'));
const historyFileReason: Record<string, string> = { history_identity_unreliable: '缺少可靠的项目或版本身份，无法打开全文。', history_metadata_invalid: '保存的恢复信息损坏或引用绑定冲突，无法打开全文。', history_snapshot_old: '该记录属于旧 snapshot，本轮不支持该版本的全文；可阅读保存片段。' };
interface CitationSelection { answerId: string; evidenceId: string | null; citations: ReadableCitation[]; index: number | null; projectId?: string; revision?: string; workspaceId?: string; fullFileReason?: string; historical?: boolean }

function CitationDrawer({ selection, revision, onClose, onSelect, onExpand, onFullFile, fullOpen, modal }: { selection: CitationSelection; onFullFile: (citation: ReadableCitation, trigger: HTMLButtonElement) => void; fullOpen: boolean; modal: boolean; revision: string; onClose: () => void; onSelect: (index: number) => void; onExpand: (source: CodeSource, trigger: HTMLButtonElement) => void }) {
  const closeButton = useRef<HTMLButtonElement>(null);
  const dialog = useRef<HTMLElement>(null);
  const citation = selection.index === null ? null : selection.citations[selection.index];
  const source: CodeSource | null = citation ? { code: citation.snippet ?? '', label: citation.qualified_name || '源码片段', path: citation.path ?? undefined, range: citationRange(citation) } : null;
  const unavailable = selection.fullFileReason || (selection.historical && citation && (!validCitationPath(citation.path) || !validCitationRange(citation)) ? '引用路径或行范围未提供或无效，无法定位全文。' : undefined);
  useEffect(() => { closeButton.current?.focus({ preventScroll: true }); }, [selection.answerId, selection.evidenceId]);
  useEffect(() => {
    if (!modal || fullOpen) return;
    const trap = (event: globalThis.KeyboardEvent) => {
      if (event.key !== 'Tab' || !dialog.current) return;
      const controls = Array.from(dialog.current.querySelectorAll<HTMLElement>('button:not(:disabled),[tabindex="0"]'));
      const first = controls[0], last = controls[controls.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus({ preventScroll: true }); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus({ preventScroll: true }); }
    };
    document.addEventListener('keydown', trap);
    return () => document.removeEventListener('keydown', trap);
  }, [modal, fullOpen]);
  return <aside ref={dialog} hidden={fullOpen} className="wb-drawer" aria-label="引用源码片段" aria-modal={modal || undefined} role="dialog"><div className="wb-drawer-head"><div><p className="wb-eyebrow">{selection.historical ? "保存引用 " : "当前引用 "}{selection.evidenceId ? `[${selection.evidenceId}]` : ''}</p><h2>{source?.label || '选择证据片段'}</h2></div><button ref={closeButton} className="wb-icon-button" type="button" onClick={onClose} aria-label="关闭源码栏"><X /></button></div>{selection.citations.length > 1 && <div className="wb-fragment-list" aria-label="选择证据片段">{selection.citations.map((part, index) => <button key={`${part.path}:${part.start_line}:${part.end_line}:${index}`} type="button" aria-pressed={selection.index === index} onClick={() => onSelect(index)}>{part.path} · {citationRange(part)}</button>)}</div>}{citation && source ? <><div className="wb-source-meta"><span>{citation.path || '路径未提供'}</span><span>{citationRange(citation)}</span>{revision && <span title={revision}>revision {revision.slice(0, 12)}</span>}</div><div className="wb-drawer-actions"><button type="button" disabled={Boolean(unavailable)} onClick={(event) => onFullFile(citation, event.currentTarget)}>查看完整文件</button>{unavailable && <p role="status">{unavailable}</p>}</div>{citation.snippet == null && <p className="wb-drawer-note">保存片段未提供。</p>}<CodeViewer key={`${selection.answerId}:${selection.evidenceId}:${selection.index}`} source={source} onExpand={(trigger) => onExpand(source, trigger)} /><p className="wb-drawer-note">高亮依据引用片段的行范围，不代表每句话的精确行。</p></> : <p className="wb-drawer-note">此证据包含多个片段，请选择要阅读的片段。</p>}</aside>;
}

function CodeModal({ source, onClose }: { source: CodeSource; onClose: () => void }) {
  const closeButton = useRef<HTMLButtonElement>(null); const dialog = useRef<HTMLElement>(null);
  useEffect(() => { closeButton.current?.focus(); const onKeyDown = (event: globalThis.KeyboardEvent) => { if (event.key === 'Escape') { event.preventDefault(); onClose(); return; } if (event.key !== 'Tab' || !dialog.current) return; const controls = Array.from(dialog.current.querySelectorAll<HTMLElement>('button,[href],[tabindex]:not([tabindex="-1"])')).filter((item) => !item.hasAttribute('disabled')); if (!controls.length) return; const first = controls[0]; const last = controls[controls.length - 1]; if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); } if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); } }; document.addEventListener('keydown', onKeyDown); return () => document.removeEventListener('keydown', onKeyDown); }, [onClose]);
  return <><button className="wb-code-modal-backdrop" type="button" aria-label="关闭放大阅读" onClick={onClose} /><section ref={dialog} className="wb-code-modal" role="dialog" aria-modal="true" aria-label="放大阅读代码"><header className="wb-code-modal-head"><div><p>{source.path || '回答中的代码块'}{source.range ? ` · ${source.range}` : ''}</p><h2>{source.label}</h2></div><button ref={closeButton} className="wb-icon-button" type="button" onClick={onClose} aria-label="关闭放大阅读"><X /></button></header><CodeViewer source={source} /></section></>;
}

const baseProgressLabels: Record<string, string> = {
  succeeded: '基础检索完成', zero_hit: '基础检索未命中',
  all_rejected: '基础检索候选均未通过验证', failed: '基础检索失败',
  rejected: '基础检索被拒绝', timed_out: '基础检索超时',
  cancelled: '基础检索已取消', deadline_exceeded: '基础检索请求超时'
};

function progressForAnswer(item: WorkbenchAnswer): AskProgressEvent[] {
  const entries = (item.progress || []).filter((event) => !(
    item.status === 'failure' && item.failure?.diagnostics.evidence_count === 0 &&
    event.type === 'citation_checked' && event.passed === true
  ));
  if (item.status === 'pending') return entries.slice(-5);
  return entries;
}

function validationProgressLabel(event: AskProgressEvent): string {
  if (event.type === 'citation_checked') {
    if (event.checkpoint === 'initial_evidence') return '初始证据身份校验完成';
    if (event.checkpoint === 'generation_input') return '生成输入证据校验完成';
    if (event.checkpoint === 'post_answer_evidence') return '回答后证据复核完成';
    return event.phase === 'before_answer' ? '证据引用预检完成' : event.phase === 'after_answer' ? '生成后引用校验完成' : '引用校验完成（阶段未注明）';
  }
  if (event.checkpoint === 'initial_evidence') return '初始关系证据校验完成';
  if (event.checkpoint === 'post_answer_evidence') return '回答后关系证据复核完成';
  return event.phase === 'before_answer' ? '关系证据预检完成' : event.phase === 'after_answer' ? '生成后关系校验完成' : '关系校验完成（阶段未注明）';
}

function answerWarnings(result: { warnings?: string[] | null; execution_summary?: ChatAnswer['execution_summary'] }): string[] {
  return (result.warnings ?? []).map((warning) =>
    warning === 'Semantic retrieval did not complete; continuing with lexical code-chunk candidates.' &&
    result.execution_summary?.base_retrieval?.new_evidence_count
      ? '语义检索未完成，本次基于词法检索证据回答。'
      :
    warning === 'Dense retrieval did not complete; continuing with lexical and symbol candidates.' &&
    result.execution_summary?.base_retrieval?.new_evidence_count
      ? '语义检索未完成，本次基于已验证的词法或符号证据回答。'
      :
    warning === 'Agent stopped after consecutive no-progress steps.' &&
    result.execution_summary?.status === 'completed' &&
    result.execution_summary.planner_termination_reason === 'no_progress'
      ? '补充检索连续未增加证据，已基于现有证据完成回答。'
      : warning
  );
}

function baseTimeoutCausedInsufficiency(item: WorkbenchAnswer): boolean {
  const diagnostics = item.failure?.diagnostics;
  return diagnostics?.failure_stage === 'retrieval' &&
    diagnostics.failure_reason_code === 'evidence_insufficient' &&
    diagnostics.base_retrieval?.attempted === true &&
    diagnostics.base_retrieval.status === 'timed_out';
}

function OutcomeNotices({ result, historical }: { result: { execution_summary?: ChatAnswer['execution_summary'] }; historical: boolean }) {
  const summary = result.execution_summary;
  if (!summary) return null;
  const statusLabels: Record<string, string> = { degraded: '降级完成', insufficient_evidence: '证据不足', budget_exhausted: '预算用尽', tool_budget_exhausted: '工具预算用尽', final_answer_failed: '回答生成失败', failed: '失败', cancelled: '已取消' };
  const notices = [
    summary.status && summary.status !== 'completed' ? (statusLabels[summary.status] || `执行状态：${summary.status}`) : null,
    summary.citation_validation_passed === false ? '引用身份与范围校验未通过' : null,
    summary.relation_validation_passed === false ? '关系校验未通过' : null,
    summary.post_generation_validation_passed === false ? '生成后证据校验未通过' : null,
  ].filter(Boolean);
  return notices.length ? <p className="wb-outcome">{historical ? '生成时记录：' : '本次状态：'}{notices.join('；')}</p> : null;
}

function HistoryMetadata({ history, revision }: { history: ChatHistoryRecord; revision?: string }) {
  const metadata = history.recovery_metadata;
  const mode = metadata?.execution_mode;
  return <>
    <div className="wb-history-meta">
      <time title={history.created_at || undefined}>{history.created_at || '保存时间未提供'}</time>
      <span>模式：{mode === 'rag' ? 'RAG' : mode === 'agent' ? 'Agent' : '未提供'}</span>
      <span>{history.identity_status !== 'reliable' ? '版本未确认' : history.is_current_snapshot ? '当前版本' : `旧版本 ${revision?.slice(0, 12) || '未提供'}`}</span>
    </div>
    {history.identity_status !== 'reliable' && <p className="wb-warnings">版本身份缺失或冲突，引用联动不可用。</p>}
    {['invalid', 'binding_conflict'].includes(history.metadata_status) && <p className="wb-warnings">恢复元数据损坏或引用绑定冲突；执行信息未提供。</p>}
    {history.metadata_status === 'missing' && <p className="wb-warnings">生成时执行信息未提供。</p>}
    {history.citations_status !== 'available' && <p className="wb-warnings">部分保存引用不可读取。</p>}
    {metadata?.metadata_truncated && <p className="wb-warnings">保存的恢复信息已裁剪。</p>}
    <details className="wb-history-details"><summary>记录详情</summary><div>
      <p>完整 revision：{revision || '未提供'}</p>
      <p>{history.revision_source === 'snapshot_link' ? '版本来自原快照关联' : history.revision_source === 'saved' ? '版本来自保存记录' : '版本身份未提供'}</p>
      <p>项目：{history.project_id || '未提供'} · 保存记录：{history.record_id}</p>
      <p>回答方式：{metadata?.answer_mode === 'llm_grounded' ? '生成时记录为基于证据生成' : metadata?.answer_mode === 'deterministic' ? '生成时记录为确定性回答' : '未提供'} · 警告信息：{metadata?.warnings == null ? '未提供' : metadata.warnings.length ? '见保存警告' : '生成时记录为空'}</p>
      <p>仅展示保存时的信息；历史读取没有重新执行或校验答案。</p>
    </div></details>
  </>;
}

function AnswerCard({ item, selected, onCitation, onExpandCode }: { item: WorkbenchAnswer; selected: CitationSelection | null; onCitation: (binding: EvidenceBinding | null, citation: ReadableCitation, index: number | null, trigger: HTMLButtonElement) => void; onExpandCode: (source: CodeSource, trigger: HTMLButtonElement) => void }) {
  const history = item.history;
  const metadata = history?.recovery_metadata;
  const result = history ? { answer: history.answer, citations: history.citations, warnings: metadata?.warnings,
    execution_mode: metadata?.execution_mode, execution_summary: metadata?.execution_summary,
    grounding_status: metadata?.grounding_status } : item.result;
  const bindings = history ? savedCitationBindings(history.citations, metadata ?? null, item.projectId, item.revision,
    history.identity_status === 'reliable' && history.metadata_status === 'available')
    : item.result ? evidenceBindings(item.result, item.projectId, item.revision) : new globalThis.Map<string, EvidenceBinding>();
  const mode = item.status === 'success' ? result?.execution_mode : item.requestedMode;
  const modeText = history ? '生成时记录的模式' : item.status === 'success' ? '执行模式' : '请求模式';
  const labels: Record<string, string> = { request_received: '请求已接收', base_started: '基础检索开始', base_completed: '基础检索完成', planner_started: '判断是否需要补充证据', planner_stopped: '补充阶段结束', tool_completed: '补充工具已执行', answer_started: '答案生成开始', citation_checked: '引用校验完成', relation_checked: '关系校验完成' };
  const progress = progressForAnswer(item);
  const progressPanel = progress.length > 0 ? <section className="wb-progress" aria-label="本次执行进度"><strong>执行进度 · {item.status === 'failure' ? '请求失败' : item.status === 'success' ? '请求已结束' : '执行中'}</strong><ol>{progress.map((event) => <li key={event.sequence}>{event.type === "citation_checked" || event.type === "relation_checked" ? validationProgressLabel(event) : event.type === 'base_completed' ? (baseProgressLabels[event.status || ''] || '基础检索结束（状态未提供）') : labels[event.type]}{event.type === 'base_completed' && ['succeeded', 'zero_hit', 'all_rejected'].includes(event.status || '') ? ` · 新增证据 ${event.new_evidence_count ?? '未提供'}` : ''}{event.type === 'tool_completed' ? ` · ${event.status === 'succeeded' ? '成功' : '未成功'}` : ''}{event.type === 'citation_checked' || event.type === 'relation_checked' ? ` · ${event.passed === true ? '通过' : event.passed === false ? '未通过' : '结果未提供'}` : ''}{event.type === 'planner_stopped' && event.reason && event.reason !== 'planner_answer' ? ' · 已按当前证据继续' : ''}</li>)}</ol></section> : null;
  const requestContext = <>{!history && <details className="wb-request-details"><summary>请求详情</summary><small className="wb-request-context">{modeText}：{mode === 'rag' ? 'RAG' : mode === 'agent' ? 'Agent' : '未提供'}{item.clientRequestId ? ` · 本地请求 #${item.clientRequestId}` : ''} · {item.revision ? item.revision.slice(0, 12) : 'revision 未提供'}</small></details>}{item.status === 'success' && progressPanel ? <details className="wb-process"><summary>执行过程 · 请求已结束</summary>{progressPanel}</details> : progressPanel}</>;
  if (item.status === 'pending') return <article className="wb-answer" data-answer-id={item.id}><div className="wb-question"><span>你</span><p>{item.question}</p></div><Waiting label={labels[progress[progress.length - 1]?.type] || '正在处理源码问题'} />{requestContext}</article>;
  if (item.status === 'failure' || !result) return <article className="wb-answer" data-answer-id={item.id}><div className="wb-question"><span>你</span><p>{item.question}</p></div><section className="wb-error" role="alert"><h2>{baseTimeoutCausedInsufficiency(item) ? '基础检索超时' : item.failure?.diagnostics.agent_status === 'insufficient_evidence' ? '证据不足' : '问答未生成可验证答案'}</h2>{baseTimeoutCausedInsufficiency(item) && <p>基础检索未在时限内完成，本次未形成足够的可验证证据；这不表示源码没有相关内容。</p>}{item.failure ? <><p>失败阶段：{item.failure.diagnostics.failure_stage} · {item.failure.diagnostics.failure_reason_code}</p><p>请求 ID：{item.failure.diagnostics.request_id}</p></> : <p>{item.errorMessage || '问答失败，请检查服务后重试。'}</p>}</section><ExecutionSummary failure={item.failure} mode={item.requestedMode} />{requestContext}</article>;
  return <article className="wb-answer" data-answer-id={item.id}><div className="wb-question"><span>你</span><p>{item.question}</p></div><div className="wb-answer-head"><Code2 /><div><strong>RepoNoesis</strong><small>{history ? `历史记录 · 生成时状态：${result.grounding_status === 'grounded' ? 'grounded' : result.grounding_status === 'degraded' ? 'degraded' : result.grounding_status === 'insufficient_evidence' ? '证据不足' : '未提供'}` : result.grounding_status === 'grounded' ? '源码引用已校验' : result.grounding_status === 'insufficient_evidence' ? '证据不足' : result.grounding_status === 'degraded' ? '降级回答' : '引用校验状态未提供'}</small></div><button type="button" onClick={() => copyText(result.answer)}><Clipboard />复制回答</button></div>{history && <HistoryMetadata history={history} revision={item.revision} />}<OutcomeNotices result={result} historical={Boolean(history)} /><MarkdownAnswer text={result.answer} bindings={bindings} selectedEvidenceId={selected?.answerId === item.id ? selected?.evidenceId : null} onEvidence={(id, trigger) => { const binding = bindings.get(id); if (binding) onCitation(binding, binding.citations[0], binding.citations.length === 1 ? 0 : null, trigger); }} onExpandCode={onExpandCode} />{(result.warnings?.length ?? 0) > 0 && <p className="wb-warnings">{answerWarnings(result).join('；')}</p>}<div className="wb-citations" aria-label={`第 ${item.clientRequestId || 1} 条回答的引用`}><span>本回答引用</span>{result.citations.length === 0 && <small>服务端未返回引用。</small>}{result.citations.map((citation, displayIndex) => { const binding = citation.evidence_id ? bindings.get(citation.evidence_id) : undefined; const partIndex = binding?.indices.indexOf(displayIndex) ?? -1; return <button key={`${citation.path}:${citation.start_line}:${citation.end_line}:${displayIndex}`} type="button" aria-pressed={selected?.answerId === item.id && selected?.index === partIndex && selected?.evidenceId === (binding?.evidenceId || null)} onClick={(event) => onCitation(binding || null, citation, binding ? partIndex : 0, event.currentTarget)}><b>{binding ? `[${binding.evidenceId}]` : `片段 ${displayIndex + 1}`}</b>{citation.qualified_name || (citation.path ? `${citation.path} · ${citationRange(citation)}` : '路径未提供')}<ChevronRight /></button>; })}</div><ExecutionSummary result={result} historical={Boolean(history)} />{requestContext}</article>;
}

function Notification({ message }: { message: string }) { const [dismissed, setDismissed] = useState(''); const persistent = actionable(message); useEffect(() => { setDismissed(''); if (!message || persistent) return; const timer = window.setTimeout(() => setDismissed(message), 5000); return () => window.clearTimeout(timer); }, [message, persistent]); if (!message || dismissed === message) return null; return <div className={`wb-notification status-line ${persistent ? 'is-actionable' : ''}`} role={persistent ? 'alert' : 'status'}><span>{message}</span><button type="button" aria-label="关闭提示" onClick={() => setDismissed(message)}><X /></button></div>; }
function Waiting({ label }: { label: string }) { const [seconds, setSeconds] = useState(0); useEffect(() => { const started = Date.now(); const timer = window.setInterval(() => setSeconds(Math.floor((Date.now() - started) / 1000)), 250); return () => window.clearInterval(timer); }, []); return <div className="wb-waiting"><span /><div><strong>{label}</strong><p>已等待 {seconds} 秒</p></div></div>; }

export function Workbench(props: WorkbenchProps) {
  const [theme, setTheme] = useState<'light' | 'dark'>(() => { try { return window.localStorage.getItem(THEME_KEY) === 'dark' ? 'dark' : 'light'; } catch { return 'light'; } });
  const [fileSelection, setFileSelection] = useState<SourceFileSelection | null>(null);
  const fileOpenId = useRef(0);
  const fileTrigger = useRef<HTMLButtonElement | null>(null);
  const [railOpen, setRailOpen] = useState(false); const [selectedCitation, setSelectedCitation] = useState<CitationSelection | null>(null); const [expandedCode, setExpandedCode] = useState<CodeSource | null>(null);
  const readerTrigger = useRef<HTMLButtonElement | null>(null); const composing = useRef(false); const textarea = useRef<HTMLTextAreaElement>(null); const scrollHost = useRef<HTMLElement>(null);
  const noProject = !props.project; const askPage = props.activePage === 'ask'; const selectedWorkspace = props.workspaces.find((item) => item.workspace_id === props.workspaceId); const projectName = props.project?.project.repo || selectedWorkspace?.display_name || '未选择项目'; const source = props.project?.project.source_location || props.project?.project.repo_url || '来源信息未提供'; const firstPrompt = askPage && !noProject && props.answers.length === 0 && !props.loading && !props.error && (!props.history || props.history.loaded) && !props.history?.loading;
  useEffect(() => { document.documentElement.dataset.theme = theme; try { window.localStorage.setItem(THEME_KEY, theme); } catch {} }, [theme]);
  useEffect(() => { setSelectedCitation(null); setExpandedCode(null); setFileSelection(null); }, [props.workspaceId, props.project?.project.id, props.revision, props.projectLoading, props.contextGeneration]);
  useEffect(() => { const handler = (event: globalThis.KeyboardEvent) => { if (event.key !== 'Escape' || event.defaultPrevented || fileSelection) return; if (expandedCode) { event.preventDefault(); closeCode(); } else if (selectedCitation) { event.preventDefault(); closeCitation(); } else if (railOpen) setRailOpen(false); }; document.addEventListener('keydown', handler); return () => document.removeEventListener('keydown', handler); });
  useEffect(() => { if (textarea.current) { textarea.current.style.height = 'auto'; textarea.current.style.height = `${Math.min(textarea.current.scrollHeight, 144)}px`; } }, [props.question]);
  useLayoutEffect(() => {
    const host = scrollHost.current;
    if (!host) return;
    const measure = () => host.closest<HTMLElement>('.wb-main')?.style.setProperty('--wb-scroll-gutter', `${host.offsetWidth - host.clientWidth}px`);
    measure();
    if (typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(measure); observer.observe(host);
    return () => observer.disconnect();
  }, []);
  const [narrow, setNarrow] = useState(() => window.matchMedia?.('(max-width: 1030px)').matches ?? false);
  useEffect(() => {
    const media = window.matchMedia?.('(max-width: 1030px)');
    if (!media) return;
    const update = () => setNarrow(media.matches);
    media.addEventListener('change', update);
    return () => media.removeEventListener('change', update);
  }, []);
  // Markdown markers can remount on selection changes. Resolve by answer and
  // entry position instead of retaining a detached DOM button or scrollTop.
  const citationAnchor = useRef<{ id: string; entry: number; offset: number } | null>(null);
  const restoreCitation = useRef(false);
  const focusCitation = useRef(false);
  const citationEntries = (card: HTMLElement) => Array.from(card.querySelectorAll<HTMLButtonElement>('.wb-evidence-marker,.wb-citations button'));
  function resolveCitationTrigger() {
    const anchor = citationAnchor.current;
    const card = Array.from(scrollHost.current?.querySelectorAll<HTMLElement>('[data-answer-id]') || []).find(node => node.dataset.answerId === anchor?.id);
    return card && anchor ? citationEntries(card)[anchor.entry] : null;
  }
  function captureCitation(trigger: HTMLButtonElement) {
    const card = trigger.closest<HTMLElement>('[data-answer-id]');
    const host = scrollHost.current;
    if (!card?.dataset.answerId || !host) return;
    citationAnchor.current = { id: card.dataset.answerId, entry: citationEntries(card).indexOf(trigger), offset: trigger.getBoundingClientRect().top - host.getBoundingClientRect().top };
    restoreCitation.current = true;
  }
  useLayoutEffect(() => {
    if (!restoreCitation.current) return;
    const host = scrollHost.current, anchor = citationAnchor.current, trigger = resolveCitationTrigger();
    if (host && anchor && trigger) {
      host.scrollTop += trigger.getBoundingClientRect().top - host.getBoundingClientRect().top - anchor.offset;
      if (focusCitation.current) trigger.focus({ preventScroll: true });
    }
    restoreCitation.current = false;
    focusCitation.current = false;
  }, [selectedCitation, fileSelection]);
  const reading = useConversationScroll(scrollHost, props.answers,
    `${props.contextGeneration ?? 0}:${props.workspaceId}:${props.project?.project.id || ''}:${props.revision}`,
    askPage, !props.history || props.history.loaded, props.projectLoading);
  function openFile(selection: SourceFileSelection, trigger: HTMLButtonElement) {
    const citationEntry = resolveCitationTrigger();
    if (citationEntry) captureCitation(citationEntry);
    fileTrigger.current = trigger;
    setExpandedCode(null);
    const identityError = selection.identityError || (!selection.projectId || !selection.revision || !selection.workspaceId
      ? '缺少可靠的项目或版本身份，无法读取完整文件。'
      : selection.projectId !== props.project?.project.id || selection.revision !== props.revision || selection.workspaceId !== props.workspaceId || props.projectLoading
        ? '版本不符：该引用不属于当前已激活快照，请重新打开项目。' : undefined);
    setFileSelection({ ...selection, identityError, openId: ++fileOpenId.current });
  }
  function closeFile() {
    const citationEntry = resolveCitationTrigger();
    if (citationEntry) captureCitation(citationEntry);
    setFileSelection(null);
    const trigger = fileTrigger.current;
    window.setTimeout(() => { if (trigger?.isConnected) trigger.focus({ preventScroll: true }); }, 0);
  }
  function closeCitation() {
    const trigger = resolveCitationTrigger();
    if (trigger) captureCitation(trigger);
    focusCitation.current = true;
    setSelectedCitation(null);
  }
  function openCode(sourceToOpen: CodeSource, trigger: HTMLButtonElement) { setFileSelection(null); readerTrigger.current = trigger; setExpandedCode(sourceToOpen); }
  function closeCode() { setExpandedCode(null); window.setTimeout(() => readerTrigger.current?.focus(), 0); }
  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) { if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing && !composing.current) { event.preventDefault(); if (!props.loading && props.question.trim()) event.currentTarget.form?.requestSubmit(); } }
  function navigate(page: WorkbenchPage) { setFileSelection(null); setRailOpen(false); setSelectedCitation(null); setExpandedCode(null); props.onNavigate(page); }
  function useExample(value: string) { props.onQuestionChange(value); window.setTimeout(() => textarea.current?.focus(), 0); }
  function submitAndReveal(event: FormEvent) { props.onSubmit(event); }
  const composer = <form className="ask-form wb-composer" onSubmit={submitAndReveal}><div>{askPage && reading.paused && props.answers.length > 0 && <button type="button" className="wb-latest" onClick={reading.revealLatest}>回到最新</button>}<textarea ref={textarea} value={props.question} disabled={noProject || props.projectLoading} aria-label="源码问题" onChange={(event) => props.onQuestionChange(event.target.value)} onCompositionStart={() => { composing.current = true; }} onCompositionEnd={() => { composing.current = false; }} onKeyDown={onKeyDown} placeholder={noProject ? '请先选择项目' : '输入你的源码问题'} rows={1} /><div className="wb-composer-foot"><div className="wb-mode-switch" role="group" aria-label="问答模式"><button type="button" aria-pressed={(props.executionMode || "agent") === "rag"} disabled={props.loading} onClick={() => props.onExecutionModeChange?.("rag")}>RAG</button><button type="button" aria-pressed={(props.executionMode || "agent") === "agent"} disabled={props.loading} onClick={() => props.onExecutionModeChange?.("agent")}>Agent</button></div><small>Enter 发送 · Shift+Enter 换行</small><button type="submit" disabled={noProject || props.projectLoading || props.loading || !props.question.trim()}><Send />发送</button></div></div></form>;
  const nav = (page: WorkbenchPage, label: string, Icon: typeof Search, enabled = true) => <button className={props.activePage === page ? 'is-active' : ''} type="button" onClick={() => navigate(page)} disabled={!enabled}><Icon />{label}</button>;
  return <SourceFileContext.Provider value={openFile}><div className={`wb-frame ${selectedCitation && !fileSelection ? 'source-open' : ''} ${railOpen ? 'rail-open' : ''}`}><aside className="wb-rail" aria-label="项目导航" inert={narrow && Boolean(selectedCitation)}><div className="wb-brand"><Code2 /><span><strong>源鉴 RepoNoesis</strong><small>真实源码工作台</small></span></div><div className="wb-rail-title"><span>项目</span><button type="button" onClick={props.onRefreshLibrary} disabled={props.libraryLoading}>刷新</button></div><div className="wb-project-list">{props.workspaces.map((item) => <button key={item.workspace_id} className={item.workspace_id === props.workspaceId ? 'is-active' : ''} disabled={!item.openable} type="button" onClick={() => { setRailOpen(false); setSelectedCitation(null); setExpandedCode(null); setFileSelection(null); props.onOpenWorkspace(item.workspace_id); }}><FolderOpen /><span><strong>{item.display_name}</strong><small>{item.repository_revision ? item.repository_revision.slice(0, 12) : 'revision 未提供'}</small></span></button>)}{!props.libraryLoading && props.workspaces.length === 0 && <p>暂无可打开的已索引项目。</p>}</div><nav className="wb-nav" aria-label="工作台导航">{nav('ask', '源码问答', Search, !noProject)}{nav('map', '项目地图', Map, !noProject)}{nav('exploration', '源码探索', Code2, !noProject)}</nav>{props.project && <ProjectMaterials project={props.project} workspaceId={props.workspaceId} revision={props.revision} loading={props.projectLoading} />}<div className="wb-secondary">{nav('manage', '项目管理', FolderCog)}<details><summary>更多</summary>{nav('report', '项目分析报告', FileCode2, !noProject)}</details></div></aside>{(railOpen || selectedCitation) && <button className="wb-backdrop" type="button" aria-label="关闭浮层" onClick={() => selectedCitation ? closeCitation() : setRailOpen(false)} />}<main className="wb-main" inert={narrow && Boolean(selectedCitation)}><header className="wb-context"><button className="wb-icon-button wb-menu" type="button" aria-label="打开项目导航" aria-expanded={railOpen} onClick={() => setRailOpen(true)}><Menu /></button><div className="wb-context-primary"><strong className="wb-context-name" title={projectName}>{projectName}</strong><span title={source}>{source}</span></div><div className="wb-context-meta"><button className="wb-theme" type="button" aria-pressed={theme === 'dark'} onClick={() => setTheme((value) => value === 'dark' ? 'light' : 'dark')} title="切换主题">{theme === 'dark' ? <Sun /> : <Moon />}<span>{theme === 'dark' ? '明亮' : '夜间'}</span></button></div></header><section ref={scrollHost} className="wb-scroll" tabIndex={0} aria-label={askPage ? "问答阅读区" : "项目内容"}><Notification message={props.loading || (askPage && !actionable(props.statusMessage)) ? '' : props.statusMessage} /><div className={`wb-content ${askPage ? 'wb-ask-content' : 'wb-page-content'} ${firstPrompt ? 'is-first-prompt' : ''}`}>{askPage ? <>{firstPrompt && <div className="wb-first-prompt"><Code2 /><h1>从源码中找到答案</h1><p>围绕当前仓库提问，结合引用核对实现。</p><div className="wb-examples">{examples.map((example) => <button key={example} type="button" onClick={() => useExample(example)}>{example}</button>)}</div>{composer}</div>}{noProject && <div className="wb-empty"><FolderOpen /><h1>选择一个已索引项目</h1><p>可从左侧打开已有项目；没有预备数据时，在项目管理中导入项目。首次导入需准备本地 BGE 权重；源码探索本身不调用生成模型。</p></div>}{props.history && !noProject && <div className="wb-history-controls" aria-label="问答历史"><span>{props.history.loading ? '正在加载历史记录…' : props.history.waiting ? '本次请求结束后展示已读取的历史。' : props.history.loaded ? '已恢复保存记录；读取历史不会重新校验答案。' : '问答历史'}</span>{props.history.error && <><p role="alert">{props.history.error}</p><button type="button" onClick={props.history.retry} disabled={props.history.loading}>重试历史加载</button></>}{!props.history.error && props.history.cursor && <button type="button" onClick={props.history.loadEarlier} disabled={props.history.loading || props.history.waiting}>加载更早</button>}</div>}<div className="wb-conversation">{props.answers.map((answer) => <AnswerCard key={answer.id || `${answer.clientRequestId || 'legacy'}:${answer.question}`} item={answer} selected={selectedCitation} onCitation={(binding, citation, index, trigger) => { captureCitation(trigger); setFileSelection(null); setExpandedCode(null); setSelectedCitation({ answerId: answer.id || String(answer.clientRequestId), evidenceId: binding?.evidenceId || null, citations: binding?.citations || [citation], index, projectId: answer.projectId, revision: answer.revision, workspaceId: answer.workspaceId, historical: Boolean(answer.history), fullFileReason: answer.history && !answer.history.full_file_available ? (historyFileReason[answer.history.full_file_unavailable_reason || ''] || '保存引用的身份不可用，无法打开全文。') : undefined }); }} onExpandCode={openCode} />)}</div></> : props.activePage === 'exploration' && props.project && !props.projectLoading ? <SourceExploration key={`${props.workspaceId}:${props.project.project.id}:${props.revision}:${props.contextGeneration ?? 0}`} projectId={props.project.project.id} workspaceId={props.workspaceId} revision={props.revision} /> : props.content}</div></section>{askPage && !firstPrompt && composer}</main>{selectedCitation && <CitationDrawer modal={narrow && !expandedCode} selection={selectedCitation} revision={selectedCitation.revision || ''} fullOpen={Boolean(fileSelection)} onFullFile={(citation, trigger) => openFile({ projectId: selectedCitation.projectId || '', revision: selectedCitation.revision || '', workspaceId: selectedCitation.workspaceId || '', path: citation.path ?? '', start: citation.start_line ?? undefined, end: citation.end_line ?? undefined }, trigger)} onClose={closeCitation} onSelect={(index) => { setExpandedCode(null); setSelectedCitation((current) => current ? { ...current, index } : null); }} onExpand={openCode} />}{expandedCode && <CodeModal source={expandedCode} onClose={closeCode} />}{fileSelection && <SourceFileReader key={fileSelection.openId} selection={fileSelection} onClose={closeFile} />}</div></SourceFileContext.Provider>;
}
