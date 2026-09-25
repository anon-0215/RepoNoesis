import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent, type ReactNode } from 'react';
import { BookOpen, ChevronRight, Clipboard, Code2, FileCode2, FolderCog, FolderOpen, LayoutDashboard, Map, Menu, Moon, Search, Send, Sun, X } from 'lucide-react';
import type { AskFailure, AskProgressEvent, ChatAnswer, Citation, ExecutionMode, ProjectResponse, WorkspaceSummary } from '../types';
import { citationLabel } from '../lib/product';
import { CodeViewer, type CodeSource } from './CodeViewer';
import { MarkdownAnswer } from './MarkdownAnswer';
import { evidenceBindings, type EvidenceBinding } from './evidenceBindings';
import { ExecutionSummary } from './ExecutionSummary';

export interface WorkbenchAnswer {
  id?: string;
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
export type WorkbenchPage = 'manage' | 'dashboard' | 'map' | 'learning' | 'ask' | 'report';
interface WorkbenchProps { project: ProjectResponse | null; workspaces: WorkspaceSummary[]; workspaceId: string; revision: string; question: string; answers: WorkbenchAnswer[]; error: AskFailure | null; loading: boolean; libraryLoading: boolean; projectLoading: boolean; statusMessage: string; activePage: WorkbenchPage; content: ReactNode; executionMode?: ExecutionMode; onExecutionModeChange?: (mode: ExecutionMode) => void; onQuestionChange: (value: string) => void; onSubmit: (event: FormEvent) => void; onOpenWorkspace: (workspaceId: string) => void; onRefreshLibrary: () => void; onNavigate: (page: WorkbenchPage) => void; }
const THEME_KEY = 'reponoesis.workbench.theme';
const examples = ['这个仓库的主要入口在哪里？', '主要模块分别负责什么？', '请解释一个函数的实现，并给出源码引用。'];
const actionable = (message: string) => /失败|无法|错误|超时|未完成|不可|证据不足/.test(message);
const copyText = (value: string) => void navigator.clipboard?.writeText(value).catch(() => undefined);
const citationRange = (citation: Citation) => citation.start_line > 0 && citation.end_line >= citation.start_line ? `行 ${citation.start_line}–${citation.end_line}` : '服务端未提供可展示的行范围';
interface CitationSelection { answerId: string; evidenceId: string | null; citations: Citation[]; index: number | null }

function CitationDrawer({ selection, revision, onClose, onSelect, onExpand }: { selection: CitationSelection; revision: string; onClose: () => void; onSelect: (index: number) => void; onExpand: (source: CodeSource, trigger: HTMLButtonElement) => void }) {
  const closeButton = useRef<HTMLButtonElement>(null);
  const citation = selection.index === null ? null : selection.citations[selection.index];
  const source: CodeSource | null = citation ? { code: citation.snippet, label: citation.qualified_name || '源码片段', path: citation.path, range: citationRange(citation) } : null;
  useEffect(() => { closeButton.current?.focus(); }, [selection.answerId, selection.evidenceId]);
  return <aside className="wb-drawer" aria-label="引用源码片段" aria-modal="true" role="dialog"><div className="wb-drawer-head"><div><p className="wb-eyebrow">当前引用 {selection.evidenceId ? `[${selection.evidenceId}]` : ''}</p><h2>{source?.label || '选择证据片段'}</h2></div><button ref={closeButton} className="wb-icon-button" type="button" onClick={onClose} aria-label="关闭源码栏"><X /></button></div>{selection.citations.length > 1 && <div className="wb-fragment-list" aria-label="选择证据片段">{selection.citations.map((part, index) => <button key={`${part.path}:${part.start_line}:${part.end_line}:${index}`} type="button" aria-pressed={selection.index === index} onClick={() => onSelect(index)}>{part.path} · {citationRange(part)}</button>)}</div>}{citation && source ? <><div className="wb-source-meta"><span>{citation.path}</span><span>{citationRange(citation)}</span>{revision && <span title={revision}>revision {revision.slice(0, 12)}</span>}</div><CodeViewer key={`${selection.answerId}:${selection.evidenceId}:${selection.index}`} source={source} onExpand={(trigger) => onExpand(source, trigger)} /><p className="wb-drawer-note">高亮依据引用片段的行范围，不代表每句话的精确行。</p></> : <p className="wb-drawer-note">此证据包含多个片段，请选择要阅读的片段。</p>}</aside>;
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
  const tail = entries.slice(-6);
  if (item.status === 'failure') {
    const base = entries.find((event) => event.type === 'base_completed');
    if (base && !tail.includes(base)) tail.unshift(base);
  }
  return tail;
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

function answerWarnings(result: ChatAnswer): string[] {
  return result.warnings.map((warning) =>
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

function AnswerCard({ item, selected, onCitation, onExpandCode }: { item: WorkbenchAnswer; selected: CitationSelection | null; onCitation: (binding: EvidenceBinding | null, citation: Citation, index: number | null, trigger: HTMLButtonElement) => void; onExpandCode: (source: CodeSource, trigger: HTMLButtonElement) => void }) {
  const result = item.result;
  const bindings = result ? evidenceBindings(result, item.projectId, item.revision) : new globalThis.Map<string, EvidenceBinding>();
  const mode = item.status === 'success' ? result?.execution_mode : item.requestedMode;
  const modeText = item.status === 'success' ? '执行模式' : '请求模式';
  const labels: Record<string, string> = { request_received: '请求已接收', base_started: '基础检索开始', base_completed: '基础检索完成', planner_started: '判断是否需要补充证据', planner_stopped: '补充阶段结束', tool_completed: '补充工具已执行', answer_started: '答案生成开始', citation_checked: '引用校验完成', relation_checked: '关系校验完成' };
  const progress = progressForAnswer(item);
  const requestContext = <><small className="wb-request-context">{modeText}：{mode === 'rag' ? 'RAG' : mode === 'agent' ? 'Agent' : '未记录'}{item.clientRequestId ? ` · 本地请求 #${item.clientRequestId}` : ''} · {item.revision ? item.revision.slice(0, 12) : 'revision 未提供'}</small>{progress.length > 0 && <section className="wb-progress" aria-label="本次执行进度"><strong>执行进度 · {item.status === 'failure' ? '请求失败' : item.status === 'success' ? '请求已结束' : '执行中'}</strong><ol>{progress.map((event) => <li key={event.sequence}>{event.type === "citation_checked" || event.type === "relation_checked" ? validationProgressLabel(event) : event.type === 'base_completed' ? (baseProgressLabels[event.status || ''] || '基础检索结束（状态未提供）') : labels[event.type]}{event.type === 'base_completed' && ['succeeded', 'zero_hit', 'all_rejected'].includes(event.status || '') ? ` · 新增证据 ${event.new_evidence_count ?? '未提供'}` : ''}{event.type === 'tool_completed' ? ` · ${event.status === 'succeeded' ? '成功' : '未成功'}` : ''}{event.type === 'citation_checked' || event.type === 'relation_checked' ? ` · ${event.passed === true ? '通过' : event.passed === false ? '未通过' : '结果未提供'}` : ''}{event.type === 'planner_stopped' && event.reason && event.reason !== 'planner_answer' ? ' · 已按当前证据继续' : ''}</li>)}</ol></section>}</>;
  if (item.status === 'pending') return <article className="wb-answer" data-answer-id={item.id}><div className="wb-question"><span>你</span><p>{item.question}</p></div><Waiting />{requestContext}</article>;
  if (item.status === 'failure' || !result) return <article className="wb-answer" data-answer-id={item.id}><div className="wb-question"><span>你</span><p>{item.question}</p></div><section className="wb-error" role="alert"><h2>{baseTimeoutCausedInsufficiency(item) ? '基础检索超时' : item.failure?.diagnostics.agent_status === 'insufficient_evidence' ? '证据不足' : '问答未生成可验证答案'}</h2>{baseTimeoutCausedInsufficiency(item) && <p>基础检索未在时限内完成，本次未形成足够的可验证证据；这不表示源码没有相关内容。</p>}{item.failure ? <><p>失败阶段：{item.failure.diagnostics.failure_stage} · {item.failure.diagnostics.failure_reason_code}</p><p>请求 ID：{item.failure.diagnostics.request_id}</p></> : <p>{item.errorMessage || '问答失败，请检查服务后重试。'}</p>}</section><ExecutionSummary failure={item.failure} mode={item.requestedMode} />{requestContext}</article>;
  return <article className="wb-answer" data-answer-id={item.id}><div className="wb-question"><span>你</span><p>{item.question}</p></div><div className="wb-answer-head"><Code2 /><div><strong>RepoNoesis</strong><small>{result.grounding_status === 'grounded' ? '源码引用已校验' : result.grounding_status === 'insufficient_evidence' ? '证据不足' : '降级回答'}</small></div><button type="button" onClick={() => copyText(result.answer)}><Clipboard />复制回答</button></div><MarkdownAnswer text={result.answer} bindings={bindings} selectedEvidenceId={selected?.answerId === item.id ? selected?.evidenceId : null} onEvidence={(id, trigger) => { const binding = bindings.get(id); if (binding) onCitation(binding, binding.citations[0], binding.citations.length === 1 ? 0 : null, trigger); }} onExpandCode={onExpandCode} />{result.warnings.length > 0 && <p className="wb-warnings">{answerWarnings(result).join('；')}</p>}<div className="wb-citations" aria-label={`第 ${item.clientRequestId || 1} 条回答的引用`}><span>本回答引用</span>{result.citations.length === 0 && <small>服务端未返回引用。</small>}{result.citations.map((citation, displayIndex) => { const binding = citation.evidence_id ? bindings.get(citation.evidence_id) : undefined; const partIndex = binding?.indices.indexOf(displayIndex) ?? -1; return <button key={`${citation.path}:${citation.start_line}:${citation.end_line}:${displayIndex}`} type="button" aria-pressed={selected?.answerId === item.id && selected?.index === partIndex && selected?.evidenceId === (binding?.evidenceId || null)} onClick={(event) => onCitation(binding || null, citation, binding ? partIndex : 0, event.currentTarget)}><b>{binding ? `[${binding.evidenceId}]` : `片段 ${displayIndex + 1}`}</b>{citation.qualified_name || citationLabel(citation)}<ChevronRight /></button>; })}</div><ExecutionSummary result={result} />{requestContext}</article>;
}

function Notification({ message }: { message: string }) { const [dismissed, setDismissed] = useState(''); const persistent = actionable(message); useEffect(() => { setDismissed(''); if (!message || persistent) return; const timer = window.setTimeout(() => setDismissed(message), 5000); return () => window.clearTimeout(timer); }, [message, persistent]); if (!message || dismissed === message) return null; return <div className={`wb-notification status-line ${persistent ? 'is-actionable' : ''}`} role={persistent ? 'alert' : 'status'}><span>{message}</span><button type="button" aria-label="关闭提示" onClick={() => setDismissed(message)}><X /></button></div>; }
function Waiting() { const [seconds, setSeconds] = useState(0); useEffect(() => { const started = Date.now(); const timer = window.setInterval(() => setSeconds(Math.floor((Date.now() - started) / 1000)), 250); return () => window.clearInterval(timer); }, []); return <div className="wb-waiting"><span /><div><strong>正在查找相关源码</strong><p>已等待 {seconds} 秒</p></div></div>; }

export function Workbench(props: WorkbenchProps) {
  const [theme, setTheme] = useState<'light' | 'dark'>(() => { try { return window.localStorage.getItem(THEME_KEY) === 'dark' ? 'dark' : 'light'; } catch { return 'light'; } });
  const [railOpen, setRailOpen] = useState(false); const [selectedCitation, setSelectedCitation] = useState<CitationSelection | null>(null); const [expandedCode, setExpandedCode] = useState<CodeSource | null>(null);
  const citationTrigger = useRef<HTMLButtonElement | null>(null); const readerTrigger = useRef<HTMLButtonElement | null>(null); const composing = useRef(false); const textarea = useRef<HTMLTextAreaElement>(null); const scrollHost = useRef<HTMLElement>(null);
  const noProject = !props.project; const askPage = props.activePage === 'ask'; const selectedWorkspace = props.workspaces.find((item) => item.workspace_id === props.workspaceId); const projectName = props.project?.project.repo || selectedWorkspace?.display_name || '未选择项目'; const source = props.project?.project.source_location || props.project?.project.repo_url || '来源信息未提供'; const firstPrompt = askPage && !noProject && props.answers.length === 0 && !props.loading && !props.error;
  useEffect(() => { document.documentElement.dataset.theme = theme; try { window.localStorage.setItem(THEME_KEY, theme); } catch {} }, [theme]);
  useEffect(() => { setSelectedCitation(null); setExpandedCode(null); }, [props.workspaceId, props.project?.project.id, props.revision]);
  useEffect(() => { const handler = (event: globalThis.KeyboardEvent) => { if (event.key !== 'Escape') return; if (expandedCode) { event.preventDefault(); closeCode(); } else if (selectedCitation) { event.preventDefault(); closeCitation(); } else if (railOpen) setRailOpen(false); }; document.addEventListener('keydown', handler); return () => document.removeEventListener('keydown', handler); });
  useEffect(() => { if (textarea.current) { textarea.current.style.height = 'auto'; textarea.current.style.height = `${Math.min(textarea.current.scrollHeight, 144)}px`; } }, [props.question]);
  function closeCitation() { setSelectedCitation(null); window.setTimeout(() => citationTrigger.current?.focus(), 0); }
  function openCode(sourceToOpen: CodeSource, trigger: HTMLButtonElement) { readerTrigger.current = trigger; setExpandedCode(sourceToOpen); }
  function closeCode() { setExpandedCode(null); window.setTimeout(() => readerTrigger.current?.focus(), 0); }
  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) { if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing && !composing.current) { event.preventDefault(); if (!props.loading && props.question.trim()) event.currentTarget.form?.requestSubmit(); } }
  function navigate(page: WorkbenchPage) { setRailOpen(false); setSelectedCitation(null); setExpandedCode(null); props.onNavigate(page); }
  function useExample(value: string) { props.onQuestionChange(value); window.setTimeout(() => textarea.current?.focus(), 0); }
  function submitAndReveal(event: FormEvent) { props.onSubmit(event); window.setTimeout(() => { const host = scrollHost.current; if (!host) return; if (typeof host.scrollTo === 'function') host.scrollTo({ top: host.scrollHeight, behavior: 'smooth' }); else host.scrollTop = host.scrollHeight; }, 0); }
  const composer = <form className="ask-form wb-composer" onSubmit={submitAndReveal}><div><textarea ref={textarea} value={props.question} disabled={noProject} onChange={(event) => props.onQuestionChange(event.target.value)} onCompositionStart={() => { composing.current = true; }} onCompositionEnd={() => { composing.current = false; }} onKeyDown={onKeyDown} placeholder={noProject ? '请先选择项目' : '输入你的源码问题'} rows={1} /><div className="wb-composer-foot"><div className="wb-mode-switch" role="group" aria-label="问答模式"><button type="button" aria-pressed={(props.executionMode || "agent") === "rag"} disabled={props.loading} onClick={() => props.onExecutionModeChange?.("rag")}>RAG</button><button type="button" aria-pressed={(props.executionMode || "agent") === "agent"} disabled={props.loading} onClick={() => props.onExecutionModeChange?.("agent")}>Agent</button></div><small>Enter 发送 · Shift+Enter 换行</small><button type="submit" disabled={noProject || props.loading || !props.question.trim()}><Send />发送</button></div></div></form>;
  const nav = (page: WorkbenchPage, label: string, Icon: typeof Search, enabled = true) => <button className={props.activePage === page ? 'is-active' : ''} type="button" onClick={() => navigate(page)} disabled={!enabled}><Icon />{label}</button>;
  return <div className={`wb-frame ${selectedCitation ? 'source-open' : ''} ${railOpen ? 'rail-open' : ''}`}><aside className="wb-rail" aria-label="项目导航"><div className="wb-brand"><Code2 /><span><strong>源鉴 RepoNoesis</strong><small>真实源码工作台</small></span></div><div className="wb-rail-title"><span>项目</span><button type="button" onClick={props.onRefreshLibrary} disabled={props.libraryLoading}>刷新</button></div><div className="wb-project-list">{props.workspaces.map((item) => <button key={item.workspace_id} className={item.workspace_id === props.workspaceId ? 'is-active' : ''} disabled={!item.openable || props.projectLoading} type="button" onClick={() => { setRailOpen(false); setSelectedCitation(null); setExpandedCode(null); props.onOpenWorkspace(item.workspace_id); }}><FolderOpen /><span><strong>{item.display_name}</strong><small>{item.repository_revision ? item.repository_revision.slice(0, 12) : 'revision 未提供'}</small></span></button>)}{!props.libraryLoading && props.workspaces.length === 0 && <p>暂无可打开的已索引项目。</p>}</div><nav className="wb-nav" aria-label="工作台导航">{nav('ask', '源码问答', Search, !noProject)}{nav('manage', '项目管理', FolderCog)}{nav('dashboard', '项目概览', LayoutDashboard, !noProject)}{nav('map', '项目地图', Map, !noProject)}{nav('learning', '学习路线', BookOpen, !noProject)}{nav('report', '报告', FileCode2, !noProject)}</nav></aside>{(railOpen || selectedCitation) && <button className="wb-backdrop" type="button" aria-label="关闭浮层" onClick={() => selectedCitation ? closeCitation() : setRailOpen(false)} />}<main className="wb-main"><header className="wb-context"><button className="wb-icon-button wb-menu" type="button" aria-label="打开项目导航" aria-expanded={railOpen} onClick={() => setRailOpen(true)}><Menu /></button><div className="wb-context-primary"><strong className="wb-context-name" title={projectName}>{projectName}</strong><span title={source}>{source}</span></div><div className="wb-context-meta"><span title={props.revision || undefined}>{props.revision ? `revision ${props.revision.slice(0, 12)}` : 'revision 未提供'}</span>{selectedWorkspace && <span title="仅为接口记录的数量，不代表当前配置完整性已额外验证">向量记录 {selectedWorkspace.embedding_count ?? 0} / {selectedWorkspace.total_chunks ?? 0}</span>}<button className="wb-theme" type="button" aria-pressed={theme === 'dark'} onClick={() => setTheme((value) => value === 'dark' ? 'light' : 'dark')} title="切换主题">{theme === 'dark' ? <Sun /> : <Moon />}<span>{theme === 'dark' ? '明亮' : '夜间'}</span></button></div></header><section ref={scrollHost} className="wb-scroll" aria-live="polite"><Notification message={props.loading ? '' : props.statusMessage} /><div className={`wb-content ${askPage ? 'wb-ask-content' : 'wb-page-content'} ${firstPrompt ? 'is-first-prompt' : ''}`}>{askPage ? <>{firstPrompt && <div className="wb-first-prompt"><Code2 /><h1>从源码中找到答案</h1><p>围绕当前仓库提问，结合引用核对实现。</p><div className="wb-examples">{examples.map((example) => <button key={example} type="button" onClick={() => useExample(example)}>{example}</button>)}</div>{composer}</div>}{noProject && <div className="wb-empty"><FolderOpen /><h1>选择一个已索引项目</h1><p>可从左侧打开已有项目；可在项目管理中导入项目。</p></div>}<div className="wb-conversation">{props.answers.map((answer) => <AnswerCard key={answer.id || `${answer.clientRequestId || 'legacy'}:${answer.question}`} item={answer} selected={selectedCitation} onCitation={(binding, citation, index, trigger) => { citationTrigger.current = trigger; setExpandedCode(null); setSelectedCitation({ answerId: answer.id || String(answer.clientRequestId), evidenceId: binding?.evidenceId || null, citations: binding?.citations || [citation], index }); }} onExpandCode={openCode} />)}</div></> : props.content}</div></section>{askPage && !firstPrompt && composer}</main>{selectedCitation && <CitationDrawer selection={selectedCitation} revision={props.revision} onClose={closeCitation} onSelect={(index) => { setExpandedCode(null); setSelectedCitation((current) => current ? { ...current, index } : null); }} onExpand={openCode} />}{expandedCode && <CodeModal source={expandedCode} onClose={closeCode} />}</div>;
}
