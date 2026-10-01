import { createContext, useContext, useEffect, useRef, useState } from 'react';
import { getSourceFile } from '../lib/api';
import type { SourceFile } from '../types';
import { CodeViewer } from './CodeViewer';
import './source-file.css';

export interface SourceFileSelection {
  projectId: string; revision: string; workspaceId: string; path: string;
  start?: number; end?: number; identityError?: string;
  openId?: number;
}
export type OpenSourceFile = (selection: SourceFileSelection, trigger: HTMLButtonElement) => void;
export const SourceFileContext = createContext<OpenSourceFile | null>(null);
export const useSourceFile = () => useContext(SourceFileContext);

export function SourceFileReader({ selection, onClose }: { selection: SourceFileSelection; onClose: () => void }) {
  const [file, setFile] = useState<SourceFile | null>(null);
  const [error, setError] = useState('');
  const [attempt, setAttempt] = useState(0);
  const [expanded, setExpanded] = useState(false);
  const dialog = useRef<HTMLElement>(null);
  const close = useRef<HTMLButtonElement>(null);
  const closeAction = useRef(onClose);
  closeAction.current = onClose;
  const request = useRef(0);
  useEffect(() => {
    const id = ++request.current;
    const controller = new AbortController();
    setFile(null); setError(selection.identityError || '');
    if (!selection.identityError) {
      getSourceFile(selection.projectId, selection.revision, selection.path, selection.workspaceId, controller.signal)
        .then((result) => { if (id === request.current && !controller.signal.aborted) setFile(result); })
        .catch((reason: unknown) => { if (id === request.current && !controller.signal.aborted) setError(reason instanceof Error ? reason.message : '全文读取失败，请重试。'); });
    }
    return () => { ++request.current; controller.abort(); };
  }, [selection, attempt]);
  useEffect(() => {
    close.current?.focus();
    function key(event: KeyboardEvent) {
      if (event.key === 'Escape') { event.preventDefault(); event.stopImmediatePropagation(); closeAction.current(); return; }
      if (event.key !== 'Tab') return;
      const controls = Array.from(dialog.current?.querySelectorAll<HTMLElement>('button:not(:disabled),input,[tabindex="0"]') || []);
      const first = controls[0], last = controls[controls.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
      else if (!dialog.current?.contains(document.activeElement)) { event.preventDefault(); first?.focus(); }
    }
    document.addEventListener('keydown', key, true);
    return () => document.removeEventListener('keydown', key, true);
  }, []);
  return <section ref={dialog} role="dialog" aria-modal="true" aria-label="已分析文件全文" className={`sf-reader ${expanded ? 'is-expanded' : ''}`}>
    <header><div><p>已保存快照中的已分析文件</p><h2>{selection.path}</h2><p>revision {selection.revision || '未提供'}</p></div><button ref={close} type="button" onClick={onClose} aria-label="关闭全文阅读器">关闭</button></header>
    <div className="sf-meta"><span>{file ? `${file.total_lines} 行 · ${file.utf8_bytes} UTF-8 字节 · 全文完整` : '只读全文'}</span><button type="button" aria-pressed={expanded} onClick={() => setExpanded((old) => !old)}>{expanded ? '恢复侧栏' : '放大阅读'}</button></div>
    {error ? <div className="sf-state" role="alert"><p>{error}</p>{!selection.identityError && <button type="button" onClick={() => setAttempt((old) => old + 1)}>重试读取</button>}</div> : !file ? <p className="sf-state" role="status">正在读取已保存文件…</p> : <CodeViewer key={`${file.project_id}:${file.revision}:${file.path}:${attempt}`} source={{ code: file.content, path: file.path, label: '已分析文件全文' }} fullFile={{ start: selection.start, end: selection.end }} />}
  </section>;
}
