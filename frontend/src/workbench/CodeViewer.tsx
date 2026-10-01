import { useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { sourceLines } from '../lib/sourceLines';
import Prism from 'prismjs';
import 'prismjs/components/prism-bash';
import 'prismjs/components/prism-json';
import 'prismjs/components/prism-python';
import 'prismjs/components/prism-typescript';
import 'prismjs/components/prism-jsx';
import { Clipboard, Expand, WrapText } from 'lucide-react';

type PrismPart = string | Prism.Token;

export interface CodeSource {
  code: string;
  language?: string;
  label: string;
  path?: string;
  range?: string;
}

interface CodeViewerProps {
  source: CodeSource;
  compact?: boolean;
  onExpand?: (trigger: HTMLButtonElement) => void;
  fullFile?: { start?: number; end?: number };
}

const languageAliases: Record<string, string> = {
  py: 'python', python: 'python', js: 'javascript', javascript: 'javascript',
  ts: 'typescript', typescript: 'typescript', tsx: 'tsx', jsx: 'jsx',
  json: 'json', sh: 'bash', shell: 'bash', bash: 'bash'
};

const extensionLanguage: Record<string, string> = {
  py: 'python', js: 'javascript', mjs: 'javascript', cjs: 'javascript',
  ts: 'typescript', tsx: 'tsx', jsx: 'jsx', json: 'json', sh: 'bash', bash: 'bash'
};

export function languageFor(value?: string): string | undefined {
  if (!value) return undefined;
  const normalized = value.trim().toLowerCase().replace(/^language-/, '');
  if (languageAliases[normalized]) return languageAliases[normalized];
  const extension = normalized.split('.').pop();
  return extension ? extensionLanguage[extension] : undefined;
}

function TokenText({ token }: { token: PrismPart | PrismPart[] }): ReactNode {
  if (Array.isArray(token)) return token.map((part, index) => <TokenText key={index} token={part} />);
  if (typeof token === 'string') return token;
  const aliases = Array.isArray(token.alias) ? token.alias : token.alias ? [token.alias] : [];
  return <span className={['token', token.type, ...aliases].join(' ')}><TokenText token={token.content as PrismPart | PrismPart[]} /></span>;
}

function FileLines({ content, wrapped, range }: { content: string; wrapped: boolean; range: { start?: number; end?: number } }) {
  const lines = useMemo(() => sourceLines(content), [content]);
  const valid = Number.isInteger(range.start) && Number.isInteger(range.end) && range.start! >= 1 && range.end! >= range.start! && range.end! <= lines.length;
  const target = valid ? range.start! : 1;
  const pageSize = 300;
  const [page, setPage] = useState(Math.floor((target - 1) / pageSize));
  const [jump, setJump] = useState(String(target));
  const [located, setLocated] = useState(target);
  const [locationRequest, setLocationRequest] = useState(0);
  const [jumpError, setJumpError] = useState(false);
  const pane = useRef<HTMLDivElement>(null);
  const first = page * pageSize;
  const lastPage = Math.max(0, Math.ceil(lines.length / pageSize) - 1);
  useLayoutEffect(() => {
    const host = pane.current;
    if (!host) return;
    const row = host.querySelector<HTMLElement>(`[data-source-line="${located}"]`);
    host.scrollTop = row ? Math.max(0, row.offsetTop - host.clientHeight / 3) : 0;
  }, [page, located, wrapped, locationRequest]);
  function go(value: number) { setPage(value); setLocated(value * pageSize + 1); setJump(String(value * pageSize + 1)); setJumpError(false); }
  return <>
    {range.start !== undefined && <p className="sf-location" role="status">{valid ? `引用行 ${range.start}–${range.end}（高亮范围）` : '无法定位该范围：行号非法或超出正文。'}</p>}
    <div className="sf-pagination" aria-label="全文分段阅读">
      <span>{lines.length ? `行 ${first + 1}–${Math.min(first + pageSize, lines.length)} / ${lines.length}` : '空文件 · 0 行'} · 纯文本分段阅读</span>
      <div><button type="button" disabled={page === 0} onClick={() => go(0)}>首段</button><button type="button" disabled={page === 0} onClick={() => go(page - 1)}>上一段</button><button type="button" disabled={page === lastPage} onClick={() => go(page + 1)}>下一段</button><button type="button" disabled={page === lastPage} onClick={() => go(lastPage)}>末段</button></div>
      <form onSubmit={(event) => { event.preventDefault(); const number = Number(jump); if (!Number.isInteger(number) || number < 1 || number > lines.length) { setJumpError(true); return; } setJumpError(false); setPage(Math.floor((number - 1) / pageSize)); setLocated(number); setLocationRequest((old) => old + 1); }}><label>定位行 <input aria-label="定位行" type="number" value={jump} onChange={(event) => setJump(event.target.value)} /></label><button type="submit">跳转</button></form>
      {jumpError && <span role="alert">无法定位该范围。</span>}
    </div>
    <div ref={pane} className={`wb-code-pane sf-lines ${wrapped ? 'is-wrapped' : ''}`} tabIndex={0} aria-label="文件全文">
      {lines.length === 0 ? <p>这是合法的空文件。</p> : <pre><code>{lines.slice(first, first + pageSize).map((line, index) => { const number = first + index + 1; return <span key={number} data-source-line={number} className={`sf-line ${valid && number >= range.start! && number <= range.end! ? 'is-cited' : ''}`}><span className="sf-line-number" aria-hidden="true">{number}</span><span className="sf-line-text">{line || '\u200b'}</span></span>; })}</code></pre>}
    </div>
  </>;
}

export function CodeViewer({ source, compact = false, onExpand, fullFile }: CodeViewerProps) {
  const [wrapped, setWrapped] = useState(false);
  const [copyStatus, setCopyStatus] = useState('');
  async function copy() {
    try {
      if (!navigator.clipboard) throw new Error('clipboard unavailable');
      await navigator.clipboard.writeText(source.code);
      setCopyStatus('已复制');
    } catch { setCopyStatus('复制失败，请检查剪贴板权限。'); }
  }
  const language = languageFor(source.language ?? source.path);
  const tokens = useMemo<PrismPart[]>(() => {
    const grammar = !fullFile && language ? Prism.languages[language] : undefined;
    return grammar ? Prism.tokenize(source.code, grammar) as PrismPart[] : [source.code];
  }, [language, source.code, fullFile]);

  return <section className={`wb-code-viewer ${compact ? 'is-compact' : ''}`} aria-label={source.label}>
    <div className="wb-code-toolbar">
      <span role={copyStatus ? 'status' : undefined} title={copyStatus || undefined}>{copyStatus || (fullFile ? '全文 · 纯文本' : language || '纯文本')}</span>
      <div>
        <button type="button" onClick={() => void copy()}><Clipboard />{fullFile ? '复制全文' : '复制代码'}</button>
        <button type="button" aria-pressed={wrapped} onClick={() => setWrapped((value) => !value)}><WrapText />折行：{wrapped ? '开' : '关'}</button>
        {onExpand && <button type="button" onClick={(event) => onExpand(event.currentTarget)}><Expand />放大阅读</button>}
      </div>
    </div>
    {fullFile ? <FileLines content={source.code} wrapped={wrapped} range={fullFile} /> : <div className={`wb-code-pane ${wrapped ? 'is-wrapped' : ''}`} tabIndex={0}>
      <pre><code className={language ? `language-${language}` : undefined}><TokenText token={tokens} /></code></pre>
    </div>}
  </section>;
}
