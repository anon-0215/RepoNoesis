import { useMemo, useState, type ReactNode } from 'react';
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

function copyRawCode(value: string) {
  void navigator.clipboard?.writeText(value).catch(() => undefined);
}

function TokenText({ token }: { token: PrismPart | PrismPart[] }): ReactNode {
  if (Array.isArray(token)) return token.map((part, index) => <TokenText key={index} token={part} />);
  if (typeof token === 'string') return token;
  const aliases = Array.isArray(token.alias) ? token.alias : token.alias ? [token.alias] : [];
  return <span className={['token', token.type, ...aliases].join(' ')}><TokenText token={token.content as PrismPart | PrismPart[]} /></span>;
}

export function CodeViewer({ source, compact = false, onExpand }: CodeViewerProps) {
  const [wrapped, setWrapped] = useState(false);
  const language = languageFor(source.language ?? source.path);
  const tokens = useMemo<PrismPart[]>(() => {
    const grammar = language ? Prism.languages[language] : undefined;
    return grammar ? Prism.tokenize(source.code, grammar) as PrismPart[] : [source.code];
  }, [language, source.code]);

  return <section className={`wb-code-viewer ${compact ? 'is-compact' : ''}`} aria-label={source.label}>
    <div className="wb-code-toolbar">
      <span>{language || '纯文本'}</span>
      <div>
        <button type="button" onClick={() => copyRawCode(source.code)}><Clipboard />复制代码</button>
        <button type="button" aria-pressed={wrapped} onClick={() => setWrapped((value) => !value)}><WrapText />折行：{wrapped ? '开' : '关'}</button>
        {onExpand && <button type="button" onClick={(event) => onExpand(event.currentTarget)}><Expand />放大阅读</button>}
      </div>
    </div>
    <div className={`wb-code-pane ${wrapped ? 'is-wrapped' : ''}`} tabIndex={0}>
      <pre><code className={language ? `language-${language}` : undefined}><TokenText token={tokens} /></code></pre>
    </div>
  </section>;
}
