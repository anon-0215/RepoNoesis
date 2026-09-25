import ReactMarkdown, { defaultUrlTransform } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { CodeViewer, languageFor, type CodeSource } from './CodeViewer';
import type { EvidenceBinding } from './evidenceBindings';

interface MarkdownAnswerProps {
  text: string;
  onExpandCode: (source: CodeSource, trigger: HTMLButtonElement) => void;
  bindings?: Map<string, EvidenceBinding>;
  selectedEvidenceId?: string | null;
  onEvidence?: (id: string, trigger: HTMLButtonElement) => void;
}

interface MarkdownNode { type: string; value?: string; url?: string; children?: MarkdownNode[] }
const marker = /\[\b(E[1-9]\d*)\]/g;

function evidencePlugin(ids: string[]) {
  const known = new Set(ids);
  return (tree: MarkdownNode) => {
    const visit = (node: MarkdownNode) => {
      if (!node.children || node.type === 'link' || node.type === 'linkReference' || node.type === 'code' || node.type === 'inlineCode') return;
      node.children = node.children.flatMap((child) => {
        if (child.type !== 'text' || !child.value) { visit(child); return [child]; }
        const parts: MarkdownNode[] = []; let from = 0;
        for (const match of child.value.matchAll(marker)) {
          const at = match.index ?? 0;
          if (!known.has(match[1])) continue;
          if (at > from) parts.push({ type: 'text', value: child.value.slice(from, at) });
          parts.push({ type: 'link', url: `evidence:${match[1]}`, children: [{ type: 'text', value: match[0] }] });
          from = at + match[0].length;
        }
        if (!parts.length) return [child];
        if (from < child.value.length) parts.push({ type: 'text', value: child.value.slice(from) });
        return parts;
      });
    };
    visit(tree);
  };
}

function safeHref(value?: string) {
  if (!value) return undefined;
  try {
    const protocol = new URL(value, 'https://repnoesis.invalid').protocol;
    return protocol === 'http:' || protocol === 'https:' || protocol === 'mailto:' ? value : undefined;
  } catch {
    return undefined;
  }
}

export function MarkdownAnswer({ text, onExpandCode, bindings, selectedEvidenceId, onEvidence }: MarkdownAnswerProps) {
  const plugin = () => evidencePlugin(Array.from(bindings?.keys() || []));
  return <div className="wb-markdown">
    <ReactMarkdown
      remarkPlugins={[remarkGfm, plugin]}
      urlTransform={(url) => /^evidence:E[1-9]\d*$/.test(url) ? url : defaultUrlTransform(url)}
      components={{
        a({ href, children }) {
          if (href?.startsWith('evidence:')) {
            const id = href.slice(9); const binding = bindings?.get(id);
            if (!binding || !onEvidence) return <span>{children}</span>;
            const citation = binding.citations[0];
            const hint = binding.citations.length > 1 ? `${id} · ${binding.citations.length} 个片段，请选择` : `${id} · ${citation.path} · ${citation.qualified_name} · 行 ${citation.start_line}–${citation.end_line}`;
            return <button className="wb-evidence-marker" type="button" title={hint} aria-label={`查看证据 ${id}${binding.citations.length > 1 ? `，${binding.citations.length} 个片段` : ''}`} aria-pressed={selectedEvidenceId === id} onClick={(event) => onEvidence(id, event.currentTarget)}>{children}</button>;
          }
          const safe = safeHref(href);
          return safe ? <a href={safe} target="_blank" rel="noreferrer">{children}</a> : <span>{children}</span>;
        },
        img() { return null; },
        code({ className, children }) {
          const code = String(children).replace(/\n$/, '');
          const language = languageFor(className);
          const block = Boolean(className?.includes('language-')) || code.includes('\n');
          if (!block) return <code className="wb-inline-code">{children}</code>;
          const source: CodeSource = { code, language, label: '回答中的代码块' };
          return <CodeViewer compact source={source} onExpand={(trigger) => onExpandCode(source, trigger)} />;
        }
      }}
    >{text}</ReactMarkdown>
  </div>;
}
