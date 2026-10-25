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

interface MarkdownNode {
  type: string; value?: string; url?: string; children?: MarkdownNode[];
  position?: { start: { offset?: number }; end: { offset?: number } };
}
const marker = /\[\b(E[1-9]\d*)\]/g;

function codeSymbolPlugin(text: string, bindings?: Map<string, EvidenceBinding>) {
  // Only repair an underscore strong node inside an exact, bound qualified
  // symbol. Bare __name__ remains ordinary Markdown because it is ambiguous.
  const symbols = new Set(Array.from(bindings?.values() || []).flatMap(binding =>
    binding.citations.map(citation => citation.qualified_name || '').filter(name =>
      /^(?:[A-Za-z_]\w*\.)+__[A-Za-z_]\w*__$/.test(name))));
  return (tree: MarkdownNode) => {
    const visit = (node: MarkdownNode) => {
      if (!node.children || ['link', 'linkReference', 'image', 'imageReference', 'code', 'inlineCode'].includes(node.type)) return;
      node.children = node.children.map(child => {
        const start = child.position?.start.offset, end = child.position?.end.offset;
        if (child.type === 'strong' && start !== undefined && end !== undefined &&
            child.children?.length === 1 && child.children[0].type === 'text') {
          const raw = text.slice(start, end);
          if (/^__[A-Za-z_]\w*__$/.test(raw) && child.children[0].value === raw.slice(2, -2)) {
            for (const symbol of symbols) {
              const symbolStart = end - symbol.length;
              if (symbolStart >= 0 && text.slice(symbolStart, end) === symbol &&
                  !/[\p{ID_Continue}.\\]/u.test(text[symbolStart - 1] || '') && !/[\p{ID_Continue}.]/u.test(text[end] || '')) {
                return { type: 'text', value: raw, position: child.position };
              }
            }
          }
        }
        visit(child); return child;
      });
    };
    visit(tree);
  };
}

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
  const symbols = () => codeSymbolPlugin(text, bindings);
  return <div className="wb-markdown">
    <ReactMarkdown
      remarkPlugins={[remarkGfm, symbols, plugin]}
      urlTransform={(url) => /^evidence:E[1-9]\d*$/.test(url) ? url : defaultUrlTransform(url)}
      components={{
        a({ href, children }) {
          if (href?.startsWith('evidence:')) {
            const id = href.slice(9); const binding = bindings?.get(id);
            if (!binding || !onEvidence) return <span>{children}</span>;
            const citation = binding.citations[0];
            const hint = binding.citations.length > 1 ? `${id} · ${binding.citations.length} 个片段，请选择` : `${id} · ${citation.path} · ${citation.qualified_name} · 行 ${citation.start_line}–${citation.end_line}`;
            return <button className="wb-evidence-marker" type="button" title={hint} aria-label={`${binding.source === 'saved_binding' ? '查看保存引用' : '查看证据'} ${id}${binding.citations.length > 1 ? `，${binding.citations.length} 个片段` : ''}`} aria-pressed={selectedEvidenceId === id} onClick={(event) => onEvidence(id, event.currentTarget)}>{children}</button>;
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
