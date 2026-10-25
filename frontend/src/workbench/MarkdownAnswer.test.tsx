// @vitest-environment jsdom
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import { MarkdownAnswer } from './MarkdownAnswer';
import type { EvidenceBinding } from './evidenceBindings';

const binding = (source: EvidenceBinding['source'], name = 'Widget.__enter__') => new Map<string, EvidenceBinding>([['E1', {
  evidenceId: 'E1', indices: [0], source,
  citations: [{ evidence_id: 'E1', path: 'widget.py', qualified_name: name, start_line: 1, end_line: 2, snippet: 'def __enter__(self): pass', summary: '' }]
}]]);
function render(text: string, bindings?: Map<string, EvidenceBinding>) {
  const host = document.createElement('div');
  host.innerHTML = renderToStaticMarkup(<MarkdownAnswer text={text} bindings={bindings} onExpandCode={vi.fn()} onEvidence={vi.fn()} />);
  return host;
}
describe('evidence-backed qualified code symbols', () => {
  for (const source of ['live_evidence', 'saved_binding'] as const) {
    it(`preserves parsed dunder characters for ${source} without changing input`, () => {
      const text = '通过 Widget.__enter__ 进入上下文 [E1]';
      const host = render(text, binding(source));
      expect(host.textContent).toBe(text);
      expect(host.querySelector('strong')).toBeNull();
      expect(host.querySelector('.wb-evidence-marker')?.textContent).toBe('[E1]');
      expect(text).toBe('通过 Widget.__enter__ 进入上下文 [E1]');
    });
  }
  it('preserves legitimate emphasis, links, code, escapes and markers', () => {
    const host = render('**name** __name__ *italic* [Widget.__enter__](https://example.com) `Widget.__enter__` \\_escape\\_ [E1]\n\n```python\nWidget.__enter__()\n```', binding('live_evidence'));
    expect(Array.from(host.querySelectorAll('strong')).map(n => n.textContent)).toEqual(['name', 'name', 'enter']);
    expect(host.querySelector('em')?.textContent).toBe('italic');
    expect(host.querySelector('a')?.getAttribute('href')).toBe('https://example.com');
    expect(host.querySelector('.wb-inline-code')?.textContent).toBe('Widget.__enter__');
    expect(host.textContent).toContain('_escape_');
    expect(host.textContent).toContain('Widget.__enter__()');
    expect(host.querySelectorAll('.wb-evidence-marker')).toHaveLength(1);
  });
  it('leaves unbound, bare, mismatched, nested and asterisk emphasis unchanged', () => {
    expect(render('Widget.__enter__').querySelector('strong')?.textContent).toBe('enter');
    const host = render('__enter__ Other.__enter__ Widget.**enter** Widget.__en*ter*__ Widget.__enter__suffix', binding('saved_binding'));
    expect(Array.from(host.querySelectorAll('strong')).map(n => n.textContent)).toEqual(['enter', 'enter', 'enter', 'enter']);
    expect(host.querySelector('em')?.textContent).toBe('ter');
    expect(host.textContent).toContain('Widget.__enter__suffix');
  });
  it('handles arbitrary qualified names but never partial suffix matches', () => {
    const host = render('pkg.Widget.__custom_method__ Other.pkg.Widget.__custom_method__ **pkg.Widget.__custom_method__**', binding('live_evidence', 'pkg.Widget.__custom_method__'));
    expect(host.textContent).toBe('pkg.Widget.__custom_method__ Other.pkg.Widget.custom_method pkg.Widget.__custom_method__');
    expect(host.querySelectorAll('strong')).toHaveLength(2);
  });
  it('keeps emphasis when an ASCII symbol is only part of a Unicode identifier', () => {
    const host = render('前缀Widget.__enter__ Widget.__enter__后缀', binding('live_evidence'));
    expect(host.textContent).toBe('前缀Widget.enter Widget.__enter__后缀');
    expect(host.querySelector('strong')?.textContent).toBe('enter');
  });
});
