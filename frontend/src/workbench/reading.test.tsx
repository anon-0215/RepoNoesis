// @vitest-environment jsdom

import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { renderToStaticMarkup } from 'react-dom/server';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { CodeViewer } from './CodeViewer';
import { MarkdownAnswer } from './MarkdownAnswer';

describe('workbench reading surface', () => {
  let container: HTMLDivElement | null = null;
  let root: Root | null = null;

  afterEach(async () => {
    if (root) await act(async () => { root?.unmount(); });
    container?.remove(); root = null; container = null;
    vi.restoreAllMocks();
  });

  it('keeps supported markdown structure and evidence text without parsing raw HTML', () => {
    const markup = renderToStaticMarkup(<MarkdownAnswer text={'# 标题\n\n**加粗** 与 `inline_code` [E1]\n\n- 第一项\n- 第二项\n\n1. 有序项\n2. 第二项\n\n| 名称 | 值 |\n| --- | --- |\n| A | B |\n\n```python\ndef answer(value):\n    # 注释\n    return value\n```\n\n<script>window.evil = true</script>\n\n![远程图](https://example.invalid/x.png)\n\n[危险](javascript:alert(1))'} onExpandCode={vi.fn()} />);
    expect(markup).toContain('<h1>标题</h1>');
    expect(markup).toContain('<strong>加粗</strong>');
    expect(markup).toContain('[E1]');
    expect(markup).toContain('<ul>');
    expect(markup).toContain('<ol>');
    expect(markup).toContain('<table>');
    expect(markup).toContain('token keyword');
    expect(markup).not.toContain('<script>');
    expect(markup).not.toContain('<img');
    expect(markup).not.toContain('javascript:');
  });

  it('copies the original code and safely degrades unknown languages', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } });
    container = document.createElement('div'); document.body.appendChild(container); root = createRoot(container);
    await act(async () => { root?.render(<CodeViewer source={{ code: 'alpha <beta>\nγ', language: 'unknown', label: '未知代码' }} />); });
    expect(container.querySelector('.wb-code-pane code')?.textContent).toBe('alpha <beta>\nγ');
    await act(async () => { (Array.from(container!.querySelectorAll('button')).find((item) => item.textContent?.includes('复制代码')) as HTMLButtonElement).click(); });
    expect(writeText).toHaveBeenCalledWith('alpha <beta>\nγ');
  });

  it('opens an answer code source only through the explicit reading control', async () => {
    const onExpand = vi.fn(); container = document.createElement('div'); document.body.appendChild(container); root = createRoot(container);
    await act(async () => { root?.render(<MarkdownAnswer text={'```python\nprint("safe")\n```'} onExpandCode={onExpand} />); });
    expect(onExpand).not.toHaveBeenCalled();
    await act(async () => { (Array.from(container!.querySelectorAll('button')).find((item) => item.textContent?.includes('放大阅读')) as HTMLButtonElement).click(); });
    expect(onExpand).toHaveBeenCalledWith(expect.objectContaining({ code: 'print("safe")', language: 'python', label: '回答中的代码块' }), expect.any(HTMLButtonElement));
  });

  it('links only bound text nodes and preserves code, normal links, and unknown markers', async () => {
    const onEvidence = vi.fn(); container = document.createElement('div'); document.body.appendChild(container); root = createRoot(container);
    const bindings = new Map([['E5', { evidenceId: 'E5', citations: [{ evidence_id: 'E5', path: 'src/a.py', qualified_name: 'a', summary: '', snippet: 'a', start_line: 5, end_line: 6 }], indices: [0] }]]);
    await act(async () => root?.render(<MarkdownAnswer text={'正文 [E5] 再次 [E5]，未知 [E1]。`[E5]`\n\n```python\n# [E5]\n```\n\n[普通链接 E5](https://example.com) 与 [[E5]](https://example.com/evidence)'} bindings={bindings} onEvidence={onEvidence} onExpandCode={vi.fn()} />));
    const markers = container.querySelectorAll<HTMLButtonElement>('.wb-evidence-marker');
    expect(markers).toHaveLength(2);
    expect(container.querySelector('.wb-inline-code')?.textContent).toBe('[E5]');
    expect(container.querySelector('.wb-code-pane')?.textContent).toContain('[E5]');
    expect(container.querySelector('a')?.getAttribute('href')).toBe('https://example.com');
    expect(container.querySelectorAll('a .wb-evidence-marker')).toHaveLength(0);
    expect(container.textContent).toContain('[E1]');
    await act(async () => markers[1].click());
    expect(onEvidence).toHaveBeenCalledWith('E5', markers[1]);
  });
});
