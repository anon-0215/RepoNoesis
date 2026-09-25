// @vitest-environment jsdom
import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { describe, expect, it, vi } from 'vitest';
import type { ProjectMap, TreeNode } from '../types';
import { ProjectMapView } from './ProjectMapView';
import { ancestorIds, indexProjectMap, nodeId, visibleMap } from './projectMapModel';

const revision = 'a'.repeat(40);
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
const file = (path: string): TreeNode => ({ name: path.split('/').slice(-1)[0], path, type: 'file' });
const dir = (path: string, children: TreeNode[]): TreeNode => ({ name: path.split('/').slice(-1)[0], path, type: 'directory', children });
const map = (children: TreeNode[], fileCount?: number): ProjectMap => ({ tree: { name: 'demo', path: '', type: 'directory', children }, modules: [], dependency_edges: [], core_files: [], project_id: 'p', repository_revision: revision, coverage: 'analyzed_files', file_count: fileCount });
const sample = map([dir('src', [dir('src/深 层', [file('src/深 层/shared.py'), file('src/深 层/很长的文件名 @#$.py')])]), dir('tests', [file('tests/shared.py')])], 3);

async function render(projectMap: ProjectMap | null, error: string | null = null) {
  const host = document.createElement('div'); document.body.appendChild(host);
  const root = createRoot(host);
  const props = { projectId: 'p', revision, projectMap, error, loading: false, onRetry: vi.fn() };
  await act(async () => root.render(<ProjectMapView {...props} />));
  return { host, root, props, done: async () => { await act(async () => root.unmount()); host.remove(); } };
}

function transform(host: HTMLElement) {
  const value = host.querySelector<HTMLElement>('.pm-world')!.style.transform;
  const match = /^translate\(([-\d.]+)px, ([-\d.]+)px\) scale\(([-\d.]+)\)$/.exec(value);
  if (!match) throw new Error(`Unexpected map transform: ${value}`);
  return { x: Number(match[1]), y: Number(match[2]), zoom: Number(match[3]) };
}

function wheel(target: Element, deltaY: number, options: WheelEventInit = {}) {
  const event = new WheelEvent('wheel', { bubbles: true, cancelable: true, clientX: 300, clientY: 200, deltaY, ...options });
  target.dispatchEvent(event);
  return event;
}

describe('project map structure and interactions', () => {
  it('commits a drag even when pointer release occurs in the same React update batch', async () => {
    const view = await render(sample);
    const canvas = view.host.querySelector<HTMLElement>('.pm-canvas')!;
    canvas.setPointerCapture = vi.fn();
    const pointer = (type: string, x: number, y: number) => canvas.dispatchEvent(new MouseEvent(type, { bubbles: true, clientX: x, clientY: y }));
    try {
      await act(async () => {
        pointer('pointerdown', 10, 10);
        pointer('pointermove', 50, 30);
        pointer('pointerup', 50, 30);
      });
      const moved = view.host.querySelector<HTMLElement>('.pm-world')!.style.transform;
      expect(moved).toContain('translate(70px, 48px)');
      await act(async () => { pointer('pointermove', 90, 90); });
      expect(view.host.querySelector<HTMLElement>('.pm-world')!.style.transform).toBe(moved);
    } finally {
      await view.done();
    }
  });

  it('zooms ordinary wheel input around the pointed map coordinate in both directions', async () => {
    const view = await render(sample);
    const canvas = view.host.querySelector<HTMLElement>('.pm-canvas')!;
    vi.spyOn(canvas, 'getBoundingClientRect').mockReturnValue({ left: 100, top: 50 } as DOMRect);
    try {
      const initial = transform(view.host);
      const pointed = { x: (200 - initial.x) / initial.zoom, y: (150 - initial.y) / initial.zoom };
      let event: WheelEvent;
      await act(async () => { event = wheel(canvas, -100); });
      const enlarged = transform(view.host);
      expect(event!.defaultPrevented).toBe(true);
      expect(enlarged.zoom).toBeGreaterThan(initial.zoom);
      expect((200 - enlarged.x) / enlarged.zoom).toBeCloseTo(pointed.x, 8);
      expect((150 - enlarged.y) / enlarged.zoom).toBeCloseTo(pointed.y, 8);
      await act(async () => { wheel(canvas, 100); });
      expect(transform(view.host).zoom).toBeCloseTo(initial.zoom, 8);
    } finally { await view.done(); }
  });

  it('accumulates rapid wheels, respects bounds without drifting, and leaves non-canvas wheel input alone', async () => {
    const view = await render(sample);
    const canvas = view.host.querySelector<HTMLElement>('.pm-canvas')!;
    vi.spyOn(canvas, 'getBoundingClientRect').mockReturnValue({ left: 100, top: 50 } as DOMRect);
    try {
      await act(async () => { wheel(canvas, -80); wheel(canvas, -80); });
      expect(transform(view.host).zoom).toBeGreaterThan(1.2);
      await act(async () => { for (let i = 0; i < 50; i++) wheel(canvas, -1000); });
      const max = transform(view.host);
      expect(max.zoom).toBe(2.2);
      await act(async () => { wheel(canvas, -1000); });
      expect(transform(view.host)).toEqual(max);
      await act(async () => { for (let i = 0; i < 50; i++) wheel(canvas, 1000, { deltaMode: 1 }); });
      const min = transform(view.host);
      expect(min.zoom).toBe(0.35);
      await act(async () => { wheel(canvas, 1000); });
      expect(transform(view.host)).toEqual(min);

      const before = transform(view.host);
      const events: WheelEvent[] = [];
      await act(async () => {
        events.push(wheel(canvas, -100, { ctrlKey: true }));
        events.push(wheel(canvas, -100, { metaKey: true }));
        events.push(wheel(canvas, 0, { deltaX: 100 }));
        events.push(wheel(canvas, -20, { deltaX: 120 }));
        events.push(wheel(view.host.querySelector('.pm-tools')!, -100));
        events.push(wheel(view.host.querySelector('.pm-search input')!, -100));
        const input = view.host.querySelector<HTMLInputElement>('.pm-search input')!;
        Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(input, 'shared.py');
        input.dispatchEvent(new Event('input', { bubbles: true }));
        (view.host.querySelector('.pm-node-select') as HTMLButtonElement).click();
      });
      await act(async () => {
        events.push(wheel(view.host.querySelector('.pm-results')!, -100));
        events.push(wheel(view.host.querySelector('.pm-detail')!, -100));
      });
      expect(events.every((event) => !event.defaultPrevented)).toBe(true);
      expect(transform(view.host)).toEqual(before);
    } finally { await view.done(); }
  });

  it('can drag after wheel zoom and stops on release without rereading cleared drag state', async () => {
    const view = await render(sample);
    const canvas = view.host.querySelector<HTMLElement>('.pm-canvas')!;
    canvas.setPointerCapture = vi.fn();
    vi.spyOn(canvas, 'getBoundingClientRect').mockReturnValue({ left: 100, top: 50 } as DOMRect);
    const pointer = (type: string, x: number, y: number) => canvas.dispatchEvent(new MouseEvent(type, { bubbles: true, clientX: x, clientY: y }));
    try {
      await act(async () => { wheel(canvas, -100); });
      const before = transform(view.host);
      let duringDrag: WheelEvent;
      await act(async () => {
        pointer('pointerdown', 150, 100);
        duringDrag = wheel(canvas, -100);
        pointer('pointermove', 190, 130);
        pointer('pointerup', 190, 130);
      });
      expect(duringDrag!.defaultPrevented).toBe(false);
      expect(transform(view.host)).toEqual({ x: before.x + 40, y: before.y + 30, zoom: before.zoom });
      await act(async () => { pointer('pointermove', 230, 160); });
      expect(transform(view.host)).toEqual({ x: before.x + 40, y: before.y + 30, zoom: before.zoom });
    } finally { await view.done(); }
  });

  it('keeps exact project, revision, type and path identity, including same names and unusual paths', () => {
    const index = indexProjectMap(sample, 'p', revision)!;
    expect(index.fileCount).toBe(3);
    const first = nodeId('p', revision, 'file', 'src/深 层/shared.py');
    const second = nodeId('p', revision, 'file', 'tests/shared.py');
    expect(first).not.toBe(second);
    expect(first).not.toBe(nodeId('p', 'b'.repeat(40), 'file', 'src/深 层/shared.py'));
    expect(index.nodes.get(first)?.name).toBe('shared.py');
    expect(ancestorIds(index, first)).toEqual([index.rootId, nodeId('p', revision, 'directory', 'src'), nodeId('p', revision, 'directory', 'src/深 层')]);
    expect(index.nodes.has(nodeId('p', revision, 'file', 'src/深 层/很长的文件名 @#$.py'))).toBe(true);
    expect(indexProjectMap(map([file('C:\\private.txt')]), 'p', revision)?.incomplete).toBe(true);
  });

  it('limits the initial graph with thousands of files without limiting the indexed search scope', () => {
    const items = Array.from({ length: 3200 }, (_, i) => file(`src/file ${i}.py`));
    const index = indexProjectMap(map([dir('src', items)], 3200), 'p', revision)!;
    const initial = visibleMap(index, new Set([index.rootId]), new Map());
    expect(index.fileCount).toBe(3200);
    expect(initial.nodes).toHaveLength(2);
    const expanded = visibleMap(index, new Set([index.rootId, nodeId('p', revision, 'directory', 'src')]), new Map());
    expect(expanded.nodes).toHaveLength(26);
    expect(expanded.more[0].remaining).toBe(3176);
    const lastPage = visibleMap(index, new Set([index.rootId, nodeId('p', revision, 'directory', 'src')]), new Map([[nodeId('p', revision, 'directory', 'src'), 3192]]));
    expect(lastPage.nodes).toHaveLength(10);
    expect(lastPage.more).toHaveLength(1);
    expect(lastPage.more[0].direction).toBe('previous');
    expect(lastPage.nodes.some((entry) => entry.node.path === 'src/file 3199.py')).toBe(true);
  });

  it('searches folded descendants, locates the exact duplicate, toggles and closes details independently', async () => {
    const view = await render(sample);
    const input = view.host.querySelector<HTMLInputElement>('input[aria-label="搜索已分析文件或目录"]')!;
    await act(async () => { const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!; setter.call(input, 'tests/shared.py'); input.dispatchEvent(new Event('input', { bubbles: true })); });
    expect(view.host.querySelectorAll('.pm-results button')).toHaveLength(1);
    await act(async () => (view.host.querySelector('.pm-results button') as HTMLButtonElement).click());
    expect(view.host.querySelector('.pm-detail dd:nth-of-type(3)')?.textContent).toBe('tests/shared.py');
    expect(view.host.querySelectorAll('.pm-node')).toHaveLength(4);
    await act(async () => (view.host.querySelector('button[aria-label="折叠目录 tests"]') as HTMLButtonElement).click());
    expect(view.host.querySelector('.pm-detail')?.textContent).toContain('tests/shared.py');
    await act(async () => (view.host.querySelector('button[aria-label="关闭地图详情"]') as HTMLButtonElement).click());
    expect(view.host.querySelector('.pm-detail')).toBeNull();
    await view.done();
  });

  it('distinguishes empty, failure, partial and legacy responses', async () => {
    const empty = await render(map([], 0));
    expect(empty.host.textContent).toContain('不代表仓库没有文件'); await empty.done();
    const failed = await render(null, '地图接口不可用');
    expect(failed.host.textContent).toContain('地图接口不可用'); expect(failed.host.textContent).not.toContain('结构为空'); await failed.done();
    const partial = await render(map([file('x.py')], 2));
    expect(partial.host.textContent).toContain('文件数与接口记录不一致'); await partial.done();
    const unresolved = await render(map([file('C:\\private.txt')], 1));
    expect(unresolved.host.textContent).toContain('无法确认分析文件是否为空'); await unresolved.done();
    const old = await render({ ...sample, coverage: undefined, file_count: undefined });
    expect(old.host.textContent).toContain('覆盖范围未声明'); await old.done();
  });
});
