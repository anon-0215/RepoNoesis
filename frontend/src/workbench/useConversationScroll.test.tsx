// @vitest-environment jsdom
import React, { act, useRef } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { useConversationScroll } from './useConversationScroll';
import type { WorkbenchAnswer } from './Workbench';

const item = (id: string, status: WorkbenchAnswer['status'] = 'success'): WorkbenchAnswer => ({ id, question: id, status });
let positions: Record<string, number> = { old: 30, latest: 1000, new: 2200 };
let host: HTMLDivElement;
let root: ReturnType<typeof createRoot>;
function Harness({ items, page = 'ask', context = 'one', ready = true }: { items: WorkbenchAnswer[]; page?: string; context?: string; ready?: boolean }) {
  const ref = useRef<HTMLElement>(null);
  const scroll = useConversationScroll(ref, items, context, page === 'ask', ready, false);
  return <><section ref={ref} tabIndex={0}>{page === 'ask' && items.map(i => <article key={i.id} data-answer-id={i.id}>{i.question}</article>)}</section>{scroll.paused && <button onClick={scroll.revealLatest}>回到最新</button>}</>;
}
async function mount() {
  vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
  host = document.createElement('div'); document.body.append(host); root = createRoot(host);
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function(this: HTMLElement) {
    const top = this.tagName === 'ARTICLE' ? (positions[this.dataset.answerId!] || 0) - (host.querySelector('section')?.scrollTop || 0) : 0;
    return { top, bottom: top + 900 } as DOMRect;
  });
  await render([item('old'), item('latest')]);
}
async function render(items: WorkbenchAnswer[], page = 'ask', context = 'one', ready = true) {
  await act(async () => root.render(<Harness items={items} page={page} context={context} ready={ready} />));
}
async function move(top: number) {
  await act(async () => { const el = host.querySelector('section')!; el.scrollTop = top; el.dispatchEvent(new Event('scroll')); });
}
afterEach(async () => { if(root) await act(async () => root.unmount()); host?.remove(); vi.restoreAllMocks(); vi.unstubAllGlobals(); positions = { old: 30, latest: 1000, new: 2200 }; });
describe('conversation reading position', () => {
  it('reveals the latest group once and keeps a long answer at its beginning', async () => {
    await mount(); expect(host.querySelector('section')!.scrollTop).toBe(984);
    await render([item('old'), item('latest'), item('new', 'pending')]);
    expect(host.querySelector('section')!.scrollTop).toBe(2184);
    await render([item('old'), item('latest'), item('new')]);
    expect(host.querySelector('section')!.scrollTop).toBe(2184);
  });
  it('pauses on upward reading and preserves pagination before returning to latest', async () => {
    await mount(); await move(984); await move(180);
    expect(host.querySelector('button')?.textContent).toBe('回到最新');
    positions.old += 400; positions.latest += 400;
    await render([item('earlier'), item('old'), item('latest')]);
    expect(host.querySelector('section')!.scrollTop).toBe(580);
    await act(async () => host.querySelector<HTMLButtonElement>('button')!.click());
    expect(host.querySelector('section')!.scrollTop).toBe(1384);
    expect(host.querySelector('button')).toBeNull();
  });
  it('restores reading when returning from another page and resets for a different project', async () => {
    await mount(); await move(984); await move(180);
    await render([item('old'), item('latest')], 'map');
    await render([item('old'), item('latest')]);
    expect(host.querySelector('section')!.scrollTop).toBe(180);
    await render([], 'ask', 'two', false);
    await render([item('new')], 'ask', 'two');
    expect(host.querySelector('section')!.scrollTop).toBe(2184);
  });
  it('waits for first history and never pulls up to it after a new question has arrived', async () => {
    await mount(); await render([], 'ask', 'three', false);
    await render([item('new', 'pending')], 'ask', 'three', false);
    expect(host.querySelector('section')!.scrollTop).toBe(2184);
    await render([item('old'), item('new')], 'ask', 'three', true);
    expect(host.querySelector('section')!.scrollTop).toBe(2184);
  });
});
