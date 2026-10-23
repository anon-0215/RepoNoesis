import { useLayoutEffect, useRef, useState, type RefObject } from 'react';
import type { WorkbenchAnswer } from './Workbench';

type Anchor = { id: string; offset: number };
/** One owner for conversation scroll; citation restoration remains entry-specific. */
export function useConversationScroll(hostRef: RefObject<HTMLElement | null>, answers: WorkbenchAnswer[], context: string,
  askPage: boolean, historyReady: boolean, projectLoading: boolean) {
  const reading = useRef({ context, initialized: false, latest: '', anchor: null as Anchor | null, top: 0, askPage });
  const expectedTop = useRef<number | null>(null);
  const [paused, setPaused] = useState(false);
  const cards = () => Array.from(hostRef.current?.querySelectorAll<HTMLElement>('[data-answer-id]') || []);
  function capture() {
    const host = hostRef.current;
    if (!host) return;
    const top = host.getBoundingClientRect().top;
    const card = cards().find(node => node.getBoundingClientRect().bottom > top);
    reading.current.anchor = card?.dataset.answerId ? { id: card.dataset.answerId, offset: card.getBoundingClientRect().top - top } : null;
    reading.current.top = host.scrollTop;
  }
  function write(top: number) {
    const host = hostRef.current;
    if (!host) return;
    // Instant positioning also respects prefers-reduced-motion. Capture the
    // clamped result so native scroll events cannot masquerade as user input.
    host.scrollTop = top;
    expectedTop.current = host.scrollTop;
    capture();
  }
  function restore() {
    const host = hostRef.current, anchor = reading.current.anchor;
    const card = cards().find(node => node.dataset.answerId === anchor?.id);
    if (host && card && anchor) write(host.scrollTop + card.getBoundingClientRect().top - host.getBoundingClientRect().top - anchor.offset);
    else write(reading.current.top);
  }
  function revealLatest() {
    const host = hostRef.current, nodes = cards(), card = nodes[nodes.length - 1];
    if (!host || !card) return;
    write(host.scrollTop + card.getBoundingClientRect().top - host.getBoundingClientRect().top - 16);
    reading.current.initialized = true;
    reading.current.latest = card.dataset.answerId || '';
    setPaused(false);
  }
  useLayoutEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    if (reading.current.context !== context) {
      reading.current = { context, initialized: false, latest: '', anchor: null, top: 0, askPage };
      expectedTop.current = null;
      setPaused(false);
    }
    const returning = askPage && !reading.current.askPage;
    reading.current.askPage = askPage;
    if (!askPage || projectLoading) return;
    const latest = answers[answers.length - 1];
    const newlyPending = latest?.status === 'pending' && latest.id !== reading.current.latest;
    if (newlyPending || (!reading.current.initialized && historyReady)) {
      revealLatest();
      if (!latest) reading.current.initialized = true;
    } else if (returning || reading.current.anchor) restore();
    reading.current.latest = latest?.id || '';
    capture();
    const onScroll = () => {
      if (expectedTop.current !== null && Math.abs(host.scrollTop - expectedTop.current) < 1) {
        expectedTop.current = null; capture(); return;
      }
      expectedTop.current = null;
      if (host.scrollTop < reading.current.top - 2) setPaused(true);
      capture();
    };
    // Explicit upward gestures pause even before the native scroll event.
    const onWheel = (event: WheelEvent) => { if (event.deltaY < 0) setPaused(true); };
    const onKey = (event: KeyboardEvent) => { if (['ArrowUp', 'PageUp', 'Home'].includes(event.key) && !(event.target instanceof HTMLTextAreaElement)) setPaused(true); };
    host.addEventListener('scroll', onScroll, { passive: true });
    host.addEventListener('wheel', onWheel, { passive: true });
    host.addEventListener('keydown', onKey);
    return () => { host.removeEventListener('scroll', onScroll); host.removeEventListener('wheel', onWheel); host.removeEventListener('keydown', onKey); };
  }, [answers, context, askPage, historyReady, projectLoading]);
  return { paused, revealLatest };
}
