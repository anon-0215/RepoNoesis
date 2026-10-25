import { useEffect, useRef, useState } from 'react';
import { getChatHistory } from './api';
import { mergeHistoryPages } from './chatHistory';
import type { ChatHistoryPage, ChatHistoryRecord } from '../types';

export interface HistoryContext { workspaceId: string; projectId: string; revision: string; generation: number }
export function useChatHistory(context: HistoryContext, pending: boolean, isAskPending: () => boolean) {
  const [items, setItems] = useState<ChatHistoryRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [cursor, setCursor] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const request = useRef(0);
  const active = useRef<AbortController | null>(null);
  const buffered = useRef<ChatHistoryPage | null>(null);
  const activationVersion = useRef<number | undefined>(undefined);
  const failedCursor = useRef<string | null>(null);
  const key = `${context.generation}:${context.workspaceId}:${context.projectId}:${context.revision}`;
  const latestKey = useRef(key); latestKey.current = key;
  const publishedKey = useRef(key);

  function publish(page: ChatHistoryPage) {
    publishedKey.current = key;
    setItems(current => mergeHistoryPages(current, page.items));
    setCursor(page.next_cursor); setLoaded(true);
    activationVersion.current = page.activation_version;
  }
  async function load(nextCursor: string | null) {
    if (!context.workspaceId || !context.projectId || active.current || buffered.current) return;
    const controller = new AbortController(); active.current = controller;
    const id = ++request.current; const owner = key; const captured = { ...context };
    failedCursor.current = nextCursor; setLoading(true); setError('');
    try {
      const page = await getChatHistory(captured.workspaceId, captured.projectId, captured.revision,
        nextCursor, activationVersion.current, controller.signal);
      if (id !== request.current || owner !== latestKey.current || controller.signal.aborted) return;
      if (isAskPending()) buffered.current = page;
      else publish(page);
    } catch (reason) {
      if (id !== request.current || owner !== latestKey.current || controller.signal.aborted) return;
      setError(reason instanceof Error ? reason.message : '历史加载失败，请重试。');
    } finally {
      if (id === request.current && owner === latestKey.current) { active.current = null; setLoading(false); }
    }
  }
  useEffect(() => {
    publishedKey.current = key;
    ++request.current; active.current?.abort(); active.current = null;
    buffered.current = null; activationVersion.current = undefined; failedCursor.current = null;
    setItems([]); setCursor(null); setError(''); setLoaded(false); setLoading(false);
    void load(null);
    return () => { ++request.current; active.current?.abort(); active.current = null; buffered.current = null; };
  }, [key]);
  useEffect(() => {
    if (!pending && !isAskPending() && buffered.current) {
      const page = buffered.current; buffered.current = null; publish(page);
    }
  }, [pending, key]);
  const owned = publishedKey.current === key;
  return { items: owned ? items : [], loading: owned && loading, error: owned ? error : '',
    cursor: owned ? cursor : null, loaded: owned && loaded, waiting: owned && buffered.current !== null,
    retry: () => void load(failedCursor.current), loadEarlier: () => { if (cursor) void load(cursor); } };
}
