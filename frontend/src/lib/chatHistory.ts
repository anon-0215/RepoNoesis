import type { ChatHistoryRecord } from '../types';
import type { WorkbenchAnswer } from '../workbench/Workbench';

export const isRecordId = (value: unknown): value is string => typeof value === 'string' && /^[1-9]\d*$/.test(value);
// Decimal strings stay strings, including SQLite IDs larger than JS's safe integer range.
export function compareRecordIds(left: string, right: string): number {
  return left.length - right.length || (left < right ? -1 : left > right ? 1 : 0);
}
export function mergeHistoryPages(current: ChatHistoryRecord[], page: ChatHistoryRecord[]): ChatHistoryRecord[] {
  const byId = new Map(current.map(item => [item.record_id, item]));
  for (const item of page) if (!byId.has(item.record_id)) byId.set(item.record_id, item);
  return Array.from(byId.values()).sort((a, b) => compareRecordIds(a.record_id, b.record_id));
}
export function historyAnswers(history: ChatHistoryRecord[], session: WorkbenchAnswer[], workspaceId: string): WorkbenchAnswer[] {
  const sessionIds = new Set(session.map(item => item.recordId).filter(isRecordId));
  return [...history.filter(item => !sessionIds.has(item.record_id)).map(item => ({
    id: `saved:${item.record_id}`, recordId: item.record_id, history: item,
    question: item.question, workspaceId, projectId: item.project_id,
    revision: item.repository_revision ?? undefined, status: 'success' as const,
  })), ...session];
}
