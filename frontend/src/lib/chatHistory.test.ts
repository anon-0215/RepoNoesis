import { afterEach, describe, expect, it, vi } from 'vitest';
import { getChatHistory } from './api';
import { compareRecordIds, historyAnswers, mergeHistoryPages } from './chatHistory';
import { savedCitationBindings } from '../workbench/evidenceBindings';
import type { ChatHistoryPage, ChatHistoryRecord } from '../types';

const citation = { evidence_id: 'E5', path: 'src/a.py', qualified_name: 'a', start_line: 10, end_line: 20, snippet: 'saved', summary: '' };
const record = (record_id: string): ChatHistoryRecord => ({ record_id, project_id: 'p', repository_revision: 'rev',
  created_at: null, question: 'same question', answer: '[E5]', revision_source: 'saved', identity_status: 'reliable',
  citations: [citation], citations_status: 'available', metadata_status: 'available', snapshot_status: 'active',
  is_current_snapshot: true, full_file_available: true, full_file_unavailable_reason: null,
  recovery_metadata: { format_version: 1, metadata_truncated: false, execution_summary: { citation_validation_passed: false },
    citation_bindings: [{ evidence_id: 'E5', project_id: 'p', repository_revision: 'rev', path: 'src/a.py', qualified_name: 'a', symbol_name: 'a', start_line: 10, end_line: 20 }] } });
const page = (): ChatHistoryPage => ({ workspace_id: 'w', active_project_id: 'p', repository_revision: 'rev', activation_version: 3,
  items: [record('9007199254740993'), record('9007199254740992')], next_cursor: 'cursor' });
afterEach(() => vi.unstubAllGlobals());
describe('history wire contract and saved bindings', () => {
  it('keeps IDs as exact decimal strings and checks every page context', async () => {
    const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify(page()), { status: 200 })); vi.stubGlobal('fetch', fetch);
    const result = await getChatHistory('w', 'p', 'rev', 'cursor', 3);
    expect(result.items[0].record_id).toBe('9007199254740993');
    expect(result.items[0].recovery_metadata?.execution_summary?.citation_validation_passed).toBe(false);
    expect(fetch.mock.calls[0][0]).toContain('activation_version=3'); expect(fetch.mock.calls[0][0]).toContain('limit=20');
  });
  it.each(['workspace', 'revision', 'activation', 'numeric_id', 'duplicate', 'full_file'])('rejects unsafe page %s', async kind => {
    const value = page();
    if (kind === 'workspace') value.workspace_id = 'other';
    if (kind === 'revision') value.repository_revision = 'other';
    if (kind === 'activation') value.activation_version = 4;
    if (kind === 'numeric_id') (value.items[0] as unknown as { record_id: unknown }).record_id = 9007199254740992;
    if (kind === 'duplicate') value.items[1].record_id = value.items[0].record_id;
    if (kind === 'full_file') value.items[0].is_current_snapshot = false;
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify(value), { status: 200 })));
    await expect(getChatHistory('w', 'p', 'rev', null, 3)).rejects.toThrow('未展示');
  });
  it('merges pages by ID and preserves session UI identity and progress instead of matching text', () => {
    const history = mergeHistoryPages([record('9007199254740993')], [record('9'), record('9007199254740992')]);
    expect(history.map(item => item.record_id)).toEqual(['9', '9007199254740992', '9007199254740993']);
    const session = { id: 'pending-ui', recordId: '9007199254740993', question: 'same question', progress: [] };
    const displayed = historyAnswers(history, [session], 'w');
    expect(displayed).toHaveLength(3); expect(displayed[2]).toBe(session);
    expect(compareRecordIds('9', '10')).toBeLessThan(0);
  });
  it('uses saved identity for sparse markers and multiple fragments without creating Evidence', () => {
    const item = record('1'); item.citations.push({ ...citation, snippet: 'second', start_line: 11, end_line: 12 });
    const binding = savedCitationBindings(item.citations, item.recovery_metadata, 'p', 'rev', true).get('E5')!;
    expect(binding.source).toBe('saved_binding'); expect(binding.indices).toEqual([0, 1]); expect(binding.citations[1].snippet).toBe('second');
    expect(savedCitationBindings(item.citations, item.recovery_metadata, 'wrong', 'rev', true).size).toBe(0);
    expect(savedCitationBindings(item.citations, item.recovery_metadata, 'p', 'wrong', true).size).toBe(0);
    expect(savedCitationBindings(item.citations, null, 'p', 'rev', true).size).toBe(0);
    expect(savedCitationBindings(item.citations, item.recovery_metadata, 'p', 'rev', false).size).toBe(0);
  });
});
