import { describe, expect, it } from 'vitest';
import type { AskProgressEvent } from '../types';
import { toolProgressFields, toolProgressLabel } from './toolProgress';

const event = (tool: string, target?: unknown, status = 'succeeded') => ({
  tool, tool_target: target, status,
} as AskProgressEvent);

describe('live tool targets', () => {
  it('names actual lookup and requested read ranges without claiming coverage', () => {
    expect(toolProgressLabel(event('lookup_symbol', { symbol: '_AtomicFile.close' }))).toBe('查找定义：_AtomicFile.close');
    expect(toolProgressLabel(event('read_source', { path: 'src/中文.py', start_line: 12, end_line: 25 }))).toBe('读取源码：src/中文.py · 请求行 12–25');
    expect(toolProgressFields(event('read_source', { path: 'src/中文.py', start_line: 12, end_line: 25 }))).toEqual({ label: '读取源码', target: 'src/中文.py', range: '请求行 12–25' });
  });
  it('keeps older, failed and unknown events readable', () => {
    expect(toolProgressLabel(event('lookup_symbol'))).toBe('查找定义');
    expect(toolProgressLabel(event('lookup_symbol', { symbol: 'secret' }, 'failed'))).toBe('查找定义');
    expect(toolProgressLabel(event('constructor', { symbol: 'secret' }))).toBe('补充工具已执行');
    expect(toolProgressLabel(event('search_code', { query: 'private query' }))).toBe('检索相关代码');
  });
  it('ignores malformed or misleading targets instead of breaking the answer', () => {
    for (const target of [null, 12, 'string', [], { symbol: 'a\nsecret' }, { symbol: 'x'.repeat(161) }, { symbol: 'a\u202eb' }, { symbol: {} }]) {
      expect(toolProgressLabel(event('lookup_symbol', target))).toBe('查找定义');
    }
    for (const path of ['/tmp/a.py', 'C:/Users/a.py', '../a.py', 'src//a.py', 'a\u202eb.py', 'x'.repeat(501), 'src\\a.py', 'a\u0000b.py']) {
      expect(toolProgressLabel(event('read_source', { path, start_line: 1, end_line: 2 }))).toBe('读取源码');
    }
    for (const [start_line, end_line] of [[true, 2], [1, false], ['1', 2], [3, 2], [0, 2], [1, 10_000_001]]) {
      expect(toolProgressLabel(event('read_source', { path: 'src/a.py', start_line, end_line }))).toBe('读取源码');
    }
  });
});
