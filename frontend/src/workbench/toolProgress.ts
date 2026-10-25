import type { AskProgressEvent } from '../types';

// Optional live details must never make an older or malformed stream unusable.
export function toolProgressFields(event: AskProgressEvent): { label: string; target?: string; range?: string } {
  const label = event.tool === 'lookup_symbol' ? '查找定义'
    : event.tool === 'read_source' ? '读取源码'
    : event.tool === 'search_code' ? '检索相关代码'
    : event.tool === 'expand_relations' ? '查看关联代码'
    : event.tool === 'validate_evidence' ? '核查引用'
    : '补充工具已执行';
  const target = event.tool_target;
  if (event.status !== 'succeeded' || !target || typeof target !== 'object' || Array.isArray(target)) return { label };
  if (event.tool === 'lookup_symbol' && 'symbol' in target && typeof target.symbol === 'string'
    && target.symbol.length <= 160 && /^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$/.test(target.symbol)) {
    return { label, target: target.symbol };
  }
  if (event.tool === 'read_source' && 'path' in target && typeof target.path === 'string'
    && target.path.length > 0 && target.path.length <= 500
    && !/[\p{Cc}\p{Cf}\p{Cs}:\\]/u.test(target.path)
    && target.path.split('/').every(part => part && part !== '.' && part !== '..')
    && Number.isInteger(target.start_line) && Number.isInteger(target.end_line)
    && target.start_line >= 1 && target.end_line >= target.start_line && target.end_line <= 10_000_000) {
    return { label, target: target.path, range: `请求行 ${target.start_line}–${target.end_line}` };
  }
  return { label };
}

export function toolProgressLabel(event: AskProgressEvent): string {
  const { label, target, range } = toolProgressFields(event);
  return `${label}${target ? `：${target}` : ''}${range ? ` · ${range}` : ''}`;
}
