import type { Citation, ConfigStatus } from '../types';

export type SourceType = 'local' | 'git_url';

export function buildAnalyzePayload(sourceType: SourceType, source: string) {
  return { source_type: sourceType, source: source.trim() };
}

export function providerSummary(status: ConfigStatus | null): string {
  if (!status) return '正在读取后端配置状态…';
  if (!status.llm.ready) return `生成模型未配置：${status.llm.missing.join(', ')}`;
  return `${status.llm.provider} / ${status.llm.model}`;
}

export function citationLabel(citation: Citation): string {
  const symbol = citation.qualified_name ? ` · ${citation.qualified_name}` : '';
  return `${citation.path}:${citation.start_line}-${citation.end_line}${symbol}`;
}
