import { describe, expect, it } from 'vitest';

import { buildAnalyzePayload, citationLabel, providerSummary } from './product';

describe('local product presentation', () => {
  it('switches between the two explicit source payloads', () => {
    expect(buildAnalyzePayload('local', ' D:\\code\\repo ')).toEqual({
      source_type: 'local',
      source: 'D:\\code\\repo'
    });
    expect(buildAnalyzePayload('git_url', ' https://example.test/repo.git ')).toEqual({
      source_type: 'git_url',
      source: 'https://example.test/repo.git'
    });
  });

  it('shows missing provider settings without credentials', () => {
    const text = providerSummary({
      llm: {
        provider: null,
        model: null,
        base_url_configured: false,
        api_key_configured: false,
        ready: false,
        missing: ['LLM_API_KEY', 'LLM_MODEL']
      },
      embedding: {
        provider: 'local_bge_m3',
        model: 'BAAI/bge-m3',
        device: 'cpu',
        offline: true,
        enabled: true,
        ready: true,
        missing: []
      }
    });
    expect(text).toBe('生成模型未配置：LLM_API_KEY, LLM_MODEL');
  });

  it('renders file symbol and exact line range for evidence', () => {
    expect(
      citationLabel({
        path: 'src/app.py',
        summary: 'answer',
        snippet: 'def answer(): pass',
        qualified_name: 'app.answer',
        start_line: 10,
        end_line: 11
      })
    ).toBe('src/app.py:10-11 · app.answer');
  });
});
