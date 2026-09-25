import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import type { AskFailure, ChatAnswer } from '../types';
import { ExecutionSummary } from './ExecutionSummary';

const answer: ChatAnswer = { answer: 'answer', citations: [], warnings: [], answer_mode: 'llm_grounded', grounding_status: 'grounded' };

describe('execution summary projection', () => {
  it('shows only independent aggregates and explicit validations on a completed request', () => {
    const markup = renderToStaticMarkup(<ExecutionSummary result={{ ...answer, execution_summary: { status: 'completed', answer_mode: 'llm_grounded', base_retrieval: { attempted: true, status: 'succeeded', new_evidence_count: 2 }, evidence_count: 2, citation_count: 1, planner_requests_attempted: 3, tool_calls_used: 1, tool_calls_attempted: 2, tool_calls_succeeded: 1, tool_calls_failed: 1, provider_logical_calls: 4, provider_http_attempt_count: 5, citation_validation_passed: true, agent_elapsed_ms: 900, planner_duration_ms: 50, finalization_duration_ms: 70 } }} />);
    expect(markup).toContain('基础检索：已完成'); expect(markup).toContain('模型逻辑调用 4 次'); expect(markup).toContain('HTTP 尝试 5 次'); expect(markup).toContain('BASE 外工具执行结果：成功 1 次 · 失败 1 次'); expect(markup).toContain('BASE 外工具执行尝试 2 次'); expect(markup).toContain('阶段耗时：Planner 50 毫秒 · 回答整理 70 毫秒'); expect(markup).toContain('不代表所有结论绝对正确'); expect(markup).toContain('服务端问答编排耗时：900');
  });
  it('keeps missing data unknown and reports truncation without counting remaining details', () => {
    const missing = renderToStaticMarkup(<ExecutionSummary result={answer} />);
    expect(missing).toContain('未提供结构化执行详情'); expect(missing).not.toContain('0 次');
    const truncated = renderToStaticMarkup(<ExecutionSummary result={{ ...answer, execution_summary: { status: 'completed', planner_termination_reason: 'planner_repair_failed', diagnostics_truncated: true } }} />);
    expect(truncated).toContain('补充检索提前结束'); expect(truncated).toContain('诊断详情已裁剪'); expect(truncated).not.toContain('工具调用 0');
  });
  it('labels BASE tool duration separately from BASE-excluded attempts', () => {
    const markup = renderToStaticMarkup(<ExecutionSummary result={{ ...answer, execution_mode: 'rag', execution_summary: {
      status: 'completed', planner_requests_attempted: 0, tool_calls_attempted: 0,
      tool_calls_used: 0, tool_duration_ms: 2719, agent_elapsed_ms: 6936,
    } }} />);
    expect(markup).toContain('BASE 外工具执行尝试 0 次');
    expect(markup).toContain('工具执行（含基础检索）2719 毫秒');
    expect(markup).toContain('服务端问答编排耗时：6936 毫秒');
    expect(markup).toContain('从编排开始至形成回答');
  });
  it('renders only the failed request diagnostic data', () => {
    const failure = { diagnostics: { agent_status: 'failed', failure_stage: 'planner', failure_reason_code: 'provider_error', request_id: 'r1', elapsed_ms: 321 } } as AskFailure;
    const markup = renderToStaticMarkup(<ExecutionSummary failure={failure} />);
    expect(markup).toContain('本次请求失败'); expect(markup).toContain('321 毫秒'); expect(markup).not.toContain('最终证据');
  });
  it('shows safe protocol subcodes and keeps an absent repair state unknown', () => {
    const diagnostics = {
      request_id: 'request-citation-1', agent_status: 'final_answer_failed',
      final_answer_attempted: true, failure_stage: 'citation_validation',
      failure_reason_code: 'citation_format_invalid', elapsed_ms: 10031,
      final_answer_initial_failure: { stable_code: 'model_supplied_location_forbidden', violation_kind: 'evidence_marker' },
      final_answer_repair_attempted: true, final_answer_repair_protocol_succeeded: false,
      final_answer_repair_succeeded: false,
      final_answer_repair_failure: { stable_code: 'citation_alias_unknown' }
    } as AskFailure['diagnostics'];
    const failure = { diagnostics } as AskFailure;
    const markup = renderToStaticMarkup(<ExecutionSummary failure={failure} mode="rag" />);
    expect(markup).toContain('执行模式：RAG');
    expect(markup).toContain('请求 ID：request-citation-1');
    expect(markup).toContain('首次协议失败子码：model_supplied_location_forbidden (evidence_marker)');
    expect(markup).toContain('有界修复：1 次');
    expect(markup).toContain('修复失败子码：citation_alias_unknown');
    const old = renderToStaticMarkup(<ExecutionSummary failure={{ diagnostics: {
      ...diagnostics, final_answer_repair_attempted: undefined,
      final_answer_repair_failure: undefined,
    } } as AskFailure} />);
    expect(old).toContain('有界修复：未提供');
    expect(old).not.toContain('0 次（未尝试）');
  });
  it('shows the safe retrieval counters for an insufficient evidence failure', () => {
    const failure = { diagnostics: { agent_status: 'insufficient_evidence', failure_stage: 'retrieval', failure_reason_code: 'evidence_insufficient', elapsed_ms: 21, base_retrieval: { attempted: true, status: 'zero_hit', retrieval_hit_count: 0, normalized_candidate_count: 0, valid_candidate_count: 0, new_evidence_count: 0, rejected_candidate_count: 0 } } } as AskFailure;
    const markup = renderToStaticMarkup(<ExecutionSummary failure={failure} />);
    expect(markup).toContain('基础检索：未检索到匹配');
    expect(markup).toContain('检索命中 0 条 · 候选通过 0 条 · 新增证据 0 条');
  });
  it('distinguishes semantic wait timeout from a stage that never started', () => {
    const timedOut = renderToStaticMarkup(<ExecutionSummary result={{ ...answer, execution_summary: {
      status: 'completed', base_retrieval: { attempted: true, status: 'succeeded', new_evidence_count: 1,
        retrieval_detail: { lexical_ms: 4, lexical_hit_count: 1, model_state: 'ready',
          semantic_status: 'timed_out', semantic_wait_ms: 210, retrieval_source: 'lexical', promotion_ms: 2 } }
    } }} />);
    expect(timedOut).toContain('语义阶段：执行中超时');
    expect(timedOut).toContain('词法检索');
    expect(timedOut).toContain('词法 4 毫秒');
    expect(timedOut).not.toContain('查询编码 0 毫秒');
    const skipped = renderToStaticMarkup(<ExecutionSummary result={{ ...answer, execution_summary: {
      status: 'completed', base_retrieval: { attempted: true, status: 'succeeded', new_evidence_count: 1,
        retrieval_detail: { semantic_status: 'skipped_budget', retrieval_source: 'lexical' } }
    } }} />);
    expect(skipped).toContain('语义阶段：剩余预算不足，未开始');
  });
});
