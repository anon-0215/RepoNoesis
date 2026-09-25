import type { AskFailure, ChatAnswer, ExecutionMode, RetrievalDetail } from '../types';

const statusLabel: Record<string, string> = { completed: '已完成', degraded: '降级完成', insufficient_evidence: '证据不足', budget_exhausted: '预算用尽', tool_budget_exhausted: '工具预算用尽', final_answer_failed: '回答生成失败', failed: '失败', cancelled: '已取消' };
const baseLabel: Record<string, string> = { succeeded: '已完成', zero_hit: '未检索到匹配', all_rejected: '候选证据均未通过验证', failed: '失败', rejected: '结果被拒绝', timed_out: '超时', cancelled: '已取消', deadline_exceeded: '请求超时' };
export function baseRetrievalStatusLabel(status: string) { return baseLabel[status] || '状态未提供'; }
const recoverable = new Set(['planner_repair_failed', 'planner_budget_exhausted', 'repeat_call', 'no_progress', 'tool_budget_exhausted', 'unknown_tool', 'invalid_parameters']);
const count = (value?: number | null) => Number.isSafeInteger(value) && value! >= 0 ? value : null;
const semanticLabel: Record<string, string> = { completed: '已完成', skipped_budget: '剩余预算不足，未开始', timed_out: '执行中超时', capacity: '执行槽已满，未开始', failed: '可恢复故障', disabled: '未启用' };
function RetrievalDiagnostics({ detail }: { detail?: RetrievalDetail | null }) {
  if (!detail) return null;
  const timings = ([
    ['词法', detail.lexical_ms], ['模型加载/身份', detail.model_load_ms ?? detail.model_identity_ms],
    ['查询编码', detail.query_encode_ms], ['向量读取', detail.vector_read_ms],
    ['向量评分', detail.vector_score_ms], ['revision 核对', detail.revision_read_ms],
    ['融合', detail.fusion_ms], ['证据晋升', detail.promotion_ms],
  ] as const).filter((entry) => count(entry[1]) !== null).map(([label, value]) => `${label} ${value} 毫秒`);
  const sourceLabel: Record<string, string> = { hybrid: '混合检索', lexical: '词法检索', lexical_symbol: '词法与符号检索', symbol: '符号检索' };
  return <><p>检索来源：{sourceLabel[detail.retrieval_source || ''] || '未记录'} · 语义阶段：{semanticLabel[detail.semantic_status || ''] || '未记录'}{detail.model_state ? ` · 模型状态 ${detail.model_state}` : ''}</p>{timings.length > 0 && <p>实测阶段耗时：{timings.join(' · ')}<small>（未结束的阶段不计时；并行耗时不可相加为总耗时）</small></p>}</>;
}

const failureCode = (value?: { stable_code: string; violation_kind?: string } | null) => value
  ? `${value.stable_code}${value.violation_kind ? ` (${value.violation_kind})` : ''}` : '未提供';
const repairCount = (attempted?: boolean | null) => attempted === true ? '1 次' : attempted === false ? '0 次（未尝试）' : '未提供';
const repairResult = (value?: boolean | null) => value === true ? '通过' : value === false ? '未通过' : '未提供';

export function ExecutionSummary({ result, failure, mode }: { result?: ChatAnswer; failure?: AskFailure; mode?: ExecutionMode }) {
  const summary = result?.execution_summary;
  if (failure) {
    const d = failure.diagnostics;
    const base = d.base_retrieval;
    const answerFailed = d.final_answer_attempted || ['final_answer', 'citation_validation', 'relation_validation'].includes(d.failure_stage);
    const repairState = (value?: boolean | null) => d.final_answer_repair_attempted === false ? '未尝试' : d.final_answer_repair_attempted === true ? repairResult(value) : '未提供';
    return <details className="wb-execution"><summary>本次执行摘要</summary><div><p>本次请求失败：{statusLabel[d.agent_status] || '未完成'}</p><p>请求 ID：{d.request_id || '未提供'} · 执行模式：{mode === 'rag' ? 'RAG' : mode === 'agent' ? 'Agent' : '未提供'}</p><p>失败阶段：{d.failure_stage} · 原因代码：{d.failure_reason_code}</p>{answerFailed && <><p>首次协议失败子码：{failureCode(d.final_answer_initial_failure)}</p><p>有界修复：{repairCount(d.final_answer_repair_attempted)} · 协议重验：{repairState(d.final_answer_repair_protocol_succeeded)} · 最终修复结果：{repairState(d.final_answer_repair_succeeded)}</p><p>修复失败子码：{d.final_answer_repair_attempted === false ? '未尝试' : failureCode(d.final_answer_repair_failure)} · 最后协议子码：{failureCode(d.final_answer_protocol_failure)}</p></>}{base && <><p>基础检索：{base.attempted ? baseRetrievalStatusLabel(base.status) : '未执行'}</p>{base.attempted && ['succeeded', 'zero_hit', 'all_rejected'].includes(base.status) && <p>检索命中 {base.retrieval_hit_count} 条 · 候选通过 {base.valid_candidate_count} 条 · 新增证据 {base.new_evidence_count} 条</p>}<RetrievalDiagnostics detail={base.retrieval_detail} /></>}<p>服务端记录耗时：{d.elapsed_ms} 毫秒</p><small>仅显示该失败请求返回的诊断信息。</small></div></details>;
  }
  if (!result) return null;
  return <details className="wb-execution"><summary>本次执行摘要</summary><div>{!summary ? <p>此回答未提供结构化执行详情。</p> : <>
    <p>最终状态：{statusLabel[summary.status || ''] || '状态未提供'} · {summary.answer_mode === 'llm_grounded' ? '基于证据生成' : summary.answer_mode === 'deterministic' ? '确定性回答' : '回答方式未提供'}</p>
    {summary.base_retrieval && <p>基础检索：{summary.base_retrieval.attempted ? (baseLabel[summary.base_retrieval.status] || '状态已记录') : '未执行'}{summary.base_retrieval.attempted ? ` · 新增证据 ${summary.base_retrieval.new_evidence_count}` : ''}</p>}
    <RetrievalDiagnostics detail={summary.base_retrieval?.retrieval_detail} />
    {(count(summary.evidence_count) !== null || count(summary.citation_count) !== null) && <p>最终证据 {count(summary.evidence_count) ?? '未提供'} 条 · 引用 {count(summary.citation_count) ?? '未提供'} 条</p>}
    {(count(summary.planner_requests_attempted) !== null || count(summary.tool_calls_used) !== null || count(summary.tool_calls_attempted) !== null) && <p>Planner 请求 {count(summary.planner_requests_attempted) ?? '未提供'} 次{count(summary.tool_calls_attempted) !== null ? ` · BASE 外工具执行尝试 ${summary.tool_calls_attempted} 次` : ''}{count(summary.tool_calls_used) !== null ? ` · BASE 外工具调用预算已用 ${summary.tool_calls_used} 次` : ''}{count(summary.planner_repair_attempts) !== null ? ` · 修复请求 ${summary.planner_repair_attempts} 次` : ''}</p>}
    {result.execution_mode === 'agent' && summary.tool_calls_attempted === 0 && <p>未调用补充工具。</p>}
    {result.execution_mode === 'rag' && summary.tool_calls_attempted === 0 && <p>固定检索流程，无自主补充工具。</p>}
    {(count(summary.tool_calls_succeeded) !== null || count(summary.tool_calls_failed) !== null) && <p>BASE 外工具执行结果：成功 {count(summary.tool_calls_succeeded) ?? '未提供'} 次 · 失败 {count(summary.tool_calls_failed) ?? '未提供'} 次</p>}
    {(count(summary.provider_logical_calls) !== null || count(summary.provider_http_attempt_count) !== null) && <p>模型逻辑调用 {count(summary.provider_logical_calls) ?? '未提供'} 次 · HTTP 尝试 {count(summary.provider_http_attempt_count) ?? '未提供'} 次</p>}
    {summary.planner_termination_reason && <p>{summary.status === 'completed' && recoverable.has(summary.planner_termination_reason) ? '补充检索提前结束，已基于现有证据完成回答。' : '补充检索已结束。'} <small>原因代码：{summary.planner_termination_reason}</small></p>}
    {typeof summary.citation_validation_passed === 'boolean' && <p>引用身份与范围校验：{summary.citation_validation_passed ? '通过' : '未通过'}<small>（不代表所有结论绝对正确）</small></p>}
    {typeof summary.relation_validation_passed === 'boolean' && <p>关系校验：{summary.relation_validation_passed ? '通过' : '未通过'}</p>}
    {typeof summary.post_generation_validation_passed === 'boolean' && <p>生成后证据校验：{summary.post_generation_validation_passed ? '通过' : '未通过'}</p>}
    {count(summary.agent_elapsed_ms) !== null && <p>服务端问答编排耗时：{summary.agent_elapsed_ms} 毫秒 <small>（从编排开始至形成回答，不含此前的路由预处理及此后的保存）</small></p>}
    {(count(summary.planner_duration_ms) !== null || count(summary.tool_duration_ms) !== null || count(summary.finalization_duration_ms) !== null) && <p>阶段耗时：{[
      count(summary.planner_duration_ms) !== null ? `Planner ${summary.planner_duration_ms} 毫秒` : null,
      count(summary.tool_duration_ms) !== null ? `工具执行（含基础检索）${summary.tool_duration_ms} 毫秒` : null,
      count(summary.finalization_duration_ms) !== null ? `回答整理 ${summary.finalization_duration_ms} 毫秒` : null
    ].filter(Boolean).join(' · ')}</p>}
    {summary.diagnostics_truncated && <small>诊断详情已裁剪；上方计数仅使用独立聚合字段。</small>}
  </>}</div></details>;
}
