import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError, askProjectStream } from './api';

const revision = 'fixture-revision';
const frame = (sequence: number, type: string, extra: Record<string, unknown> = {}) => JSON.stringify({
  request_id: 'server-1', client_request_id: 'local-1', project_id: 'p',
  repository_revision: revision, sequence, type, ...extra
}) + '\n';
const result = { execution_mode: 'rag', answer: 'answer [E5]', citations: [], answer_mode: 'llm_grounded', grounding_status: 'grounded', warnings: [] };

afterEach(() => vi.unstubAllGlobals());

describe('ask progress stream', () => {
  it('delivers intermediate events before the terminal result across split chunks', async () => {
    let controller!: ReadableStreamDefaultController<Uint8Array>;
    const body = new ReadableStream<Uint8Array>({ start(value) { controller = value; } });
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(body, { status: 200 })));
    const seen: string[] = [];
    const pending = askProjectStream('p', revision, 'question', 'rag', 'local-1', (event) => seen.push(event.type));
    const encoder = new TextEncoder();
    controller.enqueue(encoder.encode(frame(1, 'request_received') + frame(2, 'base_started').slice(0, 20)));
    await vi.waitFor(() => expect(seen).toEqual(['request_received']));
    controller.enqueue(encoder.encode(frame(2, 'base_started').slice(20) + frame(3, 'unknown_future') + frame(3, 'unknown_future') + frame(4, 'base_completed', { new_evidence_count: 1 })));
    await vi.waitFor(() => expect(seen).toEqual(['request_received', 'base_started', 'base_completed']));
    const terminal = frame(5, 'completed', { result });
    controller.enqueue(encoder.encode(terminal.slice(0, 17)));
    controller.enqueue(encoder.encode(terminal.slice(17)));
    controller.close();
    expect((await pending).execution_mode).toBe('rag');
    expect(vi.mocked(fetch)).toHaveBeenCalledTimes(1);
  });

  it('treats disconnect, mismatched mode and context as unknown rather than success', async () => {
    const run = async (contents: string) => {
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(contents, { status: 200 })));
      return askProjectStream('p', revision, 'question', 'rag', 'local-1', vi.fn());
    };
    await expect(run(frame(1, 'base_started'))).rejects.toThrow('未收到最终状态');
    await expect(run(frame(1, 'completed', { result: { ...result, execution_mode: 'agent' } }))).rejects.toThrow('模式不一致');
    await expect(run(frame(1, 'completed', { result }).replace('fixture-revision', 'other-revision'))).rejects.toThrow('上下文不匹配');
  });

  it('ignores duplicate progress and rejects a failure terminal', async () => {
    const seen: string[] = [];
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
      frame(1, 'base_started') + frame(1, 'base_started') + frame(2, 'failed', { status: 500 }), { status: 200 })));
    await expect(askProjectStream('p', revision, 'question', 'rag', 'local-1', (event) => seen.push(event.type))).rejects.toThrow('问答失败');
    expect(seen).toEqual(['base_started']);
  });

  it('keeps distinct validation checkpoints and ignores a repeated sequence', async () => {
    const seen: Array<[number, string | undefined]> = [];
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
      frame(1, 'citation_checked', { checkpoint: 'initial_evidence', passed: true }) +
      frame(1, 'citation_checked', { checkpoint: 'initial_evidence', passed: true }) +
      frame(2, 'citation_checked', { checkpoint: 'generation_input', passed: true }) +
      frame(3, 'citation_checked', { checkpoint: 'post_answer_evidence', passed: true }) +
      frame(4, 'completed', { result }), { status: 200 })));
    await askProjectStream('p', revision, 'question', 'rag', 'local-1', (event) => seen.push([event.sequence, event.checkpoint]));
    expect(seen).toEqual([[1, 'initial_evidence'], [2, 'generation_input'], [3, 'post_answer_evidence']]);
  });

  it('settles on the first terminal and ignores a later conflicting terminal', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
      frame(1, 'completed', { result }) + frame(2, 'failed', { status: 500 }), { status: 200 })));
    const onProgress = vi.fn();
    await expect(askProjectStream('p', revision, 'question', 'rag', 'local-1', onProgress)).resolves.toMatchObject({ execution_mode: 'rag' });
    expect(onProgress).not.toHaveBeenCalled();
  });

  it('projects only bounded BASE diagnostics from a failed production frame', async () => {
    const diagnostics = {
      request_id: 'server-1', agent_mode: 'bounded', agent_status: 'insufficient_evidence',
      answer_mode: 'deterministic', failure_stage: 'retrieval', failure_reason_code: 'evidence_insufficient',
      retrieval_version: 'v1', hierarchy_mode: 'off', relation_mode: 'off',
      steps_used: 0, tool_calls_used: 0, planner_logical_calls: 0, planner_repair_calls: 0,
      final_answer_attempted: true, provider_logical_calls: 0, evidence_count: 0,
      citation_count: 0, citation_failure_reason_code: null, relation_failure_reason_code: null,
      elapsed_ms: 42, base_retrieval: { attempted: true, status: 'zero_hit',
        retrieval_hit_count: 0, normalized_candidate_count: 0, valid_candidate_count: 0,
        new_evidence_count: 0, rejected_candidate_count: 0, unsafe_detail: 'PRIVATE' }
    };
    const failure = { code: 'evidence_insufficient', message: 'PRIVATE', retryable: false, diagnostics };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(frame(1, 'failed', { failure }), { status: 200 })));
    try {
      await askProjectStream('p', revision, 'question', 'rag', 'local-1', vi.fn());
      throw new Error('expected failure');
    } catch (error) {
      expect(error).toBeInstanceOf(ApiError);
      const detail = (error as ApiError).detail;
      expect(detail?.diagnostics.base_retrieval?.status).toBe('zero_hit');
      expect(detail?.diagnostics.base_retrieval?.new_evidence_count).toBe(0);
      expect(JSON.stringify(detail)).not.toContain('PRIVATE');
    }
  });
  it('projects the same safe final-answer repair fields from a failed frame', async () => {
    const diagnostics = {
      request_id: 'server-1', agent_mode: 'bounded', agent_status: 'final_answer_failed',
      answer_mode: 'deterministic', failure_stage: 'citation_validation',
      failure_reason_code: 'citation_format_invalid', retrieval_version: 'v1',
      hierarchy_mode: 'off', relation_mode: 'off', steps_used: 0, tool_calls_used: 0,
      planner_logical_calls: 0, planner_repair_calls: 0, final_answer_attempted: true,
      final_answer_repair_attempted: true, final_answer_repair_protocol_succeeded: false,
      final_answer_repair_succeeded: false, final_answer_initial_failure: {
        stable_code: 'citation_alias_unknown', raw: 'PRIVATE_RAW'
      }, final_answer_repair_failure: { stable_code: 'model_supplied_location_forbidden' },
      provider_logical_calls: 2, evidence_count: 5, citation_count: 0,
      citation_failure_reason_code: 'citation_format_invalid', relation_failure_reason_code: null,
      elapsed_ms: 10031
    };
    const failure = { code: 'citation_format_invalid', message: 'PRIVATE', retryable: false, diagnostics };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(frame(1, 'failed', { failure }), { status: 200 })));
    try {
      await askProjectStream('p', revision, 'question', 'rag', 'local-1', vi.fn());
      throw new Error('expected failure');
    } catch (error) {
      expect(error).toBeInstanceOf(ApiError);
      const detail = (error as ApiError).detail;
      expect(detail?.diagnostics.final_answer_initial_failure?.stable_code).toBe('citation_alias_unknown');
      expect(detail?.diagnostics.final_answer_repair_failure?.stable_code).toBe('model_supplied_location_forbidden');
      expect(detail?.diagnostics.final_answer_repair_attempted).toBe(true);
      expect(JSON.stringify(detail)).not.toContain('PRIVATE');
    }
  });
});
