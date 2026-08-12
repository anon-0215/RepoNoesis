import { afterEach, describe, expect, it, vi } from 'vitest';

import { ApiError, analyzeProject, askProject } from './api';

afterEach(() => {
  vi.unstubAllGlobals();
});

async function captureApiError(promise: Promise<unknown>): Promise<ApiError> {
  try {
    await promise;
  } catch (error) {
    if (error instanceof ApiError) return error;
    throw error;
  }
  throw new Error('Expected ApiError');
}

function createValidStructuredFailure() {
  return {
    detail: {
      code: 'evidence_insufficient',
      message: 'PRIVATE_VALID_STRUCTURED_MESSAGE',
      retryable: false,
      diagnostics: {
        request_id: 'request-boundary-1',
        agent_mode: 'bounded',
        agent_status: 'insufficient_evidence',
        answer_mode: 'deterministic',
        failure_stage: 'retrieval',
        failure_reason_code: 'evidence_insufficient',
        retrieval_version: 'v1',
        hierarchy_mode: 'off',
        relation_mode: 'off',
        steps_used: 1,
        tool_calls_used: 1,
        planner_logical_calls: 1,
        planner_repair_calls: 0,
        final_answer_attempted: false,
        provider_logical_calls: 1,
        evidence_count: 1,
        citation_count: 0,
        citation_failure_reason_code: null as string | null,
        relation_failure_reason_code: null as string | null,
        elapsed_ms: 10
      }
    }
  };
}

function stubJsonResponse(data: unknown, status: number) {
  const json = vi.fn().mockResolvedValue(data);
  const fetchMock = vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    json
  });
  vi.stubGlobal('fetch', fetchMock);
  return { fetchMock, json };
}

function expectSafeStructuredFallback(error: ApiError, status = 422): void {
  expect(error.status).toBe(status);
  expect(error.message).toBe(
    `请求失败（HTTP ${status}），服务端未返回可安全展示的错误详情。`
  );
  expect(error.detail).toBeNull();
  expect(JSON.stringify(error)).not.toContain('PRIVATE_');
}

function expectSafeResponseParseError(error: ApiError, status: number, marker: string): void {
  expect(error).toBeInstanceOf(ApiError);
  expect(error.status).toBe(status);
  expect(error.message).toBe(
    `响应解析失败（HTTP ${status}），服务端未返回可安全处理的数据。`
  );
  expect(error.detail).toBeNull();
  expect(error.message).not.toContain(marker);
  expect(JSON.stringify(error.detail)).not.toContain(marker);
  expect(Object.getOwnPropertyNames(error)).not.toContain('cause');
  expect(Object.getOwnPropertyNames(error)).not.toContain('body');
  expect(Object.getOwnPropertyNames(error)).not.toContain('responseBody');
  expect(
    JSON.stringify({
      name: error.name,
      message: error.message,
      status: error.status,
      detail: error.detail,
      ownProperties: Object.getOwnPropertyNames(error)
    })
  ).not.toContain(marker);
}

describe('structured ask errors', () => {
  it('projects only safe diagnostic card fields into a detached object', async () => {
    const detail = {
      code: 'citation_unknown',
      message: 'PRIVATE_SERVER_MESSAGE',
      retryable: false,
      provider_body: 'PRIVATE_PROVIDER_BODY',
      prompt: 'PRIVATE_PROMPT',
      reasoning_content: 'PRIVATE_REASONING',
      stack: 'PRIVATE_STACK',
      authorization: 'PRIVATE_AUTHORIZATION',
      headers: 'PRIVATE_HEADERS',
      api_key: 'PRIVATE_API_KEY',
      raw_response: 'PRIVATE_RAW_RESPONSE',
      candidate_answer: 'PRIVATE_CANDIDATE_ANSWER',
      diagnostics: {
        request_id: 'request-1',
        agent_mode: 'bounded',
        agent_status: 'final_answer_failed',
        answer_mode: 'deterministic',
        failure_stage: 'citation_validation',
        failure_reason_code: 'citation_unknown',
        retrieval_version: 'v1',
        hierarchy_mode: 'off',
        relation_mode: 'off',
        steps_used: 1,
        tool_calls_used: 1,
        planner_logical_calls: 1,
        planner_repair_calls: 0,
        final_answer_attempted: true,
        provider_logical_calls: 2,
        evidence_count: 1,
        citation_count: 1,
        citation_failure_reason_code: 'citation_unknown',
        relation_failure_reason_code: null,
        elapsed_ms: 10,
        prompt: 'PRIVATE_DIAGNOSTICS_PROMPT',
        provider_body: 'PRIVATE_DIAGNOSTICS_PROVIDER_BODY',
        stack: 'PRIVATE_DIAGNOSTICS_STACK',
        extra: 'PRIVATE_DIAGNOSTICS_EXTRA'
      }
    };
    const parsedFailure = { detail };
    const { json } = stubJsonResponse(parsedFailure, 502);

    const error = await captureApiError(askProject('project-1', 'where'));
    expect(json).toHaveBeenCalledTimes(1);
    expect(error.status).toBe(502);
    expect(error.message).toBe('问答未生成可验证答案，请查看安全诊断卡片。');
    expect(error.detail).not.toBe(detail);
    expect(error.detail?.diagnostics).not.toBe(detail.diagnostics);
    expect(error.detail).toEqual({
      code: 'citation_unknown',
      message: '问答未生成可验证答案，请查看安全诊断卡片。',
      retryable: false,
      diagnostics: {
        request_id: 'request-1',
        agent_mode: 'bounded',
        agent_status: 'final_answer_failed',
        answer_mode: 'deterministic',
        failure_stage: 'citation_validation',
        failure_reason_code: 'citation_unknown',
        retrieval_version: 'v1',
        hierarchy_mode: 'off',
        relation_mode: 'off',
        steps_used: 1,
        tool_calls_used: 1,
        planner_logical_calls: 1,
        planner_repair_calls: 0,
        final_answer_attempted: true,
        provider_logical_calls: 2,
        evidence_count: 1,
        citation_count: 1,
        citation_failure_reason_code: 'citation_unknown',
        relation_failure_reason_code: null,
        elapsed_ms: 10
      }
    });
    expect(JSON.stringify(error)).not.toContain('PRIVATE_');

    detail.code = 'provider_error';
    detail.message = 'PRIVATE_MUTATED_SERVER_MESSAGE';
    detail.provider_body = 'PRIVATE_MUTATED_PROVIDER_BODY';
    detail.diagnostics.request_id = 'mutated-request';
    detail.diagnostics.evidence_count = 999;
    detail.diagnostics.failure_stage = 'provider';
    detail.diagnostics.extra = 'PRIVATE_MUTATED_DIAGNOSTICS';
    expect(error.detail?.code).toBe('citation_unknown');
    expect(error.detail?.message).toBe('问答未生成可验证答案，请查看安全诊断卡片。');
    expect(error.detail?.diagnostics.request_id).toBe('request-1');
    expect(error.detail?.diagnostics.evidence_count).toBe(1);
    expect(error.detail?.diagnostics.failure_stage).toBe('citation_validation');
    expect(error.detail).not.toHaveProperty('provider_body');
    expect(error.detail?.diagnostics).not.toHaveProperty('extra');
    expect(JSON.stringify(error)).not.toContain('PRIVATE_MUTATED');
  });

  it('uses a fixed frontend message for a whitelisted legacy error code', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            detail: {
              code: 'provider_unavailable',
              message: 'PRIVATE_LEGACY_MESSAGE',
              retryable: true,
              provider_body: 'MUST-NOT-BE-SHOWN'
            }
          }),
          { status: 503, headers: { 'Content-Type': 'application/json' } }
        )
      )
    );

    const error = await captureApiError(askProject('project-1', 'where'));
    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(503);
    expect(error.message).toBe('生成服务暂时不可用，请稍后重试。');
    expect(JSON.stringify(error)).not.toContain('PRIVATE_LEGACY_MESSAGE');
    expect(JSON.stringify(error)).not.toContain('MUST-NOT-BE-SHOWN');
    expect(error.detail).toBeNull();
  });

  it('projects the current structured provider failure contract without trusting its message', async () => {
    const marker = 'PRIVATE_STRUCTURED_PROVIDER_MESSAGE';
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            detail: {
              code: 'provider_not_configured',
              message: marker,
              retryable: false,
              diagnostics: {
                request_id: 'request-provider-1',
                agent_mode: 'unknown',
                agent_status: 'unknown',
                answer_mode: 'not_available',
                failure_stage: 'provider',
                failure_reason_code: 'provider_not_configured',
                retrieval_version: 'v1',
                hierarchy_mode: 'off',
                relation_mode: 'off',
                steps_used: 0,
                tool_calls_used: 0,
                planner_logical_calls: 0,
                planner_repair_calls: 0,
                final_answer_attempted: false,
                provider_logical_calls: 0,
                evidence_count: 0,
                citation_count: 0,
                citation_failure_reason_code: null,
                relation_failure_reason_code: null,
                elapsed_ms: 1,
                provider_body: marker
              },
              raw_response: marker
            }
          }),
          { status: 503, headers: { 'Content-Type': 'application/json' } }
        )
      )
    );

    const error = await captureApiError(askProject('project-1', 'where'));
    expect(error.status).toBe(503);
    expect(error.message).toBe('问答未生成可验证答案，请查看安全诊断卡片。');
    expect(error.detail?.code).toBe('provider_not_configured');
    expect(error.detail?.message).toBe('问答未生成可验证答案，请查看安全诊断卡片。');
    expect(error.detail?.diagnostics.failure_reason_code).toBe('provider_not_configured');
    expect(JSON.stringify(error)).not.toContain(marker);
  });

  it.each([
    'unsupported_provider',
    'provider_not_configured',
    'provider_authentication_failed',
    'provider_rate_limited',
    'provider_unavailable',
    'provider_output_truncated',
    'provider_empty_content',
    'provider_invalid_response',
    'provider_request_rejected',
    'provider_error'
  ])('projects the canonical backend provider code %s', async (code) => {
    const failure = createValidStructuredFailure();
    failure.detail.code = code;
    failure.detail.diagnostics.failure_reason_code = code;
    failure.detail.diagnostics.failure_stage = 'provider';
    failure.detail.diagnostics.agent_status = 'failed';
    stubJsonResponse(failure, 502);

    const error = await captureApiError(askProject('project-1', 'where'));

    expect(error.detail?.code).toBe(code);
    expect(error.detail?.diagnostics.failure_reason_code).toBe(code);
    expect(error.detail?.diagnostics.failure_stage).toBe('provider');
  });

  it('keeps the legacy provider_error mismatch compatibility provider-only', async () => {
    const accepted = createValidStructuredFailure();
    accepted.detail.code = 'provider_not_configured';
    accepted.detail.diagnostics.failure_reason_code = 'provider_error';
    accepted.detail.diagnostics.failure_stage = 'provider';
    accepted.detail.diagnostics.agent_status = 'failed';
    stubJsonResponse(accepted, 503);

    const acceptedError = await captureApiError(askProject('project-1', 'where'));
    expect(acceptedError.detail?.code).toBe('provider_not_configured');
    expect(acceptedError.detail?.diagnostics.failure_reason_code).toBe('provider_error');

    const rejected = createValidStructuredFailure();
    rejected.detail.code = 'embedding_not_configured';
    rejected.detail.diagnostics.failure_reason_code = 'provider_error';
    rejected.detail.diagnostics.failure_stage = 'provider';
    rejected.detail.diagnostics.agent_status = 'failed';
    stubJsonResponse(rejected, 503);

    const rejectedError = await captureApiError(askProject('project-1', 'where'));
    expect(rejectedError.detail).toBeNull();
    expect(rejectedError.message).toBe('本地 Embedding 服务尚未配置，请完成配置后重试。');
  });

  it('projects response_contract_invalid at the response stage with safe diagnostics', async () => {
    const failure = createValidStructuredFailure();
    failure.detail.code = 'response_contract_invalid';
    failure.detail.message = 'PRIVATE_RESPONSE_CONTRACT_DETAIL';
    failure.detail.diagnostics.failure_reason_code = 'response_contract_invalid';
    failure.detail.diagnostics.failure_stage = 'response';
    failure.detail.diagnostics.agent_status = 'completed';
    stubJsonResponse(failure, 500);

    const error = await captureApiError(askProject('project-1', 'where'));

    expect(error.detail?.code).toBe('response_contract_invalid');
    expect(error.detail?.diagnostics.request_id).toBe('request-boundary-1');
    expect(error.detail?.diagnostics.failure_stage).toBe('response');
    expect(error.detail?.diagnostics.evidence_count).toBe(1);
    expect(JSON.stringify(error)).not.toContain('PRIVATE_RESPONSE_CONTRACT_DETAIL');
  });

  it('rejects a well-formed but non-whitelisted legacy error code', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            detail: {
              code: 'unexpected_validation_error',
              message: 'PRIVATE_UNEXPECTED_VALIDATION',
              retryable: false
            }
          }),
          { status: 422, headers: { 'Content-Type': 'application/json' } }
        )
      )
    );

    const error = await captureApiError(askProject('project-1', 'where'));
    expect(error.status).toBe(422);
    expect(error.message).toBe(
      '请求失败（HTTP 422），服务端未返回可安全展示的错误详情。'
    );
    expect(JSON.stringify(error)).not.toContain('PRIVATE_UNEXPECTED_VALIDATION');
    expect(error.detail).toBeNull();
  });

  it('falls back safely when a structured diagnostic violates numeric bounds', async () => {
    const marker = 'PRIVATE_INVALID_DIAGNOSTIC';
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            detail: {
              code: 'evidence_insufficient',
              message: marker,
              retryable: false,
              diagnostics: {
                request_id: 'request-2',
                agent_mode: 'bounded',
                agent_status: 'insufficient_evidence',
                answer_mode: 'deterministic',
                failure_stage: 'retrieval',
                failure_reason_code: 'evidence_insufficient',
                retrieval_version: 'v1',
                hierarchy_mode: 'off',
                relation_mode: 'off',
                steps_used: 1,
                tool_calls_used: 1,
                planner_logical_calls: 1,
                planner_repair_calls: 0,
                final_answer_attempted: false,
                provider_logical_calls: 1,
                evidence_count: -1,
                citation_count: 0,
                citation_failure_reason_code: null,
                relation_failure_reason_code: null,
                elapsed_ms: 10
              }
            }
          }),
          { status: 422, headers: { 'Content-Type': 'application/json' } }
        )
      )
    );

    const error = await captureApiError(askProject('project-1', 'where'));
    expect(error.status).toBe(422);
    expect(error.message).toBe(
      '请求失败（HTTP 422），服务端未返回可安全展示的错误详情。'
    );
    expect(error.detail).toBeNull();
    expect(JSON.stringify(error)).not.toContain(marker);
  });

  it.each([NaN, Infinity, -Infinity, -1, 0.5, 1_000_001])(
    'rejects an invalid representative counter value: %s',
    async (value) => {
      const failure = createValidStructuredFailure();
      const diagnostics = failure.detail.diagnostics as unknown as Record<string, unknown>;
      diagnostics.steps_used = value;
      stubJsonResponse(failure, 422);

      expectSafeStructuredFallback(await captureApiError(askProject('project-1', 'where')));
    }
  );

  it.each([
    'steps_used',
    'tool_calls_used',
    'planner_logical_calls',
    'planner_repair_calls',
    'provider_logical_calls',
    'evidence_count',
    'citation_count'
  ])('applies the bounded integer contract to %s', async (field) => {
    const failure = createValidStructuredFailure();
    const diagnostics = failure.detail.diagnostics as unknown as Record<string, unknown>;
    diagnostics[field] = -1;
    stubJsonResponse(failure, 422);

    expectSafeStructuredFallback(await captureApiError(askProject('project-1', 'where')));
  });

  it.each([0, 1_000_000])('retains the valid counter boundary %s', async (value) => {
    const failure = createValidStructuredFailure();
    failure.detail.diagnostics.steps_used = value;
    stubJsonResponse(failure, 422);

    const error = await captureApiError(askProject('project-1', 'where'));
    expect(error.detail?.diagnostics.steps_used).toBe(value);
    expect(error.message).toBe('问答未生成可验证答案，请查看安全诊断卡片。');
  });

  it.each([NaN, Infinity, -1, 0.5, 86_400_001])(
    'rejects an invalid elapsed_ms value: %s',
    async (value) => {
      const failure = createValidStructuredFailure();
      failure.detail.diagnostics.elapsed_ms = value;
      stubJsonResponse(failure, 422);

      expectSafeStructuredFallback(await captureApiError(askProject('project-1', 'where')));
    }
  );

  it.each([0, 86_400_000])('retains the valid elapsed_ms boundary %s', async (value) => {
    const failure = createValidStructuredFailure();
    failure.detail.diagnostics.elapsed_ms = value;
    stubJsonResponse(failure, 422);

    const error = await captureApiError(askProject('project-1', 'where'));
    expect(error.detail?.diagnostics.elapsed_ms).toBe(value);
  });

  it.each([
    ['', 'empty'],
    ['a'.repeat(65), 'too long'],
    ['request id', 'space'],
    ['request/id', 'slash'],
    ['request\\id', 'backslash'],
    ['request\nid', 'newline'],
    ['request_id', 'other invalid character']
  ])('rejects a request_id containing %s (%s)', async (requestId) => {
    const failure = createValidStructuredFailure();
    failure.detail.diagnostics.request_id = requestId;
    stubJsonResponse(failure, 422);

    expectSafeStructuredFallback(await captureApiError(askProject('project-1', 'where')));
  });

  it('retains a valid alphanumeric and hyphenated request_id', async () => {
    const failure = createValidStructuredFailure();
    failure.detail.diagnostics.request_id = 'Request-123-ABC';
    stubJsonResponse(failure, 422);

    const error = await captureApiError(askProject('project-1', 'where'));
    expect(error.detail?.diagnostics.request_id).toBe('Request-123-ABC');
  });

  it.each([
    ['tool_timeout', 'tool', 503],
    ['deadline_exceeded', 'deadline', 504],
    ['response_contract_invalid', 'response', 500],
    ['persistence_failed', 'persistence', 500]
  ])('projects %s without exposing the server message', async (code, stage, status) => {
    const failure = createValidStructuredFailure();
    failure.detail.code = code;
    failure.detail.message = 'PRIVATE_SERVER_FAILURE';
    failure.detail.diagnostics.failure_reason_code = code;
    failure.detail.diagnostics.failure_stage = stage;
    failure.detail.diagnostics.agent_status = code === 'persistence_failed' ? 'completed' : 'failed';
    stubJsonResponse(failure, status);

    const error = await captureApiError(askProject('project-1', 'where'));
    expect(error.status).toBe(status);
    expect(error.detail?.code).toBe(code);
    expect(error.detail?.diagnostics.request_id).toBe('request-boundary-1');
    expect(JSON.stringify(error)).not.toContain('PRIVATE_SERVER_FAILURE');
  });

  it.each([
    'agent_mode',
    'agent_status',
    'answer_mode',
    'failure_stage',
    'failure_reason_code',
    'retrieval_version',
    'hierarchy_mode',
    'relation_mode',
    'citation_failure_reason_code',
    'relation_failure_reason_code'
  ])('rejects an invalid %s enum value', async (field) => {
    const failure = createValidStructuredFailure();
    const diagnostics = failure.detail.diagnostics as unknown as Record<string, unknown>;
    const marker = `PRIVATE_INVALID_${field.toUpperCase()}`;
    diagnostics[field] = marker;
    stubJsonResponse(failure, 422);

    const error = await captureApiError(askProject('project-1', 'where'));
    expectSafeStructuredFallback(error);
    expect(error.message).not.toContain(marker);
    expect(JSON.stringify(error)).not.toContain(marker);
  });
});

describe('successful response JSON safety boundary', () => {
  it.each([
    ['invalid JSON', '{"answer":"PRIVATE_200_INVALID_JSON_BODY"', 'PRIVATE_200_INVALID_JSON_BODY'],
    ['HTML', '<html>PRIVATE_200_HTML_BODY</html>', 'PRIVATE_200_HTML_BODY'],
    ['plain text', 'PRIVATE_200_PLAIN_TEXT_BODY', 'PRIVATE_200_PLAIN_TEXT_BODY']
  ])('converts an HTTP 200 %s body to a fixed ApiError', async (_label, body, marker) => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(body, { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    const error = await captureApiError(askProject('project-1', 'where'));
    expectSafeResponseParseError(error, 200, marker);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('does not retain a controlled parser exception or its sensitive message', async () => {
    const marker = 'PRIVATE_JSON_PARSER_ERROR';
    const parserError = new Error(marker);
    const json = vi.fn().mockRejectedValue(parserError);
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 200, json });
    vi.stubGlobal('fetch', fetchMock);

    const error = await captureApiError(askProject('project-1', 'where'));
    expectSafeResponseParseError(error, 200, marker);
    expect(json).toHaveBeenCalledTimes(1);
    expect(Object.values(error)).not.toContain(parserError);
  });

  it('does not retain a parser exception that is thrown synchronously', async () => {
    const marker = 'PRIVATE_SYNC_JSON_ERROR';
    const parserError = new Error(marker);
    const json = vi.fn(() => {
      throw parserError;
    });
    const response = { ok: true, status: 200, json };
    const fetchMock = vi.fn().mockResolvedValue(response);
    vi.stubGlobal('fetch', fetchMock);

    const error = await captureApiError(askProject('project-1', 'where'));
    expectSafeResponseParseError(error, 200, marker);
    expect((error as Error & { cause?: unknown }).cause).toBeUndefined();
    expect(
      Object.getOwnPropertyNames(error).filter(
        (property) => !['stack', 'message', 'name', 'status', 'detail'].includes(property)
      )
    ).toEqual([]);
    expect(Object.values(error)).not.toContain(parserError);
    expect(Object.values(error)).not.toContain(response);
    expect(json).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('returns a valid 2xx JSON ask response unchanged', async () => {
    const answer = {
      answer: 'grounded answer',
      citations: [],
      grounding_status: 'grounded',
      evidence: [],
      warnings: []
    };
    const json = vi.fn().mockResolvedValue(answer);
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 200, json });
    vi.stubGlobal('fetch', fetchMock);

    await expect(askProject('project-1', 'where')).resolves.toEqual(answer);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(json).toHaveBeenCalledTimes(1);
  });
});

describe('safe API error fallback', () => {
  it.each([400, 404, 409, 422])('does not expose an unknown %i response body', async (status) => {
    const marker = `PRIVATE_${status}_BODY`;
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            detail: {
              code: `private_${status}_error`,
              message: marker,
              retryable: false,
              raw_response: marker
            }
          }),
          { status, headers: { 'Content-Type': 'application/json' } }
        )
      )
    );

    const error = await captureApiError(askProject('project-1', 'where'));
    expect(error.status).toBe(status);
    expect(error.message).toBe(
      `请求失败（HTTP ${status}），服务端未返回可安全展示的错误详情。`
    );
    expect(error.message).not.toContain(marker);
    expect(JSON.stringify(error.detail)).not.toContain(marker);
    expect(JSON.stringify(error)).not.toContain(marker);
  });

  it.each([
    ['plain text 404', 404, 'PRIVATE-PLAIN-404'],
    ['plain text 500', 500, 'PRIVATE-PLAIN-TEXT'],
    ['HTML 502', 502, '<html>PRIVATE-STACK</html>'],
    ['invalid JSON', 500, '{"detail":"PRIVATE-BROKEN"'],
    [
      'unknown 5xx JSON',
      503,
      JSON.stringify({ detail: { message: 'PRIVATE-UNKNOWN', stack: 'PRIVATE-STACK' } })
    ],
    [
      'unknown shaped 5xx JSON',
      502,
      JSON.stringify({
        detail: {
          code: 'unexpected_server_error',
          message: 'PRIVATE-UNKNOWN-CODE',
          retryable: false
        }
      })
    ]
  ])('uses a fixed message for %s', async (_label, status, body) => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(body, { status })));

    const error = await captureApiError(askProject('project-1', 'where'));
    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(status);
    expect(error.message).toBe(
      `请求失败（HTTP ${status}），服务端未返回可安全展示的错误详情。`
    );
    expect(error.message).not.toContain('PRIVATE');
    expect(error.detail).toBeNull();
  });

  it('uses a fixed safe message when reading an error body fails', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        status: 502,
        json: vi.fn().mockRejectedValue(new Error('PRIVATE-BODY-READ'))
      })
    );

    const error = await captureApiError(askProject('project-1', 'where'));
    expect(error.status).toBe(502);
    expect(error.message).toBe(
      '请求失败（HTTP 502），服务端未返回可安全展示的错误详情。'
    );
    expect(JSON.stringify(error)).not.toContain('PRIVATE-BODY-READ');
  });

  it('uses a fixed safe message when reading an error body throws synchronously', async () => {
    const marker = 'PRIVATE_SYNC_JSON_ERROR';
    const parserError = new Error(marker);
    const json = vi.fn(() => {
      throw parserError;
    });
    const response = { ok: false, status: 502, json };
    const fetchMock = vi.fn().mockResolvedValue(response);
    vi.stubGlobal('fetch', fetchMock);

    const error = await captureApiError(askProject('project-1', 'where'));
    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(502);
    expect(error.message).toBe(
      '请求失败（HTTP 502），服务端未返回可安全展示的错误详情。'
    );
    expect(error.message).not.toContain(marker);
    expect(error.detail).toBeNull();
    expect((error as Error & { cause?: unknown }).cause).toBeUndefined();
    expect(
      Object.getOwnPropertyNames(error).filter(
        (property) => !['stack', 'message', 'name', 'status', 'detail'].includes(property)
      )
    ).toEqual([]);
    expect(Object.values(error)).not.toContain(parserError);
    expect(Object.values(error)).not.toContain(response);
    expect(json).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('uses a fixed safe message for network failures', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('PRIVATE-NETWORK-DETAIL')));

    const error = await captureApiError(askProject('project-1', 'where'));
    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(0);
    expect(error.message).toBe('无法连接后端服务。请确认后端已启动后重试。');
    expect(error.message).not.toContain('PRIVATE-NETWORK-DETAIL');
  });
});

describe('repository import safe error contract', () => {
  it.each([
    ['git_executable_unavailable', false, '后端未找到 Git 客户端'],
    ['git_dns_failed', true, '无法解析公开 Git 主机'],
    ['git_tls_failed', false, 'TLS 或证书校验失败'],
    ['git_connection_failed', true, '连接公开 Git 仓库时中断'],
    ['git_remote_not_found', false, '未找到指定的公开 Git 仓库'],
    ['git_authentication_required', false, '当前仅支持无需凭据的公开 HTTPS 仓库'],
    ['git_clone_timeout', true, '公开 Git 仓库克隆超时'],
    ['git_clone_failed', true, '公开 Git 仓库克隆失败'],
    ['local_repository_dirty', false, '未提交、已暂存或未跟踪文件'],
    ['local_repository_root_required', false, '请选择 Git 仓库的根目录'],
    ['local_path_not_found', false, '所选本地仓库目录不存在'],
    ['git_url_invalid', false, '公开 Git 地址无效'],
    ['git_url_private_host', false, '不支持私有或本机 Git 主机']
  ])('uses a fixed message for %s', async (code, retryable, expected) => {
    const marker = 'PRIVATE_SERVER_GIT_OUTPUT';
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            detail: {
              code,
              message: marker,
              retryable,
              request_id: 'request-import-123',
              stderr: marker,
              url: marker,
              local_path: marker
            }
          }),
          { status: code === 'git_clone_timeout' ? 504 : 502 }
        )
      )
    );

    const error = await captureApiError(
      analyzeProject('git_url', 'https://public.example/repository.git')
    );
    expect(error.message).toContain(expected);
    expect(error.message).toContain('请求 ID：request-import-123');
    expect(error.message.includes('可以重试此操作。')).toBe(retryable);
    expect(JSON.stringify(error)).not.toContain(marker);
    expect(error.detail).toBeNull();
  });

  it('rejects an invalid request_id instead of displaying server text', async () => {
    const marker = 'PRIVATE_INVALID_REQUEST_ID';
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            detail: {
              code: 'git_clone_failed',
              message: marker,
              retryable: true,
              request_id: '../bad id'
            }
          }),
          { status: 502 }
        )
      )
    );

    const error = await captureApiError(
      analyzeProject('git_url', 'https://public.example/repository.git')
    );
    expect(error.message).toBe(
      '请求失败（HTTP 502），服务端未返回可安全展示的错误详情。'
    );
    expect(JSON.stringify(error)).not.toContain(marker);
  });
});
