# 源鉴 RepoNoesis Local Product Phase 1

## Product path

The local product path is separate from the frozen M5/Phase 6 evaluation
providers. It accepts either a clean local Git worktree or a public HTTPS Git
URL, performs static Python analysis, persists function/class chunks, builds a
real local BGE-M3 index, uses the existing bounded agent and validators, and
only presents a generated product answer when the configured
`openai_compatible` Chat Completions response passes the source-reference
constraints.

The application never executes the imported repository and never installs its
dependencies. Local import reads tracked files only. Public import disables
interactive credentials, submodules, Git LFS smudging, system/global Git
configuration, and repository hooks for the clone operation. URLs containing
credentials, non-HTTPS schemes, non-443 ports, query/fragment components, or
hosts resolving to non-public addresses are rejected.

## Configuration

Copy `D:\Project\RepoNoesis-v3\.env.example` to the ignored file
`D:\Project\RepoNoesis-v3\.env`. Configuration discovery is anchored to the
source tree and does not depend on the shell working directory. Existing
process variables take precedence. The backend must be restarted after any
backend setting changes; Vite must be restarted after `VITE_API_BASE_URL` or
`FRONTEND_PORT` changes.

OPS ENV1 makes this discovery explicit: ordinary imports, including
`app.main`, read configuration only from the existing process environment and
never search for `.env`. The production bootstrap `python -m app.run_server`
calls `load_environment()` before Uvicorn imports `app.main:app`. Tests provide
configuration through fixtures, explicit process variables, or temporary
sentinel files and never depend on the repository's real `.env`.

Required product generation values:

- `LLM_PROVIDER=openai_compatible`
- `LLM_BASE_URL`: provider API root that accepts `chat/completions`
- `LLM_API_KEY`: backend only; obtain it from the provider account
- `LLM_MODEL`: obtain the current model name from the provider's current
  official documentation
- `LLM_TIMEOUT_SECONDS`, `LLM_MAX_TOKENS`, `LLM_TEMPERATURE`, and
  `LLM_MAX_RETRIES`: bounded request controls
- `LLM_PLANNER_THINKING` and `LLM_ANSWER_THINKING`: optional, independent
  provider capabilities. Leave blank to omit `thinking` entirely, or set to
  `enabled` / `disabled` only when the configured provider and model document
  support. For DeepSeek V4 Gate C, set `LLM_PLANNER_THINKING=disabled`; the
  final-answer setting remains independent and may stay blank.

The client does not send `reasoning_effort`, `response_format`, or `stream` by
default. It never substitutes `reasoning_content` for final `content`. A
length-limited empty final answer is reported as `provider_output_truncated`,
while a stopped empty final answer is `provider_empty_content`; neither is
automatically retried with the unchanged token budget.

Required real embedding values:

- `EMBEDDING_ENABLED=true`
- `EMBEDDING_PROVIDER=local_bge_m3`
- `EMBEDDING_MODEL`: absolute local BGE-M3 snapshot directory, or a model ID
  already available in `EMBEDDING_CACHE_DIR`
- `EMBEDDING_DEVICE=auto`, `cpu`, `cuda`, or a valid `cuda:N`
- `EMBEDDING_OFFLINE=true`

On Windows, an unquoted absolute path such as
`D:\models\bge-m3\snapshots\<revision>` is accepted. Forward slashes are also
accepted. Do not put the API key in a frontend environment file, browser state,
database, test snapshot, command line, or support conversation.

`GET /api/config/status` and the `configuration` section of
`GET /api/health` report readiness, provider/model names, device/offline mode,
and missing variable names. They expose only the boolean
`api_key_configured`; they never return the key, its length, or any fragment.

## Persistence and duplicate imports

SQLite schema version 8 stores `source_type`, `source_location`, and a stable
identity derived from source, normalized location, and the Git commit. A
completed or in-progress import of the same revision is reused. A failed import
can be reanalyzed in the same project record. A new commit creates a new
identity, preserving the prior project and learning history. Public checkouts
are kept under the ignored `backend/data/runtime/repositories` directory.

## Startup

From the repository root, run `start_all.bat`, or start the two services with
`backend\run_backend.bat` and `frontend\run_frontend.bat`. The launchers use
repository-relative paths. Python/npm must be on `PATH`, or the backend may use
`backend\.venv\Scripts\python.exe`.

For a manual backend start, change to `backend` and run
`python -m app.run_server`. This is the sole normal application bootstrap and
preserves the required environment-before-configuration ordering.

The frontend sends only `{source_type, source}` for imports. It displays import
status, safe provider/model diagnostics, exact file/symbol/line citations, and
the warning that credentials remain backend-only.

## Verification gates

All commands below run from `D:\Project\RepoNoesis-v3\backend`. Default unit
tests remain offline and use injected fakes only.

Gate A loads real local BGE-M3 in offline mode, imports a temporary clean local
Git fixture, persists analysis/chunks, indexes embeddings and relations, and
runs the existing deterministic bounded-agent fallback to validate Evidence and
citations. It does not claim a generated-model acceptance:

```powershell
$env:EMBEDDING_ENABLED='true'; $env:EMBEDDING_PROVIDER='local_bge_m3'; $env:EMBEDDING_MODEL='D:\models\bge-m3\snapshots\<revision>'; $env:EMBEDDING_MODEL_REVISION='<revision>'; $env:EMBEDDING_DEVICE='auto'; $env:EMBEDDING_OFFLINE='true'; python -B -m app.local_product_smoke --gate-a
```

Gate B uses the same real embedding path with a public HTTPS URL:

```powershell
python -B -m app.local_product_smoke --gate-b https://public-host/owner/repository.git
```

When real provider credentials are absent, Gate C must exit with code 2 and the
safe code `provider_not_configured`; this is a conditional pass, not a real
DeepSeek pass. After the root `.env` is completed, rerun Gate C with this single
exact command:

```powershell
Set-Location D:\Project\RepoNoesis-v3\backend; D:\Programme\Anaconda\envs\gitlearnagent\python.exe -B -m app.local_product_smoke --gate-c
```

Gate C creates a temporary local fixture, uses real BGE-M3 and the configured
product provider, and passes only if the final result is both `agent_mode=bounded`
and `answer_mode=llm_grounded` with validated Evidence/citations. Deterministic
fallbacks, mock providers, and M5 evaluation providers cannot satisfy it.

### Safe Gate C diagnostics

Gate C injects a request-local, size-bounded diagnostics recorder. It reports
only fixed stage names, stable error codes, exception type names, counters,
booleans, enum values, safe HTTP/usage numbers, and response field/type
metadata. The recorder is disabled for normal API requests unless explicitly
injected, and its fields never participate in agent decisions.

Smoke diagnostics never contain prompts, messages, request or response bodies,
generated content, reasoning content, source or Evidence text, tool arguments or
results, headers, authorization data, credentials, or derived properties such as
their lengths, excerpts, summaries, or hashes. Provider and repository import
errors retain their existing safe error payloads.

Stable Gate failures use `smoke_embedding_configuration_incomplete`,
`smoke_no_python_chunks`, `smoke_validated_evidence_missing`,
`smoke_provider_grounding_failed`, or `smoke_stage_failed`. These diagnostics do
not relax any Gate C acceptance condition and do not authorize an automatic
retry of a paid provider request.

Final-answer diagnostics distinguish execution from success. The
`citation_validation_completed`, `relation_validation_completed`, and
`post_generation_validation_completed` fields mean only that the corresponding
validator returned; their optional `*_passed` partners report the accumulated
validation result. Candidate-level fields report whether a non-empty grounded
candidate was received, how many syntactically valid Evidence IDs it cited, and
whether it was ultimately accepted. The fixed
`final_answer_failure_reason_code` enum distinguishes empty output, token-budget
overflow, missing/malformed/unknown citations, missing locations, path or line
range mismatches, Evidence-to-location binding failure,
relation or post-generation validation failure, Provider failure, and deadline
exhaustion. Result-level `citation_count` remains the count of citations in the
returned answer and is deliberately separate from candidate citation counts.
`citation_validation_*` reports persisted Evidence snapshot validation through
`CitationValidator`; `grounded_reference_validation_*` separately reports the
candidate string's Evidence-ID and exact-location contract. Candidate locations
must use the exact repository-relative POSIX `path.py:start-end` supplied for
each cited Evidence ID. No path normalization or alternate absolute/Windows
display form is accepted.
The older `citation_validation_failed` value remains allowlisted for previously
recorded diagnostics, but new candidate checks emit the more specific location,
path, line-range, or binding code.

The bounded agent treats tool execution and final answer generation as separate
capacities. Exhausting `max_tool_calls`, the planner-token budget, or the bounded
planning steps stops further planning/tools, but does not consume the single
grounded finalization opportunity. If validated Evidence exists and the total
deadline has not expired, the agent may make at most one final-answer Provider
call using the existing `max_final_answer_tokens` limit. A validated grounded
answer finishes as `completed`; invalid generated references finish as
`final_answer_failed`; exhausted tools without sufficient Evidence finish as
`insufficient_evidence`. No budget value or Gate C condition is increased or
relaxed by this behavior.

## Final acceptance record

**Final status: `Local Product Phase 1: FULL PASS`.**

The code acceptance baseline is
`e07bfd16e16ecbb827ab002fb9f11274013b92e3`. The final Gate C was executed
locally on 2026-08-02 after that commit and exited with code 0. This section is
the durable acceptance summary; it deliberately excludes credentials,
environment-variable values, prompts, request or response bodies, answer text,
reasoning content, Evidence text, and source text.

| Gate | Final status | What the accepted run established |
| --- | --- | --- |
| A | PASS | A clean temporary local Git repository completed Python extraction, real offline local BGE-M3 indexing, relation indexing, bounded-agent deterministic fallback, and Evidence/citation validation. It did not claim a real generated-model result. |
| B | PASS | The public unauthenticated HTTPS Git import path completed the real-embedding product smoke with the same repository safety restrictions. It did not claim a real generated-model result. |
| C | PASS | The single final paid-provider run completed the real local-repository path with `openai_compatible`, `deepseek-v4-pro`, a bounded agent, an `llm_grounded` answer, and validated grounding. |

The accepted Gate C chain was:

```text
local Git repository
-> Python source extraction
-> local BGE-M3 CUDA embedding
-> hybrid retrieval
-> relation expansion
-> bounded agent
-> DeepSeek grounded answer
-> citation, relation, and post-generation validation
-> Gate C pass
```

Safe Gate C result metadata:

- repository input: local; resolved revision
  `496f4e32ecb954b65a0391d34b48f6c0fd0da5fb`;
- source inventory: 1 Python file and 1 code chunk;
- embedding: `local_bge_m3`, real, offline, CUDA, 1024 dimensions, 1 newly
  generated chunk;
- relation index: complete, 2 nodes and 3 edges;
- agent and answer: `agent_mode=bounded`, `answer_mode=llm_grounded`,
  `grounding_status=grounded`, 1 Evidence item and 1 citation;
- generation: `provider=openai_compatible`, `model=deepseek-v4-pro`, planner
  thinking explicitly disabled and answer thinking omitted;
- process result: `status=pass`, exit code 0.

The offline backend regression for the accepted code baseline recorded:

```text
Ran 514 tests
OK
```

There were no failures, errors, or skipped tests. The final documentation seal
did not rerun that regression because it changes no production or test code.

### What Phase 1 proves

- The configured local product can import and persist a clean local Python Git
  repository or an unauthenticated public HTTPS Git repository.
- It can extract Python function/class chunks with source locations, build a
  real offline BGE-M3 index, build static relations, and run the existing
  bounded Evidence/Citation pipeline.
- The product-neutral `openai_compatible` Chat Completions client can obtain a
  real DeepSeek answer and expose it only after grounding validation succeeds.
- Provider configuration failures, response-shape failures, agent stages, and
  Gate C assertions have bounded, redacted diagnostics.
- Project analysis, indexes, and M4 learning records are stored in SQLite and
  survive backend restarts.
- The minimal local UI can start an import and ask an evidence-grounded
  repository question without putting the API key in browser state.

### What Phase 1 does not prove

- production readiness, unattended operation, concurrent users, authentication,
  private repositories, or cloud deployment;
- correctness or performance for arbitrary repositories, large repositories,
  all Python language constructs, or languages other than the documented
  conservative analysis paths;
- a complete browser E2E over import, restart, project reopening, revision
  refresh, learning plans, tasks, attempts, and review;
- incremental analysis after a repository revision changes, partial rebuilds,
  multi-project organization, or automatic learning-history continuity across
  different project records;
- general DeepSeek quality, other OpenAI-compatible providers, arbitrary model
  configurations, token/cost behavior, or repeated paid-provider reliability;
- M5 live-pilot completion, comparative retrieval superiority, educational
  effectiveness, or mastery-prediction validity.

The Gate C fixture was intentionally tiny. Its successful real-provider and
real-embedding path is an integration acceptance, not a claim of comprehensive
quality, scale, language, platform, or production validation.

### Acceptance and documentation commits

- Code acceptance baseline:
  `e07bfd16e16ecbb827ab002fb9f11274013b92e3`.
- Documentation seal: the later commit containing this section and the Local
  Product Phase 2 plan. It records the accepted result but does not alter the
  accepted code baseline.

## Current limitations

- Local source must be the root of a clean Git worktree.
- Public import supports unauthenticated HTTPS repositories only.
- Imported repositories with no tracked Python source are rejected.
- The first product version is synchronous and intended for small/medium local
  repositories; frontend progress is coarse-grained.
- Provider-specific Base URLs and model names are intentionally not hardcoded
  and must be confirmed against the provider's current official documentation.
