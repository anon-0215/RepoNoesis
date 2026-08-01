# RepoNoesis Local Product Phase 1

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

## Current limitations

- Local source must be the root of a clean Git worktree.
- Public import supports unauthenticated HTTPS repositories only.
- Imported repositories with no tracked Python source are rejected.
- The first product version is synchronous and intended for small/medium local
  repositories; frontend progress is coarse-grained.
- Provider-specific Base URLs and model names are intentionally not hardcoded
  and must be confirmed against the provider's current official documentation.
