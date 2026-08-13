# OPS ENV1: test environment and `.env` loading isolation

## Scope

OPS ENV1 is an engineering-boundary correction, not a product capability or a
new V3 milestone. It does not change API schemas, schema version 11, provider or
embedding semantics, prompts, agent budgets, retrieval behavior, or LP2.1 to
LP2.3 behavior.

## Frozen contract

- `app.main` defines the FastAPI application and routes but does not call
  `load_environment()` during import.
- `get_env_value()` reads only the current process environment. It never falls
  back to a disk `.env`.
- `load_environment()` is the explicit disk-loading operation. It reads only
  the repository-root `.env`, and uses `os.environ.setdefault`, so existing
  process values keep precedence. Repeated calls are idempotent under that
  precedence rule. File or decoding failures stop bootstrap with a stable safe
  error that contains no configuration value.
- `app.run_server` is the normal production bootstrap. It calls
  `load_environment()` before host/port configuration is consumed and before
  Uvicorn imports `app.main:app`.
- `app.main` defers database and local learner initialization until FastAPI
  lifespan startup or first actual use. Import therefore creates no SQLite
  schema or persisted learner record.
- Tests import the safe application module and obtain configuration from
  fixtures, explicit process variables, dependency replacement, or temporary
  sentinel `.env` files. Tests do not import the production bootstrap merely to
  obtain the application.

An application factory was not introduced. The existing `app.main:app`
contract remains usable by the bootstrap, and the smaller separation was
sufficient.

## Startup paths

From the repository root:

```powershell
start_all.bat
backend\run_backend.bat
```

For a manual backend start:

```powershell
cd backend
python -m app.run_server
```

The batch and PowerShell launchers already delegate to `python -m
app.run_server`. Direct `uvicorn app.main:app` is not a supported production
startup command because it intentionally bypasses `.env` loading.

## Test design and proof boundary

The OPS ENV1 tests use an isolated subprocess, a temporary directory, a
temporary `.env`, a single meaningless sentinel variable, an isolated SQLite
path, and a file-read probe. They cover ordinary import and reload, explicit
bootstrap ordering, process-variable precedence, repeated loading, OpenAPI
generation, and the absence of socket bind/connect during application import.

The initial red run executed:

```powershell
D:\Programme\Anaconda\envs\gitlearnagent\python.exe -B -m unittest tests.test_environment_isolation -v
```

It ran 4 tests and failed 2: ordinary `get_env_value()` returned the temporary
sentinel, and importing/reloading `app.main` recorded 110 reads of the temporary
candidate `.env`. After the boundary change, the expanded 6-test module passes.

Final offline verification recorded:

- OPS ENV1 focused tests: 6/6 passed.
- LP2.1: 13/13; LP2.2: 18/18; LP2.3: 12/12.
- M1: 27/27; M2: 55/55; M3: 56/56; M4: 32/32.
- Full backend discovery: 563/563 passed.
- Frontend Vitest: 9/9 passed; TypeScript checking and Vite production build
  passed using the configured bundled Node executable and existing
  `node_modules`. The initial `npm test` attempt did not execute because `npm`
  was not on `PATH`; it is not counted as a test result.

This evidence proves behavior of project-controlled code paths, the temporary
candidate-file probe, and the isolated subprocess. It does not claim that the
operating system performed no metadata lookup whatsoever or that every present
or future third-party library can never inspect an environment file.

The repository's real `.env` was checked only for path existence and Git
ignore/tracking status. Its contents were not opened, read, printed, copied,
modified, deleted, or committed. No real network, BGE-M3, embedding, provider,
Gate A/B/C, P2 live Gate, or M5 live pilot was run for OPS ENV1.
