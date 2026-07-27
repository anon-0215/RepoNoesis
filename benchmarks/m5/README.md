# M5 real-repository pilot benchmark

This directory contains versioned definitions, schemas, and annotation tools. Runtime databases,
provider caches, raw run records, and cloned repositories are deliberately outside this directory.

`pilot-v1` contains 36 scenarios over three fixed Python repository revisions and six controlled
learning sequences. The 36 scenarios and six adaptive sequences record `user_confirmed`
provenance from the 2026-07-27 review. This remains a small pilot, not an authoritative benchmark
or evidence of teaching effectiveness.

The external checkouts are expected below an operator-supplied repository root with checkout names
`itsdangerous`, `click`, and `httpx`. The validator verifies the full commit SHA, content fingerprint,
paths, AST symbols, spans, hashes, relation identities, budgets, provenance, duplicates, and unknown
fields before a run.

From `backend`:

```powershell
python -B -m app.m5 list-modes
python -B -m app.m5 validate --dataset ..\benchmarks\m5\datasets\pilot-v1 --repository-root <root>
python -B -m app.m5 dry-run --dataset ..\benchmarks\m5\datasets\pilot-v1 --repository-root <root> --artifacts ..\artifacts\m5 --mode fixed_lexical_rag
python -B -m app.m5 run --dataset ..\benchmarks\m5\datasets\pilot-v1 --repository-root <root> --artifacts ..\artifacts\m5 --mode fixed_lexical_rag
```

The artifacts argument is a base directory, not a run directory. The runner always writes fake runs
below `<artifacts>/fake/runs` and live runs below `<artifacts>/live/runs`; their embedding caches are
isolated in the matching `<artifacts>/<run-type>/cache` tree. A manifest records `run_type`, the full
non-sensitive run identity, budgets and actual ledgers, completion state, partial state, and stop reason.
An existing manifest or checkpoint whose identity differs is rejected. Resume therefore cannot cross
dataset, live/fake, provider endpoint/model/revision, embedding configuration, evaluator identity, or
result-affecting configuration boundaries.

For local Hugging Face snapshots, embedding identity fields have distinct meanings:
`configured_revision` is the explicit configuration, `resolved_revision` is a verified plain
40-character commit SHA (or `null`), `local_snapshot_identity` is a non-sensitive path hash, and
`model_identity` is the composite identity used for run/cache/resume isolation. A local revision is
resolved only when the directory is a `models--*/snapshots/<sha>` snapshot with the required small
SentenceTransformer configuration files, the configured SHA matches the directory, and `refs/main`
also matches when present. Live runs fail closed when that revision cannot be verified. Absolute model
paths are never written to manifests.

An exact finite pilot plan uses repeated `--cell <scenario-id>::<mode>` arguments. Eighteen arguments
mean eighteen cells, not eighteen scenarios multiplied by all modes. `--batch-count N` and zero-based
`--batch-index I` deterministically partition the same complete plan; later batches use `--resume` and
do not execute already completed cells. Request, input-token, output-token, USD-cost, wall-clock, and
per-request timeout limits are explicit CLI options. Retries consume request and observed token/cost
budgets. A stopped or batch-only run is reported as partial and its checkpoint remains auditable.

Live calls require `--live`, the relevant `M5_ALLOW_*` gates, and explicit identities. No provider or
model is inferred for an M5 live run. Configure only the required variable names for the chosen plan:

- answer: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `M5_LLM_MODEL_REVISION`;
- embedding: `EMBEDDING_MODEL_NAME_OR_PATH`, `EMBEDDING_MODEL_REVISION`,
  `M5_EMBEDDING_DIMENSION`, plus the existing encoding variables such as device, batch size,
  maximum length, normalization, query prefix, and document prefix;
- evaluator: `M5_EVALUATOR_MODEL`, `M5_EVALUATOR_MODEL_REVISION`, and optionally independent
  `M5_EVALUATOR_BASE_URL` and `M5_EVALUATOR_API_KEY` (otherwise the answer endpoint/key is reused);
- pricing: `M5_ANSWER_PRICING_MODEL`, `M5_ANSWER_PRICING_CURRENCY`, `M5_ANSWER_INPUT_PRICE_PER_UNIT`,
  `M5_ANSWER_OUTPUT_PRICE_PER_UNIT`, `M5_ANSWER_PRICING_UNIT_TOKENS`,
  `M5_ANSWER_PRICING_SOURCE`, and the corresponding `M5_EVALUATOR_*` names;
- gates: `M5_ALLOW_NETWORK`, `M5_ALLOW_REAL_LLM`, `M5_ALLOW_MODEL_LOAD`, and
  `M5_ALLOW_PAID_EVAL` when the evaluator is used.

M5 evaluator independence means an independently declared evaluator identity. The manifest states
whether it uses an independent endpoint, a different model on the shared endpoint, or the same model;
the protocol does not claim process or vendor independence when an endpoint is shared. Pricing is
operator-supplied and auditable; there are no built-in model prices. Missing usage or pricing is
`unknown`, never zero. A live smoke run may retain unknown cost, while `pilot` and `full` live purposes
fail closed without explicit pricing and positive answer/evaluator cost budgets for providers they use.

Normal application startup never imports this package, runs a benchmark, downloads a model, or calls
a paid provider.
