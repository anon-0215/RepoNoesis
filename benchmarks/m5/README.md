# M5 real-repository pilot benchmark

This directory contains versioned definitions, schemas, and annotation tools. Runtime databases,
provider caches, raw run records, and cloned repositories are deliberately outside this directory.

`pilot-v1` contains 36 scenarios over three fixed Python repository revisions and six controlled
learning sequences. All gold annotations are marked
`agent_curated_pending_human_review`; this is a developer-curated pilot, not an authoritative
benchmark or evidence of teaching effectiveness.

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

Real calls additionally require `--live` and the relevant `M5_ALLOW_*` environment gates. Normal
application startup never imports this package, runs a benchmark, downloads a model, or calls a paid
provider.
