# Retrieval v2 Phase 5 results

## Acceptance

Retrieval v2 Phase 5 is completed and passed for the frozen Click pilot scope.
This is an offline retrieval evaluation, not an answer-quality, teaching-quality,
or cross-repository generalization claim.

The primary immutable run is
`artifacts/retrieval_v2_phase5/runs/retrieval-v2-phase5-e217b85b4e0ee809dce31d23`.
Its result hash is
`7ba0f395cbd8af927df8af316d212b1a6705f08d4f1880bdf038db9af7d9365a`.
Runtime artifacts are intentionally ignored by Git.

## Frozen inputs and real embedding gate

- Repository commit evaluated: `71c7631b50173b705a21188ecc4d359e6f52fa3a`.
- Dataset: `pilot-v1`, trusted `repo_id=click` subset, 12 queries: 11
  answerable and 1 unanswerable skip.
- Click revision: `00e592cea702e0b2caa0dee42489fdb1c22cd845`.
- Corpus: 663 Click chunks copied from the validated M5 engineering snapshot.
- Relation graph: complete, 687 nodes and 8,515 edges. Its hash remained
  `6205334ec1c9abb91ab61c43004fc31f31cb81d2b398b94c45a444be6fc42146`
  before and after embedding indexing.
- Real provider: `sentence-transformers`; local BGE-M3 snapshot and resolved
  revision `5617a9f61b028005a4858fdac845db406aefb181`.
- Frozen embedding configuration: CUDA, float32, normalized, 1,024 dimensions,
  batch size 8, max length 8,192, empty query/document prefixes, model-defined
  pooling.
- First index pass: 663 generated, 0 cached, 0 failed, 1,501,114.70 ms.
- Second index pass: 0 generated, 663 cached, 0 failed, 46.48 ms.
- Historical embedding rows removed from the copied multi-repository database:
  1,275. The source database was not modified; its before/after SHA-256 remained
  `a09b6314782247a1de0f3156a562933cd1e5d01e6d6ca6b9847c69808204a4a0`.
- Network attempts: 0. Dependency inventory and model snapshot metadata
  identities were unchanged after the run.
- Peak allocated CUDA memory reported by the process: 9,553,511,424 bytes.

## Five-path strict retrieval results

The denominator is the same 11 valid answerable queries for every path. Strict
matching requires revision, POSIX path, qualified symbol, exact span, and
content hash. The production result cap remains eight. Hit@10 and MRR@10 are
therefore lower bounds computed from at most eight returned candidates.

| Path | Hit@1 | Hit@3 | Hit@5 | Hit@8 | MRR@8 | Hit@10 lower bound | MRR@10 lower bound |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A: v1 | 0.181818 | 0.727273 | 0.818182 | 0.818182 | 0.427273 | 0.818182 | 0.427273 |
| B: plain v2 | 0.545455 | 0.909091 | 1.000000 | 1.000000 | 0.730303 | 1.000000 | 0.730303 |
| C: v2 + hierarchy | 0.545455 | 0.909091 | 1.000000 | 1.000000 | 0.730303 | 1.000000 | 0.730303 |
| D: v2 + relation | 0.545455 | 0.909091 | 0.909091 | 1.000000 | 0.727273 | 1.000000 | 0.727273 |
| E: v2 + hierarchy + relation | 0.545455 | 0.909091 | 0.909091 | 1.000000 | 0.725108 | 1.000000 | 0.725108 |

## Paired ablation conclusions

All comparisons use 11 paired queries, 2,000 bootstrap samples, and seed
`20260726`.

| Comparison | Mean MRR@10 lower-bound delta | Improved / unchanged / regressed | Paired bootstrap 95% CI |
| --- | ---: | ---: | ---: |
| A -> B | +0.303030 | 6 / 5 / 0 | [0.136364, 0.469697] |
| B -> C | +0.000000 | 0 / 11 / 0 | [0.000000, 0.000000] |
| B -> D | -0.003030 | 0 / 10 / 1 | [-0.009091, 0.000000] |
| C -> E | -0.005195 | 0 / 10 / 1 | [-0.015584, 0.000000] |
| D -> E | -0.002165 | 0 / 10 / 1 | [-0.006494, 0.000000] |

On this fixed pilot, plain v2 clearly outperformed v1. Hierarchy normalization
did not change any strict reciprocal rank. Relation expansion produced no new
strict gold gain and no Hit@8 gold loss, while moving `click-relation-2` down
slightly. These observations do not establish general hierarchy or relation
effectiveness.

## Relation and validator diagnostics

- Relation trigger, valid-candidate, and selected rates were all 1.0 across the
  22 valid D/E path-query cells.
- Relation-assisted strict gold hits: 5; relation-origin strict gold hits: 0.
- New strict gold gains at 8: 0; strict gold losses at 8: 0.
- All 22 relation-enabled cells reported truncation under the frozen budgets;
  direct backfill occurred 7 times.
- Final invalid Evidence count: 0. Invalid relation-chain count: 0.
- These validator results mean Evidence/Citation Contract Validity only; they do
  not mean answer correctness or citation sufficiency.

## Determinism and artifact integrity

- Normal order `A-B-C-D-E` and interleaved order `E-C-A-D-B` with reversed
  query order produced the same rank/chunk/gold identity:
  `5bfd733d1181872433c818fe3f89458965d57f0a6b4d021a5dbcb85f0b4309b4`.
- Repeating the fixed first-three-query subset across all paths produced
  `2adb71c54e88a7b182ebbdce29c4c774caba34dcd66c6ff1b1917c2870559cd4`.
- Determinism mismatches: 0. NaN/Infinity values: 0.
- The primary run contains the frozen manifest, JSON/JSONL/CSV results, paired
  comparisons, relation diagnostics, validation summary, failure taxonomy,
  Markdown report, per-file hashes, runtime completion record, and SQLite copy.

## Execution notes and boundary

An initial fresh execution encoded all 663 real vectors but was rejected before
formal retrieval because the harness required the text `bge-m3` in the runtime
model name, while the verified local provider reports `local:<revision>`. The
harness was corrected to accept only the exact frozen local revision plus its
configured revision, resolved revision, and local snapshot identity. Focused
and full regression tests passed after that correction. A resumed run then
proved that all 663 real vectors were cache hits, and the final primary run
above repeated the entire process from the source snapshot so its own manifest
contains both fresh-generation and cache-hit evidence.

The result supports only the frozen Click revision and 11 answerable queries.
It does not establish statistical stability across repositories, universal
code-understanding quality, answer correctness, teaching effectiveness,
multi-hop retrieval quality, or a reason to alter the frozen production policy.
