# Retrieval v2 Phase 6 frozen cross-repository protocol

## Scope and boundary

Retrieval v2 Phase 6 is an evaluation-harness and offline retrieval experiment.
It is not the RepoNoesis V3 product milestone M6 and does not replace or revise
the M1-M5 milestone documents. Production v1, plain v2, fusion, hierarchy,
relation, chunk identity, strict matching, Evidence, CitationValidator,
RelationValidator, database schema, and the Click benchmark remain frozen.

The phase answers whether the Phase 5 plain-v2 improvement has a limited
replication on one independent repository and diagnoses hierarchy opportunity
and relation applicability. It does not tune ranking policy and does not claim
cross-repository generality from two repositories.

## Initial scene and Phase 5 evidence protection

- Initial branch: `v3-agent-development`.
- Initial HEAD: `e6948c5b1631afcb73ef32cbbfa2d7d2f8d04f9f`.
- Initial commit: `feat(retrieval): add phase 5 offline evaluation`.
- Initial worktree: clean; V1, V2, and V3 worktrees remained isolated.
- Independently recomputed Phase 5 result hash:
  `7ba0f395cbd8af927df8af316d212b1a6705f08d4f1880bdf038db9af7d9365a`.
- Every recorded Phase 5 result-file hash matched the frozen artifact.
- Read-only external backup:
  `D:\Project\RepoNoesis-evaluation-archive\phase5\retrieval-v2-phase5-e217b85b4e0ee809dce31d23.zip`.
- Backup ZIP SHA-256:
  `649748a9d1827f045042bd5bf19c602551f81fe7a4cb5ffe78b6ab36f16a596e`.

The backup path is runtime evidence outside Git. The original Phase 5 artifact
was not modified or removed.

## Frozen repository selection

The independent repository was selected before any Phase 6 HTTPX production
retrieval. Selection used Git metadata, file counts, safe Python AST parsing,
source review, and the existing static M5 corpus only; no retrieval ranking was
run or inspected.

| Candidate | Decision | Pre-retrieval reason |
| --- | --- | --- |
| `encode/httpx` | selected | 60 Python files, 17,753 lines, 1,134 functions, 107 classes, 533 historical extractor chunks, substantial sync/async client, transport, URL, auth, calls, imports, and nested structure |
| `pallets/itsdangerous` | excluded | only 79 historical extractor chunks and insufficient independent structure for all preregistered strata |

The selected checkout already existed locally, so selection and source
acquisition used no network.

| Field | Frozen HTTPX value |
| --- | --- |
| Repository | `https://github.com/encode/httpx.git` |
| Commit | `b5addb64f0161ff6bfe94c124ef76f6a1fba5254` |
| Tree | `31ba94512339180efacceacc0646b56ee15eba63` |
| Tracked files / Python files | 125 / 60 |
| Source hash | `7f695b9bda128ffa96abd2cada5296e23419ef35ec0c26e91cb6c09a1c6af0e8` |
| Source-hash contract | SHA-256 of canonical sorted tracked path plus file-SHA entries |
| License | BSD-3-Clause, `LICENSE.md` |
| License hash | `eb237e056a490d1f290e4dfa3cc342a04c64a89701385f86a847cdbe4d21957d` |
| Checkout / submodules | clean / none |

No HTTPX dependency, test, hook, build, or repository code is executed.

## Source-first benchmark

The HTTPX benchmark was authored from the fixed source and existing static
chunk/relation metadata before production retrieval. Each answerable gold was
independently checked against the exact fixed source span and the read-only
corpus chunk. Relation-dependent queries name a reasonable seed while their
strict gold is a different symbol. Hierarchy-sensitive queries target a nested
function or member whose parent-qualified identity is relevant.

| HTTPX primary stratum | Count |
| --- | ---: |
| direct behavior/location | 6 |
| symbol-focused | 4 |
| relation-dependent | 6 |
| hierarchy-sensitive | 4 |
| unanswerable | 2 |

The 20 answerable strata are mutually exclusive. All six relation-dependent
queries have a source-reviewed resolved graph edge at freeze time, but later
coverage, selection, and rank effects remain measured outcomes rather than
inclusion criteria. Multi-gold is supported by the contract but no HTTPX query
declares alternative gold in `cross-repo-v1`.

The Click dataset stays byte-for-byte in `benchmarks/m5/datasets/pilot-v1`.
Phase 6 adds only a reporting overlay for Click strata; that overlay does not
alter Phase 5 queries, gold, or matching semantics and is not used to satisfy
the HTTPX source-first inclusion minima.

## Frozen identities

All semantic hashes use UTF-8 canonical JSON with sorted keys, compact
separators, no NaN, and SHA-256.

| Identity | SHA-256 |
| --- | --- |
| repository | `1b64c648231ffd792fd05070b8599ac73d5c944eabcc1175272353bf44355d5d` |
| dataset | `44291240a1874899adfc4978be7fbdda3fd2bfd1ef1bc8fe31cc523777a7643a` |
| query | `ddf738c7dc5a6a5cb1666a65cb371d37d6bfb81e440ebf539627351f891e4b67` |
| gold | `29b8ac2613bd33a644da09615e4d7a1f8a69e8e53314f7281137989fe4931181` |
| matcher | `169b0515ffcdd889212f88cc51430ac8b706eb25817e7647e7b064b45a405cb6` |
| strata | `d6c1c6bd31167c159ebc52b8f3d4fd65b0c7265a8537125f5f03fabe8c7ee097` |
| protocol | `c47b3745eb96a069856b1d711cc55338268c1e0ca2db2f8672cb9eff5418aed1` |

The matcher hash is exactly the Phase 5 matcher hash. Containment remains a
diagnostic only. Correct file/wrong chunk and correct symbol/wrong span remain
strict misses. Unanswerable queries are skipped and excluded from quality
denominators.

## Frozen five paths

| ID | Retrieval | Hierarchy | Relation |
| --- | --- | --- | --- |
| A | `v1` | `off` | `off` |
| B | `v2` | `off` | `off` |
| C | `v2` | `normalize_v1` | `off` |
| D | `v2` | `off` | `expand_v1` |
| E | `v2` | `normalize_v1` | `expand_v1` |

Every repository/path/query cell receives a fresh server-bound context. A
request, query, Planner, source file, prior run, path order, query order, or
repository order cannot override the frozen mode. Both repositories share the
same strict matcher, Top K 8, metric definitions, real provider contract, and
model revision.

## Metrics and paired comparisons

Primary metrics are strict Hit@1, Hit@3, Hit@5, Hit@8, MRR@8, Recall@8, and
binary nDCG@8. Any Hit@10 or MRR@10 field is explicitly named a Top-8-truncated
lower bound. Results are reported per repository, as a query-weighted micro
aggregate, as an equal-repository macro average, and by primary stratum.

Required paired comparisons are B-A, C-B, D-B, E-C, and E-D. Each reports
MRR@8 delta, Hit@8 delta, improved/unchanged/regressed query counts, 95%
bootstrap interval, query count, and repository count. The frozen bootstrap
uses seed `20260726` and 2,000 samples. Cross-repository intervals use a
repository-stratified paired bootstrap: paired answerable queries are sampled
with replacement independently within each repository before micro and macro
deltas are recomputed. Small-sample intervals are descriptive and are not
called statistically significant.

## Relation and hierarchy diagnostics

Relation diagnostics distinguish opportunity, graph coverage, expansion,
candidate production, valid targets, selected relation origin, direct gold
with relation support, new strict gold gain, strict gold loss, and rank-only
movement. Support on a direct candidate is never counted again as relation
origin or new gain. Candidate, selected, gold contribution, and noise counts
are reported by actual relation type, including calls, defines, imports, and
references. Seed, slot, row, warning, and backfill budgets remain production
`expand_v1` values.

Hierarchy diagnostics separately track eligibility, trigger, resolution,
normalization, identity change, rank change, strict gain/loss, no-op,
resolution failure, ambiguity, truncation, and warnings. Equal B/C aggregate
metrics do not imply that hierarchy did not run.

## Frozen Phase 5 diagnostic attribution

For `click-relation-2`, plain v2 path B returned strict gold
`pass_obj.new_func` at rank 5. Path D selected the non-gold nested
`pass_context.new_func` relation candidate into rank 4 through resolved
`references` and `defines` support, moving the existing direct gold to rank 6.
Path E selected that candidate and `make_pass_decorator.decorator.new_func`,
moving the same direct gold to rank 7. D and E both inspected the 96-row total
budget and reported truncation. D suppressed two candidates because their
seeds were not retained and used one direct backfill; E suppressed two for
seed-not-retained and one for the relation slot cap. Relation support attached
to the direct gold is duplicated support, not a new strict gold gain.

Hierarchy did execute on all 11 answerable Click path-C cells. Every cell
contained the frozen hierarchy audit, metadata resolution, groups, and
warnings; structural derived candidates were produced for 10 of 11 queries.
The final retained candidates were direct-origin in every cell. Seven queries
had identical final identity/rank sequences, four changed final identity, three
changed rank, and none gained or lost strict gold. Thus the Phase 5 hierarchy
result is execution with mostly no-op final selection under ambiguity and
budget/family constraints, not absence of execution.

## Formal environment, cache, and offline gate

Formal quality numbers require the existing local BGE-M3 snapshot at revision
`5617a9f61b028005a4858fdac845db406aefb181`, sentence-transformers, CUDA,
1,024 normalized float32 dimensions, batch size 8, max length 8,192, empty
query/document prefixes, and model-defined CLS pooling. The runtime must record
the observed Python, PyTorch, CUDA, GPU, provider, model identity, dimension,
allocation, reservation, process peak if available, and cache statistics; the
report may not hard-code those as observed facts.

HTTPX receives an isolated cache namespace bound to repository, revision,
chunk identity, model and revision, dimension, normalization, max length,
prefixes, pooling, and provider contract. Click reuse is allowed only when the
full Phase 5 cache identity matches. The production embedding cache is never
modified. Formal embedding and retrieval run under a socket-level offline
guard with Hugging Face and Transformers offline settings; network attempts,
downloads, dependency drift, provider fallback, graph revision drift, or
model revision drift fail the run.

## Freeze statement and execution order

At this protocol freeze, no Phase 6 production retrieval has been run for
HTTPX and no HTTPX rank has been viewed. The next allowed sequence is tests,
minimal harness extension, fake-provider contract verification, full backend
regression with embeddings disabled, real offline provider smoke, isolated
HTTPX embedding generation, two-repository five-path evaluation,
deterministic replay, immutable results, final regression, static audit, and a
separate final local commit. The frozen benchmark commit is not amended.

No push, merge, tag, reranker, multi-hop relation, filtering/weight tuning, or
Phase 7 work is authorized.
