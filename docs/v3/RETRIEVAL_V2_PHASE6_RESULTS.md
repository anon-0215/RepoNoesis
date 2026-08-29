# RepoNoesis Retrieval v2 Phase 6 results

## Status

**Completed and passed**

Phase 6 is an offline retrieval evaluation extension, not product milestone M6. It adds one fixed Python repository (`encode/httpx`) to the unchanged Phase 5 Click benchmark and evaluates the same production retrieval implementation through the same five frozen paths. No production retrieval weight, fusion rule, hierarchy policy, relation policy, matcher, Click gold, database schema, or Evidence validator was changed.

## Frozen inputs

- Benchmark freeze commit: `f559fda248015e8107fb87aa4922ca1483c739b3`
- Click revision: `00e592cea702e0b2caa0dee42489fdb1c22cd845`, 663 chunks
- HTTPX revision: `b5addb64f0161ff6bfe94c124ef76f6a1fba5254`, 533 chunks
- Total queries: 34; answerable: 31; unanswerable and excluded from retrieval-quality denominators: 3
- Historical run dataset hash (legacy CRLF identity): `44291240a1874899adfc4978be7fbdda3fd2bfd1ef1bc8fe31cc523777a7643a`
- Historical Click identity: `1e658877c9b0aa9481700fa09868242b596b4a9c818db9e3159e192825155eb3`
- Current `utf8-lf-v1` Click identity: `40bbc1f2c39a94e5c9d02e949c6fefdc5fe79c0e6b46ff0666448656c1a2f32f`
- Current derived Phase 6 dataset identity: `f65d3544d520b3d71de151f15eb7963c1e2595f0872185885fb9d003b8d380b1`
- Frozen query hash: `ddf738c7dc5a6a5cb1666a65cb371d37d6bfb81e440ebf539627351f891e4b67`
- Frozen gold hash: `29b8ac2613bd33a644da09615e4d7a1f8a69e8e53314f7281137989fe4931181`
- Frozen matcher hash: `169b0515ffcdd889212f88cc51430ac8b706eb25817e7647e7b064b45a405cb6`
- Frozen strata hash: `d6c1c6bd31167c159ebc52b8f3d4fd65b0c7265a8537125f5f03fabe8c7ee097`
- Formal top-k: 8

The HTTPX source-first benchmark contains 6 direct behavior/location, 4 symbol-focused, 6 relation-dependent, 4 hierarchy-sensitive, and 2 unanswerable queries. It was frozen before any formal HTTPX production retrieval.

Frozen JSON/JSONL transport is strict UTF-8 without BOM. Runtime component
hashing changes only CRLF and standalone CR to LF before SHA-256; it performs
no Unicode normalization, trimming, key reordering, or semantic JSON
canonicalization. The same normalized bytes are parsed without reopening the
file. Narrow `.gitattributes` rules additionally keep the M5 and Phase 6
benchmark JSON/JSONL checkouts at LF. The active policy identifier is
`utf8-lf-v1`. It migrates the Click identity that previously arose from Windows
CRLF materialization and the Phase 6 dataset identity directly derived from
it. Benchmark business content, component file hashes, query, gold, matcher,
repository, strata, and protocol identities are unchanged.

The legacy sections below were produced under the pre-`utf8-lf-v1` Click and
Phase 6 dataset identities. They remain valid historical records under those
identities and are not relabeled as current-LF evidence. At the R1 identity
migration checkpoint, no current-identity benchmark had yet been run. R3 later
completed the independent current-LF formal run documented below.

The recovered finite contract also binds formal top-k to 8, keeps the Phase 6
Path E label `hierarchy + relation` distinct from Phase 5
`v2 + hierarchy + relation`, projects every complete multi-gold span as an
atomic OR candidate, applies platform-independent baseline repository-path and
revision cross-binding checks without filesystem/network resolution, and
restricts final failure categories to the taxonomy frozen in `protocol.json`.
Unknown classifier categories fail closed before `failure_cases.json` is
written.

## Current utf8-lf-v1 formal run

This is the authoritative formal result for the active LF-normalized frozen
identity.

- Run date: 2026-08-29
- Run ID: `retrieval-v2-phase6-b87dc6090a2cb2941364c16f`
- Frozen text hash policy: `utf8-lf-v1`
- Phase 6 dataset identity: `f65d3544d520b3d71de151f15eb7963c1e2595f0872185885fb9d003b8d380b1`
- Current Click identity: `40bbc1f2c39a94e5c9d02e949c6fefdc5fe79c0e6b46ff0666448656c1a2f32f`
- Repository HEAD: `612418e843c40210022336fa4ea3115aac89c083`
- Benchmark commit: `f559fda248015e8107fb87aa4922ca1483c739b3`
- Click revision: `00e592cea702e0b2caa0dee42489fdb1c22cd845`
- HTTPX revision: `b5addb64f0161ff6bfe94c124ef76f6a1fba5254`
- Formal top-k: 8
- Matrix: 34 scenarios, 31 answerable, 3 unanswerable, 170 cells
- Encode contract: all 155 answerable cells encoded the query exactly once
- Failed cells: 0
- Production result hash: `cabd2dd0de703d40725cb7cefe7236a30ee641fd528b5868b34f55bafde11bed`
- Production validator: `load_phase6_benchmark` plus
  `validate_cross_repository_matrix`, passed
- Determinism replay: passed; no mismatches
- Invalid final Evidence / relation chains: 0 / 0
- Network attempts: 0
- Runtime source database, model snapshot metadata, and dependency inventory:
  unchanged before and after the formal run
- Local durable archive, not part of Git:
  `D:\Project\RepoNoesis-artifacts\retrieval_v2_phase6\runs\retrieval-v2-phase6-b87dc6090a2cb2941364c16f`

The supporting Backend evidence was run before the formal benchmark: 52/52
targeted tests, 760/760 ordinary Backend tests, and 13/13 environment-sensitive
tests passed. These are three separately scoped commands; overlapping targeted
nodes are not added to the complete Backend total.

### Current micro aggregate

Values are over the 31 answerable queries.

| Path | Hit@1 | Hit@3 | Hit@5 | Hit@8 | MRR@8 | Recall@8 | nDCG@8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A — v1 | 0.258065 | 0.677419 | 0.741935 | 0.741935 | 0.453763 | 0.741935 | 0.526489 |
| B — plain v2 | 0.354839 | 0.645161 | 0.741935 | 0.838710 | 0.521813 | 0.838710 | 0.599340 |
| C — v2 + hierarchy | 0.354839 | 0.612903 | 0.741935 | 0.838710 | 0.515591 | 0.838710 | 0.593799 |
| D — v2 + relation | 0.354839 | 0.645161 | 0.677419 | 0.838710 | 0.500576 | 0.838710 | 0.581964 |
| E — hierarchy + relation | 0.354839 | 0.580645 | 0.612903 | 0.774194 | 0.478879 | 0.774194 | 0.549545 |

### Current per-repository results

#### Click (11 answerable queries)

| Path | Hit@1 | Hit@3 | Hit@5 | Hit@8 | MRR@8 | Recall@8 | nDCG@8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A — v1 | 0.181818 | 0.727273 | 0.818182 | 0.818182 | 0.427273 | 0.818182 | 0.525422 |
| B — plain v2 | 0.545455 | 0.909091 | 1.000000 | 1.000000 | 0.730303 | 1.000000 | 0.798149 |
| C — v2 + hierarchy | 0.545455 | 0.909091 | 1.000000 | 1.000000 | 0.730303 | 1.000000 | 0.798149 |
| D — v2 + relation | 0.545455 | 0.909091 | 0.909091 | 1.000000 | 0.727273 | 1.000000 | 0.795363 |
| E — hierarchy + relation | 0.545455 | 0.909091 | 0.909091 | 1.000000 | 0.725108 | 1.000000 | 0.793284 |

#### HTTPX (20 answerable queries)

| Path | Hit@1 | Hit@3 | Hit@5 | Hit@8 | MRR@8 | Recall@8 | nDCG@8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A — v1 | 0.300000 | 0.650000 | 0.700000 | 0.700000 | 0.468333 | 0.700000 | 0.527075 |
| B — plain v2 | 0.250000 | 0.500000 | 0.600000 | 0.750000 | 0.407143 | 0.750000 | 0.489995 |
| C — v2 + hierarchy | 0.250000 | 0.450000 | 0.600000 | 0.750000 | 0.397500 | 0.750000 | 0.481407 |
| D — v2 + relation | 0.250000 | 0.500000 | 0.550000 | 0.750000 | 0.375893 | 0.750000 | 0.464594 |
| E — hierarchy + relation | 0.250000 | 0.400000 | 0.450000 | 0.650000 | 0.343452 | 0.650000 | 0.415488 |

### Current primary-stratum results

The three unanswerable scenarios were skipped on all five paths and have no
retrieval-quality metric denominator.

| Stratum | Path | Queries | Hit@1 | Hit@3 | Hit@5 | Hit@8 | MRR@8 | Recall@8 | nDCG@8 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct_behavior_location | A | 9 | 0.000000 | 0.555556 | 0.555556 | 0.555556 | 0.259259 | 0.555556 | 0.335969 |
| direct_behavior_location | B | 9 | 0.333333 | 0.777778 | 0.777778 | 0.888889 | 0.574074 | 0.888889 | 0.653325 |
| direct_behavior_location | C | 9 | 0.333333 | 0.777778 | 0.777778 | 0.888889 | 0.574074 | 0.888889 | 0.653325 |
| direct_behavior_location | D | 9 | 0.333333 | 0.777778 | 0.777778 | 0.888889 | 0.534392 | 0.888889 | 0.621688 |
| direct_behavior_location | E | 9 | 0.333333 | 0.777778 | 0.777778 | 0.888889 | 0.534392 | 0.888889 | 0.621688 |
| hierarchy_sensitive | A | 4 | 0.500000 | 1.000000 | 1.000000 | 1.000000 | 0.666667 | 1.000000 | 0.750000 |
| hierarchy_sensitive | B | 4 | 0.000000 | 0.250000 | 0.750000 | 1.000000 | 0.250000 | 1.000000 | 0.429390 |
| hierarchy_sensitive | C | 4 | 0.000000 | 0.000000 | 0.500000 | 0.500000 | 0.125000 | 0.500000 | 0.215338 |
| hierarchy_sensitive | D | 4 | 0.000000 | 0.250000 | 0.250000 | 1.000000 | 0.197917 | 1.000000 | 0.381970 |
| hierarchy_sensitive | E | 4 | 0.000000 | 0.000000 | 0.000000 | 0.500000 | 0.083333 | 0.500000 | 0.178104 |
| relation_dependent | A | 9 | 0.111111 | 0.333333 | 0.555556 | 0.555556 | 0.248148 | 0.555556 | 0.322737 |
| relation_dependent | B | 9 | 0.111111 | 0.333333 | 0.444444 | 0.555556 | 0.241799 | 0.555556 | 0.316791 |
| relation_dependent | C | 9 | 0.111111 | 0.333333 | 0.555556 | 0.777778 | 0.275926 | 0.777778 | 0.392841 |
| relation_dependent | D | 9 | 0.111111 | 0.333333 | 0.444444 | 0.555556 | 0.250000 | 0.555556 | 0.324201 |
| relation_dependent | E | 9 | 0.111111 | 0.222222 | 0.333333 | 0.555556 | 0.226190 | 0.555556 | 0.303141 |
| symbol_focused | A | 9 | 0.555556 | 1.000000 | 1.000000 | 1.000000 | 0.759259 | 1.000000 | 0.821421 |
| symbol_focused | B | 9 | 0.777778 | 1.000000 | 1.000000 | 1.000000 | 0.870370 | 1.000000 | 0.903437 |
| symbol_focused | C | 9 | 0.777778 | 1.000000 | 1.000000 | 1.000000 | 0.870370 | 1.000000 | 0.903437 |
| symbol_focused | D | 9 | 0.777778 | 1.000000 | 1.000000 | 1.000000 | 0.851852 | 1.000000 | 0.888889 |
| symbol_focused | E | 9 | 0.777778 | 1.000000 | 1.000000 | 1.000000 | 0.851852 | 1.000000 | 0.888889 |

### Current failure taxonomy summary

Categories are multi-label diagnostics, so their counts do not sum to 34.

| Category | Query count |
| --- | ---: |
| all paths hit | 20 |
| all paths miss | 6 |
| budget truncation | 31 |
| hierarchy gain | 2 |
| hierarchy noise | 2 |
| matcher limitation | 19 |
| relation gold loss | 4 |
| relation new strict gain | 1 |
| relation noise | 1 |
| slot-cap suppression | 24 |
| v2 fixes v1 | 3 |

### Current result interpretation

Path B has the strongest overall micro result: the highest MRR@8 and nDCG@8,
and a joint-best Hit@8. Path C shows a targeted relation-dependent gain over
B (Hit@8 0.777778 versus 0.555556 and MRR@8 0.275926 versus 0.241799), but it
does not improve the overall micro result and is materially worse on the
four-query hierarchy-sensitive stratum. D and E do not form a monotonic
improvement chain; E is below B, C, and D on overall micro MRR@8, Hit@8, and
nDCG@8.

The simultaneous hierarchy gain/noise, relation gain/loss/noise, budget
truncation, and slot-cap suppression categories show that hierarchy, relation,
and cap behavior can produce both useful recovery and ranking noise. The
result supports enabling enhancement by stratum and scenario. It does not
support a claim that the most complex Path E is generally best.

## Legacy pre-utf8-lf-v1 result — formal environment and gates

- Provider/backend: real local `sentence-transformers`
- Model: local BGE-M3 snapshot at revision `5617a9f61b028005a4858fdac845db406aefb181`
- Dimension/dtype/normalization: 1024 / float32 / normalized
- Batch/max length/prefixes: 8 / 8192 / empty query and document prefixes
- Device: CUDA, NVIDIA GeForce RTX 5060 Laptop GPU
- Peak CUDA allocation: 9,553,511,424 bytes
- Formal elapsed time: 2,787.58 seconds
- Network attempts: 0
- Invalid final Evidence: 0
- Invalid relation chains: 0
- Source database, model snapshot, and dependency inventory hashes: unchanged before/after
- Determinism: passed; normal and reordered result identity both `9007ebfcbe45ed9289395b650a492da8bdf293997413af3bdf9d4d882ad20ba3`; no mismatch

Document embedding isolation was checked by repository project/revision/chunk identity. The copied evaluation database contained exactly 663 Click and 533 HTTPX embedding rows. First indexing generated 663/533 rows with zero cache hits; the second pass generated zero rows and returned 663/533 complete cache hits.

## Legacy pre-utf8-lf-v1 result — five-path results

The table reports strict exact-span matching only. Values are over answerable queries.

### Click (11 answerable queries)

| Path | Hit@1 | Hit@3 | Hit@5 | Hit@8 | MRR@8 | Recall@8 | nDCG@8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A — v1 | 0.181818 | 0.727273 | 0.818182 | 0.818182 | 0.427273 | 0.818182 | 0.525422 |
| B — plain v2 | 0.545455 | 0.909091 | 1.000000 | 1.000000 | 0.730303 | 1.000000 | 0.798149 |
| C — v2 + hierarchy | 0.545455 | 0.909091 | 1.000000 | 1.000000 | 0.730303 | 1.000000 | 0.798149 |
| D — v2 + relation | 0.545455 | 0.909091 | 0.909091 | 1.000000 | 0.727273 | 1.000000 | 0.795363 |
| E — hierarchy + relation | 0.545455 | 0.909091 | 0.909091 | 1.000000 | 0.725108 | 1.000000 | 0.793284 |

### HTTPX (20 answerable queries)

| Path | Hit@1 | Hit@3 | Hit@5 | Hit@8 | MRR@8 | Recall@8 | nDCG@8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A — v1 | 0.300000 | 0.650000 | 0.700000 | 0.700000 | 0.468333 | 0.700000 | 0.527075 |
| B — plain v2 | 0.250000 | 0.500000 | 0.600000 | 0.750000 | 0.407143 | 0.750000 | 0.489995 |
| C — v2 + hierarchy | 0.250000 | 0.450000 | 0.600000 | 0.750000 | 0.397500 | 0.750000 | 0.481407 |
| D — v2 + relation | 0.250000 | 0.500000 | 0.550000 | 0.750000 | 0.375893 | 0.750000 | 0.464594 |
| E — hierarchy + relation | 0.250000 | 0.400000 | 0.450000 | 0.650000 | 0.343452 | 0.650000 | 0.415488 |

### Micro aggregate (31 answerable queries)

| Path | Hit@1 | Hit@3 | Hit@5 | Hit@8 | MRR@8 | Recall@8 | nDCG@8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A — v1 | 0.258065 | 0.677419 | 0.741935 | 0.741935 | 0.453763 | 0.741935 | 0.526489 |
| B — plain v2 | 0.354839 | 0.645161 | 0.741935 | 0.838710 | 0.521813 | 0.838710 | 0.599340 |
| C — v2 + hierarchy | 0.354839 | 0.612903 | 0.741935 | 0.838710 | 0.515591 | 0.838710 | 0.593799 |
| D — v2 + relation | 0.354839 | 0.645161 | 0.677419 | 0.838710 | 0.500576 | 0.838710 | 0.581964 |
| E — hierarchy + relation | 0.354839 | 0.580645 | 0.612903 | 0.774194 | 0.478879 | 0.774194 | 0.549545 |

### Macro aggregate (equal repository weight)

| Path | Hit@1 | Hit@3 | Hit@5 | Hit@8 | MRR@8 | Recall@8 | nDCG@8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A — v1 | 0.240909 | 0.688636 | 0.759091 | 0.759091 | 0.447803 | 0.759091 | 0.526249 |
| B — plain v2 | 0.397727 | 0.704545 | 0.800000 | 0.875000 | 0.568723 | 0.875000 | 0.644072 |
| C — v2 + hierarchy | 0.397727 | 0.679545 | 0.800000 | 0.875000 | 0.563902 | 0.875000 | 0.639778 |
| D — v2 + relation | 0.397727 | 0.704545 | 0.729545 | 0.875000 | 0.551583 | 0.875000 | 0.629979 |
| E — hierarchy + relation | 0.397727 | 0.654545 | 0.679545 | 0.825000 | 0.534280 | 0.825000 | 0.604386 |

## Legacy pre-utf8-lf-v1 result — repository-stratified paired comparisons

The frozen procedure independently resamples paired answerable queries within each repository, then recomputes micro and macro deltas. Seed is `20260726`; samples are 2,000.

| Comparison | Micro MRR@8 delta | Macro MRR@8 delta | Improved / unchanged / regressed | Strict gain / loss | Micro 95% CI | Macro 95% CI |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| B − A | +0.068049 | +0.120920 | 9 / 17 / 5 | 3 / 0 | [-0.042187, 0.172064] | [0.010690, 0.227453] |
| C − B | -0.006221 | -0.004821 | 3 / 26 / 2 | 2 / 2 | [-0.033602, 0.017627] | [-0.026042, 0.013661] |
| D − B | -0.021237 | -0.017140 | 1 / 20 / 10 | 1 / 1 | [-0.048388, 0.013141] | [-0.038245, 0.009351] |
| E − C | -0.036713 | -0.029621 | 0 / 20 / 11 | 0 / 2 | [-0.055031, -0.019231] | [-0.044137, -0.015889] |
| E − D | -0.021697 | -0.017302 | 1 / 26 / 4 | 1 / 3 | [-0.055876, 0.004608] | [-0.043831, 0.003387] |

Plain v2 replicated the Click improvement but not its ranking behavior on HTTPX. Click MRR@8 increased from 0.427273 to 0.730303, while HTTPX decreased from 0.468333 to 0.407143 even though HTTPX Hit@8 increased from 0.70 to 0.75. The cross-repository micro CI for B − A includes zero; the two-repository macro CI is positive but is not a stable population estimate with only two repositories.

## Legacy pre-utf8-lf-v1 result — relation applicability

Across D and E there were 62 answerable relation-enabled cells. Expansion triggered in all 62, produced candidates in all 62, selected at least one relation candidate in all 62, and was truncated in all 62. It accepted 921 edges and recorded 232 slot-cap, 105 seed-not-retained, and 92 seed-cap suppressions.

Paired B→D and C→E effects:

- 62 paired cells; every ranked identity and candidate set changed.
- MRR@8 improved in 1, was unchanged in 40, and regressed in 21.
- One strict gain and three strict losses; mean MRR@8 delta was -0.028975.
- Click: 0 gains, 0 losses, 2 rank regressions, mean delta -0.004113.
- HTTPX: 1 gain, 3 losses, 1 rank improvement, 19 rank regressions, mean delta -0.042649.
- On the relation-dependent stratum specifically: 1 strict gain, 3 losses; 1 rank improvement, 8 regressions, 9 unchanged; mean delta -0.020767.

The sole relation strict gain was `httpx-phase6-relation-05` on B→D (gold absent→rank 3). The strict losses were `httpx-phase6-relation-03` on B→D (rank 7→absent) and `httpx-phase6-relation-05` / `httpx-phase6-relation-06` on C→E (rank 8→absent). The frozen Click case `click-relation-2` reproduced the prior diagnostic: rank 5→6 for B→D and rank 5→7 for C→E, with no strict Hit@8 gain or loss.

This evidence does not support enabling frozen relation expansion universally. The mechanism is operational and can surface a missing gold, but the current one-slot/cap/budget behavior more often introduces ranking noise in this two-repository sample.

## Legacy pre-utf8-lf-v1 result — hierarchy applicability

Across C and E there were 62 answerable hierarchy-enabled cells. Normalization executed in all 62 and was truncated in all 62. Sixty cells produced 1,090 hierarchy-origin audit candidates, but none of those hierarchy-origin candidates was retained in the final top eight. Final retained candidates were direct-origin or relation-origin. Nevertheless, normalization changed direct-candidate selection/order, so it was not an identity no-op:

- B→C and D→E: 62 paired cells.
- Ranked identity changed in 37 cells and the candidate set changed in 36.
- MRR@8 improved in 4, was unchanged in 52, and regressed in 6.
- Three strict gains and five strict losses; mean MRR@8 delta was -0.013959.
- Click: no strict gain/loss; one rank regression; mean delta -0.001082.
- HTTPX: three gains, five losses; four rank improvements and five regressions; mean delta -0.021042.
- Hierarchy-sensitive stratum: zero gains, four losses, four regressions, mean delta -0.119792.
- Relation-dependent stratum: three gains, one loss, four improvements, two regressions, mean delta +0.005159.

The four hierarchy-sensitive strict losses were `httpx-phase6-hierarchy-02` and `httpx-phase6-hierarchy-04` in both B→C and D→E. The three hierarchy gains occurred on relation-dependent queries: `httpx-phase6-relation-05` and `httpx-phase6-relation-06` on B→C, and `httpx-phase6-relation-03` on D→E.

The result argues against treating the current hierarchy normalizer as generally beneficial for nested-symbol queries. It executed and explored structure, but all hierarchy-origin candidates were suppressed and its strongest negative effect occurred precisely in the frozen hierarchy-sensitive stratum.

## Legacy pre-utf8-lf-v1 result — artifact identities

Formal retrieval run:

- Run ID: `retrieval-v2-phase6-a9dfc7426605a39ad04cf749`
- Formal result hash: `cd3badd21cf84fc41fde446276269c358761c9f014fc91fc3236beabdc35bf41`
- Query-results SHA-256: `3d7e7ba3b371d5b15a87f21f8513a89e0c73baffc53b474385da4b2481a5e0cb`

During review, the first diagnostic adapter was found to read nonexistent hierarchy count keys. Strict metrics, candidates, ranks, query results, and the formal result hash were unaffected. The original immutable run was not overwritten. A corrected analysis run was generated from the byte-identical frozen `query_results.jsonl`:

- Analysis run ID: `retrieval-v2-phase6-analysis-cc0e0ffee942f5361d816620`
- Corrected analysis result hash: `add6fdfb82c158af1a50db0ab38e12eb3f16cf8ab4fb303b5c47e64918fcaa33`
- Corrected query-results SHA-256: `3d7e7ba3b371d5b15a87f21f8513a89e0c73baffc53b474385da4b2481a5e0cb` (byte-identical to the formal run)

All eight files listed in each `result_hashes.json` were independently rehashed successfully. The corrected analysis hash is the authoritative Phase 6 diagnostic result; the formal run remains the authoritative retrieval execution record.

The Phase 5 result hash was independently rechecked before Phase 6 and remained `7ba0f395cbd8af927df8af316d212b1a6705f08d4f1880bdf038db9af7d9365a`. Its external read-only ZIP backup is `D:\Project\RepoNoesis-evaluation-archive\phase5\retrieval-v2-phase5-e217b85b4e0ee809dce31d23.zip`, SHA-256 `649748a9d1827f045042bd5bf19c602551f81fe7a4cb5ffe78b6ab36f16a596e`.

## Legacy pre-utf8-lf-v1 result — verification performed

- Phase 6 focused fake-provider tests: 9 passed.
- Phase 5 + Phase 6 focused regression: 26 passed.
- Full backend regression with `EMBEDDING_ENABLED=false`: 453 passed in 86.214 seconds before the formal run.
- Real offline BGE-M3 smoke: passed; fixed identity, 1024 dimensions, zero network attempts.
- Formal two-repository five-path run plus reversed repository/path/query order and fixed-subset replay: passed.
- Independent formal and corrected-analysis artifact SHA-256 recomputation: passed.

These commands and artifacts are retained as historical evidence for the
legacy identities. They are not substituted for the current-LF R3 evidence
listed above.

## Limitations and provenance boundary

The dataset contains only 34 scenarios, including 31 answerable scenarios.
Several strata are small; the hierarchy-sensitive stratum has only four
answerable scenarios. The result proves only the behavior of this frozen
dataset, the fixed Click and HTTPX revisions, the recorded local BGE-M3
snapshot, and the evaluated implementation at the recorded repository HEAD.
HTTPX gold is agent-curated and pending independent human review. All
hierarchy- and relation-enabled cells reached frozen budgets, so the observed
applicability includes budget and cap effects.

No conclusion here establishes universal retrieval quality across repositories,
answer correctness, citation sufficiency, teaching effectiveness, or
statistical stability across repository populations. Legacy and current-LF
results retain separate identity-bound provenance; matching metric values do
not make their artifacts or identities interchangeable.

Phase 6 stops here. A future phase may use these diagnostics to propose new relation selection or hierarchy retention hypotheses, but this task does not tune, implement, or evaluate such changes.
