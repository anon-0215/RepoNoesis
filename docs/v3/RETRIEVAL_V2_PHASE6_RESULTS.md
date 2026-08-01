# RepoNoesis Retrieval v2 Phase 6 results

## Status

**Completed and passed**

Phase 6 is an offline retrieval evaluation extension, not product milestone M6. It adds one fixed Python repository (`encode/httpx`) to the unchanged Phase 5 Click benchmark and evaluates the same production retrieval implementation through the same five frozen paths. No production retrieval weight, fusion rule, hierarchy policy, relation policy, matcher, Click gold, database schema, or Evidence validator was changed.

## Frozen inputs

- Benchmark freeze commit: `f559fda248015e8107fb87aa4922ca1483c739b3`
- Click revision: `00e592cea702e0b2caa0dee42489fdb1c22cd845`, 663 chunks
- HTTPX revision: `b5addb64f0161ff6bfe94c124ef76f6a1fba5254`, 533 chunks
- Total queries: 34; answerable: 31; unanswerable and excluded from retrieval-quality denominators: 3
- Frozen dataset hash: `44291240a1874899adfc4978be7fbdda3fd2bfd1ef1bc8fe31cc523777a7643a`
- Frozen query hash: `ddf738c7dc5a6a5cb1666a65cb371d37d6bfb81e440ebf539627351f891e4b67`
- Frozen gold hash: `29b8ac2613bd33a644da09615e4d7a1f8a69e8e53314f7281137989fe4931181`
- Frozen matcher hash: `169b0515ffcdd889212f88cc51430ac8b706eb25817e7647e7b064b45a405cb6`
- Frozen strata hash: `d6c1c6bd31167c159ebc52b8f3d4fd65b0c7265a8537125f5f03fabe8c7ee097`
- Formal top-k: 8

The HTTPX source-first benchmark contains 6 direct behavior/location, 4 symbol-focused, 6 relation-dependent, 4 hierarchy-sensitive, and 2 unanswerable queries. It was frozen before any formal HTTPX production retrieval.

## Formal environment and gates

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

## Five-path results

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

## Repository-stratified paired comparisons

The frozen procedure independently resamples paired answerable queries within each repository, then recomputes micro and macro deltas. Seed is `20260726`; samples are 2,000.

| Comparison | Micro MRR@8 delta | Macro MRR@8 delta | Improved / unchanged / regressed | Strict gain / loss | Micro 95% CI | Macro 95% CI |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| B − A | +0.068049 | +0.120920 | 9 / 17 / 5 | 3 / 0 | [-0.042187, 0.172064] | [0.010690, 0.227453] |
| C − B | -0.006221 | -0.004821 | 3 / 26 / 2 | 2 / 2 | [-0.033602, 0.017627] | [-0.026042, 0.013661] |
| D − B | -0.021237 | -0.017140 | 1 / 20 / 10 | 1 / 1 | [-0.048388, 0.013141] | [-0.038245, 0.009351] |
| E − C | -0.036713 | -0.029621 | 0 / 20 / 11 | 0 / 2 | [-0.055031, -0.019231] | [-0.044137, -0.015889] |
| E − D | -0.021697 | -0.017302 | 1 / 26 / 4 | 1 / 3 | [-0.055876, 0.004608] | [-0.043831, 0.003387] |

Plain v2 replicated the Click improvement but not its ranking behavior on HTTPX. Click MRR@8 increased from 0.427273 to 0.730303, while HTTPX decreased from 0.468333 to 0.407143 even though HTTPX Hit@8 increased from 0.70 to 0.75. The cross-repository micro CI for B − A includes zero; the two-repository macro CI is positive but is not a stable population estimate with only two repositories.

## Relation applicability

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

## Hierarchy applicability

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

## Artifact identities

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

## Verification performed

- Phase 6 focused fake-provider tests: 9 passed.
- Phase 5 + Phase 6 focused regression: 26 passed.
- Full backend regression with `EMBEDDING_ENABLED=false`: 453 passed in 86.214 seconds before the formal run.
- Real offline BGE-M3 smoke: passed; fixed identity, 1024 dimensions, zero network attempts.
- Formal two-repository five-path run plus reversed repository/path/query order and fixed-subset replay: passed.
- Independent formal and corrected-analysis artifact SHA-256 recomputation: passed.

The final post-document full regression and static Git audit are reported in the task completion response after they run.

## Boundary and next decision

This result covers two fixed Python repositories and 31 answerable queries. HTTPX gold is agent-curated and pending independent human review. All hierarchy and relation enabled cells reached frozen budgets, so the observed applicability includes budget/cap effects. No conclusion here establishes universal retrieval quality, answer correctness, citation sufficiency, teaching effectiveness, or statistical stability across repository populations.

Phase 6 stops here. A future phase may use these diagnostics to propose new relation selection or hierarchy retention hypotheses, but this task does not tune, implement, or evaluate such changes.
