# Retrieval v2 Phase 5 frozen evaluation protocol

## Scope

This protocol evaluates the already-frozen Retrieval v1 and Retrieval v2
implementations on the Click subset of `benchmarks/m5/datasets/pilot-v1`.
It does not change retrieval weights, hierarchy policy, relation policy, code
chunks, the relation graph, the benchmark, gold annotations, or production
schemas. `docs/v3/M5_RESULTS.md` belongs to the separate RepoNoesis V3 M5
milestone and is not modified by this phase.

## Dataset and gold

- Dataset: `pilot-v1`, filtered only by the trusted `repo_id=click` field.
- Repository revision: `00e592cea702e0b2caa0dee42489fdb1c22cd845`.
- Query count: 12 total; 11 answerable and 1 unanswerable.
- The unanswerable query is skipped without retrieval and is excluded from
  retrieval-quality denominators.
- Strict gold identity is the declared repository revision, POSIX path,
  qualified symbol, exact start/end span, and content hash.
- Strict matching requires equality on every available strict gold field.
  Containing chunks are reported only as a diagnostic and never count as a
  strict hit.
- Dataset, selected-query, selected-gold, and matcher-definition SHA-256
  identities are recorded separately in every run manifest.

## Frozen paths

| path | retrieval_version | hierarchy_mode | relation_mode |
| --- | --- | --- | --- |
| A / v1 | `v1` | `off` | `off` |
| B / plain v2 | `v2` | `off` | `off` |
| C / v2 + hierarchy | `v2` | `normalize_v1` | `off` |
| D / v2 + relation | `v2` | `off` | `expand_v1` |
| E / v2 + hierarchy + relation | `v2` | `normalize_v1` | `expand_v1` |

Every path/query cell receives a fresh server-bound tool context. Query text,
source files, Planner output, and tool arguments cannot select or rewrite a
path mode.

## Corpus and embeddings

The run copies a previously validated M5 engineering SQLite snapshot into a
new ignored run directory. The source snapshot contains the fixed Click corpus
and its complete one-revision relation graph. Its graph rows and graph hash
must be identical before and after embedding indexing. Historical embedding
rows in the copy are removed before real indexing; the source database is
never modified.

Formal evaluation requires a complete local Hugging Face snapshot and rejects
network access, downloads, dependency changes, fake providers, injected
embedding backends, silent dense fallback, revision drift, and dimension
changes. The frozen configuration is normalized float32 BGE-M3 embeddings,
empty query/document prefixes, model-defined pooling, max length 8192, and
batch size 8. The effective model revision, device, CUDA state, dimension, and
cache identity are recorded after a one-query smoke and before retrieval.

## Ranking and metrics

Production Retrieval v1 and v2 both cap final results at 8. Formal retrieval
therefore requests top 8 and computes strict Hit@1, Hit@3, Hit@5, Hit@8,
MRR@8, Recall@1/3/5/8, and binary nDCG@1/3/5/8. For the requested Hit@10 and
MRR@10 fields, the run reports the value available from at most eight returned
candidates and marks it as a lower-bound disclosure; the production limit is
not changed for evaluation.

Empty results are misses. Multi-gold queries match if any strict gold identity
is returned; recall uses the number of distinct strict gold identities. Invalid
or skipped queries do not silently enter the denominator. Ties remain governed
by the frozen production identity tie-breaks.

Paired comparisons use the same valid query identities and report improved,
unchanged, and regressed reciprocal-rank counts plus fixed-seed paired bootstrap
95% confidence intervals (seed `20260726`, 2,000 samples).

## Relation and validator definitions

- `relation_trigger_rate`: relation expansion actually executed / valid
  answerable queries.
- `relation_candidate_rate`: queries with at least one valid unique relation
  candidate / valid answerable queries.
- `relation_selected_rate`: queries with at least one relation-derived final
  candidate / valid answerable queries.
- `relation_new_gold_gain@K`: relation-on strict hit and paired relation-off
  miss / valid paired queries.
- `relation_gold_loss@K`: relation-off strict hit and paired relation-on miss /
  valid paired queries.

Relation support attached to an already direct/hierarchy candidate is not a new
gold hit. Expansion/selection audit, edge identities, directions, relation
types, seed/target identities, suppression reasons, truncation, backfill, and
warnings are serialized. Final Evidence must pass `CitationValidator`.
Retrieval-time relation chains must pass `RelationValidator`. These measures
are named Evidence/Citation Contract Validity and do not claim answer accuracy
or citation sufficiency.

## Artifacts and determinism

Each immutable run directory contains a frozen manifest, aggregate JSON,
query-level JSONL and CSV, paired comparisons, relation diagnostics, validator
summary, failure cases, result hashes, and a Markdown report. Existing run
directories are never silently overwritten. SQLite databases, model/cache
files, and runtime artifacts remain under ignored artifact paths and are not
committed.

The run executes the five paths in normal order, interleaved order, and a
repeated fixed query subset. Final ranks, strict matches, and chunk identity
sequences must agree; floating-point diagnostic values may differ only without
changing rank or identity.
