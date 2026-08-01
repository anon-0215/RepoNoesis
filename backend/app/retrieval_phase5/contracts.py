from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.retrieval_phase5 import EVALUATION_VERSION
from app.services.embedding_service import EffectiveEmbeddingIdentity
from app.services.hierarchy_normalization import HIERARCHY_NORMALIZATION_VERSION
from app.services.relation_retrieval import (
    RELATION_EXPANSION_VERSION,
    RELATION_GRAPH_VERSION,
    RELATION_PRIORITY_VERSION,
    RELATION_SELECTION_VERSION,
    RELATION_WHITELIST_VERSION,
)
from app.services.retrieval_v2 import V2_FUSION_VERSION


FORMAL_TOP_K = 8
TOP_K_VALUES = (1, 3, 5, 8)
RANDOM_SEED = 20260726
BOOTSTRAP_SAMPLES = 2_000
BGE_M3_SNAPSHOT_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
MATCHER_NAME = "strict_source_span_identity"
MATCHER_VERSION = "strict_source_span_identity_v1@1"
GOLD_GRANULARITY = "repository_revision+path+qualified_symbol+exact_span+content_hash"


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class FrozenPath:
    path_id: str
    label: str
    retrieval_version: str
    hierarchy_mode: str
    relation_mode: str

    @property
    def request_parameters(self) -> dict[str, str]:
        return {
            "retrieval_version": self.retrieval_version,
            "hierarchy_mode": self.hierarchy_mode,
            "relation_mode": self.relation_mode,
        }


FROZEN_PATHS = (
    FrozenPath("A", "v1", "v1", "off", "off"),
    FrozenPath("B", "plain v2", "v2", "off", "off"),
    FrozenPath("C", "v2 + hierarchy", "v2", "normalize_v1", "off"),
    FrozenPath("D", "v2 + relation", "v2", "off", "expand_v1"),
    FrozenPath(
        "E",
        "v2 + hierarchy + relation",
        "v2",
        "normalize_v1",
        "expand_v1",
    ),
)


MATCHER_DEFINITION = {
    "name": MATCHER_NAME,
    "version": MATCHER_VERSION,
    "candidate_requirements": [
        "validation_status=valid",
        "repository_revision exact",
        "path exact",
        "qualified_symbol exact",
        "start_line exact",
        "end_line exact",
        "content_hash exact",
    ],
    "multi_gold": "any strict identity is a query hit; recall uses distinct identities",
    "containment": "diagnostic_only_never_a_strict_hit",
}


METRIC_DEFINITIONS = {
    "hit_at_k": "queries with at least one strict gold in the first k / valid answerable queries",
    "mrr": "mean reciprocal rank of the first strict gold; zero when absent",
    "recall_at_k": "distinct strict gold identities in first k / declared strict gold identities",
    "ndcg_at_k": "binary strict-gold DCG / ideal DCG for declared strict gold count",
    "hit_at_10": "lower bound computed from at most the production top 8",
    "mrr_at_10": "lower bound computed from at most the production top 8",
    "relation_trigger_rate": "relation expansion executed / valid answerable queries",
    "relation_candidate_rate": "queries with valid unique relation candidate / valid answerable queries",
    "relation_selected_rate": "queries with relation-derived final candidate / valid answerable queries",
    "relation_new_gold_gain_at_k": "relation-on hit and paired relation-off miss / valid paired queries",
    "relation_gold_loss_at_k": "relation-off hit and paired relation-on miss / valid paired queries",
    "evidence_citation_contract_validity": "final Evidence passing CitationValidator / final Evidence",
    "relation_contract_validity": "retrieval-time Evidence chains passing RelationValidator / emitted chains",
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_formal_embedding_identity(
    identity: EffectiveEmbeddingIdentity,
    *,
    backend: Any,
) -> None:
    backend_name = f"{type(backend).__module__}.{type(backend).__name__}".casefold()
    identity_text = " ".join(
        (
            identity.provider,
            identity.backend_type,
            identity.model_name,
            identity.model_identity,
        )
    ).casefold()
    if not identity.is_real or "fake" in identity_text or "fake" in backend_name:
        raise ManifestError("formal evaluation requires a real, non-injected BGE-M3 provider")
    if identity.provider != "sentence-transformers" or identity.backend_type != "sentence-transformers":
        raise ManifestError("formal evaluation requires the sentence-transformers backend")
    named_bge_m3 = (
        "bge-m3" in identity.model_name.casefold()
        or "bge_m3" in identity.model_name.casefold()
    )
    frozen_local_snapshot = (
        identity.model_name == f"local:{BGE_M3_SNAPSHOT_REVISION}"
        and identity.configured_revision == BGE_M3_SNAPSHOT_REVISION
        and identity.resolved_revision == BGE_M3_SNAPSHOT_REVISION
        and bool(identity.local_snapshot_identity)
    )
    if not named_bge_m3 and not frozen_local_snapshot:
        raise ManifestError("formal evaluation requires BGE-M3")
    if identity.dimension != 1024:
        raise ManifestError("formal BGE-M3 dimension must be 1024")
    if not identity.normalized or identity.dtype != "float32":
        raise ManifestError("formal embeddings must be normalized float32")
    if identity.max_length != 8192 or identity.batch_size != 8:
        raise ManifestError("formal BGE-M3 max length and batch size differ from the frozen protocol")
    if not identity.resolved_revision:
        raise ManifestError("formal embedding revision is unresolved")
    for value in identity.to_dict().values():
        if isinstance(value, float) and not math.isfinite(value):
            raise ManifestError("embedding identity contains a non-finite value")


def build_frozen_manifest(
    *,
    repository_commit: str,
    branch: str,
    corpus_project_id: str,
    corpus_repository_revision: str,
    corpus_database_hash: str,
    relation_graph_hash: str,
    dataset_hash: str,
    query_hash: str,
    gold_hash: str,
    matcher_hash: str,
    query_count: int,
    answerable_query_count: int,
    embedding_identity: EffectiveEmbeddingIdentity,
    model_local_path: str,
    cache_namespace: str,
    environment: dict[str, Any],
    source_files: dict[str, str],
    timestamp: str,
) -> dict[str, Any]:
    ensure_formal_embedding_identity(embedding_identity, backend=object())
    path_parameters = {
        item.path_id: item.request_parameters for item in FROZEN_PATHS
    }
    return {
        "evaluation_version": EVALUATION_VERSION,
        "timestamp": timestamp,
        "repository_commit": repository_commit,
        "branch": branch,
        "corpus_project_id": corpus_project_id,
        "corpus_repository_revision": corpus_repository_revision,
        "corpus_database_hash": corpus_database_hash,
        "relation_graph_hash": relation_graph_hash,
        "dataset_name": "RepoNoesis M5 real-repository pilot / Click subset",
        "dataset_version": "pilot-v1",
        "dataset_hash": dataset_hash,
        "query_file": source_files["query_file"],
        "query_hash": query_hash,
        "gold_file": source_files["gold_file"],
        "gold_hash": gold_hash,
        "matcher_name": MATCHER_NAME,
        "matcher_version": MATCHER_VERSION,
        "matcher_hash": matcher_hash,
        "query_count": int(query_count),
        "answerable_query_count": int(answerable_query_count),
        "gold_granularity": GOLD_GRANULARITY,
        "retrieval_paths": [
            {"path_id": item.path_id, "label": item.label} for item in FROZEN_PATHS
        ],
        "request_parameters": path_parameters,
        "formal_top_k": FORMAL_TOP_K,
        "top_k_values": list(TOP_K_VALUES),
        "top_10_disclosure": "computed-from-at-most-top-8-not-ten-retrieved",
        "metric_definitions": dict(METRIC_DEFINITIONS),
        "embedding_model": embedding_identity.model_name,
        "embedding_revision": embedding_identity.resolved_revision,
        "embedding_local_path": model_local_path,
        "embedding_dimension": embedding_identity.dimension,
        "pooling": embedding_identity.pooling_identity,
        "normalize": embedding_identity.normalized,
        "query_prefix": embedding_identity.query_prefix_identity,
        "document_prefix": embedding_identity.document_prefix_identity,
        "max_length": embedding_identity.max_length,
        "precision": embedding_identity.dtype,
        "device": embedding_identity.device,
        "batch_size": embedding_identity.batch_size,
        "cache_namespace": cache_namespace,
        "embedding_cache_identity": embedding_identity.cache_identity,
        "random_seed": RANDOM_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "python_version": environment.get("python_version"),
        "torch_version": environment.get("torch_version"),
        "cuda_version": environment.get("cuda_version"),
        "gpu_name": environment.get("gpu_name"),
        "cuda_initialized": environment.get("cuda_initialized"),
        "relation_graph_version": RELATION_GRAPH_VERSION,
        "relation_expansion_version": RELATION_EXPANSION_VERSION,
        "relation_selection_version": RELATION_SELECTION_VERSION,
        "relation_priority_version": RELATION_PRIORITY_VERSION,
        "relation_whitelist_version": RELATION_WHITELIST_VERSION,
        "hierarchy_normalization_version": HIERARCHY_NORMALIZATION_VERSION,
        "weighted_rrf_version": V2_FUSION_VERSION,
        "network_allowed": False,
        "model_download_allowed": False,
        "dependency_changes_allowed": False,
        "failure_taxonomy": [
            "all paths hit", "all paths miss", "v2 fixes v1", "v2 regresses v1",
            "hierarchy gain", "hierarchy noise", "relation gain", "relation noise",
            "hierarchy + relation complementary", "hierarchy + relation conflict",
            "gold retrieved but ranked below K", "correct file wrong chunk",
            "correct symbol wrong span", "lexical-only success", "semantic-only success",
            "symbol success", "fusion failure", "hierarchy resolution failure",
            "relation unavailable", "external relation", "ambiguous relation target",
            "stale relation graph", "scope conflict", "budget truncation",
            "hub suppression", "slot-cap suppression", "matcher limitation",
            "possible gold issue",
        ],
    }


def manifest_run_identity(manifest: dict[str, Any]) -> str:
    return f"retrieval-v2-phase5-{canonical_hash(manifest)[:24]}"


def immutable_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
