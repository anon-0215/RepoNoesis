from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from app.config import EmbeddingSettings
from app.m5.providers import ProviderConfigurationError, validate_vector
from app.services.embedding_service import EffectiveEmbeddingIdentity, EmbeddingService


EmbeddingIdentity = EffectiveEmbeddingIdentity


@dataclass(frozen=True)
class EmbeddingSmokeResult:
    status: str
    identity: EmbeddingIdentity
    latency_ms: int
    error_type: str | None = None


class M5EmbeddingProvider:
    """M5 wrapper around the formal M1 EmbeddingService with an isolated cache."""

    def __init__(
        self,
        settings: EmbeddingSettings,
        *,
        cache_directory: Path,
        allow_model_load: bool,
        allow_network: bool,
        backend_factory: Any = None,
        cuda_available: Any = None,
    ) -> None:
        if not allow_model_load:
            raise ProviderConfigurationError("real embedding model load requires explicit M5 opt-in")
        resolved_cache = cache_directory.resolve()
        self.settings = replace(settings, enabled=True, cache_dir=resolved_cache)
        self.allow_network = allow_network
        kwargs: dict[str, Any] = {}
        if backend_factory is not None:
            kwargs["backend_factory"] = backend_factory
        if cuda_available is not None:
            kwargs["cuda_available"] = cuda_available
        self.service = EmbeddingService(self.settings, **kwargs)
        self._dimension: int | None = None

    @property
    def identity(self) -> EmbeddingIdentity:
        identity = self.service.get_effective_embedding_identity(
            local_files_only=not self.allow_network,
        )
        self._dimension = identity.dimension
        return replace(identity, is_real=True)

    def encode_query(self, text: str, local_files_only: bool = False) -> list[float]:
        vector = self.service.encode_query(
            text, local_files_only=local_files_only or not self.allow_network
        )
        validate_vector(vector, expected_dimension=self._dimension)
        self._dimension = len(vector)
        return vector

    def encode_documents(self, texts: list[str], local_files_only: bool = False) -> list[list[float]]:
        vectors = self.service.encode_documents(
            texts, local_files_only=local_files_only or not self.allow_network
        )
        for vector in vectors:
            validate_vector(vector, expected_dimension=self._dimension)
            self._dimension = len(vector)
        return vectors

    def ensure_backend_identity(self, local_files_only: bool = False) -> Any:
        return self.service.ensure_model_identity(
            local_files_only=local_files_only or not self.allow_network
        )

    def get_backend_identity(self) -> Any:
        return self.service.get_model_identity()

    def ensure_effective_embedding_identity(
        self,
        local_files_only: bool = False,
    ) -> EmbeddingIdentity:
        identity = self.service.ensure_effective_embedding_identity(
            local_files_only=local_files_only or not self.allow_network
        )
        self._dimension = identity.dimension
        return replace(identity, is_real=True)

    def get_model_identity(self) -> EmbeddingIdentity:
        return self.ensure_effective_embedding_identity()

    def ensure_model_identity(self, local_files_only: bool = False) -> EmbeddingIdentity:
        return self.ensure_effective_embedding_identity(local_files_only=local_files_only)

    def get_embedding_dimension(self) -> int | None:
        dimension = self.service.get_embedding_dimension(
            local_files_only=not self.allow_network
        )
        if dimension is not None:
            if self._dimension is not None and self._dimension != dimension:
                raise ValueError("embedding dimension changed during the run")
            self._dimension = dimension
        return dimension

    def is_available(self) -> bool:
        return self.service.is_available()

    def smoke_check(self) -> EmbeddingSmokeResult:
        started = time.monotonic()
        try:
            self.encode_query("RepoNoesis M5 embedding smoke")
            return EmbeddingSmokeResult(
                status="succeeded",
                identity=self.identity,
                latency_ms=int((time.monotonic() - started) * 1000),
            )
        except Exception as exc:
            return EmbeddingSmokeResult(
                status="failed",
                identity=self.identity,
                latency_ms=int((time.monotonic() - started) * 1000),
                error_type=type(exc).__name__,
            )


class FakeEmbeddingBackend:
    """Deterministic backend implementing the existing EmbeddingBackend contract."""

    def __init__(self, dimension: int = 16) -> None:
        self.dimension = dimension
        self.loaded = False

    def load_model(
        self,
        model_name_or_path: str,
        device: str,
        cache_dir: Path,
        max_length: int,
        model_revision: str = "",
    ) -> None:
        self.loaded = True

    def encode(
        self,
        texts: list[str],
        batch_size: int,
        normalize: bool,
    ) -> list[list[float]]:
        if not self.loaded:
            raise RuntimeError("fake embedding backend is not loaded")
        return [_fake_vector(text, self.dimension, normalize) for text in texts]

    def get_embedding_dimension(self) -> int:
        return self.dimension

    def get_model_revision(self) -> str:
        return "fixture-v1"

    def unload_model(self) -> None:
        self.loaded = False


def fake_embedding_service(cache_directory: Path) -> EmbeddingService:
    settings = EmbeddingSettings(
        enabled=True,
        model_name_or_path="fake-bge-m3",
        model_revision="fixture-v1",
        device="cpu",
        batch_size=8,
        max_length=512,
        normalize=True,
        cache_dir=cache_directory,
        query_prefix="",
        document_prefix="",
    )
    return EmbeddingService(
        settings,
        backend_factory=FakeEmbeddingBackend,
        cuda_available=lambda: False,
    )


def _fake_vector(text: str, dimension: int, normalize: bool) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = [((digest[index % len(digest)] / 255.0) * 2.0) - 1.0 for index in range(dimension)]
    if normalize:
        norm = math.sqrt(sum(value * value for value in values))
        if norm:
            values = [value / norm for value in values]
    return values
