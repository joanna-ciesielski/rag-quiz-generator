"""Pluggable text-embedding backends.

`OpenAIEmbedder` is the production default. `HashingEmbedder` is a deterministic,
offline, dependency-free fallback used by the test suite and by anyone who wants
to try the pipeline without an API key — it is NOT for production retrieval
quality, only for wiring/tests. The interface is identical so the rest of the
app is agnostic to which one is used.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Protocol


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    """Production embeddings via the OpenAI embeddings API."""

    def __init__(self, model: str | None = None, batch_size: int | None = None) -> None:
        self.model = model or os.environ.get("EMBED_MODEL", "text-embedding-3-small")
        self.dim = int(os.environ.get("EMBED_DIM", "1536"))
        # Large documents can exceed the embeddings API's per-request input/token
        # limits, so embed in batches rather than one giant call.
        self.batch_size = batch_size or int(os.environ.get("EMBED_BATCH_SIZE", "100"))
        self._client = None

    def _client_lazy(self):
        if self._client is None:
            import openai  # lazy so tests/offline use need no SDK/key

            self._client = openai.OpenAI()
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        client = self._client_lazy()
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            resp = client.embeddings.create(model=self.model, input=batch)
            out.extend(d.embedding for d in resp.data)
        return out


class HashingEmbedder:
    """Deterministic hashing embedder — offline, no API key, for tests/demo only.

    Maps token hashes into a fixed-width vector (a hashing trick) and L2-normalizes.
    Good enough to exercise real vector search end to end; not semantically strong.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in text.lower().split():
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def get_embedder() -> Embedder:
    """Select backend from EMBEDDER env var ('openai' default, 'hashing' offline)."""
    choice = os.environ.get("EMBEDDER", "openai").lower()
    if choice == "hashing":
        return HashingEmbedder()
    return OpenAIEmbedder()
