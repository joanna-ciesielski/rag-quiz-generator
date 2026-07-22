"""Chroma-backed vector store with per-namespace (tenant) scoping.

Embeddings are computed by the app and passed in explicitly, so retrieval quality
is controlled here (not by Chroma's default model) and the store stays swappable.
Every query is scoped to a namespace via a metadata filter, so one user's/tenant's
content can never leak into another's results — a real requirement for
multi-tenant RAG.
"""

from __future__ import annotations

from dataclasses import dataclass

import chromadb

from app.embeddings import Embedder
from app.ingest import Chunk


@dataclass
class Retrieved:
    text: str
    source: str
    score: float
    chunk_id: str


class VectorStore:
    def __init__(self, embedder: Embedder, collection: str = "quiz_documents") -> None:
        self.embedder = embedder
        self._name = collection
        self._client = chromadb.Client()
        self._col = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def reset(self) -> None:
        """Drop and recreate the ENTIRE collection (all namespaces).

        Use only when you truly want a clean slate. To clear a single tenant
        without touching others, use ``clear_namespace`` — that preserves the
        multi-tenant isolation guarantee.
        """
        try:
            self._client.delete_collection(self._name)
        except Exception:  # collection may not exist yet
            pass
        self._col = self._client.get_or_create_collection(
            name=self._name, metadata={"hnsw:space": "cosine"}
        )

    def clear_namespace(self, namespace: str) -> int:
        """Delete only one namespace's documents, leaving other tenants intact.

        This is the tenant-safe way to 'start fresh' — critical because the store
        is shared across namespaces, so a full reset would wipe every tenant.
        """
        ids = self._col.get(where={"namespace": namespace}).get("ids", [])
        if ids:
            self._col.delete(ids=ids)
        return len(ids)

    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        embeddings = self.embedder.embed([c.text for c in chunks])
        self._col.upsert(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
        return len(chunks)

    def query(self, text: str, *, namespace: str = "default", k: int = 4) -> list[Retrieved]:
        """Top-k cosine retrieval, scoped to a single namespace (no cross-tenant leak)."""
        q_emb = self.embedder.embed([text])[0]
        res = self._col.query(
            query_embeddings=[q_emb],
            n_results=k,
            where={"namespace": namespace},
        )
        out: list[Retrieved] = []
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        ids = (res.get("ids") or [[]])[0]
        for doc, meta, dist, cid in zip(docs, metas, dists, ids):
            out.append(
                Retrieved(
                    text=doc,
                    source=(meta or {}).get("source", "unknown"),
                    # cosine distance (0..2) -> similarity, floored at 0 so a
                    # dissimilar hit never reports a negative score
                    score=max(0.0, 1.0 - float(dist)),
                    chunk_id=cid,
                )
            )
        return out

    def count(self, namespace: str | None = None) -> int:
        if namespace is None:
            return self._col.count()
        return len(self._col.get(where={"namespace": namespace})["ids"])
