"""Hybrid retrieval (dense + sparse) with Reciprocal Rank Fusion and MMR rerank.

Dense retrieval (semantic, via the vector store) and sparse retrieval (lexical,
via BM25) catch different things — dense finds paraphrases, sparse nails exact
terms/rare words. Fusing them with RRF is more robust than either alone. MMR then
optionally reranks for diversity so the top-k isn't several near-duplicate chunks.

Everything here is pure Python / numpy — no model downloads — so it runs offline
and its effect is measurable with the eval harness.
"""

from __future__ import annotations

import re
from collections import defaultdict

from rank_bm25 import BM25Okapi

from app.ingest import Chunk
from app.vectorstore import Retrieved, VectorStore

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    """Lexical (sparse) retrieval over chunk texts."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.ids = [c.id for c in chunks]
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in chunks]) if chunks else None

    def search(self, query: str, k: int) -> list[str]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self.ids, scores), key=lambda pair: pair[1], reverse=True)
        return [cid for cid, score in ranked[:k] if score > 0]


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked id-lists: score = sum 1/(k + rank). Robust, no score scaling."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in ranked_lists:
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def _cosine(a: list[float], b: list[float]) -> float:
    import numpy as np

    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    return float(va @ vb / (na * nb)) if na and nb else 0.0


def mmr_rerank(
    query_vec: list[float],
    candidate_ids: list[str],
    id_to_vec: dict[str, list[float]],
    *,
    lambda_mult: float = 0.6,
    k: int = 5,
) -> list[str]:
    """Maximal Marginal Relevance: balance query relevance vs diversity."""
    selected: list[str] = []
    remaining = [cid for cid in candidate_ids if cid in id_to_vec]
    while remaining and len(selected) < k:
        best_id, best_score = None, float("-inf")
        for cid in remaining:
            relevance = _cosine(query_vec, id_to_vec[cid])
            diversity = max((_cosine(id_to_vec[cid], id_to_vec[s]) for s in selected), default=0.0)
            score = lambda_mult * relevance - (1.0 - lambda_mult) * diversity
            if score > best_score:
                best_id, best_score = cid, score
        selected.append(best_id)
        remaining.remove(best_id)
    return selected


class HybridRetriever:
    """Dense + sparse retrieval with RRF fusion and optional MMR reranking.

    Bound to a single namespace so tenant isolation is preserved.
    """

    def __init__(self, store: VectorStore, chunks: list[Chunk], *, namespace: str = "default") -> None:
        self.store = store
        self.namespace = namespace
        ns_chunks = [c for c in chunks if c.namespace == namespace]
        self.bm25 = BM25Index(ns_chunks)
        self._text = {c.id: c.text for c in ns_chunks}
        self._source = {c.id: c.source for c in ns_chunks}

    def query(
        self,
        text: str,
        k: int = 5,
        *,
        candidate_k: int = 10,
        use_sparse: bool = True,
        use_mmr: bool = False,
        mmr_lambda: float = 0.6,
    ) -> list[Retrieved]:
        dense_ids = [h.chunk_id for h in self.store.query(text, namespace=self.namespace, k=candidate_k)]
        rankings = [dense_ids]
        if use_sparse:
            rankings.append(self.bm25.search(text, candidate_k))

        fused = reciprocal_rank_fusion(rankings)
        fused_ids = [cid for cid, _ in fused]
        score_map = dict(fused)

        if use_mmr and fused_ids:
            query_vec = self.store.embedder.embed([text])[0]
            id_to_vec = self.store.get_embeddings(fused_ids)
            ordered = mmr_rerank(query_vec, fused_ids, id_to_vec, lambda_mult=mmr_lambda, k=k)
        else:
            ordered = fused_ids[:k]

        return [
            Retrieved(
                text=self._text.get(cid, ""),
                source=self._source.get(cid, "unknown"),
                score=round(score_map.get(cid, 0.0), 6),
                chunk_id=cid,
            )
            for cid in ordered
        ]
