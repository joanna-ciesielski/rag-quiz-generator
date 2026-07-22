"""Tests for hybrid retrieval: BM25, RRF, MMR, and the measured lift over dense."""

from app.embeddings import HashingEmbedder
from app.evaluation import EvalCase, evaluate_retrieval
from app.ingest import chunk_text
from app.retrieval import (
    BM25Index,
    HybridRetriever,
    mmr_rerank,
    reciprocal_rank_fusion,
)
from app.vectorstore import VectorStore

CORPUS = {
    "photosynthesis.md": "Plants use sunlight and chlorophyll to turn carbon dioxide into glucose and oxygen.",
    "gravity.md": "Gravity is the force of attraction between masses that keeps planets in orbit.",
    "water_cycle.md": "Evaporation and condensation drive the water cycle producing rain and clouds.",
    "cells.md": "Mitochondria are the powerhouse of the cell and ribosomes build proteins.",
}


def _build():
    store = VectorStore(HashingEmbedder(), collection="test_hybrid")
    store.reset()
    chunks = []
    for name, text in CORPUS.items():
        cs = chunk_text(text, source=name, namespace="default", chunk_size=60, chunk_overlap=10)
        chunks += cs
        store.add(cs)
    return store, chunks


# ---- pure functions ---------------------------------------------------------

def test_reciprocal_rank_fusion_orders_by_fused_score():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])
    top2 = {item for item, _ in fused[:2]}
    assert top2 == {"a", "b"}                      # both appear high in both lists
    assert dict(fused)["a"] > dict(fused)["c"]     # a (in both) beats c (in one)


def test_mmr_picks_most_relevant_first_then_diversifies():
    query = [1.0, 0.0]
    vecs = {"a": [1.0, 0.0], "b": [0.95, 0.05], "c": [0.0, 1.0]}
    # lambda < 0.5 favors diversity, so after the most-relevant 'a' it should
    # prefer the diverse 'c' over the near-duplicate 'b'.
    order = mmr_rerank(query, ["a", "b", "c"], vecs, lambda_mult=0.2, k=2)
    assert order[0] == "a"                          # most relevant first
    assert order[1] == "c"                          # diversified away from near-duplicate b
    # and with a relevance-favoring lambda, the near-duplicate 'b' comes next
    order_rel = mmr_rerank(query, ["a", "b", "c"], vecs, lambda_mult=0.9, k=2)
    assert order_rel == ["a", "b"]


def test_bm25_finds_lexically_matching_source():
    _, chunks = _build()
    idx = BM25Index(chunks)
    hits = idx.search("planets orbit gravity", k=2)
    assert hits and any("gravity" in cid for cid in hits)


# ---- hybrid retriever -------------------------------------------------------

def test_hybrid_retriever_returns_relevant_source():
    store, chunks = _build()
    hybrid = HybridRetriever(store, chunks, namespace="default")
    hits = hybrid.query("sunlight chlorophyll glucose", k=2)
    assert hits and hits[0].source == "photosynthesis.md"


def test_hybrid_at_least_matches_dense_mrr_on_fixture():
    """The measurable point of Phase 2: hybrid should not regress dense, and
    here it improves it."""
    store, chunks = _build()
    hybrid = HybridRetriever(store, chunks, namespace="default")
    cases = [
        EvalCase("sunlight chlorophyll glucose oxygen", ["photosynthesis.md"]),
        EvalCase("force attraction planets orbit", ["gravity.md"]),
        EvalCase("evaporation condensation rain clouds", ["water_cycle.md"]),
        EvalCase("mitochondria powerhouse ribosomes proteins", ["cells.md"]),
    ]
    dense = evaluate_retrieval(lambda q, k: store.query(q, namespace="default", k=k), cases, k=3)
    hyb = evaluate_retrieval(lambda q, k: hybrid.query(q, k=k), cases, k=3)
    assert hyb["mrr"] >= dense["mrr"]              # hybrid does not regress relevance
    assert hyb["recall_at_k"] >= dense["recall_at_k"]
