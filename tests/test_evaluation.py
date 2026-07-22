"""Tests for the retrieval-quality evaluation harness (offline, deterministic)."""

from pathlib import Path

from app.embeddings import HashingEmbedder
from app.evaluation import (
    EvalCase,
    evaluate,
    load_eval_set,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.ingest import chunk_text
from app.vectorstore import VectorStore

EVAL = Path(__file__).resolve().parents[1] / "eval"


# ---- pure metric functions (hand-computed expectations) ---------------------

def test_precision_at_k():
    assert precision_at_k(["a", "b", "a", "c"], {"a"}, 4) == 0.5
    assert precision_at_k(["a", "a"], {"a"}, 5) == 1.0   # denom is len(topk)
    assert precision_at_k([], {"a"}, 3) == 0.0


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a", "d"}, 3) == 0.5   # 1 of 2 relevant found
    assert recall_at_k(["a", "b"], {"a", "b"}, 5) == 1.0
    assert recall_at_k(["x"], set(), 3) == 0.0                  # no relevant -> 0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "a", "b"], {"a"}) == 0.5       # first relevant at rank 2
    assert reciprocal_rank(["a", "b"], {"a"}) == 1.0            # rank 1
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0           # none


# ---- end-to-end evaluation over a small labeled store -----------------------

def _labeled_store() -> VectorStore:
    store = VectorStore(HashingEmbedder(), collection="test_eval_store")
    store.reset()
    store.add(chunk_text(
        "Photosynthesis lets plants use sunlight and chlorophyll to make glucose and oxygen.",
        source="photo.md", chunk_size=120, chunk_overlap=20))
    store.add(chunk_text(
        "Gravity is the force of attraction between masses that keeps planets in orbit.",
        source="gravity.md", chunk_size=120, chunk_overlap=20))
    return store


def test_evaluate_returns_sensible_metrics():
    store = _labeled_store()
    cases = [
        EvalCase("plants sunlight chlorophyll glucose", ["photo.md"]),
        EvalCase("force attraction planets orbit gravity", ["gravity.md"]),
    ]
    report = evaluate(store, cases, k=3)
    assert report.n_queries == 2
    assert report.recall_at_k == 1.0          # each relevant doc is retrieved
    assert report.mrr > 0.5                    # relevant doc ranks near the top
    assert 0.0 <= report.precision_at_k <= 1.0
    assert 0.0 <= report.grounding_rate <= 1.0


def test_bundled_eval_set_loads_and_runs():
    """The shipped eval set parses and produces a full report."""
    cases = load_eval_set(EVAL / "eval_set.jsonl")
    assert len(cases) >= 3
    assert all(c.query and c.relevant_sources for c in cases)


def test_report_is_serializable():
    store = _labeled_store()
    report = evaluate(store, [EvalCase("gravity orbit", ["gravity.md"])], k=2, check_grounding=False)
    d = report.to_dict()
    assert set(d) == {"precision_at_k", "recall_at_k", "mrr", "grounding_rate", "k", "n_queries"}
