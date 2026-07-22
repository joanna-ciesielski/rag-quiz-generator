"""Retrieval + grounding evaluation for the RAG pipeline.

Turns "the retrieval seems fine" into measured numbers: precision@k, recall@k,
and MRR against a labeled eval set, plus a grounding rate for generation. This
is what lets you prove a retrieval change actually helped, and it runs as a CI
quality gate.

Relevance is defined at the source-document level: a retrieved chunk counts as
relevant if it came from a document labeled relevant for that query.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean

from app.quiz import generate_quiz
from app.vectorstore import Retrieved, VectorStore


# ---- core metric functions (pure, unit-testable) ----------------------------

def precision_at_k(retrieved_sources: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-k retrieved chunks that came from a relevant source."""
    topk = retrieved_sources[:k]
    if not topk:
        return 0.0
    return sum(1 for s in topk if s in relevant) / len(topk)


def recall_at_k(retrieved_sources: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant sources represented anywhere in the top-k."""
    if not relevant:
        return 0.0
    return len(set(retrieved_sources[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved_sources: list[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant retrieved chunk (0 if none in the list)."""
    for i, source in enumerate(retrieved_sources, start=1):
        if source in relevant:
            return 1.0 / i
    return 0.0


# ---- aggregate evaluation ---------------------------------------------------

@dataclass
class EvalReport:
    precision_at_k: float
    recall_at_k: float
    mrr: float
    grounding_rate: float
    k: int
    n_queries: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvalCase:
    query: str
    relevant_sources: list[str]


def load_eval_set(path: str | Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            obj = json.loads(line)
            cases.append(EvalCase(query=obj["query"], relevant_sources=list(obj["relevant_sources"])))
    return cases


def evaluate_retrieval(
    retrieve_fn: Callable[[str, int], list[Retrieved]],
    cases: list[EvalCase],
    *,
    k: int = 5,
) -> dict[str, float]:
    """Retrieval-only metrics for any retriever exposed as ``fn(query, k)``.

    Lets us score and compare different retrievers (dense vs hybrid vs
    hybrid+MMR) on the same eval set with identical metrics.
    """
    precisions, recalls, rrs = [], [], []
    for case in cases:
        relevant = set(case.relevant_sources)
        sources = [h.source for h in retrieve_fn(case.query, k)]
        precisions.append(precision_at_k(sources, relevant, k))
        recalls.append(recall_at_k(sources, relevant, k))
        rrs.append(reciprocal_rank(sources, relevant))
    return {
        "precision_at_k": round(mean(precisions), 4) if precisions else 0.0,
        "recall_at_k": round(mean(recalls), 4) if recalls else 0.0,
        "mrr": round(mean(rrs), 4) if rrs else 0.0,
        "k": k,
        "n_queries": len(cases),
    }


def evaluate(
    store: VectorStore,
    cases: list[EvalCase],
    *,
    k: int = 5,
    namespace: str = "default",
    check_grounding: bool = True,
) -> EvalReport:
    """Run the eval set and return aggregate retrieval + grounding metrics."""
    precisions, recalls, rrs = [], [], []
    grounded, generated = 0, 0

    for case in cases:
        relevant = set(case.relevant_sources)
        hits = store.query(case.query, namespace=namespace, k=k)
        sources = [h.source for h in hits]
        precisions.append(precision_at_k(sources, relevant, k))
        recalls.append(recall_at_k(sources, relevant, k))
        rrs.append(reciprocal_rank(sources, relevant))

        if check_grounding:
            # mock generation keeps this offline; each question's source should
            # be a relevant document -> measures whether generation drew on the
            # right material.
            for q in generate_quiz(store, case.query, namespace=namespace, num_questions=2, mock=True):
                generated += 1
                if q.source in relevant:
                    grounded += 1

    return EvalReport(
        precision_at_k=round(mean(precisions), 4) if precisions else 0.0,
        recall_at_k=round(mean(recalls), 4) if recalls else 0.0,
        mrr=round(mean(rrs), 4) if rrs else 0.0,
        grounding_rate=round(grounded / generated, 4) if generated else 0.0,
        k=k,
        n_queries=len(cases),
    )
