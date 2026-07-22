"""Evaluate + compare retrievers, and (optionally) gate CI on thresholds.

    python eval/run_eval.py                       # compare dense vs hybrid vs hybrid+MMR
    python eval/run_eval.py --gate-on hybrid --min-mrr 0.9 --min-recall 0.9

Runs offline with the deterministic HashingEmbedder (EMBEDDER=hashing) for
reproducible CI; point EMBEDDER=openai (with a key) to evaluate real embeddings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.embeddings import get_embedder  # noqa: E402
from app.evaluation import evaluate_retrieval, load_eval_set  # noqa: E402
from app.ingest import ingest_file  # noqa: E402
from app.retrieval import HybridRetriever  # noqa: E402
from app.vectorstore import VectorStore  # noqa: E402

ROOT = Path(__file__).resolve().parent


def build(namespace: str = "default"):
    store = VectorStore(get_embedder(), collection="eval_corpus")
    store.reset()
    chunks = []
    for doc in sorted((ROOT / "corpus").glob("*.md")):
        cs = ingest_file(doc, namespace=namespace, chunk_size=200, chunk_overlap=40)
        chunks += cs
        store.add(cs)
    hybrid = HybridRetriever(store, chunks, namespace=namespace)
    return store, hybrid


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate + compare retrievers.")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--gate-on", choices=["dense", "hybrid", "hybrid_mmr"], default=None)
    ap.add_argument("--min-precision", type=float, default=None)
    ap.add_argument("--min-recall", type=float, default=None)
    ap.add_argument("--min-mrr", type=float, default=None)
    args = ap.parse_args()

    store, hybrid = build()
    cases = load_eval_set(ROOT / "eval_set.jsonl")

    retrievers = {
        "dense": lambda q, k: store.query(q, namespace="default", k=k),
        "hybrid": lambda q, k: hybrid.query(q, k=k, use_sparse=True, use_mmr=False),
        "hybrid_mmr": lambda q, k: hybrid.query(q, k=k, use_sparse=True, use_mmr=True),
    }
    results = {name: evaluate_retrieval(fn, cases, k=args.k) for name, fn in retrievers.items()}
    print(json.dumps(results, indent=2))

    if args.gate_on:
        report = results[args.gate_on]
        checks = {"precision_at_k": args.min_precision, "recall_at_k": args.min_recall, "mrr": args.min_mrr}
        failures = [
            f"{m}={report[m]} < required {t}"
            for m, t in checks.items()
            if t is not None and report[m] < t
        ]
        if failures:
            print(f"\nQUALITY GATE FAILED ({args.gate_on}):", *failures, sep="\n  ", file=sys.stderr)
            return 1
        print(f"\nQuality gate passed ({args.gate_on}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
