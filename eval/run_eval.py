"""Run the retrieval-quality evaluation and (optionally) gate CI on thresholds.

    python eval/run_eval.py                 # print metrics
    python eval/run_eval.py --min-mrr 0.8   # exit non-zero if below threshold

Runs offline with the deterministic HashingEmbedder (set EMBEDDER=hashing) so it
is reproducible in CI; point EMBEDDER=openai (with a key) to evaluate the real
embedding model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# allow running as a script (python eval/run_eval.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.embeddings import get_embedder  # noqa: E402
from app.evaluation import evaluate, load_eval_set  # noqa: E402
from app.ingest import ingest_file  # noqa: E402
from app.vectorstore import VectorStore  # noqa: E402

ROOT = Path(__file__).resolve().parent


def build_corpus_store(namespace: str = "default") -> VectorStore:
    store = VectorStore(get_embedder(), collection="eval_corpus")
    store.reset()
    for doc in sorted((ROOT / "corpus").glob("*.md")):
        # finer chunks so each relevant document contributes several passages,
        # making precision@k meaningful (not capped by a single relevant chunk).
        store.add(ingest_file(doc, namespace=namespace, chunk_size=200, chunk_overlap=40))
    return store


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate retrieval quality against the eval set.")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--min-precision", type=float, default=None)
    ap.add_argument("--min-recall", type=float, default=None)
    ap.add_argument("--min-mrr", type=float, default=None)
    ap.add_argument("--min-grounding", type=float, default=None)
    args = ap.parse_args()

    store = build_corpus_store()
    cases = load_eval_set(ROOT / "eval_set.jsonl")
    report = evaluate(store, cases, k=args.k)

    print(json.dumps(report.to_dict(), indent=2))

    checks = {
        "precision_at_k": args.min_precision,
        "recall_at_k": args.min_recall,
        "mrr": args.min_mrr,
        "grounding_rate": args.min_grounding,
    }
    failures = []
    for metric, threshold in checks.items():
        if threshold is not None and getattr(report, metric) < threshold:
            failures.append(f"{metric}={getattr(report, metric)} < required {threshold}")
    if failures:
        print("\nQUALITY GATE FAILED:", *failures, sep="\n  ", file=sys.stderr)
        return 1
    print("\nQuality gate passed." if any(v is not None for v in checks.values()) else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
