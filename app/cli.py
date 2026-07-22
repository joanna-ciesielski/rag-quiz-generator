"""Command-line entrypoint — run the pipeline without the Streamlit UI.

    # offline smoke test (no API key):
    python -m app.cli eval/corpus/*.md --topic "the water cycle" --offline

    # real run against a persistent store:
    export OPENAI_API_KEY=sk-...
    python -m app.cli notes.pdf --topic "chapter 3" --persist-dir .chroma

Prints questions (text or --json) to stdout and a metrics line to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys

from app.embeddings import HashingEmbedder, get_embedder
from app.pipeline import run_with_metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rag-quiz", description="Generate a grounded quiz from documents.")
    p.add_argument("files", nargs="+", help="Document paths (.pdf/.md/.txt)")
    p.add_argument("--topic", required=True, help="Quiz topic to retrieve context for")
    p.add_argument("--namespace", default="default", help="Tenant namespace (isolates retrieval)")
    p.add_argument("--num-questions", type=int, default=5)
    p.add_argument("--type", dest="qtype", choices=["multiple_choice", "open_ended"], default="multiple_choice")
    p.add_argument("--retrieval", choices=["dense", "hybrid"], default="hybrid")
    p.add_argument("--mmr", action="store_true", help="Enable MMR reranking (diversity; off by default)")
    p.add_argument("--offline", action="store_true",
                   help="No API key: deterministic hashing embedder + mock questions")
    p.add_argument("--persist-dir", default=None, help="Directory for a durable on-disk index")
    p.add_argument("--json", action="store_true", help="Emit questions as JSON")
    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument("--chunk-overlap", type=int, default=150)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    embedder = HashingEmbedder() if args.offline else get_embedder()

    questions, metrics = run_with_metrics(
        args.files, args.topic,
        namespace=args.namespace, num_questions=args.num_questions,
        question_type=args.qtype, embedder=embedder, retrieval=args.retrieval,
        use_mmr=args.mmr, mock=args.offline, persist_dir=args.persist_dir,
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
    )

    if args.json:
        print(json.dumps([q.model_dump() for q in questions], indent=2))
    else:
        if not questions:
            print("No relevant content found for that topic.")
        for i, q in enumerate(questions, 1):
            print(f"\nQ{i}. {q.question}")
            for j, choice in enumerate(q.choices):
                print(f"   {chr(97 + j)}) {choice}")
            if q.answer is not None:
                print(f"   -> answer: {q.answer}")
            if q.source:
                print(f"   [source: {q.source}]")

    print(f"\n[metrics] {metrics.summary()}", file=sys.stderr)
    return 0 if questions else 1


if __name__ == "__main__":
    raise SystemExit(main())
