"""End-to-end orchestration: ingest -> index -> (optionally hybrid) retrieve -> generate."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from app.budget import TokenBudget
from app.cache import CachingEmbedder
from app.embeddings import Embedder, get_embedder
from app.ingest import Chunk, ingest_file
from app.metrics import RunMetrics, Timer, estimate_tokens
from app.quiz import QuestionType, QuizQuestion, generate_quiz
from app.retrieval import HybridRetriever
from app.vectorstore import VectorStore

RetrievalMode = Literal["dense", "hybrid"]


def ingest_all(
    files: list[str | Path],
    *,
    namespace: str = "default",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for f in files:
        chunks.extend(
            ingest_file(f, namespace=namespace, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )
    return chunks


def build_store(
    files: list[str | Path],
    *,
    namespace: str = "default",
    embedder: Embedder | None = None,
    collection: str = "quiz_documents",
    fresh: bool = False,
    persist_dir: str | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> VectorStore:
    """Ingest and index one or more documents into a namespace-scoped store.

    ``fresh=True`` clears only THIS namespace's prior documents first (tenant-safe).
    ``persist_dir`` makes the index durable on disk (survives restarts).
    """
    store = VectorStore(embedder or get_embedder(), collection=collection, persist_dir=persist_dir)
    if fresh:
        store.clear_namespace(namespace)
    store.add(ingest_all(files, namespace=namespace, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    return store


def run(
    files: list[str | Path],
    topic: str,
    *,
    namespace: str = "default",
    num_questions: int = 5,
    question_type: QuestionType = "multiple_choice",
    embedder: Embedder | None = None,
    retrieval: RetrievalMode = "hybrid",
    use_mmr: bool = False,
    mock: bool = False,
    persist_dir: str | None = None,
    cache_path: str | None = None,
    token_budget: int | None = None,
    concurrency: int = 1,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[QuizQuestion]:
    """Build the index and generate a quiz. Defaults to hybrid retrieval (the
    eval-measured better retriever); ``retrieval='dense'`` for the simple path."""
    questions, _ = run_with_metrics(
        files, topic,
        namespace=namespace, num_questions=num_questions, question_type=question_type,
        embedder=embedder, retrieval=retrieval, use_mmr=use_mmr, mock=mock,
        persist_dir=persist_dir, cache_path=cache_path, token_budget=token_budget,
        concurrency=concurrency, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
    )
    return questions


def run_with_metrics(
    files: list[str | Path],
    topic: str,
    *,
    namespace: str = "default",
    num_questions: int = 5,
    question_type: QuestionType = "multiple_choice",
    embedder: Embedder | None = None,
    retrieval: RetrievalMode = "hybrid",
    use_mmr: bool = False,
    mock: bool = False,
    persist_dir: str | None = None,
    cache_path: str | None = None,
    token_budget: int | None = None,
    concurrency: int = 1,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> tuple[list[QuizQuestion], RunMetrics]:
    """Same as ``run`` but also returns a :class:`RunMetrics` with stage timings,
    counts, and an estimated token/cost figure — used by the CLI and for ops.

    ``cache_path`` memoizes embeddings on disk (skips re-embedding unchanged
    chunks). ``token_budget`` caps a namespace's estimated token spend — for BOTH
    embeddings and generation — and fails fast with ``BudgetExceeded`` before
    paying. ``concurrency`` generates questions in parallel.
    """
    metrics = RunMetrics(documents=len(files))

    chunks = ingest_all(files, namespace=namespace, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    metrics.chunks_indexed = len(chunks)
    full_embed_est = sum(estimate_tokens(c.text) for c in chunks)
    metrics.embed_tokens_est = full_embed_est

    # Pre-flight budget check: reject before spending on embeddings (conservative —
    # charges the full estimate even though the cache may cover some of it).
    budget = TokenBudget(token_budget)
    budget.charge(full_embed_est, label="embeddings")

    resolved_cache = cache_path or os.environ.get("EMBED_CACHE_PATH") or None
    base_embedder = embedder or get_embedder()
    active_embedder = CachingEmbedder(base_embedder, path=resolved_cache) if resolved_cache else base_embedder

    try:
        with Timer() as t_index:
            store = VectorStore(active_embedder, persist_dir=persist_dir)
            store.add(chunks)
        metrics.index_seconds = t_index.seconds
        if isinstance(active_embedder, CachingEmbedder):
            metrics.extra["cache_hits"] = active_embedder.hits
            metrics.extra["cache_misses"] = active_embedder.misses
            # ~tokens we DIDN'T have to re-embed because they were already cached
            metrics.extra["embed_tokens_saved"] = active_embedder.chars_hit // 4
            metrics.extra["embed_tokens_full_estimate"] = full_embed_est
            # cost should reflect what was ACTUALLY sent to the embedder (misses),
            # not the baseline — otherwise a warm cache still looks expensive.
            metrics.embed_tokens_est = active_embedder.chars_missed // 4

        retrieve_fn = None
        if retrieval == "hybrid":
            hybrid = HybridRetriever(store, chunks, namespace=namespace)
            retrieve_fn = lambda q, k: hybrid.query(q, k=k, use_mmr=use_mmr)  # noqa: E731

        # Pre-flight budget for generation too — LLM calls are usually the larger
        # cost, so guard them before spending, not just the embeddings.
        avg_chunk_tokens = full_embed_est // max(1, len(chunks))
        llm_pre_est = num_questions * (avg_chunk_tokens + 200)
        budget.charge(llm_pre_est, label="generation")

        with Timer() as t_gen:
            questions = generate_quiz(
                store, topic,
                namespace=namespace, num_questions=num_questions,
                question_type=question_type, mock=mock, retrieve_fn=retrieve_fn,
                concurrency=concurrency,
            )
        metrics.generate_seconds = t_gen.seconds
        metrics.generation_calls = len(questions)
        metrics.questions_produced = len(questions)
        # estimate LLM tokens from the produced Q/A text plus a fixed prompt overhead
        metrics.llm_tokens_est = sum(
            estimate_tokens(q.question + " " + " ".join(q.choices) + " " + (q.answer or ""))
            for q in questions
        ) + 120 * max(1, len(questions))
        metrics.extra["budget_spent"] = budget.spent
        if budget.remaining is not None:
            metrics.extra["budget_remaining"] = budget.remaining

        return questions, metrics
    finally:
        # Always release the cache's SQLite connection — otherwise repeated runs
        # in a long-lived process (e.g. Streamlit) leak open connections.
        if isinstance(active_embedder, CachingEmbedder):
            active_embedder.close()
