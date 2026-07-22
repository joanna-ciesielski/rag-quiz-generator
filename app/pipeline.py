"""End-to-end orchestration: ingest -> index -> (optionally hybrid) retrieve -> generate."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

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
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[QuizQuestion]:
    """Build the index and generate a quiz. Defaults to hybrid retrieval (the
    eval-measured better retriever); ``retrieval='dense'`` for the simple path."""
    questions, _ = run_with_metrics(
        files, topic,
        namespace=namespace, num_questions=num_questions, question_type=question_type,
        embedder=embedder, retrieval=retrieval, use_mmr=use_mmr, mock=mock,
        persist_dir=persist_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
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
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> tuple[list[QuizQuestion], RunMetrics]:
    """Same as ``run`` but also returns a :class:`RunMetrics` with stage timings,
    counts, and an estimated token/cost figure — used by the CLI and for ops."""
    metrics = RunMetrics(documents=len(files))

    with Timer() as t_index:
        chunks = ingest_all(files, namespace=namespace, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        store = VectorStore(embedder or get_embedder(), persist_dir=persist_dir)
        store.add(chunks)
    metrics.index_seconds = t_index.seconds
    metrics.chunks_indexed = len(chunks)
    metrics.embed_tokens_est = sum(estimate_tokens(c.text) for c in chunks)

    retrieve_fn = None
    if retrieval == "hybrid":
        hybrid = HybridRetriever(store, chunks, namespace=namespace)
        retrieve_fn = lambda q, k: hybrid.query(q, k=k, use_mmr=use_mmr)  # noqa: E731

    with Timer() as t_gen:
        questions = generate_quiz(
            store, topic,
            namespace=namespace, num_questions=num_questions,
            question_type=question_type, mock=mock, retrieve_fn=retrieve_fn,
        )
    metrics.generate_seconds = t_gen.seconds
    metrics.generation_calls = len(questions)
    metrics.questions_produced = len(questions)
    # estimate LLM tokens from the produced Q/A text plus a fixed prompt overhead
    metrics.llm_tokens_est = sum(
        estimate_tokens(q.question + " " + " ".join(q.choices) + " " + (q.answer or ""))
        for q in questions
    ) + 120 * max(1, len(questions))

    return questions, metrics
