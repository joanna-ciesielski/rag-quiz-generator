"""End-to-end orchestration: ingest -> index -> (optionally hybrid) retrieve -> generate."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from app.embeddings import Embedder, get_embedder
from app.ingest import Chunk, ingest_file
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
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> VectorStore:
    """Ingest and index one or more documents into a namespace-scoped store.

    ``fresh=True`` clears only THIS namespace's prior documents first (tenant-safe).
    """
    store = VectorStore(embedder or get_embedder(), collection=collection)
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
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[QuizQuestion]:
    """Build the index and generate a quiz. Defaults to hybrid retrieval (the
    eval-measured better retriever); ``retrieval='dense'`` for the simple path."""
    chunks = ingest_all(files, namespace=namespace, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    store = VectorStore(embedder or get_embedder())
    store.add(chunks)

    retrieve_fn = None
    if retrieval == "hybrid":
        hybrid = HybridRetriever(store, chunks, namespace=namespace)
        retrieve_fn = lambda q, k: hybrid.query(q, k=k, use_mmr=use_mmr)  # noqa: E731

    return generate_quiz(
        store,
        topic,
        namespace=namespace,
        num_questions=num_questions,
        question_type=question_type,
        mock=mock,
        retrieve_fn=retrieve_fn,
    )
