"""End-to-end orchestration: ingest -> index -> generate quiz."""

from __future__ import annotations

from pathlib import Path

from app.embeddings import Embedder, get_embedder
from app.ingest import ingest_file
from app.quiz import QuestionType, QuizQuestion, generate_quiz
from app.vectorstore import VectorStore


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

    ``fresh=True`` clears only THIS namespace's prior documents first (tenant-safe,
    so other namespaces are untouched); leave False to persist and accumulate
    across calls, which is the multi-tenant production default.
    """
    store = VectorStore(embedder or get_embedder(), collection=collection)
    if fresh:
        store.clear_namespace(namespace)
    for f in files:
        chunks = ingest_file(
            f, namespace=namespace, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        store.add(chunks)
    return store


def run(
    files: list[str | Path],
    topic: str,
    *,
    namespace: str = "default",
    num_questions: int = 5,
    question_type: QuestionType = "multiple_choice",
    embedder: Embedder | None = None,
    mock: bool = False,
) -> list[QuizQuestion]:
    store = build_store(files, namespace=namespace, embedder=embedder)
    return generate_quiz(
        store,
        topic,
        namespace=namespace,
        num_questions=num_questions,
        question_type=question_type,
        mock=mock,
    )
