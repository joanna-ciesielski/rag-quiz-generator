"""Concurrent generation must produce the SAME ordered, deduplicated result as
the sequential path — parallelism is a speed optimization, not a behavior change."""

from app.embeddings import HashingEmbedder
from app.ingest import chunk_text
from app.quiz import generate_quiz
from app.vectorstore import VectorStore

DOC = (
    "Alpha section covers introduction basics and setup steps in detail. "
    "Beta section covers intermediate topics and worked examples in depth. "
    "Gamma section covers advanced material, theory, and proofs at length. "
    "Delta section covers the summary, review, and conclusion thoroughly. "
)


def _store():
    store = VectorStore(HashingEmbedder(), collection="concurrency_test")
    store.reset()
    store.add(chunk_text(DOC, source="doc.txt", namespace="default", chunk_size=90, chunk_overlap=10))
    return store


def test_concurrent_matches_sequential():
    store = _store()
    seq = generate_quiz(store, "sections", num_questions=3, question_type="open_ended",
                        mock=True, concurrency=1)
    par = generate_quiz(store, "sections", num_questions=3, question_type="open_ended",
                        mock=True, concurrency=4)
    assert [q.question for q in par] == [q.question for q in seq]
    assert [q.source for q in par] == [q.source for q in seq]


def test_concurrent_respects_num_questions_cap():
    store = _store()
    par = generate_quiz(store, "sections", num_questions=2, question_type="open_ended",
                        mock=True, concurrency=4)
    assert len(par) <= 2
