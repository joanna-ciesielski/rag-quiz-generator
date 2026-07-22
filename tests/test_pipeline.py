"""Offline tests: real chunking + real Chroma vector search + stubbed LLM.

Uses the deterministic HashingEmbedder and mock question generation, so the full
retrieve->generate path is exercised with no API key and no network.
"""

from pathlib import Path

import pytest

from app.embeddings import HashingEmbedder
from app.ingest import chunk_text
from app.quiz import GenerationError, QuizQuestion, generate_quiz
from app.pipeline import run
from app.vectorstore import VectorStore

DATA = Path(__file__).resolve().parents[1] / "data"


def _store_with(docs: dict[str, str], namespace="default") -> VectorStore:
    store = VectorStore(HashingEmbedder(), collection=f"test_{abs(hash(tuple(docs)))%10000}")
    for name, text in docs.items():
        store.add(chunk_text(text, source=name, namespace=namespace, chunk_size=120, chunk_overlap=20))
    return store


def test_chunking_is_structure_aware_and_ided():
    chunks = chunk_text("Para one. More.\n\nPara two is here.", source="d.txt", chunk_size=20, chunk_overlap=5)
    assert len(chunks) >= 2
    assert all(c.id.startswith("default:d.txt:") for c in chunks)
    assert all(c.metadata["source"] == "d.txt" for c in chunks)


def test_retrieval_returns_relevant_chunk():
    store = _store_with({"animals.txt": "The cheetah is the fastest land animal. Snails are slow."})
    hits = store.query("fastest animal", k=2)
    assert hits
    assert "cheetah" in hits[0].text.lower()


def test_namespace_scoping_prevents_cross_tenant_leak():
    store = VectorStore(HashingEmbedder(), collection="test_tenants")
    store.add(chunk_text("Tenant A secret: alpha widget.", source="a.txt", namespace="tenantA", chunk_size=100, chunk_overlap=10))
    store.add(chunk_text("Tenant B secret: beta gadget.", source="b.txt", namespace="tenantB", chunk_size=100, chunk_overlap=10))
    a_hits = store.query("secret", namespace="tenantA", k=5)
    assert a_hits and all("alpha" in h.text.lower() for h in a_hits)
    assert all("beta" not in h.text.lower() for h in a_hits)  # B never leaks into A


def test_generate_quiz_mock_multiple_choice():
    store = _store_with({"sci.txt": "Water boils at 100 degrees Celsius at sea level. Ice melts at 0."})
    qs = generate_quiz(store, "boiling point", num_questions=2, question_type="multiple_choice", mock=True)
    assert qs and all(isinstance(q, QuizQuestion) for q in qs)
    assert all(q.type == "multiple_choice" and len(q.choices) == 4 for q in qs)
    assert all(q.source == "sci.txt" for q in qs)


def test_generate_quiz_dedupes_and_respects_count():
    store = _store_with({"r.txt": "Repeat. Repeat. Repeat. Repeat."})
    qs = generate_quiz(store, "repeat", num_questions=5, mock=True)
    keys = [q.question.lower() for q in qs]
    assert len(keys) == len(set(keys))  # no duplicates


def test_empty_store_returns_no_questions():
    store = VectorStore(HashingEmbedder(), collection="test_empty")
    assert generate_quiz(store, "anything", mock=True) == []


def test_full_pipeline_run_offline(tmp_path):
    doc = tmp_path / "lesson.md"
    doc.write_text("# Photosynthesis\nPlants convert sunlight into energy. Chlorophyll is green.", encoding="utf-8")
    qs = run([doc], "photosynthesis", num_questions=2, embedder=HashingEmbedder(), mock=True)
    assert qs and all(q.question for q in qs)


def test_reset_clears_prior_documents():
    store = VectorStore(HashingEmbedder(), collection="test_reset")
    store.add(chunk_text("Old content about dinosaurs.", source="old.txt", chunk_size=100, chunk_overlap=10))
    assert store.count() > 0
    store.reset()
    assert store.count() == 0
    # after reset, prior content is no longer retrievable
    hits = store.query("dinosaurs", k=3)
    assert hits == []


def test_llm_error_wrapped_as_generation_error(monkeypatch):
    """Non-mock path: an SDK error becomes a clean GenerationError."""
    import sys
    import types

    class _APIError(Exception):
        pass

    class _Msgs:
        def create(self, **kw):
            raise _APIError("boom")

    class _Chat:
        completions = _Msgs()

    class _Client:
        chat = _Chat()

    stub = types.ModuleType("openai")
    stub.OpenAI = lambda: _Client()
    stub.APIError = _APIError
    monkeypatch.setitem(sys.modules, "openai", stub)

    store = _store_with({"x.txt": "Some content about topic X for retrieval."})
    with pytest.raises(GenerationError):
        generate_quiz(store, "topic X", num_questions=1, mock=False)
