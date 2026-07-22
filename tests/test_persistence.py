"""Persistence: an on-disk store survives being reopened in a new process-like
VectorStore instance (the core Phase-3 reliability guarantee)."""

from app.embeddings import HashingEmbedder
from app.ingest import chunk_text
from app.vectorstore import VectorStore

CORPUS = {
    "a.md": "Photosynthesis converts sunlight and carbon dioxide into glucose and oxygen.",
    "b.md": "Gravity keeps planets in orbit around the sun.",
}


def _chunks():
    out = []
    for name, text in CORPUS.items():
        out += chunk_text(text, source=name, namespace="default", chunk_size=80, chunk_overlap=10)
    return out


def test_index_survives_reopen(tmp_path):
    path = str(tmp_path / "chroma")
    chunks = _chunks()

    # First "session": index and close.
    store1 = VectorStore(HashingEmbedder(), collection="persist_test", persist_dir=path)
    store1.clear_namespace("default")
    added = store1.add(chunks)
    assert added == len(chunks)
    del store1

    # Second "session": reopen the SAME dir — data is still there, no re-embedding.
    store2 = VectorStore(HashingEmbedder(), collection="persist_test", persist_dir=path)
    assert store2.count("default") == len(chunks)
    hits = store2.query("sunlight glucose oxygen", namespace="default", k=1)
    assert hits and hits[0].source == "a.md"


def test_in_memory_default_is_isolated():
    # No persist_dir -> in-memory; a fresh instance does not see prior data.
    s = VectorStore(HashingEmbedder(), collection="mem_test")
    s.reset()
    assert s.count("default") == 0
