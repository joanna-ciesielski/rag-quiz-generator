"""Streamlit UI — thin layer over the RAG pipeline.

Run:  streamlit run streamlit_app.py
Set OPENAI_API_KEY for real embeddings + generation, or set EMBEDDER=hashing
and use the mock toggle to try it offline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from app.embeddings import HashingEmbedder, get_embedder
from app.pipeline import ingest_all
from app.quiz import generate_quiz
from app.retrieval import HybridRetriever
from app.vectorstore import VectorStore

st.set_page_config(page_title="RAG Quiz Generator", page_icon="📝")
st.title("📝 RAG Quiz Generator")
st.caption("Upload documents → retrieve relevant context → generate quiz questions grounded in your material.")

with st.sidebar:
    st.header("Settings")
    offline = st.toggle("Offline mode (no API key)", value=False,
                        help="Uses a deterministic local embedder + mock questions to try the flow without OpenAI.")
    namespace = st.text_input("Namespace (tenant)", value="default",
                              help="Retrieval is scoped to this namespace — content never leaks across namespaces.")
    num_q = st.slider("Number of questions", 1, 20, 5)
    qtype = st.radio("Question type", ["multiple_choice", "open_ended"])
    st.divider()
    hybrid_on = st.toggle(
        "Hybrid retrieval (dense + keyword)", value=True,
        help="Fuses semantic (dense) and BM25 (lexical) search — the eval-measured better retriever.",
    )
    mmr_on = st.toggle(
        "MMR reranking (diversity)", value=False, disabled=not hybrid_on,
        help="Trades some relevance for diversity. Off by default; lowered scores on the eval set.",
    )

files = st.file_uploader("Upload PDF / Markdown / text", type=["pdf", "md", "txt"], accept_multiple_files=True)
topic = st.text_input("Quiz topic", placeholder="e.g. cell biology, chapter 3")

if st.button("Generate quiz", type="primary"):
    if not files or not topic:
        st.warning("Upload at least one document and enter a topic.")
        st.stop()

    saved: list[Path] = []
    for f in files:
        tmp = Path(tempfile.mkstemp(suffix=Path(f.name).suffix)[1])
        tmp.write_bytes(f.read())
        saved.append(tmp)

    embedder = HashingEmbedder() if offline else get_embedder()
    with st.spinner("Indexing documents…"):
        # clear_namespace clears only THIS namespace's prior docs (tenant-safe), so a
        # rerun uses this run's uploads without wiping other namespaces.
        chunks = ingest_all(saved, namespace=namespace)
        store = VectorStore(embedder)
        store.clear_namespace(namespace)
        store.add(chunks)
    st.success(f"Indexed {store.count(namespace)} chunks in namespace '{namespace}'.")

    retrieve_fn = None
    if hybrid_on:
        hybrid = HybridRetriever(store, chunks, namespace=namespace)
        retrieve_fn = lambda q, k: hybrid.query(q, k=k, use_mmr=mmr_on)  # noqa: E731

    with st.spinner("Generating questions…"):
        questions = generate_quiz(
            store, topic, namespace=namespace,
            num_questions=num_q, question_type=qtype, mock=offline,
            retrieve_fn=retrieve_fn,
        )

    if not questions:
        st.error("No relevant content found for that topic. Try a different topic or document.")
    for i, q in enumerate(questions, 1):
        st.subheader(f"Question {i}")
        st.write(q.question)
        if q.type == "multiple_choice":
            st.radio("Choose:", q.choices, key=f"q{i}", index=None)
            with st.expander("Answer & source"):
                st.write(f"**Answer:** {q.answer}")
                st.caption(f"Source: {q.source}")
        else:
            st.text_area("Your answer", key=f"a{i}")
            with st.expander("Model answer & source"):
                st.write(q.answer)
                st.caption(f"Source: {q.source}")
