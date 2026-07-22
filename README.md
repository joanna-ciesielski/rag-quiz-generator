# RAG Quiz Generator

![CI](https://github.com/joanna-ciesielski/rag-quiz-generator/actions/workflows/ci.yml/badge.svg)

A production-grade Retrieval-Augmented Generation app that turns your documents
(PDF, Markdown, text) into quiz questions **grounded in the source material** —
upload content, and it retrieves the most relevant passages and generates
validated questions with answers and citations.

Built with real components throughout: structure-aware chunking, real vector
embeddings, a Chroma vector store with **per-namespace (tenant) isolation**,
structured/validated LLM output, retries, and a full offline-testable test suite.

## Architecture

```
documents ──▶ ingest (chunk) ──▶ embed ──▶ Chroma vector store (namespace-scoped)
                                                     │
topic ─────────────────────────▶ retrieve top-k ─────┘──▶ LLM (JSON) ──▶ validated questions
```

- **Ingestion** (`app/ingest.py`) — PDF/MD/text → structure-aware chunks
  (`RecursiveCharacterTextSplitter`, paragraph→line→word boundaries) with source
  metadata for citation.
- **Embeddings** (`app/embeddings.py`) — pluggable backend: `OpenAIEmbedder`
  (production) or a deterministic offline `HashingEmbedder` (tests/demo, no key).
- **Vector store** (`app/vectorstore.py`) — Chroma with cosine similarity;
  embeddings are supplied by the app, and **every query is scoped to a namespace**
  so one tenant's content can't leak into another's results.
- **Generation** (`app/quiz.py`) — retrieves top-k context, asks the LLM for
  strict JSON, and **validates it with pydantic** (no fragile text parsing);
  deduplicates questions; wraps API failures in a clean `GenerationError`.
- **UI** (`streamlit_app.py`) — a thin layer over the pipeline.

## Quickstart

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
export OPENAI_API_KEY=sk-...          # for real embeddings + generation
streamlit run streamlit_app.py
```

Try it **offline** (no API key) with the deterministic embedder + mock questions:

```bash
EMBEDDER=hashing streamlit run streamlit_app.py   # then tick "Offline mode" in the sidebar
```

## Programmatic use

```python
from app.pipeline import run
questions = run(["data/sample_lesson.md"], topic="the water cycle", num_questions=5)
for q in questions:
    print(q.question, "->", q.answer, f"[{q.source}]")
```

## Tests

The suite exercises real chunking and **real Chroma vector search** with the
offline embedder and stubbed LLM — no API key, no network — and also runs in CI
on Python 3.11/3.12.

```bash
python -m pytest -q
```

Coverage includes chunking, relevant-chunk retrieval, **cross-tenant isolation**,
structured multiple-choice generation, deduplication, the empty-store case, the
full pipeline, and clean error handling on API failure.

## Configuration

| Variable         | Default                  | Purpose                                   |
| ---------------- | ------------------------ | ----------------------------------------- |
| `OPENAI_API_KEY` | —                        | Required for real embeddings + generation |
| `EMBEDDER`       | `openai`                 | `hashing` for offline/no-key mode         |
| `EMBED_MODEL`    | `text-embedding-3-small` | OpenAI embedding model                    |
| `QUIZ_MODEL`     | `gpt-4o-mini`            | Generation model                          |

## Beyond this version (deliberate extension points)

Documented, not implemented, to keep the core focused: hybrid (dense + sparse)
retrieval and a reranking stage for higher precision; a persistent/hosted vector
store (Chroma server, pgvector, or Qdrant) for scale; and a regression eval set
with quality metrics wired into CI.

---

*History: this began as a mocked prototype (stubbed embeddings + substring
"search"); it was rebuilt into this real, tested RAG pipeline.*
