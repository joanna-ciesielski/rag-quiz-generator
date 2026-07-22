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
  so one tenant's content can't leak into another's results. Optionally
  **persistent on disk** (`persist_dir` / `CHROMA_PERSIST_DIR`) so an index
  survives restarts and documents aren't re-embedded every run.
- **Reliability** (`app/retry.py`) — API calls (embeddings + generation) retry
  transient failures (rate limits, timeouts, 5xx) with **exponential backoff**;
  auth/bad-request errors are not retried, since waiting won't fix them.
- **Observability** (`app/metrics.py`) — each run reports stage timings, counts,
  and an estimated token/cost figure, so cost and latency are visible, not guessed.
- **Hybrid retrieval** (`app/retrieval.py`) — dense (semantic) + BM25 (lexical)
  fused with Reciprocal Rank Fusion, with optional MMR reranking for diversity.
  Measured to beat dense-only on the eval set (see below); the default retriever.
- **Generation** (`app/quiz.py`) — retrieves top-k context, asks the LLM for
  strict JSON, and **validates it with pydantic** (no fragile text parsing);
  deduplicates questions; wraps API failures in a clean `GenerationError`.
- **Interfaces** — a Streamlit UI (`streamlit_app.py`) and a headless
  **CLI** (`app/cli.py`, `python -m app.cli`), both thin layers over the pipeline.

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

## Command line

Run the whole pipeline without the UI. Offline mode needs no API key:

```bash
# offline smoke test (deterministic embedder + mock questions)
EMBEDDER=hashing python -m app.cli eval/corpus/*.md --topic "the water cycle" --offline

# real run against a durable on-disk index
export OPENAI_API_KEY=sk-...
python -m app.cli notes.pdf --topic "chapter 3" --persist-dir .chroma --json
```

Questions print to stdout (`--json` for machine-readable); a metrics line
(timings, counts, estimated cost) goes to stderr.

## Programmatic use

```python
from app.pipeline import run, run_with_metrics
questions = run(["data/sample_lesson.md"], topic="the water cycle", num_questions=5)
for q in questions:
    print(q.question, "->", q.answer, f"[{q.source}]")

# run_with_metrics also returns stage timings + an estimated token/cost figure
questions, metrics = run_with_metrics(["data/sample_lesson.md"], topic="the water cycle")
print(metrics.summary())
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

## Evaluation (measured retrieval quality)

Retrieval quality is **measured, not assumed**. A labeled eval set
(`eval/eval_set.jsonl`) over a small corpus (`eval/corpus/`) is scored on
**precision@k, recall@k, MRR**, and a generation **grounding rate**:

```bash
EMBEDDER=hashing python eval/run_eval.py            # print metrics
EMBEDDER=hashing python eval/run_eval.py --k 5 --min-mrr 0.6 --min-recall 0.9 --min-precision 0.3
```

The runner **compares retrievers** (dense vs hybrid vs hybrid+MMR) on the same
eval set, and `--gate-on hybrid` makes it a **CI quality gate** — it exits
non-zero if metrics fall below thresholds, so a retrieval regression fails the
build. This let us *prove* the Phase-2 hybrid retriever helped rather than
assume it (offline hashing-embedder numbers):

| retriever        | precision@5 | recall@5 | MRR |
| ---------------- | ----------- | -------- | --- |
| dense (baseline) | 0.40        | 1.0      | 0.80 |
| **hybrid (dense+BM25)** | **0.44** | **1.0** | **1.00** |
| hybrid + MMR     | 0.40        | 1.0      | 0.80 |

Hybrid is a clear win. MMR *lowered* the score here — it trades relevance for
diversity, which hurts when relevance is concentrated — so it's **off by
default**, enabled only when redundant near-duplicate chunks are a problem. That
tradeoff is exactly the kind of thing the eval harness is for.

Metrics run offline in CI with the deterministic `HashingEmbedder` (a weak,
keyword-hash embedder — real numbers with `EMBEDDER=openai` are higher); the
harness is the same either way. Relevance is defined at the source-document
level: a retrieved chunk is relevant if it came from a document labeled relevant
for that query.

## Configuration

| Variable         | Default                  | Purpose                                   |
| ---------------- | ------------------------ | ----------------------------------------- |
| `OPENAI_API_KEY`     | —                        | Required for real embeddings + generation |
| `EMBEDDER`           | `openai`                 | `hashing` for offline/no-key mode         |
| `EMBED_MODEL`        | `text-embedding-3-small` | OpenAI embedding model                    |
| `QUIZ_MODEL`         | `gpt-4o-mini`            | Generation model                          |
| `CHROMA_PERSIST_DIR` | — (in-memory)            | Directory for a durable on-disk index     |
| `EMBED_MAX_RETRIES`  | `3`                      | Retries for transient embedding failures  |
| `LLM_MAX_RETRIES`    | `3`                      | Retries for transient generation failures |
| `RETRY_BASE_DELAY`   | `0.5`                    | Base seconds for exponential backoff      |

## Beyond this version (deliberate extension points)

Documented, not implemented, to keep the core focused: a **neural cross-encoder
reranker** (stronger than MMR, but needs a downloadable model); a hosted/scale-out
vector store (Chroma server, pgvector, or Qdrant); embedding **caching** and
per-tenant **token budgets**; and multi-tier model routing plus richer tracing.

*Already implemented:* a regression eval set with retrieval metrics + CI gate
(see [Evaluation](#evaluation-measured-retrieval-quality)); **hybrid retrieval
(dense + BM25 + optional MMR)** proven to beat dense-only on that eval;
**persistent on-disk storage**, **retry-with-backoff** on API calls, per-run
**metrics** (timings + estimated cost), and a headless **CLI**.

---

*History: this began as a mocked prototype (stubbed embeddings + substring
"search"); it was rebuilt into this real, tested RAG pipeline.*
