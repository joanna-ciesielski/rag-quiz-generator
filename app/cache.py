"""Embedding cache — skip re-embedding chunks whose text hasn't changed.

Wraps any ``Embedder``. Each text is keyed by ``sha256(model_id + text)`` so the
cache is content-addressed and safe across models (different models never collide).
With a ``path`` it is backed by SQLite (durable across runs — the big cost win when
re-processing the same documents); without one it caches within the process only.

Only cache *misses* are sent to the wrapped embedder, and identical texts within a
call are embedded once. ``hits``/``misses`` counters make the savings measurable.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3

from app.embeddings import Embedder


def _key(model_id: str, text: str) -> str:
    return hashlib.sha256(f"{model_id}\x00{text}".encode()).hexdigest()


class CachingEmbedder:
    """An ``Embedder`` that memoizes embeddings by content hash."""

    def __init__(self, inner: Embedder, *, path: str | None = None) -> None:
        self.inner = inner
        self.dim = inner.dim
        # Distinguish backends/models so a hashing-embedder vector is never served
        # for an OpenAI request (or vice versa).
        self.model_id = f"{getattr(inner, 'model', type(inner).__name__)}:{inner.dim}"
        self.path = path
        self.hits = 0
        self.misses = 0
        self.chars_hit = 0      # chars we did NOT have to re-embed (cache savings)
        self.chars_missed = 0   # chars actually sent to the wrapped embedder
        self._mem: dict[str, list[float]] = {}
        self._db: sqlite3.Connection | None = None
        if path:
            self._db = sqlite3.connect(path)
            self._db.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
            self._db.commit()

    def _get(self, key: str) -> list[float] | None:
        if key in self._mem:
            return self._mem[key]
        if self._db is not None:
            row = self._db.execute("SELECT v FROM emb WHERE k = ?", (key,)).fetchone()
            if row is not None:
                vec = json.loads(row[0])
                self._mem[key] = vec
                return vec
        return None

    def _put(self, key: str, vec: list[float]) -> None:
        self._mem[key] = vec
        if self._db is not None:
            self._db.execute("INSERT OR REPLACE INTO emb (k, v) VALUES (?, ?)", (key, json.dumps(vec)))

    def embed(self, texts: list[str]) -> list[list[float]]:
        keys = [_key(self.model_id, t) for t in texts]
        out: list[list[float] | None] = [self._get(k) for k in keys]

        # Collect unique misses (dedupe repeated text within this call).
        missing_unique: dict[str, str] = {}  # key -> text
        for key, text, cached in zip(keys, texts, out):
            if cached is None and key not in missing_unique:
                missing_unique[key] = text

        self.hits += sum(1 for c in out if c is not None)
        self.misses += len(missing_unique)
        self.chars_hit += sum(len(t) for t, c in zip(texts, out) if c is not None)
        self.chars_missed += sum(len(t) for t in missing_unique.values())

        if missing_unique:
            miss_keys = list(missing_unique.keys())
            miss_vecs = self.inner.embed([missing_unique[k] for k in miss_keys])
            for key, vec in zip(miss_keys, miss_vecs):
                self._put(key, vec)
            if self._db is not None:
                self._db.commit()
            resolved = dict(zip(miss_keys, miss_vecs))
            out = [resolved[k] if v is None else v for k, v in zip(keys, out)]

        # Must return exactly one vector per input, in order. Any leftover None
        # means the inner embedder returned too few vectors — fail loudly rather
        # than silently drop a vector and misalign embeddings against chunks.
        if any(v is None for v in out):
            missing = sum(1 for v in out if v is None)
            raise RuntimeError(
                f"CachingEmbedder resolved {len(out) - missing}/{len(out)} vectors — "
                "inner embedder returned too few"
            )
        return out  # type: ignore[return-value]

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None
