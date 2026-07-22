"""Embedding cache: hits avoid re-embedding, dedupe within a call, and the
SQLite backing is durable across instances."""

from app.cache import CachingEmbedder


class CountingEmbedder:
    """Records how many texts it was actually asked to embed."""

    dim = 3

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts):
        self.calls += len(texts)
        # deterministic vector from text length so we can assert correctness
        return [[float(len(t)), 1.0, 0.0] for t in texts]


def test_second_pass_is_all_hits():
    inner = CountingEmbedder()
    cache = CachingEmbedder(inner)
    texts = ["alpha", "beta", "gamma"]

    v1 = cache.embed(texts)
    assert inner.calls == 3 and cache.misses == 3 and cache.hits == 0

    v2 = cache.embed(texts)          # identical input -> served entirely from cache
    assert inner.calls == 3          # inner was NOT called again
    assert cache.hits == 3
    assert v2 == v1                  # same vectors returned


def test_dedupes_repeats_within_one_call():
    inner = CountingEmbedder()
    cache = CachingEmbedder(inner)
    out = cache.embed(["x", "x", "x"])
    assert inner.calls == 1          # embedded the unique text once
    assert len(out) == 3            # but returns a vector per input position
    assert out[0] == out[1] == out[2]


def test_mixed_hits_and_misses_stay_aligned():
    inner = CountingEmbedder()
    cache = CachingEmbedder(inner)
    cache.embed(["seen"])                     # prime one entry
    assert inner.calls == 1

    # order: cached, new, cached-repeat, new — result must be per-input, in order
    out = cache.embed(["seen", "fresh", "seen", "brand new"])
    assert len(out) == 4                       # exactly one vector per input
    assert out[0] == out[2]                    # both "seen" positions match
    assert inner.calls == 1 + 2                # only "fresh" and "brand new" embedded
    # vectors are the length-encoded ones from CountingEmbedder
    assert out[1] == [float(len("fresh")), 1.0, 0.0]


def test_persists_across_instances(tmp_path):
    path = str(tmp_path / "emb.sqlite")
    inner1 = CountingEmbedder()
    c1 = CachingEmbedder(inner1, path=path)
    c1.embed(["durable one", "durable two"])
    assert inner1.calls == 2
    c1.close()

    # New instance, same file: no re-embedding needed.
    inner2 = CountingEmbedder()
    c2 = CachingEmbedder(inner2, path=path)
    c2.embed(["durable one", "durable two"])
    assert inner2.calls == 0
    assert c2.hits == 2
    assert c2.chars_hit > 0          # savings are tracked


def test_close_is_idempotent_and_safe(tmp_path):
    cache = CachingEmbedder(CountingEmbedder(), path=str(tmp_path / "e.sqlite"))
    cache.embed(["one"])
    cache.close()
    cache.close()                              # second close must not raise
    # cached vectors are still served from memory after the db is closed
    assert cache.embed(["one"]) == [[3.0, 1.0, 0.0]]


def test_different_backends_do_not_collide(tmp_path):
    path = str(tmp_path / "emb.sqlite")
    a = CachingEmbedder(CountingEmbedder(), path=path)
    a.embed(["shared text"])
    # a different model_id must not read the first backend's vector
    b = CachingEmbedder(CountingEmbedder(), path=path)
    b.model_id = "other-model:3"
    inner_b = b.inner
    b.embed(["shared text"])
    assert inner_b.calls == 1         # miss, because the key namespace differs
