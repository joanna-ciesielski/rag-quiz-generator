"""Pipeline-level cache integration: a warm run reuses the durable cache (no
re-embedding) and the run reports the savings. Exercises run_with_metrics with a
cache_path end to end (the branch verified manually before, now under test)."""

from app.embeddings import HashingEmbedder
from app.pipeline import run_with_metrics

CORPUS = "eval/corpus/photosynthesis.md"


def test_cache_path_cold_then_warm(tmp_path):
    cache = str(tmp_path / "emb.sqlite")

    # Cold: nothing cached yet -> misses > 0, no savings.
    _, m_cold = run_with_metrics(
        [CORPUS], "photosynthesis",
        embedder=HashingEmbedder(), mock=True, cache_path=cache,
    )
    assert m_cold.extra["cache_misses"] >= 1
    assert m_cold.extra["cache_hits"] == 0
    assert m_cold.extra["embed_tokens_saved"] == 0

    # Warm: same content + same cache file -> all hits, zero re-embedding.
    _, m_warm = run_with_metrics(
        [CORPUS], "photosynthesis",
        embedder=HashingEmbedder(), mock=True, cache_path=cache,
    )
    assert m_warm.extra["cache_misses"] == 0
    assert m_warm.extra["cache_hits"] >= 1
    assert m_warm.extra["embed_tokens_saved"] > 0
    # cost reflects the savings: actual embed tokens drop to zero on the warm run
    assert m_warm.embed_tokens_est == 0
    assert m_warm.extra["embed_tokens_full_estimate"] > 0
