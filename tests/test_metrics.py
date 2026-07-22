"""Tests for the metrics helpers: token estimate, Timer, and cost math."""

import time

from app.metrics import RunMetrics, Timer, estimate_tokens


def test_estimate_tokens_scales_with_length():
    assert estimate_tokens("") == 1                     # floored at 1
    assert estimate_tokens("a" * 40) == 10              # ~4 chars/token


def test_timer_records_elapsed():
    with Timer() as t:
        time.sleep(0.01)
    assert t.seconds >= 0.01


def test_cost_and_summary_math():
    m = RunMetrics(
        documents=2, chunks_indexed=10, questions_produced=3,
        index_seconds=0.5, generate_seconds=1.5,
        embed_tokens_est=1_000_000, llm_tokens_est=1_000_000,
        embed_price_per_m=0.02, llm_price_per_m=0.60,
    )
    assert m.total_seconds == 2.0
    assert round(m.estimated_cost_usd, 4) == 0.62       # 0.02 + 0.60 per 1M each
    d = m.as_dict()
    assert d["estimated_cost_usd"] == 0.62
    assert "questions=3" in m.summary()
