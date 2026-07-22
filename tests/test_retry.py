"""Tests for the retry/backoff helper — no real sleeping, injected clock."""

import pytest

from app.retry import RetryError, call_with_retries


def test_returns_immediately_on_success():
    calls = []
    out = call_with_retries(lambda: calls.append(1) or "ok", sleep=lambda _: None)
    assert out == "ok"
    assert len(calls) == 1                      # no retries when it works first time


def test_retries_then_succeeds():
    attempts = {"n": 0}
    delays = []

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ValueError("transient")
        return "recovered"

    out = call_with_retries(
        flaky, retries=3, base_delay=1.0, exceptions=(ValueError,), sleep=delays.append
    )
    assert out == "recovered"
    assert attempts["n"] == 3
    assert delays == [1.0, 2.0]                 # exponential backoff before attempts 2 and 3


def test_exhausts_and_raises_retry_error_chained():
    def always_fails():
        raise ValueError("nope")

    with pytest.raises(RetryError) as ei:
        call_with_retries(always_fails, retries=2, exceptions=(ValueError,), sleep=lambda _: None)
    assert isinstance(ei.value.__cause__, ValueError)   # preserves the underlying error


def test_unlisted_exception_is_not_retried():
    attempts = {"n": 0}

    def bug():
        attempts["n"] += 1
        raise KeyError("a real bug, not transient")

    with pytest.raises(KeyError):
        call_with_retries(bug, retries=5, exceptions=(ValueError,), sleep=lambda _: None)
    assert attempts["n"] == 1                   # propagated immediately, no retry


def test_backoff_is_capped_at_max_delay():
    delays = []

    def always_fails():
        raise ValueError("x")

    with pytest.raises(RetryError):
        call_with_retries(
            always_fails, retries=5, base_delay=1.0, max_delay=3.0,
            exceptions=(ValueError,), sleep=delays.append,
        )
    assert max(delays) <= 3.0                   # 1,2,3,3,3 — never exceeds the cap
