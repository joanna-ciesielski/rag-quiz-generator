"""Retry-with-exponential-backoff for flaky network calls — no dependencies.

Real embedding/LLM APIs rate-limit and occasionally blip. Production code retries
transient failures with growing backoff instead of failing the first time. The
``sleep`` function is injectable so tests can exercise the retry logic without
actually waiting.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryError(RuntimeError):
    """Raised when all retry attempts are exhausted."""


# The OpenAI SDK exceptions worth retrying: rate limits, timeouts, connection
# blips, and 5xx. Auth/bad-request errors are deliberately excluded — retrying
# them just wastes time. Resolved by name so this tolerates SDK version drift
# (and test stubs that only define a subset).
_TRANSIENT_NAMES = (
    "RateLimitError",
    "APITimeoutError",
    "APIConnectionError",
    "InternalServerError",
)


def openai_transient_exceptions(openai_module) -> tuple[type[BaseException], ...]:
    """Return the retryable OpenAI exception classes present on ``openai_module``.

    Empty if none are present (e.g. a minimal test stub) — meaning "retry nothing",
    so a stubbed error surfaces immediately instead of sleeping through retries.
    """
    return tuple(
        getattr(openai_module, name)
        for name in _TRANSIENT_NAMES
        if isinstance(getattr(openai_module, name, None), type)
    )


def call_with_retries(
    fn: Callable[[], T],
    *,
    retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    exceptions: Iterable[type[BaseException]] = (Exception,),
    sleep: Callable[[float], None] = time.sleep,
    label: str = "operation",
) -> T:
    """Call ``fn`` up to ``retries + 1`` times, backing off exponentially.

    Delay before attempt *n* (1-indexed) is ``min(base_delay * 2**(n-1), max_delay)``.
    Only the listed ``exceptions`` trigger a retry; anything else propagates
    immediately (we don't want to retry a genuine bug or a validation error).
    Raises ``RetryError`` (chained to the last failure) once attempts run out.
    """
    exc_types = tuple(exceptions)
    last: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except exc_types as exc:  # noqa: PERF203 - retry loop, cost is negligible
            last = exc
            if attempt == retries:
                break
            delay = min(base_delay * (2 ** attempt), max_delay)
            logger.warning(
                "%s failed (attempt %d/%d): %s — retrying in %.2fs",
                label, attempt + 1, retries + 1, exc, delay,
            )
            sleep(delay)
    raise RetryError(f"{label} failed after {retries + 1} attempts") from last
