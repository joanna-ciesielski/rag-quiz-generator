"""Lightweight run observability: timings, counts, and a token/cost estimate.

No external tracing dependency — just enough to answer "how long did each stage
take, how much work did it do, and roughly what did it cost?" for a single run.
Cost is a transparent estimate from a char/4 token heuristic and configurable
per-1M-token prices; it is a planning figure, not a billing record.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 chars/token). Good enough for a cost estimate without
    pulling in a tokenizer; real usage numbers would come from API responses."""
    return max(1, len(text) // 4)


@dataclass
class Timer:
    """Context manager that records wall-clock seconds into ``.seconds``."""

    seconds: float = 0.0
    _start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self._start


@dataclass
class RunMetrics:
    """Counts, stage timings, and an estimated cost for one pipeline run."""

    documents: int = 0
    chunks_indexed: int = 0
    retrieval_calls: int = 0
    generation_calls: int = 0
    questions_produced: int = 0
    questions_skipped: int = 0
    index_seconds: float = 0.0
    retrieve_seconds: float = 0.0
    generate_seconds: float = 0.0
    embed_tokens_est: int = 0
    llm_tokens_est: int = 0
    # default prices are illustrative (USD per 1M tokens); override via config.
    embed_price_per_m: float = 0.02
    llm_price_per_m: float = 0.60
    extra: dict = field(default_factory=dict)

    @property
    def total_seconds(self) -> float:
        return self.index_seconds + self.retrieve_seconds + self.generate_seconds

    @property
    def estimated_cost_usd(self) -> float:
        return (
            self.embed_tokens_est / 1_000_000 * self.embed_price_per_m
            + self.llm_tokens_est / 1_000_000 * self.llm_price_per_m
        )

    def as_dict(self) -> dict:
        d = asdict(self)
        d["total_seconds"] = round(self.total_seconds, 4)
        d["estimated_cost_usd"] = round(self.estimated_cost_usd, 6)
        return d

    def summary(self) -> str:
        return (
            f"docs={self.documents} chunks={self.chunks_indexed} "
            f"questions={self.questions_produced} (skipped {self.questions_skipped}) | "
            f"index={self.index_seconds:.3f}s retrieve={self.retrieve_seconds:.3f}s "
            f"generate={self.generate_seconds:.3f}s total={self.total_seconds:.3f}s | "
            f"~{self.embed_tokens_est + self.llm_tokens_est} tokens "
            f"~${self.estimated_cost_usd:.4f}"
        )
