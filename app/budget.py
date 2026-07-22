"""Per-tenant token budgets — a guardrail so one namespace can't run up an
unbounded bill. Uses the same token estimates as the metrics module; a run that
would exceed its cap fails fast with ``BudgetExceeded`` *before* spending, rather
than discovering the overage after the API calls are already paid for.
"""

from __future__ import annotations


class BudgetExceeded(RuntimeError):
    """Raised when a run's estimated token use exceeds its budget."""

    def __init__(self, requested: int, spent: int, limit: int, label: str) -> None:
        self.requested = requested
        self.spent = spent
        self.limit = limit
        self.label = label
        super().__init__(
            f"token budget exceeded at '{label}': {spent}+{requested} > {limit}"
        )


class TokenBudget:
    """A simple accumulating token budget. ``None`` limit means unlimited."""

    def __init__(self, limit: int | None) -> None:
        self.limit = limit
        self.spent = 0

    @property
    def remaining(self) -> int | None:
        return None if self.limit is None else max(0, self.limit - self.spent)

    def charge(self, tokens: int, *, label: str = "tokens") -> int:
        """Reserve ``tokens`` against the budget. Raises ``BudgetExceeded`` (and
        charges nothing) if this would push past the limit."""
        if self.limit is not None and self.spent + tokens > self.limit:
            raise BudgetExceeded(tokens, self.spent, self.limit, label)
        self.spent += tokens
        return self.spent
