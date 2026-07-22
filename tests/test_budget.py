"""Token budget: accumulates, rejects overspend before charging, and the
pipeline enforces it pre-flight (before paying to embed)."""

import pytest

from app.budget import BudgetExceeded, TokenBudget
from app.embeddings import HashingEmbedder
from app.pipeline import run_with_metrics


def test_unlimited_when_none():
    b = TokenBudget(None)
    b.charge(1_000_000)
    assert b.remaining is None
    assert b.spent == 1_000_000


def test_charges_accumulate_within_limit():
    b = TokenBudget(100)
    b.charge(40)
    b.charge(30)
    assert b.spent == 70
    assert b.remaining == 30


def test_overspend_raises_and_does_not_charge():
    b = TokenBudget(50)
    b.charge(40)
    with pytest.raises(BudgetExceeded):
        b.charge(20)               # 40 + 20 > 50
    assert b.spent == 40           # rejected charge left the balance untouched


def test_pipeline_rejects_before_embedding(tmp_path):
    corpus = tmp_path / "doc.md"
    corpus.write_text("The water cycle moves water through evaporation and condensation. " * 40)
    with pytest.raises(BudgetExceeded):
        run_with_metrics(
            [str(corpus)], "water cycle",
            embedder=HashingEmbedder(), mock=True, token_budget=1,   # absurdly low -> reject
        )


def test_pipeline_allows_within_budget(tmp_path):
    corpus = tmp_path / "doc.md"
    corpus.write_text("Photosynthesis turns sunlight and carbon dioxide into glucose and oxygen.")
    questions, metrics = run_with_metrics(
        [str(corpus)], "photosynthesis",
        embedder=HashingEmbedder(), mock=True, token_budget=100_000,
    )
    assert metrics.extra["budget_spent"] > 0
    assert metrics.extra["budget_remaining"] >= 0


def test_pipeline_budget_also_guards_generation(tmp_path):
    """A budget large enough to embed but not to generate must still be rejected —
    the LLM step is usually the bigger cost, so it is guarded too."""
    corpus = tmp_path / "doc.md"
    corpus.write_text("Photosynthesis turns sunlight and carbon dioxide into glucose and oxygen.")
    # embedding this tiny doc is ~10-20 tokens; generation pre-charge for 5
    # questions is ~1000+. A budget of 50 clears embeddings but not generation.
    with pytest.raises(BudgetExceeded) as ei:
        run_with_metrics(
            [str(corpus)], "photosynthesis",
            embedder=HashingEmbedder(), mock=True, num_questions=5, token_budget=50,
        )
    assert ei.value.label == "generation"
