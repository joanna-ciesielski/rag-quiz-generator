"""Optional live smoke test against the REAL OpenAI API.

Skipped unless OPENAI_API_KEY is set AND RUN_LIVE=1, so it never runs in CI or
in normal offline test runs (it costs money and needs network). Run locally with:

    RUN_LIVE=1 OPENAI_API_KEY=sk-... python -m pytest tests/test_live_smoke.py -v

It exercises the real embedding + generation code paths end to end — the one
surface the deterministic offline suite deliberately mocks.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    not (os.environ.get("OPENAI_API_KEY") and os.environ.get("RUN_LIVE") == "1"),
    reason="live test: set RUN_LIVE=1 and OPENAI_API_KEY to run",
)


def test_real_embed_and_generate(tmp_path):
    from app.embeddings import OpenAIEmbedder
    from app.pipeline import run_with_metrics

    doc = tmp_path / "lesson.md"
    doc.write_text(
        "The water cycle moves water through evaporation, condensation, and "
        "precipitation. Evaporation turns liquid water into vapor; condensation "
        "forms clouds; precipitation returns water to the surface as rain or snow."
    )

    questions, metrics = run_with_metrics(
        [str(doc)], "the water cycle",
        embedder=OpenAIEmbedder(), num_questions=2, question_type="multiple_choice",
    )
    assert questions, "expected the real model to return at least one question"
    for q in questions:
        assert q.question.strip()
        assert q.answer in q.choices        # structured-output contract holds live
        assert q.source == "lesson.md"
    assert metrics.llm_tokens_est > 0
