"""Quiz-question generation over retrieved context, with structured output.

The LLM is asked to return strict JSON (validated by pydantic), so questions,
choices, and answers are machine-checkable rather than parsed from free text.
A `mock=True` mode returns deterministic questions with no API call, so the
whole pipeline runs offline and in tests.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from app.vectorstore import Retrieved, VectorStore

logger = logging.getLogger(__name__)

QuestionType = Literal["multiple_choice", "open_ended"]


class QuizQuestion(BaseModel):
    question: str
    type: QuestionType
    choices: list[str] = Field(default_factory=list)
    answer: str | None = None
    source: str | None = None

    @model_validator(mode="after")
    def _validate_multiple_choice(self) -> "QuizQuestion":
        """A multiple-choice question must have distinct choices AND its answer
        must be one of them — otherwise the question is unusable."""
        if self.type == "multiple_choice":
            if len(self.choices) < 2:
                raise ValueError("multiple_choice requires at least 2 choices")
            if len(set(self.choices)) != len(self.choices):
                raise ValueError("multiple_choice choices must be unique")
            if self.answer is None or self.answer not in self.choices:
                raise ValueError("multiple_choice answer must be one of the choices")
        return self


class GenerationError(RuntimeError):
    """Raised when the LLM call fails or returns unusable output."""


DEFAULT_MODEL = os.environ.get("QUIZ_MODEL", "gpt-4o-mini")

_SYSTEM = (
    "You write quiz questions strictly from the provided context. "
    "Return ONLY valid JSON, no prose. For multiple_choice, provide exactly 4 "
    "choices and set answer to the correct choice text. Never invent facts beyond "
    "the context."
)


def _prompt(context: str, qtype: QuestionType, topic: str) -> str:
    shape = (
        '{"question": "...", "type": "multiple_choice", "choices": ["...","...","...","..."], "answer": "..."}'
        if qtype == "multiple_choice"
        else '{"question": "...", "type": "open_ended", "answer": "..."}'
    )
    return (
        f"Topic: {topic}\n\nContext:\n{context}\n\n"
        f"Write one {qtype} question answerable purely from the context. "
        f"Respond as JSON exactly matching: {shape}"
    )


def _mock_question(context: str, qtype: QuestionType) -> QuizQuestion:
    head = context.strip().split(".")[0][:80] or "the material"
    if qtype == "multiple_choice":
        return QuizQuestion(
            question=f"Which statement is supported by: '{head}'?",
            type="multiple_choice",
            choices=["It is supported", "It is contradicted", "Not mentioned", "None"],
            answer="It is supported",
        )
    return QuizQuestion(
        question=f"In your own words, explain: '{head}'.",
        type="open_ended",
        answer="A correct summary of the referenced context.",
    )


def _call_llm(context: str, qtype: QuestionType, topic: str, model: str) -> QuizQuestion:
    import openai

    from app.retry import RetryError, call_with_retries, openai_transient_exceptions

    client = openai.OpenAI()
    transient = openai_transient_exceptions(openai)
    max_retries = int(os.environ.get("LLM_MAX_RETRIES", "3"))
    base_delay = float(os.environ.get("RETRY_BASE_DELAY", "0.5"))
    try:
        resp = call_with_retries(
            lambda: client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": _prompt(context, qtype, topic)},
                ],
                temperature=0.5,
                max_tokens=400,
            ),
            retries=max_retries,
            base_delay=base_delay,
            exceptions=transient,
            label="chat.completions.create",
        )
    except (openai.APIError, RetryError) as exc:  # non-transient, or retries exhausted
        raise GenerationError(f"LLM call failed: {exc}") from exc
    raw = resp.choices[0].message.content or ""
    try:
        return QuizQuestion(**json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise GenerationError(f"Model returned invalid question JSON: {exc}") from exc


def _generate_one(ctx: Retrieved, question_type: QuestionType, topic: str,
                  model: str, mock: bool) -> QuizQuestion | None:
    """Generate a single question for one context, or None if it failed/was invalid."""
    try:
        q = (
            _mock_question(ctx.text, question_type)
            if mock
            else _call_llm(ctx.text, question_type, topic, model)
        )
    except GenerationError as exc:
        # Skip a single bad/invalid generation rather than failing the whole quiz.
        logger.warning("Skipping a question for source %s: %s", ctx.source, exc)
        return None
    q.source = ctx.source
    return q


def generate_quiz(
    store: VectorStore,
    topic: str,
    *,
    namespace: str = "default",
    num_questions: int = 5,
    question_type: QuestionType = "multiple_choice",
    mock: bool = False,
    model: str = DEFAULT_MODEL,
    retrieve_fn=None,
    concurrency: int = 1,
) -> list[QuizQuestion]:
    """Retrieve context for the topic, then generate deduplicated questions.

    ``retrieve_fn(query, k) -> list[Retrieved]`` lets generation use any
    retriever (e.g. the hybrid one); defaults to plain dense store retrieval.
    ``concurrency > 1`` generates questions in parallel (bounded thread pool),
    cutting wall-clock time; results are still ordered, deduplicated, and capped
    deterministically, and per-question failures are still skipped.
    """
    k = max(num_questions, 4)
    contexts: list[Retrieved] = (
        retrieve_fn(topic, k) if retrieve_fn is not None
        else store.query(topic, namespace=namespace, k=k)
    )
    if not contexts:
        return []

    if concurrency > 1 and len(contexts) > 1:
        results = _generate_concurrent(contexts, question_type, topic, model, mock, concurrency)
    else:
        results = (
            _generate_one(ctx, question_type, topic, model, mock) for ctx in contexts
        )

    questions: list[QuizQuestion] = []
    seen: set[str] = set()
    for q in results:
        if len(questions) >= num_questions:
            break
        if q is None:
            continue
        key = q.question.strip().lower()
        if key and key not in seen:
            seen.add(key)
            questions.append(q)
    return questions


def _generate_concurrent(contexts: list[Retrieved], question_type: QuestionType, topic: str,
                         model: str, mock: bool, concurrency: int) -> list[QuizQuestion | None]:
    """Run per-context generation in a bounded pool, returning results in the
    SAME order as ``contexts`` so dedup/capping stay deterministic."""
    from concurrent.futures import ThreadPoolExecutor

    workers = min(concurrency, len(contexts))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(
            lambda ctx: _generate_one(ctx, question_type, topic, model, mock), contexts
        ))
