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

    client = openai.OpenAI()
    try:
        resp = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _prompt(context, qtype, topic)},
            ],
            temperature=0.5,
            max_tokens=400,
        )
    except openai.APIError as exc:  # auth, rate limit, bad model, network
        raise GenerationError(f"LLM call failed: {exc}") from exc
    raw = resp.choices[0].message.content or ""
    try:
        return QuizQuestion(**json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise GenerationError(f"Model returned invalid question JSON: {exc}") from exc


def generate_quiz(
    store: VectorStore,
    topic: str,
    *,
    namespace: str = "default",
    num_questions: int = 5,
    question_type: QuestionType = "multiple_choice",
    mock: bool = False,
    model: str = DEFAULT_MODEL,
) -> list[QuizQuestion]:
    """Retrieve context for the topic, then generate deduplicated questions."""
    contexts: list[Retrieved] = store.query(topic, namespace=namespace, k=max(num_questions, 4))
    if not contexts:
        return []

    questions: list[QuizQuestion] = []
    seen: set[str] = set()
    for ctx in contexts:
        if len(questions) >= num_questions:
            break
        try:
            q = (
                _mock_question(ctx.text, question_type)
                if mock
                else _call_llm(ctx.text, question_type, topic, model)
            )
        except GenerationError as exc:
            # Skip a single bad/invalid generation rather than failing the whole
            # quiz; keep going so one malformed response doesn't lose the batch.
            logger.warning("Skipping a question for source %s: %s", ctx.source, exc)
            continue
        q.source = ctx.source
        key = q.question.strip().lower()
        if key and key not in seen:
            seen.add(key)
            questions.append(q)
    return questions
