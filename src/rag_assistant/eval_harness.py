"""Evaluation harness: retrieval hit-rate and answer accuracy against a golden question set."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from rag_assistant.generator import generate_answer
from rag_assistant.retriever import EmptyStoreError, RetrievedChunk, retrieve
from rag_assistant.store import ChunkStore

if TYPE_CHECKING:
    import anthropic

FALLBACK_RESPONSE = "I don't have information about that in the indexed documents."


@dataclass
class GoldenQuestion:
    question: str
    expected_substring: str | None = None
    expected_source_file: str | None = None
    expected_fallback: bool = False
    notes: str = ""


@dataclass
class RetrievalResult:
    hit: bool  # source_hit AND text_hit
    top_k_sources: list[str]
    top_k_texts: list[str]
    chunks: list[RetrievedChunk]


@dataclass
class AnswerEvalResult:
    hit: bool
    answer_text: str
    was_fallback: bool
    input_tokens: int
    output_tokens: int


@dataclass
class QuestionResult:
    question: GoldenQuestion
    retrieval: RetrievalResult | None  # None for unanswerable questions
    answer: AnswerEvalResult


@dataclass
class EvalReport:
    retrieval_hit_rate: float
    answer_hit_rate: float
    fallback_rate: float
    per_question: list[QuestionResult]
    total_input_tokens: int
    total_output_tokens: int


def load_golden_set(path: Path) -> list[GoldenQuestion]:
    """Load golden questions from a YAML file."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [
        GoldenQuestion(
            question=q["question"],
            expected_substring=q.get("expected_substring"),
            expected_source_file=q.get("expected_source_file"),
            expected_fallback=q.get("expected_fallback", False),
            notes=q.get("notes", ""),
        )
        for q in data["questions"]
    ]


def evaluate_retrieval(question: GoldenQuestion, store: ChunkStore) -> RetrievalResult:
    """Retrieve top-k chunks; hit = expected_source_file present AND expected_substring found.

    Both conditions must hold for a retrieval HIT (conjunction, not OR).
    source comparison uses Path(...).name so full paths don't affect matching.
    """
    chunks = retrieve(question.question, store=store)
    sources = [Path(c.source_file).name for c in chunks]
    texts = [c.text for c in chunks]

    source_hit = question.expected_source_file in sources
    text_hit = any(
        (question.expected_substring or "").lower() in text.lower() for text in texts
    )
    return RetrievalResult(
        hit=source_hit and text_hit,
        top_k_sources=sources,
        top_k_texts=texts,
        chunks=chunks,
    )


def evaluate_answer(
    question: GoldenQuestion,
    retrieved_chunks: list[RetrievedChunk],
    client: anthropic.Anthropic | None = None,
) -> AnswerEvalResult:
    """Generate an answer and evaluate it against the golden criterion."""
    result = generate_answer(question.question, retrieved_chunks, client=client)
    was_fallback = FALLBACK_RESPONSE.lower() in result.answer_text.lower()

    if question.expected_fallback:
        hit = was_fallback
    else:
        hit = (
            (question.expected_substring or "").lower() in result.answer_text.lower()
            and not was_fallback
        )

    return AnswerEvalResult(
        hit=hit,
        answer_text=result.answer_text,
        was_fallback=was_fallback,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


def run_full_eval(
    golden_set: list[GoldenQuestion],
    store: ChunkStore,
    client: anthropic.Anthropic | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> EvalReport:
    """Orchestrate retrieval + answer evaluation for every golden question."""
    per_question: list[QuestionResult] = []
    total_input = 0
    total_output = 0
    n = len(golden_set)

    for i, question in enumerate(golden_set, start=1):
        if progress_callback:
            progress_callback(f"Question {i}/{n}: {question.question[:60]}...")

        if question.expected_fallback:
            # Still retrieve — tests that Claude doesn't hallucinate against irrelevant context
            try:
                chunks = retrieve(question.question, store=store)
            except EmptyStoreError:
                chunks = []
            retrieval = None
        else:
            retrieval = evaluate_retrieval(question, store)
            chunks = retrieval.chunks

        answer = evaluate_answer(question, chunks, client)
        total_input += answer.input_tokens
        total_output += answer.output_tokens
        per_question.append(QuestionResult(question=question, retrieval=retrieval, answer=answer))

    answerable = [r for r in per_question if not r.question.expected_fallback]
    unanswerable = [r for r in per_question if r.question.expected_fallback]

    retrieval_hit_rate = (
        sum(1 for r in answerable if r.retrieval and r.retrieval.hit) / len(answerable)
        if answerable
        else 0.0
    )
    answer_hit_rate = (
        sum(1 for r in answerable if r.answer.hit) / len(answerable) if answerable else 0.0
    )
    fallback_rate = (
        sum(1 for r in unanswerable if r.answer.was_fallback) / len(unanswerable)
        if unanswerable
        else 0.0
    )

    return EvalReport(
        retrieval_hit_rate=retrieval_hit_rate,
        answer_hit_rate=answer_hit_rate,
        fallback_rate=fallback_rate,
        per_question=per_question,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
    )
