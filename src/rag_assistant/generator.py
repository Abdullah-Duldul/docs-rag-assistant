"""Generate a cited answer from retrieved chunks using Claude (haiku-4-5)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import anthropic
from dotenv import load_dotenv

from rag_assistant.prompts import RAG_SYSTEM_PROMPT, build_user_message

if TYPE_CHECKING:
    from rag_assistant.retriever import RetrievedChunk

_MODEL = "claude-haiku-4-5-20251001"
_MAX_TOKENS = 1024


@dataclass
class AnswerResult:
    answer_text: str
    sources: list[RetrievedChunk]
    input_tokens: int
    output_tokens: int


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    client: anthropic.Anthropic | None = None,
) -> AnswerResult:
    """Send retrieved chunks + question to Claude and return a cited answer.

    Raises ValueError if ANTHROPIC_API_KEY is not set.
    client defaults to anthropic.Anthropic() when None (reads key from env).
    """
    if client is None:
        load_dotenv()
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file."
            )
        client = anthropic.Anthropic(api_key=api_key)

    user_message = build_user_message(question, chunks)

    response = client.messages.create(
        model=_MODEL,
        max_tokens=_MAX_TOKENS,
        system=RAG_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    return AnswerResult(
        answer_text=response.content[0].text,
        sources=chunks,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
