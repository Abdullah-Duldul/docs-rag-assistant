"""Tests for the answer generator. Mocks the Anthropic client — no network calls."""

import pytest

from rag_assistant.generator import AnswerResult, generate_answer
from rag_assistant.prompts import RAG_SYSTEM_PROMPT, build_user_message
from rag_assistant.retriever import RetrievedChunk


class _FakeUsage:
    input_tokens = 100
    output_tokens = 42


class _FakeContent:
    text = "The answer is 42 [1]."


class _FakeMessage:
    content = [_FakeContent()]
    usage = _FakeUsage()


class _FakeMessages:
    def __init__(self):
        self.last_kwargs: dict = {}

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeMessage()


class _FakeClient:
    def __init__(self):
        self.messages = _FakeMessages()


def _make_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        text="The answer is found in document section 4.2.",
        source_file="reference.pdf",
        page_number=5,
        chunk_index=3,
        distance=0.05,
    )


@pytest.mark.slow
def test_generate_answer_returns_answer_result():
    client = _FakeClient()
    chunk = _make_chunk()
    result = generate_answer("What is the answer?", [chunk], client=client)

    assert isinstance(result, AnswerResult)
    assert result.answer_text == "The answer is 42 [1]."
    assert result.input_tokens == 100
    assert result.output_tokens == 42
    assert result.sources == [chunk]


@pytest.mark.slow
def test_generate_passes_correct_prompts():
    client = _FakeClient()
    chunk = _make_chunk()
    question = "What is the answer?"
    generate_answer(question, [chunk], client=client)

    kwargs = client.messages.last_kwargs
    assert kwargs["system"] == RAG_SYSTEM_PROMPT
    assert kwargs["messages"][0]["content"] == build_user_message(question, [chunk])
    assert kwargs["messages"][0]["role"] == "user"
