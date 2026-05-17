"""Tests for the RAG system prompt and user message builder."""

from rag_assistant.prompts import RAG_SYSTEM_PROMPT, build_user_message
from rag_assistant.retriever import RetrievedChunk

_FALLBACK = "I don't have information about that in the indexed documents."


def test_system_prompt_contains_citation_rule():
    assert "[1]" in RAG_SYSTEM_PROMPT


def test_system_prompt_contains_antihallucination_fallback():
    assert _FALLBACK in RAG_SYSTEM_PROMPT


def test_system_prompt_contains_spartan_tone_directive():
    assert "Spartan" in RAG_SYSTEM_PROMPT


def test_system_prompt_contains_worked_examples():
    assert "Good answer:" in RAG_SYSTEM_PROMPT
    assert "Bad answer" in RAG_SYSTEM_PROMPT


def test_build_user_message_structure():
    chunks = [
        RetrievedChunk(
            text="The REST API rate limit is 60 req/min.",
            source_file="api_guide.pdf",
            page_number=3,
            chunk_index=7,
            distance=0.1,
        ),
        RetrievedChunk(
            text="Rate limits apply at the account level.",
            source_file="faq.md",
            page_number=None,
            chunk_index=2,
            distance=0.2,
        ),
    ]
    question = "What is the rate limit?"
    msg = build_user_message(question, chunks)

    assert "Source [1]:" in msg
    assert "Source [2]:" in msg
    assert f"Question: {question}" in msg
    assert chunks[0].text in msg
    assert chunks[1].text in msg
    # page number shown for chunk with a page
    assert "page 3" in msg
    # page info omitted for chunk without a page
    assert "page None" not in msg
