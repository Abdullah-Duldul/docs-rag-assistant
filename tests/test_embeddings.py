"""Tests for the lazy-loaded embeddings module."""

import pytest


def test_embed_empty_returns_empty():
    from rag_assistant.embeddings import embed

    assert embed([]) == []


def test_embedding_dimension_constant():
    from rag_assistant.embeddings import embedding_dimension

    assert embedding_dimension() == 384


@pytest.mark.slow
def test_embed_returns_correct_shape():
    from rag_assistant.embeddings import embed

    vecs = embed(["hello", "world", "foo"])
    assert len(vecs) == 3
    assert all(len(v) == 384 for v in vecs)


@pytest.mark.slow
def test_embed_is_deterministic():
    from rag_assistant.embeddings import embed

    text = "The quick brown fox jumps over the lazy dog"
    assert embed([text]) == embed([text])
