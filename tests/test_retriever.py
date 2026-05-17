"""Tests for the retriever (embed query → top-k RetrievedChunks)."""

import pytest

from rag_assistant.retriever import EmptyStoreError, RetrievedChunk, retrieve


@pytest.mark.slow
def test_retrieve_returns_relevant_chunk(tmp_path):
    from rag_assistant.chunker import Chunk
    from rag_assistant.embeddings import embed
    from rag_assistant.store import ChunkStore

    chunks = [
        Chunk(
            text=(
                "To make pasta, boil salted water, add spaghetti, "
                "cook for 8-10 minutes, and serve with sauce."
            ),
            chunk_index=0,
            char_start=0,
            char_end=90,
            source_file="cooking.txt",
            page_number=None,
        ),
        Chunk(
            text=(
                "Python's asyncio module enables writing concurrent code "
                "using the async/await syntax and an event loop."
            ),
            chunk_index=0,
            char_start=0,
            char_end=95,
            source_file="programming.txt",
            page_number=None,
        ),
    ]

    store = ChunkStore(db_path=tmp_path / "chroma")
    embeddings = embed([c.text for c in chunks])
    store.add_chunks(chunks, embeddings)

    results = retrieve("how do I cook pasta?", top_k=1, store=store)

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, RetrievedChunk)
    assert result.source_file == "cooking.txt"
    assert result.text != ""
    assert isinstance(result.distance, float)
    assert result.page_number is None
    assert result.chunk_index == 0


@pytest.mark.slow
def test_retrieve_raises_on_empty_store(tmp_path):
    from rag_assistant.store import ChunkStore

    store = ChunkStore(db_path=tmp_path / "chroma")
    with pytest.raises(EmptyStoreError):
        retrieve("any question", store=store)
