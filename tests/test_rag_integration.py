"""Integration test: embed → store → query returns topically relevant results."""

import pytest


@pytest.mark.slow
def test_retrieves_topically_relevant_chunks(tmp_path):
    """Three documents on three distinct topics. Each query must retrieve
    the correct source — proving semantic search works end-to-end."""
    from rag_assistant.chunker import Chunk
    from rag_assistant.embeddings import embed
    from rag_assistant.store import ChunkStore

    chunks = [
        Chunk(
            text=(
                "Python's asyncio library enables concurrent execution of "
                "I/O-bound tasks using coroutines and event loops."
            ),
            chunk_index=0,
            char_start=0,
            char_end=100,
            source_file="programming.txt",
            page_number=None,
        ),
        Chunk(
            text=(
                "To make sourdough bread, mix flour and water, let the "
                "dough autolyse, then add starter and salt before bulk fermentation."
            ),
            chunk_index=0,
            char_start=0,
            char_end=130,
            source_file="cooking.txt",
            page_number=None,
        ),
        Chunk(
            text=(
                "Tomato plants need at least six hours of direct sunlight, "
                "consistent watering, and well-drained soil to thrive."
            ),
            chunk_index=0,
            char_start=0,
            char_end=110,
            source_file="gardening.txt",
            page_number=None,
        ),
    ]

    store = ChunkStore(db_path=tmp_path / "chroma")
    embeddings = embed([c.text for c in chunks])
    store.add_chunks(chunks, embeddings)

    # cooking query
    query_vec = embed(["how do I bake bread at home?"])[0]
    results = store.query(query_vec, top_k=1)
    assert len(results) == 1
    assert results[0]["source_file"] == "cooking.txt", (
        f"Expected cooking.txt to rank highest; got {results[0]['source_file']}"
    )

    # gardening query
    query_vec = embed(["what conditions do vegetables need to grow?"])[0]
    results = store.query(query_vec, top_k=1)
    assert results[0]["source_file"] == "gardening.txt", (
        f"Expected gardening.txt to rank highest; got {results[0]['source_file']}"
    )

    # programming query
    query_vec = embed(["concurrent programming patterns"])[0]
    results = store.query(query_vec, top_k=1)
    assert results[0]["source_file"] == "programming.txt", (
        f"Expected programming.txt to rank highest; got {results[0]['source_file']}"
    )
