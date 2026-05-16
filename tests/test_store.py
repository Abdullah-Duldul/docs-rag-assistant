"""Tests for ChunkStore (ChromaDB wrapper). All tests use tmp_path — never ./data/chroma/."""

import pytest

from rag_assistant.chunker import Chunk
from rag_assistant.store import ChunkStore


@pytest.fixture
def store(tmp_path) -> ChunkStore:
    return ChunkStore(db_path=tmp_path / "chroma")


def _make_chunks(n: int, source: str = "test.txt") -> tuple[list[Chunk], list[list[float]]]:
    """Synthetic chunks with deterministic fake 384-dim embeddings."""
    chunks = [
        Chunk(
            text=f"chunk {i}",
            chunk_index=i,
            char_start=i * 10,
            char_end=i * 10 + 9,
            source_file=source,
            page_number=None,
        )
        for i in range(n)
    ]
    embeddings = [[float(i)] * 384 for i in range(n)]
    return chunks, embeddings


def test_store_starts_empty(store):
    assert store.count() == 0
    assert store.list_sources() == []


def test_add_chunks_increments_count(store):
    chunks, embeds = _make_chunks(3)
    store.add_chunks(chunks, embeds)
    assert store.count() == 3


def test_add_chunks_rejects_length_mismatch(store):
    chunks, embeds = _make_chunks(3)
    with pytest.raises(ValueError, match="same length"):
        store.add_chunks(chunks, embeds[:2])


def test_upsert_updates_existing_chunks(store):
    chunks, embeds = _make_chunks(3)
    store.add_chunks(chunks, embeds)

    # mutate chunk 0's text, re-upsert with same ID
    chunks[0] = Chunk(
        text="updated text",
        chunk_index=0,
        char_start=0,
        char_end=12,
        source_file="test.txt",
        page_number=None,
    )
    store.add_chunks(chunks, embeds)

    # idempotent on count
    assert store.count() == 3
    # AND content actually updated (the whole point of upsert)
    result = store._collection.get(ids=["test.txt::0"], include=["documents"])
    assert result["documents"][0] == "updated text"


def test_list_sources_returns_unique_sorted(store):
    chunks_b, embeds_b = _make_chunks(2, source="b.txt")
    chunks_a, embeds_a = _make_chunks(2, source="a.txt")
    store.add_chunks(chunks_b, embeds_b)
    store.add_chunks(chunks_a, embeds_a)
    assert store.list_sources() == ["a.txt", "b.txt"]


def test_reset_clears_store(store):
    chunks, embeds = _make_chunks(3)
    store.add_chunks(chunks, embeds)
    assert store.count() == 3
    store.reset()
    assert store.count() == 0
    assert store.list_sources() == []
