"""Tests for the token-based text chunker."""

from rag_assistant.chunker import chunk_pages, chunk_text
from rag_assistant.parsers import ParsedPage

# ~2000 tokens of text (repeat a paragraph many times)
_LONG_PARA = (
    "The quick brown fox jumps over the lazy dog near the riverbank. "
    "Scientists have discovered that regular exercise improves cognitive function "
    "and reduces the risk of chronic diseases. "
)
LONG_TEXT = _LONG_PARA * 60  # ~2000 tokens


def test_empty_text_returns_no_chunks():
    assert chunk_text("", "test.txt") == []
    assert chunk_text("   \n  ", "test.txt") == []


def test_short_text_returns_single_chunk():
    chunks = chunk_text("Hello world, this is a short sentence.", "test.txt")
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].source_file == "test.txt"
    assert chunks[0].page_number is None


def test_long_text_returns_multiple_chunks():
    chunks = chunk_text(LONG_TEXT, "doc.txt", chunk_size=100, overlap=10)
    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i
    assert chunks[-1].chunk_index == len(chunks) - 1


def test_chunk_overlap_is_correct():
    chunks = chunk_text(LONG_TEXT, "doc.txt", chunk_size=100, overlap=20)
    assert len(chunks) >= 3
    for i in range(len(chunks) - 1):
        tail = chunks[i].text[-200:]
        head = chunks[i + 1].text[:200]
        overlap_found = any(tail[j : j + 30] in head for j in range(0, len(tail) - 30))
        assert overlap_found, f"No overlap detected between chunk {i} and chunk {i + 1}"


def test_chunk_offsets_round_trip():
    text = LONG_TEXT
    chunks = chunk_text(text, "doc.txt", chunk_size=100, overlap=10)
    for chunk in chunks:
        # decoded text should appear in a slightly widened slice (±10 chars)
        slack = 10
        start = max(0, chunk.char_start - slack)
        end = min(len(text), chunk.char_end + slack)
        assert chunk.text in text[start:end], (
            f"chunk {chunk.chunk_index} text not found near offsets "
            f"[{chunk.char_start}:{chunk.char_end}]"
        )


def test_chunk_pages_increments_indices_across_pages():
    page1 = ParsedPage(text=LONG_TEXT[:500], page_number=1, source_file="doc.pdf")
    page2 = ParsedPage(text=LONG_TEXT[500:1000], page_number=2, source_file="doc.pdf")
    chunks = chunk_pages([page1, page2], chunk_size=50, overlap=5)
    assert len(chunks) > 2
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i, f"chunk_index gap at position {i}"


def test_chunk_pages_preserves_page_numbers():
    page1 = ParsedPage(text=LONG_TEXT[:500], page_number=1, source_file="doc.pdf")
    page2 = ParsedPage(text=LONG_TEXT[500:1000], page_number=2, source_file="doc.pdf")
    chunks = chunk_pages([page1, page2], chunk_size=50, overlap=5)
    page1_chunks = [c for c in chunks if c.page_number == 1]
    page2_chunks = [c for c in chunks if c.page_number == 2]
    assert len(page1_chunks) >= 1
    assert len(page2_chunks) >= 1
    # all chunks belong to one page or the other
    assert len(page1_chunks) + len(page2_chunks) == len(chunks)
