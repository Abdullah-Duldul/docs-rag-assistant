"""Token-based text splitter: ~500-token chunks with 50-token overlap."""

from dataclasses import dataclass

import tiktoken

from rag_assistant.parsers import ParsedPage

_ENCODING = "cl100k_base"


@dataclass
class Chunk:
    text: str
    chunk_index: int  # 0-indexed, global across the whole source file
    char_start: int  # character offset into the page's source text
    char_end: int
    source_file: str
    page_number: int | None


def chunk_text(
    text: str,
    source_file: str,
    page_number: int | None = None,
    chunk_size: int = 500,
    overlap: int = 50,
    start_chunk_index: int = 0,
) -> list[Chunk]:
    """Split text into overlapping token-bounded chunks.

    char_start/char_end are offsets into the supplied text string.
    """
    if not text.strip():
        return []

    enc = tiktoken.get_encoding(_ENCODING)
    tokens = enc.encode(text)
    step = chunk_size - overlap
    chunks: list[Chunk] = []
    ptr = 0  # running search pointer into original text

    for idx, start in enumerate(range(0, len(tokens), step)):
        chunk_tokens = tokens[start : start + chunk_size]
        decoded = enc.decode(chunk_tokens)

        pos = text.find(decoded, ptr)
        if pos == -1:
            pos = ptr  # fallback for rare decode boundary mismatches

        char_start = pos
        char_end = pos + len(decoded)
        chunks.append(
            Chunk(
                text=decoded,
                chunk_index=start_chunk_index + idx,
                char_start=char_start,
                char_end=char_end,
                source_file=source_file,
                page_number=page_number,
            )
        )
        ptr = pos + 1  # advance by 1 to avoid re-matching on repeated text

    return chunks


def chunk_pages(
    pages: list[ParsedPage],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """Chunk all pages, keeping chunk_index globally sequential across pages."""
    all_chunks: list[Chunk] = []
    running_index = 0
    for page in pages:
        page_chunks = chunk_text(
            page.text,
            page.source_file,
            page.page_number,
            chunk_size=chunk_size,
            overlap=overlap,
            start_chunk_index=running_index,
        )
        all_chunks.extend(page_chunks)
        running_index += len(page_chunks)
    return all_chunks
