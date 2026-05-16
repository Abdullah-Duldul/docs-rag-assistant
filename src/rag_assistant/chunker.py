"""Token-based text splitter: ~500-token chunks with 50-token overlap."""

# TODO: implement in step 2


def chunk_text(text: str, source_file: str, page_number: int | None = None) -> list[dict]:
    """Split text into overlapping chunks and return chunk metadata dicts.

    Each dict contains: text, source_file, chunk_index, page_number, char_start, char_end.
    """
    raise NotImplementedError
