"""ChromaDB wrapper: persistent collection 'documents' stored at ./data/chroma/."""

# TODO: implement in step 3


def add_chunks(chunks: list[dict]) -> None:
    """Embed and upsert a list of chunk dicts into the ChromaDB collection."""
    raise NotImplementedError


def query(embedding: list[float], top_k: int = 5) -> list[dict]:
    """Return the top-k most similar chunks for the given query embedding."""
    raise NotImplementedError


def count() -> int:
    """Return total number of chunks currently stored."""
    raise NotImplementedError


def list_sources() -> list[str]:
    """Return deduplicated list of source_file values in the collection."""
    raise NotImplementedError


def reset() -> None:
    """Delete and recreate the ChromaDB collection, wiping all data."""
    raise NotImplementedError
