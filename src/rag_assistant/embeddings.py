"""Lazy-loaded sentence-transformers singleton (model: all-MiniLM-L6-v2, 384-dim)."""

# TODO: implement in step 3


def get_model():
    """Return the shared SentenceTransformer instance, loading it on first call."""
    raise NotImplementedError


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings and return a list of 384-dim float vectors."""
    raise NotImplementedError
