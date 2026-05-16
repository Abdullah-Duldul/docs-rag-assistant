"""Lazy-loaded sentence-transformers singleton (model: all-MiniLM-L6-v2, 384-dim)."""

from __future__ import annotations

_model = None  # SentenceTransformer | None; loaded on first call to _get_model()


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts and return 384-dim float vectors.

    Empty input returns empty output without loading the model.
    """
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return vectors.tolist()


def embedding_dimension() -> int:
    """Return the embedding vector dimension (all-MiniLM-L6-v2 = 384)."""
    return 384
