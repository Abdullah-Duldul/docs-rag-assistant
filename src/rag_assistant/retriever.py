"""Embed a query and retrieve the top-k matching chunks with metadata."""

from dataclasses import dataclass

from rag_assistant.embeddings import embed
from rag_assistant.store import DEFAULT_DB_PATH, ChunkStore


class EmptyStoreError(Exception):
    """Raised when the vector store has no indexed chunks to query."""


@dataclass
class RetrievedChunk:
    text: str
    source_file: str
    page_number: int | None
    chunk_index: int
    distance: float  # cosine distance: 0 = identical, 2 = opposite


def retrieve(
    question: str,
    top_k: int = 5,
    store: ChunkStore | None = None,
) -> list[RetrievedChunk]:
    """Embed question, query the store, and return the top-k RetrievedChunks.

    Raises EmptyStoreError if the store has no indexed documents.
    store defaults to ChunkStore(DEFAULT_DB_PATH) when None.
    """
    if store is None:
        store = ChunkStore(db_path=DEFAULT_DB_PATH)

    if store.count() == 0:
        raise EmptyStoreError(
            "No documents indexed. Run `rag ingest <path>` first."
        )

    query_embedding = embed([question])[0]
    results = store.query(query_embedding, top_k=top_k)

    return [
        RetrievedChunk(
            text=r["text"],
            source_file=r["source_file"],
            page_number=r["page_number"],
            chunk_index=r["chunk_index"],
            distance=r["distance"],
        )
        for r in results
    ]
