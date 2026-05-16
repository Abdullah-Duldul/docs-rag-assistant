"""ChromaDB persistent store: single 'documents' collection at ./data/chroma/."""

from pathlib import Path

import chromadb

from rag_assistant.chunker import Chunk

DEFAULT_DB_PATH = Path("./data/chroma")
COLLECTION_NAME = "documents"

_PAGE_NUMBER_NONE_SENTINEL = -1


class ChunkStore:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        """Open or create the persistent ChromaDB collection at db_path."""
        self._db_path = db_path
        db_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(db_path))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Upsert chunks with their embeddings.

        IDs are deterministic ({source_file}::{chunk_index}) so re-ingesting
        the same file overwrites existing entries without creating duplicates.
        Raises ValueError if len(chunks) != len(embeddings).
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks and embeddings must have the same length "
                f"(got {len(chunks)} chunks and {len(embeddings)} embeddings)"
            )
        if not chunks:
            return

        ids = [f"{c.source_file}::{c.chunk_index}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "source_file": c.source_file,
                "chunk_index": c.chunk_index,
                "char_start": c.char_start,
                "char_end": c.char_end,
                "page_number": c.page_number
                if c.page_number is not None
                else _PAGE_NUMBER_NONE_SENTINEL,
            }
            for c in chunks
        ]
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """Return the top-k most similar chunks for the given query embedding.

        Each result dict has keys: text, source_file, chunk_index, page_number,
        char_start, char_end, distance.
        Distance is cosine distance: 0 = identical, 2 = opposite.
        """
        n_results = min(top_k, self._collection.count())
        if n_results == 0:
            return []
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        output = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            page_number = meta["page_number"]
            if page_number == _PAGE_NUMBER_NONE_SENTINEL:
                page_number = None
            output.append(
                {
                    "text": doc,
                    "source_file": meta["source_file"],
                    "chunk_index": meta["chunk_index"],
                    "page_number": page_number,
                    "char_start": meta["char_start"],
                    "char_end": meta["char_end"],
                    "distance": dist,
                }
            )
        return output

    def count(self) -> int:
        """Return total number of chunks currently stored."""
        return self._collection.count()

    def list_sources(self) -> list[str]:
        """Return sorted unique source_file values currently in the store."""
        result = self._collection.get(include=["metadatas"])
        metadatas = result.get("metadatas") or []
        sources = {m["source_file"] for m in metadatas if m and "source_file" in m}
        return sorted(sources)

    def reset(self) -> None:
        """Delete the collection and recreate it empty."""
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
