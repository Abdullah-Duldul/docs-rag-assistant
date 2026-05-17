"""Walk a directory, route files to the correct parser, chunk, embed, and store."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from rag_assistant.chunker import chunk_pages
from rag_assistant.embeddings import embed
from rag_assistant.parsers import parse_file
from rag_assistant.store import ChunkStore


@dataclass
class IngestSummary:
    files_processed: int = 0
    files_skipped: int = 0  # unsupported file types
    files_failed: int = 0  # parse/embed errors
    chunks_added: int = 0
    failed_files: list[tuple[str, str]] = field(default_factory=list)  # (path, error)


def ingest_path(
    path: Path,
    store: ChunkStore,
    chunk_size: int = 500,
    overlap: int = 50,
    progress_callback: Callable[[str], None] | None = None,
) -> IngestSummary:
    """Ingest all supported files from path into the store.

    path may be a single file or a directory (recursively walked).
    One file failure never aborts the batch — errors are collected and returned.
    """
    if path.is_file():
        files = [path]
    else:
        files = sorted([f for f in path.rglob("*") if f.is_file()], key=str)

    summary = IngestSummary()
    for file in files:
        if progress_callback:
            progress_callback(f"Processing {file.name}")
        try:
            pages = parse_file(file)
        except ValueError:
            # Unsupported file type — not an error, just skip
            summary.files_skipped += 1
            continue
        except Exception as e:
            summary.files_failed += 1
            summary.failed_files.append((str(file), str(e)))
            continue

        chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            continue

        embeddings = embed([c.text for c in chunks])
        store.add_chunks(chunks, embeddings)
        summary.files_processed += 1
        summary.chunks_added += len(chunks)

    return summary
