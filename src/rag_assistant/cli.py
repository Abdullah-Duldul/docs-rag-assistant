"""Typer CLI for rag-assistant.

Commands: ingest, ask, status, reset.
"""

import sys
from pathlib import Path

import typer

from rag_assistant.store import DEFAULT_DB_PATH

app = typer.Typer(help="RAG over your local documents.")


@app.command()
def ingest(
    path: str = typer.Argument(..., help="File or directory to ingest"),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, "--db-path", help="Path to ChromaDB store"),
) -> None:
    """Ingest documents from a path into the vector store."""
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from rag_assistant.ingest import ingest_path
    from rag_assistant.store import ChunkStore

    console = Console(file=sys.stdout)
    target = Path(path).expanduser().resolve()
    if not target.exists():
        console.print(f"[red]Error:[/red] Path does not exist: {target}")
        raise typer.Exit(code=1)

    store = ChunkStore(db_path=db_path)

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Starting ingest...", total=None)

        def update(msg: str) -> None:
            progress.update(task, description=msg)

        summary = ingest_path(target, store, progress_callback=update)

    console.print("\nDone.")
    console.print(f"  Files processed: {summary.files_processed}")
    console.print(f"  Files skipped (unsupported): {summary.files_skipped}")
    console.print(f"  Files failed (errors): {summary.files_failed}")
    console.print(f"  Chunks added: {summary.chunks_added}")
    console.print(f"  Total chunks in store: {store.count()}")
    if summary.failed_files:
        console.print("\nFailures:")
        for path_str, error in summary.failed_files:
            console.print(f"  {path_str}: {error}")


@app.command()
def status(
    db_path: Path = typer.Option(DEFAULT_DB_PATH, "--db-path", help="Path to ChromaDB store"),
) -> None:
    """Show store status: chunk count, source files, DB path."""
    from rich.console import Console

    from rag_assistant.store import ChunkStore

    console = Console(file=sys.stdout)
    store = ChunkStore(db_path=db_path)
    console.print("docs-rag-assistant store status")
    console.print(f"  DB path: {db_path.resolve()}")
    console.print(f"  Total chunks: {store.count()}")
    sources = store.list_sources()
    console.print(f"  Unique sources: {len(sources)}")
    if sources:
        console.print("  Files:")
        for src in sources[:20]:
            console.print(f"    - {src}")
        if len(sources) > 20:
            console.print(f"    ... and {len(sources) - 20} more")


@app.command()
def reset(
    db_path: Path = typer.Option(DEFAULT_DB_PATH, "--db-path", help="Path to ChromaDB store"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Wipe the vector store."""
    from rich.console import Console

    from rag_assistant.store import ChunkStore

    console = Console(file=sys.stdout)
    store = ChunkStore(db_path=db_path)
    count = store.count()
    if count == 0:
        console.print("Store is already empty.")
        return
    if not yes:
        confirm = typer.confirm(f"This will delete {count} chunks from the store. Continue?")
        if not confirm:
            console.print("Aborted.")
            return
    store.reset()
    console.print(f"Store reset. Removed {count} chunks.")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to answer"),
    top_k: int = typer.Option(5, "--top-k", help="Number of chunks to retrieve"),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, "--db-path", help="Path to ChromaDB store"),
) -> None:
    """Retrieve relevant chunks and generate a cited answer with Claude."""
    from rich.console import Console
    from rich.panel import Panel

    from rag_assistant.generator import generate_answer
    from rag_assistant.retriever import EmptyStoreError, retrieve
    from rag_assistant.store import ChunkStore

    console = Console(file=sys.stdout)
    store = ChunkStore(db_path=db_path)

    try:
        chunks = retrieve(question, top_k=top_k, store=store)
    except EmptyStoreError:
        console.print("[red]Error:[/red] Store is empty. Run `rag ingest <path>` first.")
        raise typer.Exit(code=1)

    try:
        result = generate_answer(question, chunks)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)
    except Exception as exc:
        console.print(f"[red]Error:[/red] API call failed: {exc}")
        raise typer.Exit(code=1)

    console.print(Panel(result.answer_text, title="Answer", border_style="blue"))
    console.print("\n  Sources:", style="dim")
    for i, chunk in enumerate(result.sources, start=1):
        page_info = f"  page {chunk.page_number}" if chunk.page_number is not None else ""
        console.print(
            f"    [{i}]  {chunk.source_file}{page_info}  chunk {chunk.chunk_index}",
            style="dim",
        )
    console.print(
        f"\n  Tokens: {result.input_tokens} in / {result.output_tokens} out",
        style="dim",
    )


if __name__ == "__main__":
    app()
