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


@app.command()
def eval(
    corpus_path: Path = typer.Option(
        Path("eval/corpus"), "--corpus-path", help="Directory of corpus .md files"
    ),
    golden_path: Path = typer.Option(
        Path("eval/golden_questions.yaml"), "--golden-path", help="Golden questions YAML"
    ),
    eval_db_path: Path = typer.Option(
        Path("./data/chroma_eval"), "--eval-db-path", help="Isolated ChromaDB path for eval"
    ),
    threshold: float = typer.Option(0.80, "--threshold", help="Hit-rate pass threshold (0–1)"),
) -> None:
    """Run the RAG evaluation harness against the golden question set."""
    import os

    import anthropic
    from dotenv import load_dotenv
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table

    from rag_assistant.eval_harness import load_golden_set, run_full_eval
    from rag_assistant.ingest import ingest_path
    from rag_assistant.store import ChunkStore

    console = Console(file=sys.stdout)

    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]Error:[/red] ANTHROPIC_API_KEY not set. Add it to your .env file.")
        raise typer.Exit(code=1)

    if not corpus_path.exists():
        console.print(f"[red]Error:[/red] Corpus path does not exist: {corpus_path}")
        raise typer.Exit(code=1)
    if not golden_path.exists():
        console.print(f"[red]Error:[/red] Golden questions file not found: {golden_path}")
        raise typer.Exit(code=1)

    golden_set = load_golden_set(golden_path)
    store = ChunkStore(db_path=eval_db_path)
    client = anthropic.Anthropic(api_key=api_key)

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"Ingesting corpus from {corpus_path}...", total=None)
        ingest_path(corpus_path, store)

        def _update(msg: str) -> None:
            progress.update(task, description=msg)

        progress.update(task, description="Running evaluation...")
        report = run_full_eval(golden_set, store, client=client, progress_callback=_update)

    # Per-question table
    table = Table(title="Evaluation Results", show_lines=False)
    table.add_column("Q#", style="dim", width=3)
    table.add_column("Question", max_width=46)
    table.add_column("Retrieve", justify="center", width=9)
    table.add_column("Answer", justify="center", width=7)
    table.add_column("Fallback?", justify="center", width=9)

    for i, qr in enumerate(report.per_question, start=1):
        q_short = qr.question.question
        if len(q_short) > 46:
            q_short = q_short[:43] + "..."
        if qr.question.expected_fallback:
            ret_cell = "[dim]—[/dim]"
            ans_cell = "[dim]—[/dim]"
            fb_cell = "[green]✓[/green]" if qr.answer.was_fallback else "[red]✗[/red]"
        else:
            ret_cell = "[green]✓[/green]" if (qr.retrieval and qr.retrieval.hit) else "[red]✗[/red]"
            ans_cell = "[green]✓[/green]" if qr.answer.hit else "[red]✗[/red]"
            fb_cell = "[dim]—[/dim]"
        table.add_row(str(i), q_short, ret_cell, ans_cell, fb_cell)

    console.print()
    console.print(table)
    console.print()

    n_ans = len([q for q in golden_set if not q.expected_fallback])
    n_una = len([q for q in golden_set if q.expected_fallback])
    r_hits = round(report.retrieval_hit_rate * n_ans)
    a_hits = round(report.answer_hit_rate * n_ans)
    f_hits = round(report.fallback_rate * n_una)

    r_ok = report.retrieval_hit_rate >= threshold
    a_ok = report.answer_hit_rate >= threshold
    f_ok = report.fallback_rate == 1.0

    r_mark = "[green]✓[/green]" if r_ok else "[red]✗[/red]"
    a_mark = "[green]✓[/green]" if a_ok else "[red]✗[/red]"
    f_mark = "[green]✓[/green]" if f_ok else "[red]✗[/red]"

    console.print(
        f"  Retrieval hit-rate:   {r_hits}/{n_ans}  ({report.retrieval_hit_rate:.1%})  {r_mark}"
        f"  threshold: {threshold:.0%}"
    )
    console.print(
        f"  Answer hit-rate:      {a_hits}/{n_ans}  ({report.answer_hit_rate:.1%})  {a_mark}"
        f"  threshold: {threshold:.0%}"
    )
    console.print(
        f"  Anti-hallucination:   {f_hits}/{n_una}  ({report.fallback_rate:.1%})  {f_mark}"
    )
    console.print()

    # Cost estimate (Haiku 4.5: $1/MTok in, $5/MTok out)
    est_cost = (report.total_input_tokens / 1_000_000 * 1.0) + (
        report.total_output_tokens / 1_000_000 * 5.0
    )
    console.print(
        f"  Tokens: {report.total_input_tokens:,} in / {report.total_output_tokens:,} out"
        f"  |  Est. cost: ${est_cost:.4f}",
        style="dim",
    )
    console.print()

    if not r_ok or not a_ok or not f_ok:
        console.print("[red]FAIL[/red] — one or more metrics below threshold.")
        raise typer.Exit(code=1)
    console.print("[green]PASS[/green] — all metrics meet threshold.")


if __name__ == "__main__":
    app()
