"""Typer CLI for rag-assistant.

Commands: ingest, ask, status, reset.
Full implementation in steps 4-5.
"""

import typer

app = typer.Typer(help="RAG over your local documents.")


@app.command()
def ingest(path: str = typer.Argument(..., help="Directory to ingest")) -> None:
    """Ingest documents from a path. TODO: implement in step 4."""
    typer.echo(f"[stub] Would ingest from: {path}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to answer"),
    top_k: int = typer.Option(5, "--top-k", help="Number of chunks to retrieve"),
) -> None:
    """Ask a question. TODO: implement in step 5."""
    typer.echo(f"[stub] Would answer: {question}")


@app.command()
def status() -> None:
    """Show store status. TODO: implement in step 4."""
    typer.echo("[stub] Status not yet implemented.")


@app.command()
def reset() -> None:
    """Reset the vector store. TODO: implement in step 4."""
    typer.echo("[stub] Reset not yet implemented.")


if __name__ == "__main__":
    app()
