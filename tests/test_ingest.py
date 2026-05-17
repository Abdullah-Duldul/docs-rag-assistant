"""CLI integration tests for ingest, status, and reset commands."""

from pathlib import Path

from typer.testing import CliRunner

from rag_assistant.cli import app

FIXTURES = Path(__file__).parent / "fixtures"
runner = CliRunner()


def _run(args: list[str], db_path: Path, **kwargs):
    """Invoke the CLI, automatically appending --db-path to args."""
    return runner.invoke(app, args + ["--db-path", str(db_path)], **kwargs)


# ---------------------------------------------------------------------------
# ingest command
# ---------------------------------------------------------------------------


def test_ingest_processes_supported_files(tmp_path, fake_embed):
    result = _run(["ingest", str(FIXTURES)], db_path=tmp_path / "chroma")
    assert result.exit_code == 0

    # Behavioral: data actually landed in the store
    from rag_assistant.store import ChunkStore

    store = ChunkStore(db_path=tmp_path / "chroma")
    assert store.count() > 0
    assert len(store.list_sources()) == 2  # sample.txt and sample.md


def test_ingest_handles_nonexistent_path(tmp_path):
    result = _run(["ingest", "/no/such/path/exists"], db_path=tmp_path / "chroma")
    assert result.exit_code == 1
    assert "does not exist" in result.stdout


def test_ingest_skips_unsupported_files(tmp_path, fake_embed):
    good = tmp_path / "input" / "doc.txt"
    bad = tmp_path / "input" / "data.csv"
    good.parent.mkdir()
    good.write_text("This is valid text content. " * 20)
    bad.write_text("col1,col2\n1,2\n")

    result = _run(["ingest", str(tmp_path / "input")], db_path=tmp_path / "chroma")
    assert result.exit_code == 0
    assert "Files skipped (unsupported): 1" in result.stdout

    from rag_assistant.store import ChunkStore

    store = ChunkStore(db_path=tmp_path / "chroma")
    assert store.count() > 0  # doc.txt was processed


def test_ingest_continues_on_parse_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "rag_assistant.ingest.embed", lambda texts: [[0.1] * 384] * len(texts)
    )

    from rag_assistant.parsers import ParsedPage

    def fake_parse(path: Path):
        if path.name == "bad.txt":
            raise RuntimeError("Simulated parse error")
        return [ParsedPage(text=path.read_text(), page_number=None, source_file=str(path))]

    monkeypatch.setattr("rag_assistant.ingest.parse_file", fake_parse)

    good = tmp_path / "good.txt"
    bad = tmp_path / "bad.txt"
    good.write_text("Good content that should be processed. " * 20)
    bad.write_text("Bad content.")

    result = _run(["ingest", str(tmp_path)], db_path=tmp_path / "chroma")
    assert result.exit_code == 0
    assert "Files processed: 1" in result.stdout
    assert "Files failed (errors): 1" in result.stdout
    assert "bad.txt" in result.stdout  # listed in failures


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------


def test_status_empty_store(tmp_path):
    result = _run(["status"], db_path=tmp_path / "chroma")
    assert result.exit_code == 0
    assert "Total chunks: 0" in result.stdout


def test_status_after_ingest(tmp_path, fake_embed):
    _run(["ingest", str(FIXTURES)], db_path=tmp_path / "chroma")
    result = _run(["status"], db_path=tmp_path / "chroma")
    assert result.exit_code == 0
    assert "Total chunks" in result.stdout

    from rag_assistant.store import ChunkStore

    store = ChunkStore(db_path=tmp_path / "chroma")
    assert store.count() > 0
    assert len(store.list_sources()) >= 2


# ---------------------------------------------------------------------------
# reset command
# ---------------------------------------------------------------------------


def test_reset_with_yes_flag(tmp_path, fake_embed):
    _run(["ingest", str(FIXTURES)], db_path=tmp_path / "chroma")

    from rag_assistant.store import ChunkStore

    assert ChunkStore(db_path=tmp_path / "chroma").count() > 0

    result = _run(["reset", "--yes"], db_path=tmp_path / "chroma")
    assert result.exit_code == 0
    assert "reset" in result.stdout.lower()
    assert ChunkStore(db_path=tmp_path / "chroma").count() == 0


def test_reset_aborts_on_no_confirmation(tmp_path):
    # Populate the store directly — no need to go through CLI ingest
    from rag_assistant.chunker import Chunk
    from rag_assistant.store import ChunkStore

    store = ChunkStore(db_path=tmp_path / "chroma")
    chunk = Chunk(
        text="hello world",
        chunk_index=0,
        char_start=0,
        char_end=11,
        source_file="test.txt",
        page_number=None,
    )
    store.add_chunks([chunk], [[0.1] * 384])
    assert store.count() == 1

    result = _run(["reset"], db_path=tmp_path / "chroma", input="n\n")
    assert result.exit_code == 0
    assert "Aborted" in result.stdout
    assert ChunkStore(db_path=tmp_path / "chroma").count() == 1
