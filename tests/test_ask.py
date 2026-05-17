"""CLI tests for the `rag ask` command."""

from pathlib import Path

from typer.testing import CliRunner

from rag_assistant.cli import app

runner = CliRunner()


def _run(args: list[str], db_path: Path, **kwargs):
    return runner.invoke(app, args + ["--db-path", str(db_path)], **kwargs)


def test_ask_empty_store_exits_nonzero(tmp_path, monkeypatch):
    from rag_assistant.retriever import EmptyStoreError

    def _raise(*args, **kwargs):
        raise EmptyStoreError("No documents indexed.")

    monkeypatch.setattr("rag_assistant.retriever.retrieve", _raise)
    result = _run(["ask", "What is the answer?"], db_path=tmp_path / "chroma")
    assert result.exit_code == 1
    assert "ingest" in result.stdout


def test_ask_returns_answer(tmp_path, monkeypatch):
    from rag_assistant.generator import AnswerResult
    from rag_assistant.retriever import RetrievedChunk

    mock_chunk = RetrievedChunk(
        text="The answer is 42.",
        source_file="doc.txt",
        page_number=None,
        chunk_index=0,
        distance=0.1,
    )
    mock_result = AnswerResult(
        answer_text="The answer is 42 [1].",
        sources=[mock_chunk],
        input_tokens=50,
        output_tokens=10,
    )

    monkeypatch.setattr(
        "rag_assistant.retriever.retrieve",
        lambda *args, **kwargs: [mock_chunk],
    )
    monkeypatch.setattr(
        "rag_assistant.generator.generate_answer",
        lambda *args, **kwargs: mock_result,
    )

    result = _run(["ask", "What is the answer?"], db_path=tmp_path / "chroma")
    assert result.exit_code == 0
    assert "The answer is 42 [1]." in result.stdout
    assert "Sources" in result.stdout
