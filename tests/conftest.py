"""Shared pytest fixtures for docs-rag-assistant tests."""

from pathlib import Path

import pytest
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas


def _make_two_page_pdf(path: Path) -> None:
    """Write a 2-page PDF with extractable text to path."""
    c = canvas.Canvas(str(path), pagesize=LETTER)
    c.drawString(100, 700, "Page one of the sample PDF document.")
    c.drawString(100, 680, "This text is on the first page and should be extractable.")
    c.showPage()
    c.drawString(100, 700, "Page two of the sample PDF document.")
    c.drawString(100, 680, "This text is on the second page and should be extractable.")
    c.showPage()
    c.save()


@pytest.fixture(scope="session")
def sample_pdf_path(tmp_path_factory) -> Path:
    """Return path to a temporary 2-page PDF with extractable text."""
    path = tmp_path_factory.mktemp("fixtures") / "sample.pdf"
    _make_two_page_pdf(path)
    return path


@pytest.fixture
def blank_page_pdf(tmp_path) -> Path:
    """Return path to a 2-page PDF where page 2 is blank (no text)."""
    path = tmp_path / "blank_page.pdf"
    c = canvas.Canvas(str(path), pagesize=LETTER)
    c.drawString(100, 700, "Only page one has text content.")
    c.showPage()
    c.showPage()  # page 2: blank
    c.save()
    return path


@pytest.fixture
def fake_embed(monkeypatch):
    """Patch embed() in rag_assistant.ingest to avoid model load in CLI tests."""

    def _fake(texts: list[str]) -> list[list[float]]:
        return [[float(i % 7) / 10.0] * 384 for i in range(len(texts))]

    monkeypatch.setattr("rag_assistant.ingest.embed", _fake)
