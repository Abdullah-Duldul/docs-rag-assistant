"""Tests for PDF, Markdown, and plain-text parsers."""

from pathlib import Path

import pytest

from rag_assistant.parsers import ParsedPage, parse_file, parse_markdown, parse_pdf, parse_text

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_text_returns_single_page():
    pages = parse_text(FIXTURES / "sample.txt")
    assert len(pages) == 1
    page = pages[0]
    assert page.page_number is None
    assert "Lorem ipsum" in page.text
    assert page.source_file.endswith("sample.txt")


def test_parse_markdown_returns_single_page():
    pages = parse_markdown(FIXTURES / "sample.md")
    assert len(pages) == 1
    page = pages[0]
    assert page.page_number is None
    assert "# Sample Markdown Document" in page.text
    assert page.source_file.endswith("sample.md")


def test_parse_pdf_returns_one_page_per_pdf_page(sample_pdf_path):
    pages = parse_pdf(sample_pdf_path)
    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert pages[1].page_number == 2
    assert "Page one" in pages[0].text
    assert "Page two" in pages[1].text


def test_parse_pdf_skips_empty_pages(blank_page_pdf):
    pages = parse_pdf(blank_page_pdf)
    assert len(pages) == 1
    assert pages[0].page_number == 1


@pytest.mark.parametrize(
    "filename",
    ["sample.txt", "sample.md"],
)
def test_parse_file_dispatches_by_suffix_static(filename):
    pages = parse_file(FIXTURES / filename)
    assert isinstance(pages, list)
    assert len(pages) >= 1
    assert all(isinstance(p, ParsedPage) for p in pages)


def test_parse_file_dispatches_pdf(sample_pdf_path):
    pages = parse_file(sample_pdf_path)
    assert isinstance(pages, list)
    assert len(pages) >= 1
    assert all(isinstance(p, ParsedPage) for p in pages)


def test_parse_file_rejects_unsupported_suffix(tmp_path):
    bad_file = tmp_path / "data.csv"
    bad_file.write_text("col1,col2\n1,2\n")
    with pytest.raises(ValueError, match="Unsupported file type"):
        parse_file(bad_file)
