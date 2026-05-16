"""Parse raw text from PDF, Markdown, and plain-text files."""

from dataclasses import dataclass
from pathlib import Path

import pypdf


@dataclass
class ParsedPage:
    text: str
    page_number: int | None  # None for non-paginated formats
    source_file: str  # absolute path as string


def parse_pdf(path: Path) -> list[ParsedPage]:
    """Extract text from a PDF file, one ParsedPage per non-empty page."""
    pages: list[ParsedPage] = []
    reader = pypdf.PdfReader(path)
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(ParsedPage(text=text, page_number=i, source_file=str(path.resolve())))
    return pages


def parse_markdown(path: Path) -> list[ParsedPage]:
    """Return a single ParsedPage for a Markdown file."""
    text = path.read_text(encoding="utf-8")
    return [ParsedPage(text=text, page_number=None, source_file=str(path.resolve()))]


def parse_text(path: Path) -> list[ParsedPage]:
    """Return a single ParsedPage for a plain-text file."""
    text = path.read_text(encoding="utf-8")
    return [ParsedPage(text=text, page_number=None, source_file=str(path.resolve()))]


def parse_file(path: Path) -> list[ParsedPage]:
    """Dispatch to the correct parser based on file suffix.

    Raises ValueError for unsupported file types.
    Supported: .pdf, .md, .markdown, .txt
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path)
    if suffix in {".md", ".markdown"}:
        return parse_markdown(path)
    if suffix == ".txt":
        return parse_text(path)
    raise ValueError(f"Unsupported file type: {suffix!r}")
