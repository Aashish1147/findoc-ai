"""Unit tests for document parser (PDF, TXT, MD) and financial text normalization."""

import pymupdf
import pytest

from findoc.errors import DocumentParsingError
from findoc.ingestion.parser import (
    normalize_financial_text,
    parse_document_bytes,
    parse_pdf_bytes,
    parse_text_bytes,
)


def create_in_memory_pdf(pages_text: list[str]) -> bytes:
    """Helper to generate a clean synthetic multi-page PDF in memory using PyMuPDF."""
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        if text.strip():
            page.insert_text((72, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_normalize_financial_text_preserves_crucial_formatting():
    """Verify normalization does NOT strip currency symbols, signs, decimals, or dates."""
    raw = "  Net   income   in FY2023 was   +$1,250.50M   (-14.2%   YoY)   on 2023-12-31.   "
    normalized = normalize_financial_text(raw)
    assert "+$1,250.50M" in normalized
    assert "-14.2%" in normalized
    assert "2023-12-31" in normalized
    assert "  " not in normalized  # multiple spaces collapsed


def test_parse_multi_page_pdf():
    """Verify PyMuPDF extracts text page by page with 1-based page provenance."""
    p1_content = "Annual Revenue was $45.6M with gross margin of 68.2%."
    p2_content = "Total operating expenses were $14.1M for the current year."
    pdf_bytes = create_in_memory_pdf([p1_content, p2_content])

    pages, warning = parse_pdf_bytes(pdf_bytes, "doc-test-1", "annual_report.pdf")

    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert pages[0].document_id == "doc-test-1"
    assert pages[0].source_name == "annual_report.pdf"
    assert "$45.6M" in pages[0].text

    assert pages[1].page_number == 2
    assert "$14.1M" in pages[1].text
    assert warning is None


def test_parse_scanned_or_blank_pdf_warning():
    """Verify that a PDF without extractable text produces an honest, actionable warning."""
    # Blank PDF page (simulating scanned page without OCR)
    pdf_bytes = create_in_memory_pdf(["   "])

    pages, warning = parse_pdf_bytes(pdf_bytes, "doc-blank", "scanned.pdf")
    assert len(pages) == 1
    assert warning is not None
    assert "scanned or image-only" in warning


def test_parse_corrupt_pdf():
    """Verify that corrupted PDF bytes raise DocumentParsingError."""
    corrupted_bytes = b"%PDF-1.4\nCorrupted content completely invalid binary data"
    with pytest.raises(DocumentParsingError):
        parse_pdf_bytes(corrupted_bytes, "doc-corrupt", "corrupt.pdf")


def test_parse_txt_and_md_have_none_page_number():
    """Verify TXT and Markdown files do NOT invent PDF-style page numbers."""
    txt_content = b"Eligibility clause: Account balance must exceed $500."
    pages_txt, warning_txt = parse_text_bytes(txt_content, "doc-txt", "terms.txt")
    assert len(pages_txt) == 1
    assert pages_txt[0].page_number is None
    assert "$500" in pages_txt[0].text
    assert warning_txt is None

    md_content = b"# Fee Schedule\n- Maintenance fee: $12.00/month\n- Overdraft fee: $35.00"
    pages_md, warning_md = parse_document_bytes(md_content, "md", "doc-md", "fees.md")
    assert len(pages_md) == 1
    assert pages_md[0].page_number is None
    assert "$12.00/month" in pages_md[0].text


def test_parse_empty_text_document():
    """Verify parsing an empty text document returns a warning."""
    pages, warning = parse_text_bytes(b"   \n  \n  ", "doc-empty", "empty.txt")
    assert len(pages) == 1
    assert warning is not None
    assert "no readable text" in warning
