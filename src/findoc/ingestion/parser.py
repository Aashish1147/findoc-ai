"""Page-aware document parser for FinDoc AI.

Extracts text from PDF, TXT, and Markdown files while strictly preserving
page-level provenance and financial formatting (currency, units, decimals, signs).
"""

import re
from typing import List, Optional, Tuple
import pymupdf

from findoc.errors import DocumentParsingError
from findoc.schemas import FileType, PageText


def normalize_financial_text(text: str) -> str:
    """Normalize extracted text whitespace without stripping financial symbols or numerals.

    Preserves:
      - Currency symbols: $, €, £, ¥, ₹, etc.
      - Mathematical and accounting signs: +, -, %, (), [], decimal points.
      - Date separators and financial terms.
    """
    if not text:
        return ""

    # Replace null bytes or form feed characters
    text = text.replace("\x00", "").replace("\x0c", "\n")

    # Normalize CRLF / CR to standard LF
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace non-breaking spaces with standard space
    text = text.replace("\u00a0", " ")

    # Collapse multiple spaces on the same line while preserving line breaks for tables/rows
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse more than 3 consecutive newlines to 2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def parse_pdf_bytes(
    content: bytes,
    document_id: str,
    source_name: str,
) -> Tuple[List[PageText], Optional[str]]:
    """Extract text from PDF page by page using PyMuPDF.

    Returns:
        Tuple of (list_of_PageText, warning_message_if_any)

    Raises:
        DocumentParsingError: If PDF structure is malformed or unreadable.
    """
    try:
        doc = pymupdf.open(stream=content, filetype="pdf")
    except Exception as e:
        raise DocumentParsingError(f"Failed to open PDF document: {e}") from e

    try:
        total_pages = len(doc)
        if total_pages == 0:
            raise DocumentParsingError("PDF document contains 0 pages.")

        pages: List[PageText] = []
        total_extracted_chars = 0
        empty_page_numbers: List[int] = []

        for page_idx in range(total_pages):
            physical_page_num = page_idx + 1
            try:
                page = doc[page_idx]
                raw_text = page.get_text("text") or ""
            except Exception as e:
                raw_text = ""
                empty_page_numbers.append(physical_page_num)

            clean_text = normalize_financial_text(raw_text)
            total_extracted_chars += len(clean_text)

            if not clean_text:
                empty_page_numbers.append(physical_page_num)

            pages.append(
                PageText(
                    document_id=document_id,
                    source_name=source_name,
                    page_number=physical_page_num,
                    text=clean_text,
                )
            )

        warning: Optional[str] = None
        if total_extracted_chars < 30:
            warning = (
                "No selectable text found in PDF. This document appears to be scanned or image-only. "
                "Version 1 does not execute OCR; please supply a text-based PDF."
            )
        elif empty_page_numbers:
            warning = (
                f"Pages with no extractable text detected: {empty_page_numbers[:5]}"
                + ("..." if len(empty_page_numbers) > 5 else "")
            )

        return pages, warning
    finally:
        doc.close()


def parse_text_bytes(
    content: bytes,
    document_id: str,
    source_name: str,
) -> Tuple[List[PageText], Optional[str]]:
    """Decode and extract text from TXT or Markdown files with safe encoding fallback.

    Returns:
        Tuple of (list_of_PageText_with_none_page_number, warning_message_if_any)
    """
    # Safe decoding strategy: UTF-8 first, fallback to latin-1 (which never throws decode errors)
    try:
        raw_text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            raw_text = content.decode("cp1252")
        except UnicodeDecodeError:
            raw_text = content.decode("latin-1", errors="replace")

    clean_text = normalize_financial_text(raw_text)

    warning: Optional[str] = None
    if not clean_text:
        warning = "Document contains no readable text."

    page = PageText(
        document_id=document_id,
        source_name=source_name,
        page_number=None,  # No PDF pages for plain text/markdown
        text=clean_text,
    )

    return [page], warning


def parse_document_bytes(
    content: bytes,
    file_type: FileType,
    document_id: str,
    source_name: str,
) -> Tuple[List[PageText], Optional[str]]:
    """Dispatcher to parse document bytes according to validated file type."""
    if file_type == "pdf":
        return parse_pdf_bytes(content, document_id, source_name)
    elif file_type in ("txt", "md"):
        return parse_text_bytes(content, document_id, source_name)
    else:
        raise DocumentParsingError(f"Unsupported file type for parsing: {file_type}")
