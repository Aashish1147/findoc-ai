"""Unit tests for file validation, sanitization, and duplicate detection."""

import pytest

from findoc.errors import FileValidationError
from findoc.ingestion.validation import (
    compute_sha256,
    generate_document_id,
    sanitize_display_name,
    validate_file_upload,
)


def test_sanitize_display_name_removes_path_traversal():
    """Verify that path traversal attempts are stripped from filenames."""
    assert sanitize_display_name("../../etc/passwd.pdf") == "passwd.pdf"
    assert sanitize_display_name("..\\..\\Windows\\System32\\file.txt") == "file.txt"
    assert sanitize_display_name("/var/log/annual_report.pdf") == "annual_report.pdf"


def test_sanitize_display_name_strips_control_characters():
    """Verify that null bytes and control chars are removed."""
    raw = "test\x00file\x1f_2023.pdf"
    cleaned = sanitize_display_name(raw)
    assert "\x00" not in cleaned
    assert "\x1f" not in cleaned
    assert cleaned.endswith(".pdf")


def test_compute_sha256_and_deterministic_id():
    """Verify deterministic hash and document ID generation."""
    sample_bytes = b"FinDoc AI sample content for deterministic hashing."
    hash1 = compute_sha256(sample_bytes)
    hash2 = compute_sha256(sample_bytes)
    assert hash1 == hash2

    doc_id1 = generate_document_id(hash1)
    doc_id2 = generate_document_id(hash2)
    assert doc_id1 == doc_id2
    assert doc_id1.startswith("doc_")
    assert len(doc_id1) == 20  # "doc_" (4 chars) + 16 hex chars


def test_validate_file_upload_empty_file():
    """Verify that 0-byte files raise FileValidationError."""
    with pytest.raises(FileValidationError) as exc_info:
        validate_file_upload("empty.pdf", b"")
    assert "Uploaded file is empty" in str(exc_info.value)


def test_validate_file_upload_oversized():
    """Verify that files exceeding maximum size limit are rejected."""
    # 2 MB content with a 1 MB limit
    oversized = b"X" * (2 * 1024 * 1024)
    with pytest.raises(FileValidationError) as exc_info:
        validate_file_upload("large.txt", oversized, max_file_mb=1)
    assert "exceeds maximum allowed limit" in str(exc_info.value)


def test_validate_file_upload_unsupported_extension():
    """Verify that disallowed extensions (.exe, .docx, .csv) are rejected."""
    for ext in ("script.exe", "document.docx", "data.csv", "archive.zip"):
        with pytest.raises(FileValidationError) as exc_info:
            validate_file_upload(ext, b"some content")
        assert "Unsupported file extension" in str(exc_info.value)


def test_validate_file_upload_pdf_signature_check():
    """Verify that PDF files must start with the valid %PDF- magic bytes."""
    # Fake PDF without %PDF-
    fake_pdf = b"This is not a real PDF file header."
    with pytest.raises(FileValidationError) as exc_info:
        validate_file_upload("fake.pdf", fake_pdf)
    assert "lacks a valid PDF header signature" in str(exc_info.value)

    # Valid PDF signature
    valid_pdf_bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj"
    doc_id, name, ftype, content_hash = validate_file_upload("valid.pdf", valid_pdf_bytes)
    assert ftype == "pdf"
    assert name == "valid.pdf"
    assert doc_id.startswith("doc_")


def test_validate_file_upload_txt_and_md():
    """Verify valid .txt and .md files pass validation."""
    txt_bytes = b"Sample policy text."
    _, name_txt, ftype_txt, _ = validate_file_upload("policy.txt", txt_bytes)
    assert ftype_txt == "txt"
    assert name_txt == "policy.txt"

    md_bytes = b"# Sample Financial Disclosure\nRevenue was $10M."
    _, name_md, ftype_md, _ = validate_file_upload("disclosure.md", md_bytes)
    assert ftype_md == "md"
    assert name_md == "disclosure.md"
