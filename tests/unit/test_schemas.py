"""Unit tests for FinDoc AI Pydantic schemas and data contracts."""

import pytest
from pydantic import ValidationError

from findoc.schemas import (
    AnswerResult,
    ChunkRecord,
    Citation,
    ComparisonFinding,
    ComparisonResult,
    DocumentRecord,
    PageText,
)


def test_valid_document_record():
    """Verify DocumentRecord creation with valid data and default values."""
    doc = DocumentRecord(
        document_id="doc-123456",
        display_name="Annual_Report_2023.pdf",
        file_type="pdf",
        content_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        status="indexed",
        page_count=24,
        chunk_count=65,
    )
    assert doc.document_id == "doc-123456"
    assert doc.display_name == "Annual_Report_2023.pdf"
    assert doc.file_type == "pdf"
    assert doc.status == "indexed"
    assert doc.page_count == 24
    assert doc.chunk_count == 65
    assert doc.created_at is not None
    assert doc.warning_message is None


def test_document_record_invalid_status():
    """Verify that an unsupported status string raises ValidationError."""
    with pytest.raises(ValidationError):
        DocumentRecord(
            document_id="doc-1",
            display_name="file.pdf",
            file_type="pdf",
            content_hash="abc1234567890def123",
            status="unknown_status",  # invalid
        )


def test_document_record_immutability():
    """Verify that DocumentRecord instances are frozen and cannot be mutated."""
    doc = DocumentRecord(
        document_id="doc-1",
        display_name="file.pdf",
        file_type="pdf",
        content_hash="abc1234567890def123",
        status="pending",
    )
    with pytest.raises(ValidationError):
        doc.status = "indexed"


def test_page_text_pdf_and_text_file():
    """Verify PageText records with page numbers for PDF and None for text files."""
    # PDF page
    pdf_page = PageText(
        document_id="doc-1",
        source_name="Report.pdf",
        page_number=3,
        text="Revenue was $10M in FY2023.",
    )
    assert pdf_page.page_number == 3

    # Text file (no page number)
    txt_page = PageText(
        document_id="doc-2",
        source_name="Policy.txt",
        page_number=None,
        text="All applicants must have an active checking account.",
    )
    assert txt_page.page_number is None


def test_chunk_record_validation():
    """Verify ChunkRecord fields and constraints."""
    chunk = ChunkRecord(
        chunk_id="chunk-001",
        document_id="doc-1",
        source_name="Report.pdf",
        page_number=1,
        chunk_index=0,
        text="Overview of Company operations for the fiscal year ended December 31, 2023.",
        content_hash="hash-123456",
    )
    assert chunk.chunk_id == "chunk-001"
    assert chunk.chunk_index == 0


def test_citation_creation():
    """Verify Citation model attributes."""
    citation = Citation(
        chunk_id="chunk-001",
        source_name="Report.pdf",
        page_number=4,
        quote="Net income increased by 14% to $2.3 million.",
    )
    assert citation.chunk_id == "chunk-001"
    assert citation.page_number == 4


def test_answer_result_valid_citations():
    """Verify that AnswerResult accepts citations when chunk_id is in retrieved_chunk_ids."""
    result = AnswerResult(
        status="answered",
        answer="The company reported $2.3M net income.",
        citations=[
            Citation(
                chunk_id="chunk-10",
                source_name="Report.pdf",
                page_number=4,
                quote="Net income increased to $2.3 million.",
            )
        ],
        retrieved_chunk_ids=["chunk-10", "chunk-11", "chunk-12"],
        model_mode="mock",
        limitations=["Demo output"],
    )
    assert result.status == "answered"
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == "chunk-10"


def test_answer_result_rejects_hallucinated_citation():
    """Verify that AnswerResult rejects citations referencing chunk IDs not in retrieved_chunk_ids."""
    with pytest.raises(ValidationError) as exc_info:
        AnswerResult(
            status="answered",
            answer="The company reported $2.3M net income.",
            citations=[
                Citation(
                    chunk_id="chunk-999-fabricated",
                    source_name="Report.pdf",
                    page_number=4,
                    quote="Net income increased to $2.3 million.",
                )
            ],
            retrieved_chunk_ids=["chunk-10", "chunk-11"],
            model_mode="mock",
        )
    assert "Citation references chunk_id 'chunk-999-fabricated' which was not in retrieved evidence pack" in str(
        exc_info.value
    )


def test_comparison_models():
    """Verify ComparisonFinding and ComparisonResult schema validation."""
    finding = ComparisonFinding(
        category="different",
        topic="Annual Maintenance Fee",
        summary="Document A charges $50 annually whereas Document B waives the fee with $1,000 balance.",
        evidence_a=["chunk-a-1"],
        evidence_b=["chunk-b-2"],
        caveat=None,
    )
    assert finding.category == "different"
    assert finding.evidence_a == ["chunk-a-1"]

    result = ComparisonResult(
        status="complete",
        document_a_id="doc-a",
        document_b_id="doc-b",
        question="Compare maintenance fees",
        findings=[finding],
        limitations=["Assistive comparison tool"],
    )
    assert result.status == "complete"
    assert len(result.findings) == 1
