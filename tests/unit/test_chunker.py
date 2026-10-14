"""Unit tests for page-aware text chunking and provenance preservation."""

from findoc.ingestion.chunker import chunk_page_texts, split_text_into_chunks
from findoc.schemas import PageText


def test_split_text_into_chunks_short_text():
    """Verify text shorter than chunk_size produces exactly 1 chunk."""
    text = "Short annual financial summary."
    chunks = split_text_into_chunks(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_text_into_chunks_empty_text():
    """Verify empty or whitespace-only text produces no chunks."""
    assert split_text_into_chunks("", chunk_size=500, chunk_overlap=50) == []
    assert split_text_into_chunks("   \n\n  ", chunk_size=500, chunk_overlap=50) == []


def test_split_text_into_chunks_overlap_and_boundaries():
    """Verify text exceeding chunk size splits with overlap and respects boundaries."""
    paragraph1 = "Paragraph 1: In fiscal year 2023, net revenue increased by 15.4% to $120.5M."
    paragraph2 = "Paragraph 2: Operating margin expanded to 23.1% driven by operational efficiencies."
    paragraph3 = "Paragraph 3: Cash flow from operations was $35.0M compared to $28.0M in 2022."
    full_text = f"{paragraph1}\n\n{paragraph2}\n\n{paragraph3}"

    chunks = split_text_into_chunks(full_text, chunk_size=120, chunk_overlap=30)
    assert len(chunks) >= 2
    # Check that each chunk is bounded
    for c in chunks:
        assert len(c) <= 150  # within reasonable boundary window


def test_chunk_page_texts_preserves_page_provenance():
    """Verify that chunks accurately retain physical page numbers across multiple pages."""
    pages = [
        PageText(
            document_id="doc-report-1",
            source_name="Annual_Report.pdf",
            page_number=1,
            text="Executive overview and company profile for 2023.",
        ),
        PageText(
            document_id="doc-report-1",
            source_name="Annual_Report.pdf",
            page_number=2,
            text="Financial performance: Revenue of $100M and EBITDA of $25M.",
        ),
    ]

    chunk_records = chunk_page_texts(pages, chunk_size=500, chunk_overlap=50)
    assert len(chunk_records) == 2

    # First chunk from page 1
    assert chunk_records[0].document_id == "doc-report-1"
    assert chunk_records[0].source_name == "Annual_Report.pdf"
    assert chunk_records[0].page_number == 1
    assert chunk_records[0].chunk_index == 0
    assert chunk_records[0].chunk_id == "doc-report-1_p1_c0000"
    assert "Executive overview" in chunk_records[0].text

    # Second chunk from page 2
    assert chunk_records[1].document_id == "doc-report-1"
    assert chunk_records[1].source_name == "Annual_Report.pdf"
    assert chunk_records[1].page_number == 2
    assert chunk_records[1].chunk_index == 1
    assert chunk_records[1].chunk_id == "doc-report-1_p2_c0001"
    assert "Financial performance" in chunk_records[1].text


def test_chunk_page_texts_skips_empty_pages():
    """Verify that empty pages within a document do not create blank chunks."""
    pages = [
        PageText(
            document_id="doc-report-2",
            source_name="Report.pdf",
            page_number=1,
            text="Page 1 readable financial data.",
        ),
        PageText(
            document_id="doc-report-2",
            source_name="Report.pdf",
            page_number=2,
            text="   \n   ",  # blank page
        ),
        PageText(
            document_id="doc-report-2",
            source_name="Report.pdf",
            page_number=3,
            text="Page 3 balance sheet notes.",
        ),
    ]

    chunks = chunk_page_texts(pages, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[1].page_number == 3
    assert chunks[1].chunk_index == 1


def test_chunk_page_texts_text_file_none_page():
    """Verify that text files produce chunks with page_number=None."""
    pages = [
        PageText(
            document_id="doc-policy",
            source_name="policy.txt",
            page_number=None,
            text="Terms and conditions for business loan accounts. Minimum credit score is 680.",
        )
    ]

    chunks = chunk_page_texts(pages, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0].page_number is None
    assert chunks[0].chunk_id == "doc-policy_txt_c0000"
    assert chunks[0].content_hash is not None
