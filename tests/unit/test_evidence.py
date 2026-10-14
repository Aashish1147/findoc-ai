"""Unit tests for EvidencePack and citation derivation."""

from findoc.retrieval.evidence import EvidenceItem, EvidencePack, build_evidence_pack
from findoc.schemas import ChunkRecord


def test_empty_evidence_pack():
    """Verify empty evidence pack flags no usable evidence and formats cleanly."""
    pack = EvidencePack(
        question="What was net revenue?",
        selected_document_ids=["doc-1"],
        items=[],
    )
    assert not pack.has_usable_evidence
    assert pack.chunk_ids == []
    assert pack.evidence_ids == []
    assert "NO EVIDENCE" in pack.format_for_prompt()
    assert pack.to_citation("E1", "some quote") is None


def test_build_evidence_pack_and_formatting():
    """Verify build_evidence_pack assigns sequential E1, E2 IDs and formats delimited context."""
    chunks = [
        (
            ChunkRecord(
                chunk_id="chunk-100",
                document_id="doc-1",
                source_name="annual_report.pdf",
                page_number=3,
                chunk_index=0,
                text="In FY2023, net revenue was $45.6M.",
                content_hash="hash-100",
            ),
            0.15,
        ),
        (
            ChunkRecord(
                chunk_id="chunk-101",
                document_id="doc-1",
                source_name="annual_report.pdf",
                page_number=4,
                chunk_index=1,
                text="Operating income totaled $12.3M.",
                content_hash="hash-101",
            ),
            0.22,
        ),
    ]

    pack = build_evidence_pack("What was the income?", ["doc-1"], chunks)
    assert pack.has_usable_evidence
    assert len(pack.items) == 2
    assert pack.items[0].evidence_id == "E1"
    assert pack.items[0].chunk_id == "chunk-100"
    assert pack.items[1].evidence_id == "E2"
    assert pack.items[1].chunk_id == "chunk-101"

    prompt_text = pack.format_for_prompt()
    assert "--- EVIDENCE [E1] ---" in prompt_text
    assert "Source: annual_report.pdf" in prompt_text
    assert "Page 3" in prompt_text
    assert "In FY2023, net revenue was $45.6M." in prompt_text
    assert "--- EVIDENCE [E2] ---" in prompt_text


def test_to_citation_trusted_metadata_mapping():
    """Verify to_citation derives source filename and page strictly from trusted backend items."""
    chunks = [
        (
            ChunkRecord(
                chunk_id="chunk-trusted",
                document_id="doc-99",
                source_name="verified_report.pdf",
                page_number=14,
                chunk_index=5,
                text="The company maintains a reserve fund of $50,000,000 for unforeseen contingencies.",
                content_hash="hash-trusted",
            ),
            0.10,
        )
    ]
    pack = build_evidence_pack("Reserve fund", ["doc-99"], chunks)

    # Citation using evidence ID "E1"
    cit1 = pack.to_citation("E1", "maintains a reserve fund of $50,000,000")
    assert cit1 is not None
    assert cit1.chunk_id == "chunk-trusted"
    assert cit1.source_name == "verified_report.pdf"
    assert cit1.page_number == 14
    assert cit1.quote == "maintains a reserve fund of $50,000,000"

    # Citation using bracketed "[E1]"
    cit2 = pack.to_citation("[E1]", "reserve fund")
    assert cit2 is not None
    assert cit2.chunk_id == "chunk-trusted"

    # Citation using stable chunk ID "chunk-trusted"
    cit3 = pack.to_citation("chunk-trusted", "reserve fund")
    assert cit3 is not None
    assert cit3.source_name == "verified_report.pdf"

    # Reject unknown/hallucinated ID
    assert pack.to_citation("E99", "fabricated") is None
    assert pack.to_citation("non_existent_chunk", "fabricated") is None
