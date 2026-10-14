"""Unit tests for reporting and export utilities."""

from findoc.reporting import export_answer_markdown, export_comparison_markdown
from findoc.schemas import AnswerResult, Citation, ComparisonFinding, ComparisonResult


def test_export_answer_markdown_with_citations():
    """Verify export_answer_markdown includes title, answer, citations, and disclaimer."""
    result = AnswerResult(
        status="answered",
        answer="Operating margin was 22.3% and revenue was $145.2M.",
        citations=[
            Citation(
                chunk_id="chunk_1",
                source_name="Apex_Report.pdf",
                page_number=2,
                quote="Operating margin expanded to 22.3%.",
            )
        ],
        retrieved_chunk_ids=["chunk_1"],
        model_mode="mock-mode",
        limitations=["Generated from synthetic disclosure."],
    )

    md = export_answer_markdown("What was the operating margin?", result)

    assert "# FinDoc AI — Grounded Answer Report" in md
    assert "**Question:** What was the operating margin?" in md
    assert "Operating margin was 22.3%" in md
    assert "Apex_Report.pdf" in md
    assert "Page 2" in md
    assert "Operating margin expanded to 22.3%" in md
    assert "Generated from synthetic disclosure." in md
    assert "assistive document reading prototype" in md


def test_export_answer_markdown_insufficient_evidence():
    """Verify export_answer_markdown formats abstention clearly."""
    result = AnswerResult(
        status="insufficient_evidence",
        answer="The documents do not contain revenue forecasts for 2035.",
        citations=[],
        retrieved_chunk_ids=[],
        model_mode="mock-mode",
        limitations=["No matching candidate passages."],
    )

    md = export_answer_markdown("What is the 2035 revenue forecast?", result)

    assert "INSUFFICIENT EVIDENCE (ABSTAINED)" in md
    assert "The documents do not contain revenue forecasts for 2035." in md


def test_export_comparison_markdown():
    """Verify export_comparison_markdown generates clean comparison table."""
    result = ComparisonResult(
        status="complete",
        document_a_id="doc_a_123",
        document_b_id="doc_b_456",
        question="Compare origination fees",
        findings=[
            ComparisonFinding(
                category="different",
                topic="Origination Fee",
                summary="Doc A charges 1.50% while Doc B charges 1.00%.",
                evidence_a=["A1"],
                evidence_b=["B1"],
                caveat=None,
            )
        ],
        limitations=["Synthesized from retrieved passages."],
    )

    md = export_comparison_markdown(result, doc_a_name="Policy_A.pdf", doc_b_name="Policy_B.pdf")

    assert "# FinDoc AI — Two-Document Comparison Report" in md
    assert "Policy_A.pdf" in md
    assert "Policy_B.pdf" in md
    assert "**DIFFERENT**" in md
    assert "Origination Fee" in md
    assert "Doc A charges 1.50%" in md
    assert "`A1`" in md
    assert "`B1`" in md
    assert "Absence in retrieved passages does not verify absence" in md
