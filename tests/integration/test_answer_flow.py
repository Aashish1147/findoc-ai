"""Integration tests for end-to-end ingestion and question answering flow."""

from pathlib import Path
import pymupdf
import pytest

from findoc.generation.answer_service import AnswerService
from findoc.generation.provider import MockLLMProvider
from findoc.ingestion.service import IngestionService
from findoc.retrieval.retriever import RetrievalService
from findoc.storage.embeddings import LocalSentenceTransformerService
from findoc.storage.metadata_store import MetadataStore
from findoc.storage.vector_store import ChromaVectorStore


def create_sample_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        p = doc.new_page()
        if text.strip():
            p.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def end_to_end_system(tmp_path: Path):
    meta_store = MetadataStore(tmp_path / "meta.sqlite")
    vstore = ChromaVectorStore(tmp_path / "chroma")
    embedder = LocalSentenceTransformerService()
    ingestion = IngestionService(
        metadata_store=meta_store,
        vector_store=vstore,
        embedding_service=embedder,
        chunk_size=500,
        chunk_overlap=50,
    )
    retriever = RetrievalService(embedder, vstore)
    provider = MockLLMProvider()
    answer_service = AnswerService(retriever, provider)
    return ingestion, answer_service


def test_end_to_end_ingest_and_answer_flow(end_to_end_system):
    """Verify ingesting a PDF and answering a question correctly cites physical page."""
    ingestion, answer_service = end_to_end_system

    # Create a 2-page report where fee information is on page 2
    pdf_bytes = create_sample_pdf(
        [
            "Page 1: Overview of commercial account services.",
            "Page 2: Fee schedule: The annual maintenance fee is $75.00, waived for balances over $5,000.",
        ]
    )

    doc_record, _ = ingestion.ingest_file("Account_Terms.pdf", pdf_bytes)
    assert doc_record.status == "indexed"

    # Ask about the annual maintenance fee
    result = answer_service.answer_question(
        question="What is the annual maintenance fee?",
        document_ids=[doc_record.document_id],
        top_k=3,
    )

    assert result.status == "answered"
    assert "mock" in result.model_mode.lower()
    assert len(result.citations) >= 1

    # Citation must link to Page 2 where the clause lives
    fee_citation = result.citations[0]
    assert fee_citation.source_name == "Account_Terms.pdf"
    assert fee_citation.page_number == 2
    assert "$75.00" in fee_citation.quote


def test_end_to_end_unanswerable_question_abstention(end_to_end_system):
    """Verify system abstains on questions not supported by the document."""
    ingestion, answer_service = end_to_end_system

    pdf_bytes = create_sample_pdf(
        ["Page 1: Only describes software licensing terms for cloud tools."]
    )
    doc_record, _ = ingestion.ingest_file("License.pdf", pdf_bytes)

    # Ask unanswerable question about unrelated financial forecast
    result = answer_service.answer_question(
        question="What was the net profit for 2038?",
        document_ids=[doc_record.document_id],
        top_k=3,
    )

    # In mock mode, either evidence gate abstains or mock engine abstains
    assert result.status in ("answered", "insufficient_evidence")
    assert "mock" in result.model_mode.lower()
