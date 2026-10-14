"""Integration tests for the complete document ingestion pipeline."""

from pathlib import Path
import pymupdf
import pytest

from findoc.ingestion.service import IngestionService
from findoc.storage.embeddings import DeterministicMockEmbeddingService
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
def ingestion_pipeline(tmp_path: Path):
    meta_store = MetadataStore(tmp_path / "meta.sqlite")
    vstore = ChromaVectorStore(tmp_path / "chroma")
    embedder = DeterministicMockEmbeddingService(dimension=64)
    service = IngestionService(
        metadata_store=meta_store,
        vector_store=vstore,
        embedding_service=embedder,
        chunk_size=500,
        chunk_overlap=50,
    )
    return service, meta_store, vstore


def test_end_to_end_pdf_ingestion_and_deduplication(ingestion_pipeline):
    """Verify full PDF ingestion, metadata registration, vector indexing, and duplicate detection."""
    service, meta_store, vstore = ingestion_pipeline

    pdf_bytes = create_sample_pdf(
        [
            "Page 1: Annual financial statement for Q4 2023. Gross revenue was $85.4M.",
            "Page 2: Balance sheet summary. Total current assets reached $42.1M.",
        ]
    )

    # 1. Ingest document
    doc_record, is_duplicate = service.ingest_file("Q4_2023_Report.pdf", pdf_bytes)
    assert not is_duplicate
    assert doc_record.status == "indexed"
    assert doc_record.page_count == 2
    assert doc_record.chunk_count >= 2
    assert vstore.count() == doc_record.chunk_count

    # 2. Attempt duplicate ingestion with same bytes
    dup_record, is_dup = service.ingest_file("Different_Filename_Same_Bytes.pdf", pdf_bytes)
    assert is_dup
    assert dup_record.document_id == doc_record.document_id
    assert vstore.count() == doc_record.chunk_count  # no duplicate chunks added


def test_scanned_or_empty_pdf_ingestion_warning(ingestion_pipeline):
    """Verify empty/scanned PDF receives warning status and produces zero vector chunks."""
    service, meta_store, vstore = ingestion_pipeline

    empty_pdf = create_sample_pdf(["   "])  # 1 blank page
    record, is_dup = service.ingest_file("scanned_receipt.pdf", empty_pdf)

    assert not is_dup
    assert record.status == "warning"
    assert record.chunk_count == 0
    assert record.warning_message is not None
    assert "scanned" in record.warning_message.lower()


def test_delete_document_removes_metadata_and_vectors(ingestion_pipeline):
    """Verify document deletion removes chunks from Chroma and records from SQLite."""
    service, meta_store, vstore = ingestion_pipeline

    pdf_bytes = create_sample_pdf(["Content for doc that will be removed."])
    record, _ = service.ingest_file("temp.pdf", pdf_bytes)
    assert vstore.count() >= 1

    # Delete
    deleted = service.delete_document(record.document_id)
    assert deleted is True
    assert meta_store.get_document(record.document_id) is None
    assert vstore.count() == 0
