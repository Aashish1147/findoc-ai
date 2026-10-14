"""Unit tests for SQLite metadata repository."""

from datetime import datetime, timezone
from pathlib import Path
import pytest

from findoc.errors import StorageError
from findoc.schemas import DocumentRecord
from findoc.storage.metadata_store import MetadataStore


@pytest.fixture
def temp_meta_store(tmp_path: Path) -> MetadataStore:
    db_file = tmp_path / "test_metadata.sqlite"
    return MetadataStore(db_file)


def test_register_and_get_document(temp_meta_store: MetadataStore):
    """Verify document registration and retrieval by ID."""
    doc = DocumentRecord(
        document_id="doc-test-1",
        display_name="annual_report_2023.pdf",
        file_type="pdf",
        content_hash="hash-1234567890abcdef",
        status="pending",
        page_count=12,
        chunk_count=0,
    )
    temp_meta_store.register_document(doc)

    retrieved = temp_meta_store.get_document("doc-test-1")
    assert retrieved is not None
    assert retrieved.document_id == "doc-test-1"
    assert retrieved.display_name == "annual_report_2023.pdf"
    assert retrieved.file_type == "pdf"
    assert retrieved.page_count == 12
    assert retrieved.status == "pending"


def test_duplicate_content_hash_raises_storage_error(temp_meta_store: MetadataStore):
    """Verify duplicate content hash insertion is prevented."""
    doc1 = DocumentRecord(
        document_id="doc-1",
        display_name="file1.pdf",
        file_type="pdf",
        content_hash="unique-hash-111111",
        status="indexed",
    )
    temp_meta_store.register_document(doc1)

    # Lookup by hash
    existing = temp_meta_store.get_by_content_hash("unique-hash-111111")
    assert existing is not None
    assert existing.document_id == "doc-1"

    # Attempt inserting another doc with the same content hash
    doc2 = DocumentRecord(
        document_id="doc-2",
        display_name="file2.pdf",
        file_type="pdf",
        content_hash="unique-hash-111111",
        status="pending",
    )
    with pytest.raises(StorageError) as exc_info:
        temp_meta_store.register_document(doc2)
    assert "already exists" in str(exc_info.value)


def test_update_document_status_and_counts(temp_meta_store: MetadataStore):
    """Verify updating lifecycle status, chunk counts, and warnings."""
    doc = DocumentRecord(
        document_id="doc-update",
        display_name="policy.txt",
        file_type="txt",
        content_hash="hash-999999999999",
        status="processing",
    )
    temp_meta_store.register_document(doc)

    temp_meta_store.update_document_status(
        document_id="doc-update",
        status="warning",
        chunk_count=5,
        warning_message="Partially indexed with empty sections.",
    )

    updated = temp_meta_store.get_document("doc-update")
    assert updated is not None
    assert updated.status == "warning"
    assert updated.chunk_count == 5
    assert updated.warning_message == "Partially indexed with empty sections."


def test_list_and_delete_documents(temp_meta_store: MetadataStore):
    """Verify document listing and deletion."""
    doc1 = DocumentRecord(
        document_id="doc-a",
        display_name="doc_a.pdf",
        file_type="pdf",
        content_hash="hash-aaaaaaaaaaaaaaaa",
        status="indexed",
    )
    doc2 = DocumentRecord(
        document_id="doc-b",
        display_name="doc_b.pdf",
        file_type="pdf",
        content_hash="hash-bbbbbbbbbbbbbbbb",
        status="indexed",
    )
    temp_meta_store.register_document(doc1)
    temp_meta_store.register_document(doc2)

    docs = temp_meta_store.list_documents()
    assert len(docs) == 2

    # Delete doc-a
    deleted = temp_meta_store.delete_document("doc-a")
    assert deleted is True

    # Confirm doc-a is gone, doc-b remains
    assert temp_meta_store.get_document("doc-a") is None
    assert temp_meta_store.get_document("doc-b") is not None
    assert len(temp_meta_store.list_documents()) == 1
