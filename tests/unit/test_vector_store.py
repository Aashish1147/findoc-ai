"""Unit and integration tests for persistent Chroma vector store adapter."""

from pathlib import Path
import pytest

from findoc.schemas import ChunkRecord
from findoc.storage.embeddings import DeterministicMockEmbeddingService
from findoc.storage.vector_store import ChromaVectorStore


@pytest.fixture
def temp_vector_store(tmp_path: Path) -> ChromaVectorStore:
    chroma_dir = tmp_path / "test_chroma"
    return ChromaVectorStore(chroma_dir)


def test_upsert_and_count(temp_vector_store: ChromaVectorStore):
    """Verify upserting chunk records updates collection count."""
    embedder = DeterministicMockEmbeddingService(dimension=64)
    chunks = [
        ChunkRecord(
            chunk_id="chunk-1",
            document_id="doc-1",
            source_name="report.pdf",
            page_number=1,
            chunk_index=0,
            text="Revenue increased to $100M.",
            content_hash="hash-1",
        ),
        ChunkRecord(
            chunk_id="chunk-2",
            document_id="doc-1",
            source_name="report.pdf",
            page_number=2,
            chunk_index=1,
            text="Net profit margin reached 18.5%.",
            content_hash="hash-2",
        ),
    ]
    embeddings = embedder.embed_documents([c.text for c in chunks])

    temp_vector_store.upsert_chunks(chunks, embeddings)
    assert temp_vector_store.count() == 2


def test_query_strictly_filters_by_document_id(temp_vector_store: ChromaVectorStore):
    """Verify that querying for doc-A never returns chunks from doc-B."""
    embedder = DeterministicMockEmbeddingService(dimension=64)

    # Chunks from doc-A
    chunks_a = [
        ChunkRecord(
            chunk_id="chunk-a-1",
            document_id="doc-A",
            source_name="doc_a.pdf",
            page_number=1,
            chunk_index=0,
            text="Interest rate is 4.5% annually.",
            content_hash="hash-a-1",
        )
    ]
    # Chunks from doc-B
    chunks_b = [
        ChunkRecord(
            chunk_id="chunk-b-1",
            document_id="doc-B",
            source_name="doc_b.pdf",
            page_number=1,
            chunk_index=0,
            text="Interest rate is 6.0% annually.",
            content_hash="hash-b-1",
        )
    ]

    all_chunks = chunks_a + chunks_b
    embeddings = embedder.embed_documents([c.text for c in all_chunks])
    temp_vector_store.upsert_chunks(all_chunks, embeddings)

    # Query embedding matching interest rate
    q_emb = embedder.embed_query("What is the interest rate?")

    # Query filtered strictly to doc-A
    results_a = temp_vector_store.query_similar(q_emb, top_k=5, document_ids=["doc-A"])
    assert len(results_a) == 1
    assert results_a[0][0].document_id == "doc-A"
    assert results_a[0][0].chunk_id == "chunk-a-1"

    # Query filtered strictly to doc-B
    results_b = temp_vector_store.query_similar(q_emb, top_k=5, document_ids=["doc-B"])
    assert len(results_b) == 1
    assert results_b[0][0].document_id == "doc-B"
    assert results_b[0][0].chunk_id == "chunk-b-1"


def test_unpaged_text_file_metadata_restoration(temp_vector_store: ChromaVectorStore):
    """Verify text file chunks with page_number=None are restored with page_number=None."""
    embedder = DeterministicMockEmbeddingService(dimension=64)
    chunks = [
        ChunkRecord(
            chunk_id="chunk-txt-1",
            document_id="doc-text",
            source_name="terms.txt",
            page_number=None,
            chunk_index=0,
            text="General eligibility criteria for loan accounts.",
            content_hash="hash-txt",
        )
    ]
    embeddings = embedder.embed_documents([c.text for c in chunks])
    temp_vector_store.upsert_chunks(chunks, embeddings)

    q_emb = embedder.embed_query("eligibility")
    results = temp_vector_store.query_similar(q_emb, top_k=1, document_ids=["doc-text"])
    assert len(results) == 1
    assert results[0][0].page_number is None


def test_delete_document_chunks(temp_vector_store: ChromaVectorStore):
    """Verify deleting a document removes its chunks while preserving other documents."""
    embedder = DeterministicMockEmbeddingService(dimension=64)
    chunks = [
        ChunkRecord(
            chunk_id="chunk-del-1",
            document_id="doc-delete-me",
            source_name="delete.pdf",
            page_number=1,
            chunk_index=0,
            text="Temporary document content.",
            content_hash="hash-del",
        ),
        ChunkRecord(
            chunk_id="chunk-keep-1",
            document_id="doc-keep-me",
            source_name="keep.pdf",
            page_number=1,
            chunk_index=0,
            text="Permanent document content.",
            content_hash="hash-keep",
        ),
    ]
    embeddings = embedder.embed_documents([c.text for c in chunks])
    temp_vector_store.upsert_chunks(chunks, embeddings)
    assert temp_vector_store.count() == 2

    # Delete doc-delete-me
    temp_vector_store.delete_document("doc-delete-me")
    assert temp_vector_store.count() == 1

    # Verify query
    q_emb = embedder.embed_query("document content")
    results = temp_vector_store.query_similar(q_emb, top_k=5)
    assert len(results) == 1
    assert results[0][0].document_id == "doc-keep-me"


def test_cross_session_persistence(tmp_path: Path):
    """Verify persistent client re-reads vector index from disk after recreation."""
    chroma_dir = tmp_path / "persistent_chroma_test"
    embedder = DeterministicMockEmbeddingService(dimension=64)

    # First session: write to disk
    store1 = ChromaVectorStore(chroma_dir)
    chunk = ChunkRecord(
        chunk_id="persisted-chunk",
        document_id="doc-persistent",
        source_name="report.pdf",
        page_number=5,
        chunk_index=0,
        text="Persistent data across restarts.",
        content_hash="hash-p",
    )
    store1.upsert_chunks([chunk], embedder.embed_documents([chunk.text]))
    assert store1.count() == 1
    del store1

    # Second session: open existing directory
    store2 = ChromaVectorStore(chroma_dir)
    assert store2.count() == 1
    q_emb = embedder.embed_query("Persistent data")
    results = store2.query_similar(q_emb, top_k=1, document_ids=["doc-persistent"])
    assert len(results) == 1
    assert results[0][0].chunk_id == "persisted-chunk"
    assert results[0][0].page_number == 5
