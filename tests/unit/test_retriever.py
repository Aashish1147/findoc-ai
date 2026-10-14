"""Unit tests for RetrievalService validation, filtering, deduplication, and evidence gating."""

from pathlib import Path
import pytest

from findoc.errors import RetrievalError
from findoc.retrieval.retriever import (
    RetrievalService,
    calculate_lexical_overlap,
    deduplicate_overlap_chunks,
)
from findoc.schemas import ChunkRecord
from findoc.storage.embeddings import DeterministicMockEmbeddingService
from findoc.storage.vector_store import ChromaVectorStore


@pytest.fixture
def mock_retrieval_setup(tmp_path: Path):
    embedder = DeterministicMockEmbeddingService(dimension=64)
    vstore = ChromaVectorStore(tmp_path / "retrieval_chroma")
    service = RetrievalService(embedder, vstore)
    return embedder, vstore, service


def test_retriever_validation_errors(mock_retrieval_setup):
    """Verify input validation: empty question, empty documents, invalid top_k."""
    _, _, service = mock_retrieval_setup

    with pytest.raises(RetrievalError) as exc:
        service.retrieve("", document_ids=["doc-1"])
    assert "cannot be empty" in str(exc.value)

    with pytest.raises(RetrievalError) as exc:
        service.retrieve("   ", document_ids=["doc-1"])
    assert "cannot be empty" in str(exc.value)

    with pytest.raises(RetrievalError) as exc:
        service.retrieve("What is revenue?", document_ids=[])
    assert "At least one document must be selected" in str(exc.value)

    with pytest.raises(RetrievalError) as exc:
        service.retrieve("What is revenue?", document_ids=["doc-1"], top_k=0)
    assert "top_k must be a positive integer" in str(exc.value)


def test_retriever_document_filtering(mock_retrieval_setup):
    """Verify retrieval filters candidates strictly by selected document IDs."""
    embedder, vstore, service = mock_retrieval_setup

    chunks_doc1 = [
        ChunkRecord(
            chunk_id="c1",
            document_id="doc-A",
            source_name="DocA.pdf",
            page_number=1,
            chunk_index=0,
            text="Document A reports annual dividend of $2.50 per share.",
            content_hash="hash-c1",
        )
    ]
    chunks_doc2 = [
        ChunkRecord(
            chunk_id="c2",
            document_id="doc-B",
            source_name="DocB.pdf",
            page_number=1,
            chunk_index=0,
            text="Document B reports annual dividend of $4.10 per share.",
            content_hash="hash-c2",
        )
    ]

    all_chunks = chunks_doc1 + chunks_doc2
    vstore.upsert_chunks(all_chunks, embedder.embed_documents([c.text for c in all_chunks]))

    # Retrieve only from doc-A
    pack_a = service.retrieve("dividend", document_ids=["doc-A"], top_k=5)
    assert pack_a.has_usable_evidence
    assert len(pack_a.items) == 1
    assert pack_a.items[0].document_id == "doc-A"
    assert "$2.50" in pack_a.items[0].text

    # Retrieve only from doc-B
    pack_b = service.retrieve("dividend", document_ids=["doc-B"], top_k=5)
    assert pack_b.has_usable_evidence
    assert len(pack_b.items) == 1
    assert pack_b.items[0].document_id == "doc-B"
    assert "$4.10" in pack_b.items[0].text


def test_retriever_evidence_gate_empty_results(mock_retrieval_setup):
    """Verify that querying an empty vector index returns an EvidencePack with has_usable_evidence=False."""
    _, _, service = mock_retrieval_setup

    pack = service.retrieve("What are the quarterly profits?", document_ids=["doc-empty"], top_k=5)
    assert not pack.has_usable_evidence
    assert len(pack.items) == 0


def test_deduplicate_overlap_chunks():
    """Verify deduplicate_overlap_chunks prunes near-identical chunks from the same page."""
    base_text = "The quarterly earnings report indicates a 12% rise in gross commercial lending balances."
    similar_text = "The quarterly earnings report indicates a 12% rise in gross commercial lending balances. Further notes follow."

    c1 = (
        ChunkRecord(
            chunk_id="c1",
            document_id="doc-1",
            source_name="rep.pdf",
            page_number=2,
            chunk_index=0,
            text=base_text,
            content_hash="h1",
        ),
        0.1,
    )
    c2 = (
        ChunkRecord(
            chunk_id="c2",
            document_id="doc-1",
            source_name="rep.pdf",
            page_number=2,
            chunk_index=1,
            text=similar_text,
            content_hash="h2",
        ),
        0.12,
    )
    # Distinct chunk from another page
    c3 = (
        ChunkRecord(
            chunk_id="c3",
            document_id="doc-1",
            source_name="rep.pdf",
            page_number=3,
            chunk_index=2,
            text="Executive officers list and board member compensation disclosures.",
            content_hash="h3",
        ),
        0.2,
    )

    deduped = deduplicate_overlap_chunks([c1, c2, c3], overlap_threshold=0.8)
    assert len(deduped) == 2
    assert deduped[0][0].chunk_id == "c1"
    assert deduped[1][0].chunk_id == "c3"
