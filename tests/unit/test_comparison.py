"""Unit tests for two-document comparison service and source citation isolation."""

import json
from pathlib import Path
import pytest

from findoc.comparison.service import ComparisonService
from findoc.errors import FinDocError
from findoc.generation.provider import BaseLLMProvider, MockLLMProvider
from findoc.retrieval.retriever import RetrievalService
from findoc.schemas import ChunkRecord, DocumentRecord
from findoc.storage.embeddings import DeterministicMockEmbeddingService
from findoc.storage.metadata_store import MetadataStore
from findoc.storage.vector_store import ChromaVectorStore


class CustomComparisonMockProvider(BaseLLMProvider):
    def __init__(self, raw_json: str) -> None:
        self.raw_json = raw_json

    @property
    def model_mode(self) -> str:
        return "custom-comparison-mock"

    def generate(self, user_prompt: str, system_instruction: str) -> str:
        return self.raw_json


@pytest.fixture
def comparison_setup(tmp_path: Path):
    meta_store = MetadataStore(tmp_path / "comp_meta.sqlite")
    vstore = ChromaVectorStore(tmp_path / "comp_chroma")
    embedder = DeterministicMockEmbeddingService(dimension=64)
    retriever = RetrievalService(embedder, vstore)

    # Register metadata for Doc A and Doc B
    doc_a = DocumentRecord(
        document_id="doc_policy_v1",
        display_name="Policy_2022.pdf",
        file_type="pdf",
        content_hash="hash_policy_v1_12345678",
        status="indexed",
    )
    doc_b = DocumentRecord(
        document_id="doc_policy_v2",
        display_name="Policy_2023.pdf",
        file_type="pdf",
        content_hash="hash_policy_v2_12345678",
        status="indexed",
    )
    meta_store.register_document(doc_a)
    meta_store.register_document(doc_b)

    # Index chunks
    chunk_a = ChunkRecord(
        chunk_id="chunk_a_fee",
        document_id="doc_policy_v1",
        source_name="Policy_2022.pdf",
        page_number=2,
        chunk_index=0,
        text="The annual membership fee is $50.00 with no balance exemptions.",
        content_hash="h_a",
    )
    chunk_b = ChunkRecord(
        chunk_id="chunk_b_fee",
        document_id="doc_policy_v2",
        source_name="Policy_2023.pdf",
        page_number=3,
        chunk_index=0,
        text="The annual membership fee is $75.00, waived for balances exceeding $5,000.",
        content_hash="h_b",
    )

    all_chunks = [chunk_a, chunk_b]
    vstore.upsert_chunks(all_chunks, embedder.embed_documents([c.text for c in all_chunks]))

    provider = MockLLMProvider()
    service = ComparisonService(retriever, provider, meta_store)
    return service, vstore, meta_store


def test_comparison_requires_distinct_documents(comparison_setup):
    """Verify that comparing a document to itself raises FinDocError."""
    service, _, _ = comparison_setup
    with pytest.raises(FinDocError) as exc:
        service.compare_documents("doc_policy_v1", "doc_policy_v1", "Compare fees")
    assert "must be distinct" in str(exc.value)


def test_comparison_both_sides_complete(comparison_setup):
    """Verify complete comparison with evidence from Document A and Document B."""
    service, _, _ = comparison_setup

    result, pack_a, pack_b = service.compare_documents(
        document_a_id="doc_policy_v1",
        document_b_id="doc_policy_v2",
        question="Compare the annual membership fees",
        top_k=3,
    )

    assert result.status == "complete"
    assert len(result.findings) >= 1
    assert pack_a.has_usable_evidence
    assert pack_b.has_usable_evidence

    finding = result.findings[0]
    assert finding.category in ("different", "same", "unclear")
    assert len(finding.evidence_a) >= 1
    assert len(finding.evidence_b) >= 1

    # Verify A chunk cited for A, B chunk cited for B
    assert finding.evidence_a[0] == "chunk_a_fee"
    assert finding.evidence_b[0] == "chunk_b_fee"


def test_cross_source_citation_contamination_rejected(tmp_path: Path):
    """Verify citations leaked across documents (e.g. Document A citing Document B) are discarded."""
    meta_store = MetadataStore(tmp_path / "cross_meta.sqlite")
    vstore = ChromaVectorStore(tmp_path / "cross_chroma")
    embedder = DeterministicMockEmbeddingService(dimension=64)
    retriever = RetrievalService(embedder, vstore)

    meta_store.register_document(
        DocumentRecord(
            document_id="doc_x",
            display_name="DocX.pdf",
            file_type="pdf",
            content_hash="hash_x_1234567890",
            status="indexed",
        )
    )
    meta_store.register_document(
        DocumentRecord(
            document_id="doc_y",
            display_name="DocY.pdf",
            file_type="pdf",
            content_hash="hash_y_1234567890",
            status="indexed",
        )
    )

    c_x = ChunkRecord(
        chunk_id="chunk_x1",
        document_id="doc_x",
        source_name="DocX.pdf",
        page_number=1,
        chunk_index=0,
        text="Rule X content.",
        content_hash="hx",
    )
    c_y = ChunkRecord(
        chunk_id="chunk_y1",
        document_id="doc_y",
        source_name="DocY.pdf",
        page_number=1,
        chunk_index=0,
        text="Rule Y content.",
        content_hash="hy",
    )
    vstore.upsert_chunks([c_x, c_y], embedder.embed_documents([c_x.text, c_y.text]))

    # Malicious/hallucinating provider putting Doc B ID into evidence_a and non-existent ID
    contaminated_response = json.dumps(
        {
            "status": "complete",
            "findings": [
                {
                    "category": "different",
                    "topic": "Rules",
                    "summary": "Rules differ.",
                    "evidence_a": ["B1", "E_HALLUCINATED"],  # Invalid for side A!
                    "evidence_b": ["B1"],
                    "caveat": None,
                }
            ],
            "limitations": [],
        }
    )

    provider = CustomComparisonMockProvider(contaminated_response)
    service = ComparisonService(retriever, provider, meta_store)

    result, _, _ = service.compare_documents("doc_x", "doc_y", "Compare rules")

    finding = result.findings[0]
    # B1 and E_HALLUCINATED must be discarded from evidence_a
    assert len(finding.evidence_a) == 0
    # Because evidence_a is empty, category must be downgraded to "unclear"
    assert finding.category == "unclear"
    assert any("Discarded invalid/cross-contaminated" in lim for lim in result.limitations)
