"""Unit tests for local embedding service and deterministic mock adapter."""

import math
from findoc.storage.embeddings import (
    DeterministicMockEmbeddingService,
    get_embedding_service,
)


def test_mock_embedding_dimension_and_normalization():
    """Verify mock embedding outputs correct dimension and normalized unit vectors."""
    service = DeterministicMockEmbeddingService(dimension=384)
    assert service.dimension == 384

    vec = service.embed_query("Annual report revenue 2023")
    assert len(vec) == 384

    # Verify unit length (magnitude ≈ 1.0)
    magnitude = math.sqrt(sum(x * x for x in vec))
    assert abs(magnitude - 1.0) < 0.01


def test_mock_embedding_determinism():
    """Verify that identical inputs produce identical embeddings, and different inputs differ."""
    service = DeterministicMockEmbeddingService(dimension=128)
    query = "Operating expenses in Q4"

    vec1 = service.embed_query(query)
    vec2 = service.embed_query(query)
    assert vec1 == vec2

    vec_diff = service.embed_query("Different question regarding tax liabilities")
    assert vec1 != vec_diff


def test_mock_embedding_batch_documents():
    """Verify batch embedding of documents."""
    service = DeterministicMockEmbeddingService(dimension=64)
    texts = [
        "First chunk text about assets.",
        "Second chunk text about liabilities.",
        "Third chunk text about equity.",
    ]

    embeddings = service.embed_documents(texts)
    assert len(embeddings) == 3
    assert len(embeddings[0]) == 64
    assert len(embeddings[1]) == 64
    assert len(embeddings[2]) == 64


def test_mock_embedding_empty_batch():
    """Verify empty document list returns empty list."""
    service = DeterministicMockEmbeddingService()
    assert service.embed_documents([]) == []


def test_embedding_factory_mock():
    """Verify get_embedding_service returns DeterministicMockEmbeddingService when requested."""
    service = get_embedding_service(mock=True)
    assert isinstance(service, DeterministicMockEmbeddingService)
    assert service.dimension == 384
