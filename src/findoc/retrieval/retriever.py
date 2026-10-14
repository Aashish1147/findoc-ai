"""Retrieval service for FinDoc AI.

Performs question validation, query embedding, document-filtered semantic search,
overlap deduplication, and evidence pack construction.
"""

from typing import List, Optional, Tuple
from findoc.errors import RetrievalError
from findoc.retrieval.evidence import EvidencePack, build_evidence_pack
from findoc.schemas import ChunkRecord
from findoc.storage.embeddings import BaseEmbeddingService
from findoc.storage.vector_store import ChromaVectorStore


def calculate_lexical_overlap(text1: str, text2: str) -> float:
    """Calculate simple token overlap ratio between two passages."""
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1.intersection(tokens2)
    smaller = min(len(tokens1), len(tokens2))
    return len(intersection) / smaller if smaller > 0 else 0.0


def deduplicate_overlap_chunks(
    candidates: List[Tuple[ChunkRecord, float]],
    overlap_threshold: float = 0.85,
) -> List[Tuple[ChunkRecord, float]]:
    """Filter out near-identical adjacent overlap chunks from the same page."""
    deduped: List[Tuple[ChunkRecord, float]] = []

    for chunk, score in candidates:
        is_duplicate = False
        for kept_chunk, _ in deduped:
            # Check if from same document and page
            if (
                chunk.document_id == kept_chunk.document_id
                and chunk.page_number == kept_chunk.page_number
            ):
                # If identical hash or high token overlap
                if chunk.content_hash == kept_chunk.content_hash:
                    is_duplicate = True
                    break
                if calculate_lexical_overlap(chunk.text, kept_chunk.text) >= overlap_threshold:
                    is_duplicate = True
                    break

        if not is_duplicate:
            deduped.append((chunk, score))

    return deduped


class RetrievalService:
    """Coordinates semantic retrieval and evidence extraction."""

    def __init__(
        self,
        embedding_service: BaseEmbeddingService,
        vector_store: ChromaVectorStore,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(
        self,
        question: str,
        document_ids: List[str],
        top_k: int = 5,
    ) -> EvidencePack:
        """Retrieve relevant evidence passages from the selected documents.

        Args:
            question: User's natural language question.
            document_ids: Explicit list of document IDs to search across.
            top_k: Number of ranked passages to return.

        Returns:
            EvidencePack containing verified candidates and metadata.

        Raises:
            RetrievalError: If inputs are invalid or search fails.
        """
        clean_q = question.strip() if question else ""
        if not clean_q:
            raise RetrievalError("Search question cannot be empty or whitespace.")

        if not document_ids or len(document_ids) == 0:
            raise RetrievalError(
                "At least one document must be selected. Document-ID filtering is required."
            )

        if top_k <= 0:
            raise RetrievalError(f"top_k must be a positive integer, got {top_k}")

        clean_doc_ids = [d.strip() for d in document_ids if d and d.strip()]
        if not clean_doc_ids:
            raise RetrievalError("No valid document IDs provided in search filter.")

        # 1. Embed query
        query_vector = self.embedding_service.embed_query(clean_q)

        # 2. Query vector store with generous candidate pool for deduplication
        candidate_k = min(top_k * 2, 20)
        raw_candidates = self.vector_store.query_similar(
            query_embedding=query_vector,
            top_k=candidate_k,
            document_ids=clean_doc_ids,
        )

        # 3. Deduplicate redundant overlap passages
        deduped = deduplicate_overlap_chunks(raw_candidates, overlap_threshold=0.85)

        # 4. Limit to requested top_k
        final_candidates = deduped[:top_k]

        # 5. Build and return typed EvidencePack
        return build_evidence_pack(
            question=clean_q,
            selected_document_ids=clean_doc_ids,
            retrieved_chunks=final_candidates,
        )
