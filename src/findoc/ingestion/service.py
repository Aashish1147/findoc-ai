"""Ingestion coordinator service for FinDoc AI.

Orchestrates file validation, page extraction, chunking, embedding generation,
persistent vector indexing, and metadata tracking.
"""

from pathlib import Path
from typing import List, Optional, Tuple

from findoc.errors import FileValidationError, StorageError
from findoc.ingestion.chunker import chunk_page_texts
from findoc.ingestion.parser import parse_document_bytes
from findoc.ingestion.validation import validate_file_upload
from findoc.schemas import DocumentRecord
from findoc.storage.embeddings import BaseEmbeddingService
from findoc.storage.metadata_store import MetadataStore
from findoc.storage.vector_store import ChromaVectorStore


class IngestionService:
    """Manages full document ingestion lifecycle from raw bytes to persistent vector index."""

    def __init__(
        self,
        metadata_store: MetadataStore,
        vector_store: ChromaVectorStore,
        embedding_service: BaseEmbeddingService,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        max_file_mb: int = 20,
    ) -> None:
        self.metadata_store = metadata_store
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_file_mb = max_file_mb

    def ingest_file(
        self,
        raw_filename: str,
        content: bytes,
    ) -> Tuple[DocumentRecord, bool]:
        """Process and index an uploaded document.

        Returns:
            Tuple of (DocumentRecord, is_duplicate)
        """
        # 1. Validate file format, size, signature, and generate deterministic ID
        document_id, sanitized_name, file_type, content_hash = validate_file_upload(
            raw_filename=raw_filename,
            content=content,
            max_file_mb=self.max_file_mb,
        )

        # 2. Duplicate Detection: check if identical content has already been indexed
        existing_doc = self.metadata_store.get_by_content_hash(content_hash)
        if existing_doc:
            if existing_doc.status in ("indexed", "warning"):
                return existing_doc, True

        # 3. Register initial record in SQLite as 'processing'
        doc_record = DocumentRecord(
            document_id=document_id,
            display_name=sanitized_name,
            file_type=file_type,
            content_hash=content_hash,
            status="processing",
            page_count=None,
            chunk_count=0,
        )

        try:
            if existing_doc:
                self.metadata_store.update_document_status(
                    document_id=document_id,
                    status="processing",
                )
            else:
                self.metadata_store.register_document(doc_record)
        except Exception:
            # If already exists in DB, update status
            self.metadata_store.update_document_status(
                document_id=document_id,
                status="processing",
            )

        try:
            # 4. Parse pages page-by-page
            pages, parse_warning = parse_document_bytes(
                content=content,
                file_type=file_type,
                document_id=document_id,
                source_name=sanitized_name,
            )

            page_count = len(pages) if file_type == "pdf" else None

            # 5. Chunk page text while preserving page provenance
            chunks = chunk_page_texts(
                pages=pages,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

            # If no chunks produced (e.g. empty or scanned PDF)
            if not chunks:
                status = "warning"
                warning_msg = parse_warning or "No extractable text chunks could be produced."
                self.metadata_store.update_document_status(
                    document_id=document_id,
                    status=status,
                    chunk_count=0,
                    page_count=page_count,
                    warning_message=warning_msg,
                )
                return self.metadata_store.get_document(document_id) or doc_record, False

            # 6. Generate local embeddings
            chunk_texts = [c.text for c in chunks]
            embeddings = self.embedding_service.embed_documents(chunk_texts)

            # 7. Upsert chunks and vectors into Chroma
            self.vector_store.upsert_chunks(chunks, embeddings)

            # 8. Mark document as successfully indexed (or warning if minor issues noted)
            final_status = "warning" if parse_warning else "indexed"
            self.metadata_store.update_document_status(
                document_id=document_id,
                status=final_status,
                chunk_count=len(chunks),
                page_count=page_count,
                warning_message=parse_warning,
            )

            updated_record = self.metadata_store.get_document(document_id)
            return updated_record or doc_record, False

        except Exception as e:
            # Handle failure gracefully by updating document status to 'failed'
            self.metadata_store.update_document_status(
                document_id=document_id,
                status="failed",
                warning_message=f"Ingestion failed: {e}",
            )
            failed_record = self.metadata_store.get_document(document_id)
            raise StorageError(f"Failed to ingest document '{sanitized_name}': {e}") from e

    def delete_document(self, document_id: str) -> bool:
        """Remove document chunks from Chroma and record from SQLite consistently."""
        try:
            # Remove vector chunks
            self.vector_store.delete_document(document_id)
            # Remove metadata record
            return self.metadata_store.delete_document(document_id)
        except Exception as e:
            raise StorageError(f"Failed to delete document '{document_id}': {e}") from e

    def list_documents(self) -> List[DocumentRecord]:
        """List all indexed document records."""
        return self.metadata_store.list_documents()
