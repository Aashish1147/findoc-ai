"""Persistent Chroma vector store adapter for FinDoc AI.

Encapsulates ChromaDB operations, embedding indexing, document-filtered querying,
and document chunk deletion.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings

from findoc.errors import StorageError
from findoc.schemas import ChunkRecord


class ChromaVectorStore:
    """Isolated Chroma vector store adapter supporting filtered similarity search."""

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str = "findoc_chunks",
    ) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._init_client()

    def _init_client(self) -> None:
        """Initialize Chroma PersistentClient and get/create collection."""
        try:
            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    is_persistent=True,
                ),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            raise StorageError(f"Failed to initialize Chroma vector store: {e}") from e

    def upsert_chunks(
        self,
        chunks: List[ChunkRecord],
        embeddings: List[List[float]],
    ) -> None:
        """Upsert a batch of chunk records and corresponding embedding vectors.

        Raises:
            StorageError: If chunk and embedding counts differ or insertion fails.
        """
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise StorageError(
                f"Mismatch between chunks count ({len(chunks)}) and embeddings count ({len(embeddings)})."
            )

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            metadatas.append(
                {
                    "document_id": chunk.document_id,
                    "source_name": chunk.source_name,
                    # Chroma requires primitive metadata; use -1 for unpaged text files
                    "page_number": chunk.page_number if chunk.page_number is not None else -1,
                    "chunk_index": chunk.chunk_index,
                    "content_hash": chunk.content_hash,
                }
            )

        try:
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as e:
            raise StorageError(f"Failed to upsert chunks to Chroma: {e}") from e

    def query_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
    ) -> List[Tuple[ChunkRecord, float]]:
        """Search Chroma for the most similar chunks, strictly filtered by document_ids if specified.

        Returns:
            List of (ChunkRecord, distance) tuples ordered by similarity.
        """
        if top_k <= 0:
            return []

        where_filter: Optional[Dict[str, Any]] = None
        if document_ids:
            clean_ids = [d.strip() for d in document_ids if d and d.strip()]
            if len(clean_ids) == 1:
                where_filter = {"document_id": clean_ids[0]}
            elif len(clean_ids) > 1:
                where_filter = {"document_id": {"$in": clean_ids}}

        try:
            total_items = self._collection.count()
            if total_items == 0:
                return []

            actual_k = min(top_k, total_items)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=actual_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            retrieved: List[Tuple[ChunkRecord, float]] = []
            if not results or not results.get("ids") or not results["ids"][0]:
                return []

            res_ids = results["ids"][0]
            res_docs = results["documents"][0] if results.get("documents") else []
            res_metas = results["metadatas"][0] if results.get("metadatas") else []
            res_dists = results["distances"][0] if results.get("distances") else []

            for i, chunk_id in enumerate(res_ids):
                meta = res_metas[i] if i < len(res_metas) else {}
                doc_text = res_docs[i] if i < len(res_docs) else ""
                dist = float(res_dists[i]) if i < len(res_dists) else 0.0

                raw_page = meta.get("page_number", -1)
                page_number = None if raw_page == -1 else int(raw_page)

                chunk = ChunkRecord(
                    chunk_id=chunk_id,
                    document_id=str(meta.get("document_id", "")),
                    source_name=str(meta.get("source_name", "")),
                    page_number=page_number,
                    chunk_index=int(meta.get("chunk_index", 0)),
                    text=doc_text,
                    content_hash=str(meta.get("content_hash", "")),
                )
                retrieved.append((chunk, dist))

            return retrieved
        except Exception as e:
            raise StorageError(f"Chroma similarity search failed: {e}") from e

    def delete_document(self, document_id: str) -> None:
        """Delete all chunks belonging to a document ID from the vector store."""
        try:
            self._collection.delete(where={"document_id": document_id})
        except Exception as e:
            raise StorageError(f"Failed to delete chunks for document '{document_id}': {e}") from e

    def count(self) -> int:
        """Return total number of chunks currently stored in the collection."""
        try:
            return self._collection.count()
        except Exception as e:
            raise StorageError(f"Failed to retrieve collection count: {e}") from e

    def health_check(self) -> Dict[str, Any]:
        """Perform a quick operational health check on the vector store."""
        return {
            "status": "healthy",
            "persist_dir": str(self.persist_dir),
            "collection_name": self.collection_name,
            "total_chunks": self.count(),
        }
