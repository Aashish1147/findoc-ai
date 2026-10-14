"""Local embedding service adapter for FinDoc AI.

Provides local vector embedding generation using SentenceTransformers or a deterministic
offline mock for fast, isolated unit testing.
"""

import hashlib
import math
from typing import List, Optional
from findoc.errors import RetrievalError


class BaseEmbeddingService:
    """Abstract interface for FinDoc AI embedding providers."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of document chunk texts."""
        raise NotImplementedError

    def embed_query(self, query: str) -> List[float]:
        """Embed a single search query."""
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        """Return embedding vector dimensionality."""
        raise NotImplementedError


class DeterministicMockEmbeddingService(BaseEmbeddingService):
    """Deterministic, zero-dependency embedding service for unit testing.

    Generates reproducible unit-normalized vectors from text hashes without
    requiring PyTorch weights or internet access.
    """

    def __init__(self, dimension: int = 384) -> None:
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def _embed_single(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self._dim

        # Generate deterministic floats from SHA-256 rounds
        vector: List[float] = []
        seed = text.encode("utf-8")
        current_hash = hashlib.sha256(seed).digest()

        while len(vector) < self._dim:
            for b in current_hash:
                # scale byte 0-255 to -1.0 to 1.0
                vector.append((b / 127.5) - 1.0)
                if len(vector) == self._dim:
                    break
            current_hash = hashlib.sha256(current_hash).digest()

        # Normalize to unit length
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [round(x / norm, 6) for x in vector]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_single(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        return self._embed_single(query)


class LocalSentenceTransformerService(BaseEmbeddingService):
    """Local embedding service using SentenceTransformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._dimension: Optional[int] = None

    def _load_model(self):
        """Lazy load the sentence transformer model into memory."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
                # Determine dimension
                test_emb = self._model.encode("test", convert_to_numpy=True)
                self._dimension = len(test_emb)
            except Exception as e:
                raise RetrievalError(
                    f"Failed to load local embedding model '{self.model_name}': {e}"
                ) from e

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load_model()
        return self._dimension or 384

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        self._load_model()
        try:
            embeddings = self._model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return embeddings.tolist()
        except Exception as e:
            raise RetrievalError(f"Failed to generate document embeddings: {e}") from e

    def embed_query(self, query: str) -> List[float]:
        self._load_model()
        try:
            embedding = self._model.encode(
                query,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return embedding.tolist()
        except Exception as e:
            raise RetrievalError(f"Failed to generate query embedding: {e}") from e


def get_embedding_service(
    model_name: Optional[str] = None,
    mock: bool = False,
) -> BaseEmbeddingService:
    """Factory helper to obtain an embedding service."""
    if mock:
        return DeterministicMockEmbeddingService()
    return LocalSentenceTransformerService(
        model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2"
    )
