"""Evidence pack definitions and citation mapping utilities for FinDoc AI.

Structures retrieved candidate passages into trusted, numbered evidence packs
and derives display citations exclusively from backend metadata.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from findoc.schemas import ChunkRecord, Citation


class EvidenceItem(BaseModel):
    """A single retrieved evidence passage carrying verified source metadata."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str = Field(..., description="Query-scoped identifier like 'E1', 'E2'")
    chunk_id: str = Field(..., description="Underlying stable chunk ID from storage")
    document_id: str = Field(..., description="Parent document identifier")
    source_name: str = Field(..., description="Display filename from backend records")
    page_number: Optional[int] = Field(
        default=None,
        description="Physical page number for PDF, or None for text files",
    )
    text: str = Field(..., description="Exact textual content of the chunk")
    search_score: float = Field(
        default=0.0,
        description="Raw vector distance score (not a probability of correctness)",
    )


class EvidencePack(BaseModel):
    """Collection of retrieved evidence items compiled for a single user query."""

    model_config = ConfigDict(frozen=True)

    question: str = Field(..., min_length=1, description="User's original question")
    selected_document_ids: List[str] = Field(
        default_factory=list,
        description="Documents selected by user for search scope",
    )
    items: List[EvidenceItem] = Field(
        default_factory=list,
        description="Ranked candidate evidence passages",
    )

    @property
    def has_usable_evidence(self) -> bool:
        """Evidence gate check: returns True if at least one evidence passage is present."""
        return len(self.items) > 0

    @property
    def chunk_ids(self) -> List[str]:
        """List of all chunk IDs in this evidence pack."""
        return [item.chunk_id for item in self.items]

    @property
    def evidence_ids(self) -> List[str]:
        """List of all evidence IDs in this evidence pack."""
        return [item.evidence_id for item in self.items]

    def get_by_chunk_id(self, chunk_id: str) -> Optional[EvidenceItem]:
        """Lookup evidence item by stable chunk ID."""
        for item in self.items:
            if item.chunk_id == chunk_id:
                return item
        return None

    def get_by_evidence_id(self, evidence_id: str) -> Optional[EvidenceItem]:
        """Lookup evidence item by query-scoped evidence ID (e.g. 'E1' or '[E1]')."""
        clean_id = evidence_id.strip("[]").strip()
        for item in self.items:
            if item.evidence_id == clean_id:
                return item
        return None

    def to_citation(self, id_reference: str, quote: str) -> Optional[Citation]:
        """Map a model-returned ID reference (evidence_id or chunk_id) to a validated Citation.

        Derives filename and page number strictly from backend records rather than LLM text.
        Returns None if ID reference does not match any retrieved evidence item.
        """
        item = self.get_by_evidence_id(id_reference) or self.get_by_chunk_id(id_reference)
        if not item:
            return None

        # Clean quotation string
        clean_quote = quote.strip().strip('"').strip("'")
        if not clean_quote:
            # Fallback to leading snippet from trusted chunk text
            clean_quote = item.text[:120].strip() + ("..." if len(item.text) > 120 else "")

        return Citation(
            chunk_id=item.chunk_id,
            source_name=item.source_name,
            page_number=item.page_number,
            quote=clean_quote,
        )

    def format_for_prompt(self) -> str:
        """Format evidence items into a delimited, untrusted context block for the LLM."""
        if not self.items:
            return "NO EVIDENCE AVAILABLE."

        formatted_blocks: List[str] = []
        for item in self.items:
            page_str = f"Page {item.page_number}" if item.page_number is not None else "Text File"
            block = (
                f"--- EVIDENCE [{item.evidence_id}] ---\n"
                f"Source: {item.source_name} | Location: {page_str} | Chunk ID: {item.chunk_id}\n"
                f"Content:\n{item.text}\n"
                f"--- END EVIDENCE [{item.evidence_id}] ---"
            )
            formatted_blocks.append(block)

        return "\n\n".join(formatted_blocks)


def build_evidence_pack(
    question: str,
    selected_document_ids: List[str],
    retrieved_chunks: List[tuple[ChunkRecord, float]],
) -> EvidencePack:
    """Construct an EvidencePack from ranked retrieval candidates."""
    items: List[EvidenceItem] = []
    for idx, (chunk, score) in enumerate(retrieved_chunks):
        evidence_id = f"E{idx + 1}"
        items.append(
            EvidenceItem(
                evidence_id=evidence_id,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                source_name=chunk.source_name,
                page_number=chunk.page_number,
                text=chunk.text,
                search_score=round(score, 4),
            )
        )

    return EvidencePack(
        question=question,
        selected_document_ids=selected_document_ids,
        items=items,
    )
