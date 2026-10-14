"""Pydantic schemas and data models for FinDoc AI.

Defines validated contracts for documents, pages, chunks, citations, answers,
and comparison results.
"""

from datetime import datetime, timezone
from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DocumentStatus = Literal["pending", "processing", "indexed", "warning", "failed"]
FileType = Literal["pdf", "txt", "md"]
AnswerStatus = Literal["answered", "insufficient_evidence", "error"]
ComparisonCategory = Literal["same", "different", "only_in_a", "only_in_b", "unclear"]
ComparisonStatus = Literal["complete", "partial_evidence", "insufficient_evidence", "error"]


class DocumentRecord(BaseModel):
    """Metadata record for an uploaded document."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(..., description="Unique internal document identifier (e.g. SHA-based or UUID)")
    display_name: str = Field(..., min_length=1, description="Sanitized filename for user-facing display")
    file_type: FileType = Field(..., description="Normalized document file type")
    content_hash: str = Field(..., min_length=16, description="Cryptographic SHA-256 hash of original file bytes")
    status: DocumentStatus = Field(default="pending", description="Current ingestion lifecycle status")
    page_count: Optional[int] = Field(default=None, ge=1, description="Number of physical PDF pages, or None for text files")
    chunk_count: int = Field(default=0, ge=0, description="Total number of searchable chunks indexed")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when document record was registered",
    )
    warning_message: Optional[str] = Field(
        default=None,
        description="Actionable warning message (e.g. empty pages or scanned content without selectable text)",
    )


class PageText(BaseModel):
    """Extracted text unit from a single physical document page or text file."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(..., description="ID of parent document")
    source_name: str = Field(..., description="Sanitized source display filename")
    page_number: Optional[int] = Field(
        default=None,
        ge=1,
        description="1-based physical PDF page number, or None for unpaged text files",
    )
    text: str = Field(..., description="Extracted textual content")


class ChunkRecord(BaseModel):
    """Searchable text chunk carrying traceable source provenance."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(..., description="Deterministic or unique chunk identifier")
    document_id: str = Field(..., description="ID of parent document")
    source_name: str = Field(..., description="Source display filename")
    page_number: Optional[int] = Field(
        default=None,
        ge=1,
        description="Physical page number where chunk originates, or None for text files",
    )
    chunk_index: int = Field(..., ge=0, description="Zero-based sequence index within document ingestion")
    text: str = Field(..., min_length=1, description="Searchable text content of the chunk")
    content_hash: str = Field(..., description="Hash of chunk text or document for consistency checking")


class Citation(BaseModel):
    """Verified evidence citation referencing a specific retrieved passage."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(..., description="Verified chunk ID from the retrieval evidence pack")
    source_name: str = Field(..., description="Display filename derived from backend chunk metadata")
    page_number: Optional[int] = Field(
        default=None,
        ge=1,
        description="Physical page number derived from backend chunk metadata",
    )
    quote: str = Field(..., min_length=1, description="Exact or concise excerpt supporting the claim")


class AnswerResult(BaseModel):
    """Structured response from the FinDoc AI Q&A service."""

    model_config = ConfigDict(frozen=True)

    status: AnswerStatus = Field(..., description="Outcome of the Q&A request")
    answer: str = Field(..., description="Synthesized grounded answer or insufficient evidence explanation")
    citations: list[Citation] = Field(
        default_factory=list,
        description="List of verified citations backing the claims in the answer",
    )
    retrieved_chunk_ids: list[str] = Field(
        default_factory=list,
        description="All chunk IDs provided in the evidence pack for this query",
    )
    model_mode: str = Field(..., description="Model descriptor (e.g. 'mock' or provider model ID)")
    limitations: list[str] = Field(
        default_factory=list,
        description="Transparency notes, uncertainties, or boundary disclaimers",
    )

    @model_validator(mode="after")
    def validate_citations_against_retrieved(self) -> "AnswerResult":
        """Ensure that every citation chunk_id is present in retrieved_chunk_ids if provided."""
        if self.retrieved_chunk_ids:
            for citation in self.citations:
                if citation.chunk_id not in self.retrieved_chunk_ids:
                    raise ValueError(
                        f"Citation references chunk_id '{citation.chunk_id}' which was not in retrieved evidence pack"
                    )
        return self


class ComparisonFinding(BaseModel):
    """A single compared topic or clause between Document A and Document B."""

    model_config = ConfigDict(frozen=True)

    category: ComparisonCategory = Field(
        ...,
        description="Classification: 'same', 'different', 'only_in_a', 'only_in_b', or 'unclear'",
    )
    topic: str = Field(..., min_length=1, description="Topic, fee, requirement, or clause heading")
    summary: str = Field(..., min_length=1, description="Concise explanation of the comparison finding")
    evidence_a: list[str] = Field(
        default_factory=list,
        description="Retrieved chunk IDs from Document A supporting this finding",
    )
    evidence_b: list[str] = Field(
        default_factory=list,
        description="Retrieved chunk IDs from Document B supporting this finding",
    )
    caveat: Optional[str] = Field(
        default=None,
        description="Clarification or limitation (e.g. unverified absence in the other document)",
    )


class ComparisonResult(BaseModel):
    """Structured comparison between two documents."""

    model_config = ConfigDict(frozen=True)

    status: ComparisonStatus = Field(..., description="Status of the comparison")
    document_a_id: str = Field(..., description="Internal ID of Document A")
    document_b_id: str = Field(..., description="Internal ID of Document B")
    question: str = Field(..., min_length=1, description="Comparison focus question or topic")
    findings: list[ComparisonFinding] = Field(
        default_factory=list,
        description="Structured findings comparing Document A and Document B",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Overall caveats, e.g. notice that comparison is an assistive reading aid",
    )
