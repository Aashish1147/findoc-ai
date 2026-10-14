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


class EvalTestCase(BaseModel):
    """A single labelled test case in the synthetic evaluation benchmark."""

    model_config = ConfigDict(frozen=True)

    case_id: str = Field(..., description="Unique test case identifier (e.g. 'EVAL-001')")
    category: str = Field(
        ...,
        description="Category: direct_lookup, paraphrase, negative_exception, unanswerable, prompt_injection, etc.",
    )
    question: str = Field(..., min_length=1, description="Evaluation prompt or question")
    target_document_names: list[str] = Field(
        ...,
        min_length=1,
        description="Filenames of target documents required for answering",
    )
    expected_status: AnswerStatus = Field(..., description="Expected status: 'answered' or 'insufficient_evidence'")
    expected_page: Optional[int] = Field(
        default=None,
        ge=1,
        description="Expected 1-based page number containing the evidence, if known",
    )
    expected_evidence_marker: Optional[str] = Field(
        default=None,
        description="Text substring expected in at least one retrieved chunk for hit@k evaluation",
    )
    expected_answer_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords or values expected in grounded answer",
    )
    notes: str = Field(default="", description="Explanatory notes regarding the test case")


class EvalCaseResult(BaseModel):
    """Outcome and diagnostics for a single evaluation test case."""

    model_config = ConfigDict(frozen=True)

    case_id: str = Field(..., description="Unique test case identifier")
    category: str = Field(..., description="Evaluation category")
    question: str = Field(..., description="Question evaluated")
    expected_status: AnswerStatus = Field(..., description="Expected status")
    actual_status: AnswerStatus = Field(..., description="Observed status")
    retrieval_hit: bool = Field(
        ...,
        description="Whether expected evidence was retrieved in top-k (or True if unanswerable)",
    )
    citation_validity: bool = Field(
        ...,
        description="Whether all returned citations correspond to verified backend chunk IDs",
    )
    abstention_correct: bool = Field(
        ...,
        description="Whether system correctly abstained if unanswerable, or answered if answerable",
    )
    injection_resistant: bool = Field(
        ...,
        description="Whether system safely resisted prompt injection attempt",
    )
    passed: bool = Field(
        ...,
        description="Whether the test case satisfied all grounding, citation, and abstention criteria",
    )
    latency_ms: float = Field(..., ge=0.0, description="Execution time in milliseconds")
    actual_answer: str = Field(default="", description="Synthesized answer text or abstention message")
    citation_count: int = Field(default=0, ge=0, description="Number of valid citations returned")
    retrieved_chunk_count: int = Field(default=0, ge=0, description="Total chunks retrieved")
    notes: str = Field(default="", description="Diagnostic notes or error observations")


class EvalSummary(BaseModel):
    """Aggregate benchmark results and quality diagnostics."""

    model_config = ConfigDict(frozen=True)

    total_cases: int = Field(..., ge=0, description="Total number of evaluated test cases")
    passed_cases: int = Field(..., ge=0, description="Number of test cases meeting all criteria")
    overall_pass_rate: float = Field(..., ge=0.0, le=1.0, description="Overall benchmark pass rate")
    retrieval_hit_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Retrieval Hit@K rate across answerable cases with labelled evidence",
    )
    citation_validity_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of citations referencing valid retrieved chunk IDs",
    )
    abstention_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Abstention accuracy across unanswerable test cases",
    )
    injection_resistance_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Resistance rate across prompt injection attempts",
    )
    avg_latency_ms: float = Field(..., ge=0.0, description="Average execution latency per query in milliseconds")
    case_results: list[EvalCaseResult] = Field(
        default_factory=list,
        description="Individual test case outcomes",
    )
    disclaimer: str = Field(
        default="Synthetic benchmark evaluates retrieval recall, citation-ID integrity, and explicit abstention. It does not replace human domain review for semantic correctness.",
        description="Mandatory evaluation disclaimer",
    )

