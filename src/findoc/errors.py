"""Domain exceptions for FinDoc AI.

Defines a clean hierarchy of application errors for safe, predictable handling.
"""


class FinDocError(Exception):
    """Base exception for all FinDoc AI errors."""


class ConfigurationError(FinDocError):
    """Raised when application configuration or environment settings are invalid."""


class FileValidationError(FinDocError):
    """Raised when an uploaded file fails validation (size, extension, signature, empty)."""


class DocumentParsingError(FinDocError):
    """Raised when a document cannot be parsed into usable text records."""


class StorageError(FinDocError):
    """Raised when vector storage or document metadata storage encounters an error."""


class RetrievalError(FinDocError):
    """Raised when candidate retrieval fails or query embedding cannot be produced."""


class CitationValidationError(FinDocError):
    """Raised when model citations refer to non-existent or un-retrieved evidence IDs."""


class GenerationError(FinDocError):
    """Raised when LLM adapter fails, times out, or returns unparseable outputs."""


class InsufficientEvidenceError(FinDocError):
    """Raised or signaled when retrieved passages do not provide enough evidence to answer."""
