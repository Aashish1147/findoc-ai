"""File upload validation and safe identifier utilities for FinDoc AI.

Ensures uploaded files conform to allowed formats, size limits, and binary signatures
without trusting client filenames as filesystem paths.
"""

import hashlib
import os
import re
from pathlib import Path
from typing import Tuple

from findoc.errors import FileValidationError
from findoc.schemas import FileType

ALLOWED_EXTENSIONS = {
    ".pdf": "pdf",
    ".txt": "txt",
    ".md": "md",
}

PDF_MAGIC_BYTES = b"%PDF-"


def compute_sha256(data: bytes) -> str:
    """Compute standard SHA-256 hex digest of file bytes for integrity and deduplication."""
    return hashlib.sha256(data).hexdigest()


def generate_document_id(content_hash: str) -> str:
    """Generate deterministic, collision-resistant document identifier from content hash."""
    return f"doc_{content_hash[:16]}"


def sanitize_display_name(raw_filename: str) -> str:
    """Sanitize user-provided filename for safe display.

    Removes directory traversal components, path separators, control characters,
    and limits length while preserving the valid extension.
    """
    if not raw_filename or not raw_filename.strip():
        return "unnamed_document"

    # Extract base name to eliminate directory traversal (Windows & POSIX separators)
    basename = os.path.basename(raw_filename.replace("\\", "/"))

    # Remove null bytes and non-printable characters
    cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", basename).strip()

    # Replace suspicious characters with underscores while allowing standard filename chars
    cleaned = re.sub(r'[^a-zA-Z0-9_\-\. ()]', "_", cleaned)

    # Collapse multiple spaces or underscores
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned or cleaned.startswith("."):
        return f"document_{cleaned}".strip("_")

    return cleaned[:100]


def validate_file_upload(
    raw_filename: str,
    content: bytes,
    max_file_mb: int = 20,
) -> Tuple[str, str, FileType, str]:
    """Validate uploaded file bytes, extension, size, and header signature.

    Returns:
        Tuple of (document_id, sanitized_display_name, file_type, content_hash)

    Raises:
        FileValidationError: If file violates extension, size, or signature constraints.
    """
    if not content or len(content) == 0:
        raise FileValidationError("Uploaded file is empty (0 bytes).")

    max_bytes = max_file_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise FileValidationError(
            f"File size ({len(content) / (1024 * 1024):.2f} MB) exceeds maximum allowed "
            f"limit of {max_file_mb} MB."
        )

    sanitized_name = sanitize_display_name(raw_filename)
    extension = Path(sanitized_name).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(ALLOWED_EXTENSIONS.keys())
        raise FileValidationError(
            f"Unsupported file extension '{extension}'. Allowed file types are: {allowed}"
        )

    file_type: FileType = ALLOWED_EXTENSIONS[extension]  # type: ignore[assignment]

    # Verify signature for binary formats
    if file_type == "pdf":
        if not content.startswith(PDF_MAGIC_BYTES):
            raise FileValidationError(
                "File has a .pdf extension but lacks a valid PDF header signature (%PDF-)."
            )

    content_hash = compute_sha256(content)
    document_id = generate_document_id(content_hash)

    return document_id, sanitized_name, file_type, content_hash
