"""SQLite metadata repository for FinDoc AI document library.

Tracks document identity, content hashes for deduplication, processing status,
page counts, chunk counts, and actionable warnings.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from findoc.errors import StorageError
from findoc.schemas import DocumentRecord, DocumentStatus, FileType


class MetadataStore:
    """Thread-safe SQLite store for DocumentRecord metadata."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create documents table and indices if they do not exist."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        document_id TEXT PRIMARY KEY,
                        display_name TEXT NOT NULL,
                        file_type TEXT NOT NULL,
                        content_hash TEXT NOT NULL UNIQUE,
                        status TEXT NOT NULL,
                        page_count INTEGER,
                        chunk_count INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL,
                        warning_message TEXT
                    );
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_doc_content_hash ON documents(content_hash);"
                )
                conn.commit()
        except Exception as e:
            raise StorageError(f"Failed to initialize SQLite metadata store: {e}") from e

    def register_document(self, doc: DocumentRecord) -> None:
        """Register a new document record.

        Raises:
            StorageError: If registration fails or content_hash already exists.
        """
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO documents (
                        document_id, display_name, file_type, content_hash,
                        status, page_count, chunk_count, created_at, warning_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        doc.document_id,
                        doc.display_name,
                        doc.file_type,
                        doc.content_hash,
                        doc.status,
                        doc.page_count,
                        doc.chunk_count,
                        doc.created_at.isoformat(),
                        doc.warning_message,
                    ),
                )
                conn.commit()
        except sqlite3.IntegrityError as e:
            raise StorageError(
                f"Document with content hash '{doc.content_hash}' already exists."
            ) from e
        except Exception as e:
            raise StorageError(f"Failed to register document metadata: {e}") from e

    def update_document_status(
        self,
        document_id: str,
        status: DocumentStatus,
        chunk_count: Optional[int] = None,
        page_count: Optional[int] = None,
        warning_message: Optional[str] = None,
    ) -> None:
        """Update processing status, counts, or warnings for an existing document."""
        try:
            with self._get_connection() as conn:
                fields = ["status = ?"]
                params = [status]

                if chunk_count is not None:
                    fields.append("chunk_count = ?")
                    params.append(chunk_count)
                if page_count is not None:
                    fields.append("page_count = ?")
                    params.append(page_count)
                if warning_message is not None:
                    fields.append("warning_message = ?")
                    params.append(warning_message)

                params.append(document_id)
                query = f"UPDATE documents SET {', '.join(fields)} WHERE document_id = ?;"
                cursor = conn.execute(query, params)
                if cursor.rowcount == 0:
                    raise StorageError(f"Document with ID '{document_id}' not found.")
                conn.commit()
        except StorageError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to update document status: {e}") from e

    def get_document(self, document_id: str) -> Optional[DocumentRecord]:
        """Retrieve a DocumentRecord by its document_id."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM documents WHERE document_id = ?;",
                    (document_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return None
                return self._row_to_record(row)
        except Exception as e:
            raise StorageError(f"Failed to retrieve document '{document_id}': {e}") from e

    def get_by_content_hash(self, content_hash: str) -> Optional[DocumentRecord]:
        """Find an existing document by its content hash to detect duplicates."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM documents WHERE content_hash = ?;",
                    (content_hash,),
                )
                row = cursor.fetchone()
                if not row:
                    return None
                return self._row_to_record(row)
        except Exception as e:
            raise StorageError(f"Failed to query document by hash: {e}") from e

    def list_documents(self) -> List[DocumentRecord]:
        """List all registered document records ordered by creation date."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM documents ORDER BY created_at DESC;"
                )
                return [self._row_to_record(row) for row in cursor.fetchall()]
        except Exception as e:
            raise StorageError(f"Failed to list documents: {e}") from e

    def delete_document(self, document_id: str) -> bool:
        """Delete a document record by its ID. Returns True if deleted, False if not found."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM documents WHERE document_id = ?;",
                    (document_id,),
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            raise StorageError(f"Failed to delete document '{document_id}': {e}") from e

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> DocumentRecord:
        raw_dt = row["created_at"]
        try:
            created_at = datetime.fromisoformat(raw_dt)
        except Exception:
            created_at = datetime.now(timezone.utc)

        return DocumentRecord(
            document_id=row["document_id"],
            display_name=row["display_name"],
            file_type=row["file_type"],
            content_hash=row["content_hash"],
            status=row["status"],
            page_count=row["page_count"],
            chunk_count=row["chunk_count"],
            created_at=created_at,
            warning_message=row["warning_message"],
        )
