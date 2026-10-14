"""Page-aware text chunking for FinDoc AI.

Splits extracted page text into overlapping, bounded chunks while strictly preserving
physical page numbers, document IDs, chunk indices, and content hashes.
"""

from typing import List, Optional

from findoc.ingestion.validation import compute_sha256
from findoc.schemas import ChunkRecord, PageText


def split_text_into_chunks(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[str]:
    """Split a continuous text string into overlapping chunks using natural boundaries.

    Tries splitting on paragraphs, then sentences, then words, falling back to characters.
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        if end >= text_len:
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Look for natural breaking points near the end boundary
        search_window_start = max(start + chunk_overlap, end - 150)
        search_window = text[search_window_start:end]

        split_pos = -1
        # Try paragraph break
        para_idx = search_window.rfind("\n\n")
        if para_idx != -1:
            split_pos = search_window_start + para_idx + 2
        else:
            # Try sentence break (. followed by space or newline)
            sent_idx = -1
            for sep in (". ", ".\n", "? ", "! "):
                idx = search_window.rfind(sep)
                if idx > sent_idx:
                    sent_idx = idx + len(sep)
            if sent_idx != -1:
                split_pos = search_window_start + sent_idx
            else:
                # Try newline or space
                for sep in ("\n", " "):
                    idx = search_window.rfind(sep)
                    if idx != -1:
                        split_pos = search_window_start + idx + 1
                        break

        # Fallback to hard character cutoff if no separator found
        if split_pos == -1 or split_pos <= start:
            split_pos = end

        chunk = text[start:split_pos].strip()
        if chunk:
            chunks.append(chunk)

        # Advance start position taking overlap into account
        start = max(split_pos - chunk_overlap, start + 1)

    return chunks


def chunk_page_texts(
    pages: List[PageText],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[ChunkRecord]:
    """Chunk a sequence of PageText records while preserving page-level provenance.

    Each chunk retains:
      - Stable deterministic chunk_id: '{document_id}_p{page}_c{index}'
      - document_id & source_name
      - physical page_number (int for PDF, None for TXT/MD)
      - global document chunk_index
      - content_hash of the chunk text
    """
    chunk_records: List[ChunkRecord] = []
    chunk_index = 0

    for page in pages:
        clean_text = page.text.strip()
        if not clean_text:
            continue

        raw_chunks = split_text_into_chunks(
            clean_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk_text in raw_chunks:
            chunk_hash = compute_sha256(chunk_text.encode("utf-8"))
            page_label = f"p{page.page_number}" if page.page_number is not None else "txt"
            chunk_id = f"{page.document_id}_{page_label}_c{chunk_index:04d}"

            chunk_records.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    document_id=page.document_id,
                    source_name=page.source_name,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    content_hash=chunk_hash,
                )
            )
            chunk_index += 1

    return chunk_records
