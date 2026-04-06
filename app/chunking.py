"""
chunking.py — Intelligent text chunking for semantic search.
================================================================================
Splits extracted page text into overlapping chunks that respect sentence
boundaries. Each chunk is sized for optimal embedding quality with
sentence-transformers (~300 tokens ≈ ~1200 chars, with 50-token overlap).
"""

import re
from typing import Dict, List


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_CHUNK_CHARS = 1200       # ~300 tokens
OVERLAP_CHARS = 200          # ~50 tokens overlap between consecutive chunks
MIN_CHUNK_CHARS = 80         # Don't create tiny fragment chunks


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------

# Regex that splits on sentence-ending punctuation followed by whitespace
_SENTENCE_RE = re.compile(
    r'(?<=[.!?])\s+|'           # Standard sentence endings
    r'(?<=\n)\s*\n+|'           # Double newlines (paragraph breaks)
    r'(?<=:)\s*\n'              # Colon followed by newline (list headers)
)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentence-level fragments."""
    parts = _SENTENCE_RE.split(text.strip())
    # Filter out pure whitespace fragments
    return [p.strip() for p in parts if p and p.strip()]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> List[str]:
    """
    Split text into overlapping chunks that respect sentence boundaries.

    Args:
        text: The raw text to chunk.
        max_chars: Maximum character length per chunk (~300 tokens).
        overlap_chars: Number of characters to overlap between chunks.

    Returns:
        List of chunk strings. Guaranteed non-empty (returns [text] for
        very short inputs).
    """
    text = text.strip()
    if not text:
        return []

    # Short text: return as single chunk
    if len(text) <= max_chars:
        return [text]

    sentences = _split_sentences(text)
    if not sentences:
        return [text]

    chunks: List[str] = []
    current_sentences: List[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # If a single sentence exceeds max_chars, force-split it
        if sentence_len > max_chars:
            # Flush current buffer first
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_len = 0

            # Hard-split the long sentence
            for i in range(0, sentence_len, max_chars - overlap_chars):
                fragment = sentence[i:i + max_chars]
                if len(fragment) >= MIN_CHUNK_CHARS:
                    chunks.append(fragment)
            continue

        # Would adding this sentence exceed the limit?
        if current_len + sentence_len + 1 > max_chars and current_sentences:
            # Flush current chunk
            chunks.append(" ".join(current_sentences))

            # Create overlap: take trailing sentences that fit within overlap_chars
            overlap_sentences: List[str] = []
            overlap_len = 0
            for s in reversed(current_sentences):
                if overlap_len + len(s) + 1 > overlap_chars:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s) + 1

            current_sentences = overlap_sentences
            current_len = overlap_len

        current_sentences.append(sentence)
        current_len += sentence_len + 1  # +1 for space

    # Flush remaining
    if current_sentences:
        final_chunk = " ".join(current_sentences)
        # Only add if it's substantially different from the last chunk
        if len(final_chunk) >= MIN_CHUNK_CHARS:
            if not chunks or final_chunk != chunks[-1]:
                chunks.append(final_chunk)

    return chunks if chunks else [text]


def chunk_page(
    page_text: str,
    visual_description: str | None,
    filename: str,
    page_num: int,
) -> List[Dict]:
    """
    Chunk a single page's content into embedding-ready records.

    Returns:
        List of dicts with keys:
            - id: unique chunk ID (e.g., "doc.pdf_3_chunk_0")
            - text: the chunk text for embedding
            - metadata: {filename, page_num, chunk_index, chunk_type}
    """
    records: List[Dict] = []

    # Chunk the main text
    text_chunks = chunk_text(page_text)
    for i, chunk in enumerate(text_chunks):
        # Prepend filename context for better retrieval
        embedding_text = f"[{filename} | Page {page_num}] {chunk}"
        records.append({
            "id": f"{filename}_{page_num}_text_{i}",
            "text": embedding_text,
            "raw_text": chunk,
            "metadata": {
                "filename": filename,
                "page_num": page_num,
                "chunk_index": i,
                "chunk_type": "text",
            },
        })

    # Chunk visual descriptions separately
    if visual_description and visual_description.strip():
        vis_chunks = chunk_text(visual_description)
        for i, chunk in enumerate(vis_chunks):
            embedding_text = f"[{filename} | Page {page_num} | Visual Content] {chunk}"
            records.append({
                "id": f"{filename}_{page_num}_visual_{i}",
                "text": embedding_text,
                "raw_text": chunk,
                "metadata": {
                    "filename": filename,
                    "page_num": page_num,
                    "chunk_index": i,
                    "chunk_type": "visual",
                },
            })

    return records
