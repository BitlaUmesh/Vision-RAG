"""
engine.py — ChromaDB-powered semantic search engine (Phase 2).
================================================================================
Replaces the BM25 keyword-matching engine with:
  - ChromaDB for persistent vector storage
  - sentence-transformers (all-mpnet-base-v2) for high-quality embeddings
  - Intelligent overlapping text chunking via app.chunking
"""

import logging
import os
import time
import threading
import uuid
from typing import Callable, Dict, List, Optional

import pymupdf as fitz

from app.ocr import extract_text_hybrid, extract_visual_description, page_to_base64
from app.chunking import chunk_page

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded globals (heavy imports deferred to first use)
# ---------------------------------------------------------------------------
_embedding_model = None
_embedding_lock = threading.Lock()

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chromadb_data")
COLLECTION_NAME = "vision_rag_chunks"
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"  # 420MB, high quality


def _get_embedding_model():
    """Lazy-load the sentence-transformers model (one-time ~420MB download)."""
    global _embedding_model
    if _embedding_model is None:
        with _embedding_lock:
            if _embedding_model is None:
                logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
                from sentence_transformers import SentenceTransformer
                _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
                logger.info("Embedding model loaded.")
    return _embedding_model


class VisionEngine:
    """Core RAG engine with ChromaDB vector search + local embeddings."""

    def __init__(self):
        import chromadb

        self._client = chromadb.PersistentClient(path=CHROMA_DIR)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # In-memory image cache (not persisted — rebuilt on demand)
        self._page_images: Dict[str, str] = {}

        # Track indexed documents (derive from ChromaDB metadata)
        self._indexed_filenames: set = set()
        self._refresh_document_list()

        # Threading
        self._current_run_id = str(uuid.uuid4())
        self._lock = threading.Lock()
        self._index_ready = self._collection.count() > 0

        logger.info(
            f"VisionEngine initialized. ChromaDB at '{CHROMA_DIR}', "
            f"{self._collection.count()} existing chunks, "
            f"{len(self._indexed_filenames)} documents."
        )

    # ── Document management ────────────────────────────────────────────────

    def _refresh_document_list(self):
        """Derive the list of indexed filenames from ChromaDB metadata."""
        try:
            if self._collection.count() == 0:
                self._indexed_filenames = set()
                return
            # Peek at all documents' metadata to extract filenames
            # ChromaDB .get() with no IDs returns all
            results = self._collection.get(
                include=["metadatas"],
                limit=self._collection.count(),
            )
            filenames = set()
            if results and results["metadatas"]:
                for meta in results["metadatas"]:
                    if meta and "filename" in meta:
                        filenames.add(meta["filename"])
            self._indexed_filenames = filenames
        except Exception as e:
            logger.warning(f"Failed to refresh document list: {e}")
            self._indexed_filenames = set()

    def get_document_list(self) -> List[str]:
        with self._lock:
            return list(self._indexed_filenames)

    def remove_document(self, filename: str):
        with self._lock:
            logger.info(f"Removing document {filename} from ChromaDB...")

            # Delete all chunks belonging to this file
            try:
                self._collection.delete(
                    where={"filename": filename}
                )
            except Exception as e:
                logger.warning(f"ChromaDB delete failed: {e}")

            # Clean image cache
            prefix = f"{filename}_"
            self._page_images = {
                k: v for k, v in self._page_images.items()
                if not k.startswith(prefix)
            }

            self._indexed_filenames.discard(filename)
            self._index_ready = self._collection.count() > 0

            logger.info(
                f"Document {filename} removed. "
                f"{self._collection.count()} chunks remaining."
            )

    # ── Indexing ───────────────────────────────────────────────────────────

    def load_or_create_index(
        self,
        pdf_path: str,
        status_callback: Optional[Callable[[str, int, str], None]] = None,
    ):
        run_id = str(uuid.uuid4())
        self._current_run_id = run_id
        filename = os.path.basename(pdf_path)

        def _report(step_id: str, progress: int, msg: str):
            if status_callback:
                status_callback(step_id, progress, msg)

        logger.info(f"Starting indexing pipeline for {filename}")
        _report("upload", 15, "Starting extraction...")

        # Accumulate locally before committing to ChromaDB
        all_chunk_records: List[Dict] = []
        local_page_images: Dict[str, str] = {}
        local_ocr_pages: List[str] = []
        local_visual_pages: List[str] = []

        doc = None
        try:
            doc = fitz.open(pdf_path)
            if len(doc) == 0:
                raise ValueError(f"PDF has 0 pages: {pdf_path}")

            total = len(doc)

            for i, page in enumerate(doc):
                page_num = i + 1
                page_id = f"{filename}_{page_num}"

                # Abort check
                if self._current_run_id != run_id:
                    logger.warning(f"Indexing aborted for {filename}")
                    _report("error", 0, "Indexing aborted by a newer action.")
                    return

                logger.info(f"Processing {filename} page {page_num}/{total}...")
                base_progress = 15 + int((i / total) * 50)  # 15% → 65%
                _report("text", base_progress, f"Extracting text (Page {page_num}/{total})...")

                # Rate-limit API calls
                if i > 0:
                    time.sleep(3)

                # --- Text extraction ---
                embedded = page.get_text() or ""
                text = extract_text_hybrid(page)

                if len(embedded.strip()) < 50 and len(text.strip()) > len(embedded.strip()):
                    local_ocr_pages.append(page_id)

                # --- Visual content analysis ---
                _report("visual", base_progress + int(25 / total),
                        f"Analyzing visuals (Page {page_num}/{total})...")
                visual_desc = None
                try:
                    visual_desc = extract_visual_description(page)
                    if visual_desc:
                        local_visual_pages.append(page_id)
                except Exception as e:
                    logger.warning(f"Page {page_num}: visual analysis failed: {e}")

                # --- Store page image ---
                try:
                    img_b64 = page_to_base64(page, dpi=150)
                    local_page_images[page_id] = img_b64
                except Exception as e:
                    logger.warning(f"Page {page_num}: failed to store image: {e}")

                # --- Chunk the page ---
                page_chunks = chunk_page(
                    page_text=text,
                    visual_description=visual_desc,
                    filename=filename,
                    page_num=page_num,
                )
                all_chunk_records.extend(page_chunks)

            # --- Generate embeddings ---
            if self._current_run_id != run_id:
                _report("error", 0, "Indexing aborted by a newer action.")
                return

            _report("index", 70, f"Generating embeddings for {len(all_chunk_records)} chunks...")

            model = _get_embedding_model()
            texts_to_embed = [r["text"] for r in all_chunk_records]

            # Batch embed all chunks at once
            logger.info(f"Embedding {len(texts_to_embed)} chunks...")
            embeddings = model.encode(
                texts_to_embed,
                show_progress_bar=False,
                batch_size=32,
                normalize_embeddings=True,
            ).tolist()

            _report("index", 85, "Storing in vector database...")

            # --- Commit to ChromaDB ---
            with self._lock:
                # Remove existing chunks for this filename (re-upload case)
                try:
                    self._collection.delete(where={"filename": filename})
                except Exception:
                    pass  # No existing entries — fine

                # Upsert in batches (ChromaDB has a batch limit)
                batch_size = 100
                for batch_start in range(0, len(all_chunk_records), batch_size):
                    batch_end = min(batch_start + batch_size, len(all_chunk_records))
                    batch = all_chunk_records[batch_start:batch_end]
                    batch_embeddings = embeddings[batch_start:batch_end]

                    self._collection.add(
                        ids=[r["id"] for r in batch],
                        embeddings=batch_embeddings,
                        documents=[r["raw_text"] for r in batch],
                        metadatas=[r["metadata"] for r in batch],
                    )

                # Update image cache
                self._page_images.update(local_page_images)
                self._indexed_filenames.add(filename)
                self._index_ready = True

            ocr_msg = f" (OCR on {len(local_ocr_pages)} pages)" if local_ocr_pages else ""
            vis_msg = f" (Visuals on {len(local_visual_pages)} pages)" if local_visual_pages else ""
            final_msg = (
                f"Indexed {total} pages → {len(all_chunk_records)} chunks "
                f"of {filename}{ocr_msg}{vis_msg}"
            )
            logger.info(f"Index complete: {final_msg}")
            _report("completed", 100, final_msg)

        except Exception as e:
            _report("error", 0, f"Indexing failed: {e}")
            raise
        finally:
            if doc:
                doc.close()

    # ── Search ─────────────────────────────────────────────────────────────

    def search(
        self,
        queries: List[str],
        k_total: int = 5,
        prioritize_visuals: bool = False,
    ) -> List[Dict]:
        """
        Semantic search across all indexed documents.

        Returns a deduplicated list of:
            {"filename": str, "page_num": int, "score": float, "text": str}
        """
        with self._lock:
            if not self._index_ready or not queries:
                return []

            model = _get_embedding_model()

            # Embed all queries
            query_embeddings = model.encode(
                queries,
                normalize_embeddings=True,
            ).tolist()

            # Accumulate results across all sub-queries
            page_scores: Dict[str, float] = {}      # page_id → score
            page_texts: Dict[str, List[str]] = {}    # page_id → [chunk texts]
            page_info: Dict[str, Dict] = {}           # page_id → {filename, page_num}

            n_results = min(k_total * 3, self._collection.count())
            if n_results == 0:
                return []

            for q_embedding in query_embeddings:
                results = self._collection.query(
                    query_embeddings=[q_embedding],
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"],
                )

                if not results or not results["ids"] or not results["ids"][0]:
                    continue

                for idx in range(len(results["ids"][0])):
                    meta = results["metadatas"][0][idx]
                    doc_text = results["documents"][0][idx]
                    distance = results["distances"][0][idx]
                    # ChromaDB cosine distance: 0 = identical, 2 = opposite
                    # Convert to similarity score: 1 - (distance / 2)
                    similarity = 1.0 - (distance / 2.0)

                    page_id = f"{meta['filename']}_{meta['page_num']}"

                    # Boost visual chunks if requested
                    boost = 1.0
                    if prioritize_visuals and meta.get("chunk_type") == "visual":
                        boost = 1.5

                    score = similarity * boost

                    # Accumulate best score per page
                    if page_id not in page_scores or score > page_scores[page_id]:
                        page_scores[page_id] = score

                    page_info[page_id] = {
                        "filename": meta["filename"],
                        "page_num": meta["page_num"],
                    }

                    if page_id not in page_texts:
                        page_texts[page_id] = []
                    if doc_text and doc_text not in page_texts[page_id]:
                        page_texts[page_id].append(doc_text)

            if not page_scores:
                return []

            # Sort by score, return top k
            sorted_pages = sorted(
                page_scores.keys(),
                key=lambda pid: page_scores[pid],
                reverse=True,
            )[:k_total]

            results = []
            for page_id in sorted_pages:
                info = page_info[page_id]
                results.append({
                    "filename": info["filename"],
                    "page_num": info["page_num"],
                    "score": round(page_scores[page_id], 4),
                    "retrieved_chunks": page_texts.get(page_id, []),
                })

            return results

    # ── Page data access ───────────────────────────────────────────────────

    def get_page_images(self, pages_to_fetch: List[Dict]) -> Dict[str, str]:
        """Fetch page images. Input is list of dicts with 'filename' and 'page_num'."""
        with self._lock:
            result = {}
            for p in pages_to_fetch:
                page_id = f"{p['filename']}_{p['page_num']}"
                if page_id in self._page_images:
                    result[page_id] = self._page_images[page_id]
            return result

    def get_page_info(self, filename: str, page_num: int) -> Optional[Dict]:
        """
        Return stored chunk text for a specific page.
        Queries ChromaDB for all chunks belonging to this page.
        """
        try:
            results = self._collection.get(
                where={
                    "$and": [
                        {"filename": filename},
                        {"page_num": page_num},
                    ]
                },
                include=["documents", "metadatas"],
            )

            if not results or not results["documents"]:
                return None

            # Combine all chunk texts for this page
            text_parts = []
            visual_parts = []
            for doc, meta in zip(results["documents"], results["metadatas"]):
                if meta.get("chunk_type") == "visual":
                    visual_parts.append(doc)
                else:
                    text_parts.append(doc)

            return {
                "filename": filename,
                "page_num": page_num,
                "text": "\n".join(text_parts),
                "visual_description": "\n".join(visual_parts) if visual_parts else "",
            }
        except Exception as e:
            logger.warning(f"get_page_info failed: {e}")
            return None

    def rebuild_index(
        self,
        pdf_path: str,
        status_callback: Optional[Callable[[str, int, str], None]] = None,
    ):
        # Invalidate current run → causes any running thread to abort
        self._current_run_id = str(uuid.uuid4())
        self.load_or_create_index(pdf_path, status_callback=status_callback)

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._index_ready

    @property
    def chunk_count(self) -> int:
        return self._collection.count()

    @property
    def ocr_page_count(self) -> int:
        # Approximate — not tracked persistently in Phase 2
        return 0

    @property
    def visual_page_count(self) -> int:
        # Approximate — not tracked persistently in Phase 2
        return 0