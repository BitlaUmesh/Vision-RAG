import logging
import time
import threading
from typing import Callable, Dict, List, Optional
from collections import defaultdict

import pymupdf as fitz
from rank_bm25 import BM25Okapi

from app.ocr import extract_text_hybrid, extract_visual_description, page_to_base64

logger = logging.getLogger(__name__)


class VisionEngine:
    """Core RAG engine managing BM25 search over extracted text and metadata across multiple PDFs."""

    def __init__(self, index_root: str = ".byaldi", index_name: str = "vision_rag_index"):
        # State across all documents
        self._bm25 = None
        self._pages_info: List[Dict] = []
        self._tokenized_corpus: List[List[str]] = []
        self._index_ready = False
        
        self._ocr_pages: List[str] = []     # e.g. "filename.pdf_1"
        self._visual_pages: List[str] = []  # e.g. "filename.pdf_1"
        self._page_images: Dict[str, str] = {}  # "filename.pdf_1" -> base64
        
        import uuid
        self._current_run_id = str(uuid.uuid4())
        self._lock = threading.Lock()

    def get_document_list(self) -> List[str]:
        with self._lock:
            # Extract unique filenames from current index
            filenames = set(info["filename"] for info in self._pages_info)
            return list(filenames)

    def remove_document(self, filename: str):
        with self._lock:
            logger.info(f"Removing document {filename} from index...")
            
            # Find indices to keep
            keep_indices = [i for i, info in enumerate(self._pages_info) if info["filename"] != filename]
            
            if len(keep_indices) == len(self._pages_info):
                logger.warning(f"Document {filename} not found in index.")
                return

            self._pages_info = [self._pages_info[i] for i in keep_indices]
            self._tokenized_corpus = [self._tokenized_corpus[i] for i in keep_indices]
            
            # Filter other lists/dicts
            prefix = f"{filename}_"
            self._ocr_pages = [p for p in self._ocr_pages if not p.startswith(prefix)]
            self._visual_pages = [p for p in self._visual_pages if not p.startswith(prefix)]
            self._page_images = {k: v for k, v in self._page_images.items() if not k.startswith(prefix)}
            
            if self._tokenized_corpus:
                self._bm25 = BM25Okapi(self._tokenized_corpus)
                self._index_ready = True
            else:
                self._bm25 = None
                self._index_ready = False
            
            logger.info(f"Document {filename} removed. {len(self._pages_info)} total pages remaining.")

    def load_or_create_index(self, pdf_path: str, status_callback: Optional[Callable[[str, int, str], None]] = None):
        import uuid
        import os
        run_id = str(uuid.uuid4())
        self._current_run_id = run_id
        
        filename = os.path.basename(pdf_path)

        def _report(step_id: str, progress: int, msg: str):
            if status_callback:
                status_callback(step_id, progress, msg)

        # Build into LOCAL variables initially — so we don't hold the lock during expensive processing
        logger.info(f"Extracting text and analyzing visuals from {pdf_path}")
        _report("upload", 15, "Starting extraction...")
        
        local_pages_info = []
        local_ocr_pages = []
        local_visual_pages = []
        local_page_images = {}
        local_tokenized_corpus = []

        doc = None
        try:
            doc = fitz.open(pdf_path)
            if len(doc) == 0:
                raise ValueError(f"PDF has 0 pages: {pdf_path}")

            total = len(doc)
            for i, page in enumerate(doc):
                page_num = i + 1
                page_id = f"{filename}_{page_num}"

                # Check for race conditions before processing page
                if self._current_run_id != run_id:
                    logger.warning(f"Indexing aborted for {filename}: run cancelled.")
                    _report("error", 0, "Indexing aborted by a newer action.")
                    return

                logger.info(f"Processing {filename} page {page_num}/{total}...")
                base_progress = 15 + int((i / total) * 65)  # 15% to 80%
                
                _report("text", base_progress, f"Extracting text (Page {page_num}/{total})...")

                # Pace API calls to avoid rate limits
                if i > 0:
                    time.sleep(3)

                # --- Step 1: Text extraction (hybrid: embedded + OCR fallback) ---
                embedded = page.get_text() or ""
                text = extract_text_hybrid(page)

                # Track if OCR was used
                if len(embedded.strip()) < 50 and len(text.strip()) > len(embedded.strip()):
                    local_ocr_pages.append(page_id)

                # --- Step 2: Visual content analysis ---
                _report("visual", base_progress + int(32 / total), f"Analyzing visuals (Page {page_num}/{total})...")
                visual_desc = None
                try:
                    visual_desc = extract_visual_description(page)
                    if visual_desc:
                        local_visual_pages.append(page_id)
                        logger.info(f"Page {page_num}: visual content analyzed "
                                    f"({len(visual_desc)} chars)")
                except Exception as e:
                    logger.warning(f"Page {page_num}: visual analysis failed: {e}")

                # --- Step 3: Store page image for query-time multimodal calls ---
                try:
                    img_b64 = page_to_base64(page, dpi=150)  # lower DPI for storage
                    local_page_images[page_id] = img_b64
                except Exception as e:
                    logger.warning(f"Page {page_num}: failed to store image: {e}")

                # --- Step 4: Store page info ---
                local_pages_info.append({
                    "filename": filename,
                    "page_num": page_num,
                    "text": text,
                    "visual_description": visual_desc or "",
                })

                # --- Step 5: Build combined corpus for BM25 ---
                # Include the filename so queries referencing the document name match
                combined = f"Filename: {filename}\n{text}"
                if visual_desc:
                    combined += "\n\n[VISUAL CONTENT]\n" + visual_desc

                # Replace underscores and hyphens to ensure filename tokenization (e.g. Umesh_Resume -> umesh resume)
                import re
                clean_combined = re.sub(r'[_\-]', ' ', combined)
                tokens = clean_combined.lower().split()
                if tokens:
                    local_tokenized_corpus.append(tokens)
                else:
                    local_tokenized_corpus.append([""])  # keep alignment with page index

            # Check one more time before mutating global state
            if self._current_run_id != run_id:
                logger.warning(f"Indexing aborted at commit for {filename}: run cancelled.")
                _report("error", 0, "Indexing aborted by a newer action.")
                return

            _report("index", 85, "Merging into global search index...")
            
            # --- ATOMIC MERGE ---
            with self._lock:
                # Remove any existing pages for this filename (in case they uploaded it again)
                # This ensures we don't duplicate indexing if they re-upload the same file
                keep_indices = [i for i, info in enumerate(self._pages_info) if info["filename"] != filename]
                self._pages_info = [self._pages_info[i] for i in keep_indices] + local_pages_info
                self._tokenized_corpus = [self._tokenized_corpus[i] for i in keep_indices] + local_tokenized_corpus
                
                prefix = f"{filename}_"
                self._ocr_pages = [p for p in self._ocr_pages if not p.startswith(prefix)] + local_ocr_pages
                self._visual_pages = [p for p in self._visual_pages if not p.startswith(prefix)] + local_visual_pages
                
                # Update images directly
                self._page_images.update(local_page_images)
                
                if self._tokenized_corpus:
                    self._bm25 = BM25Okapi(self._tokenized_corpus)
                else:
                    self._bm25 = None
                self._index_ready = True

            ocr_msg = ""
            if local_ocr_pages:
                ocr_msg = f" (OCR used on {len(local_ocr_pages)} pages)"
            visual_msg = ""
            if local_visual_pages:
                visual_msg = f" (Visual content on {len(local_visual_pages)} pages)"
            
            final_msg = f"Indexed {len(local_pages_info)} pages of {filename}{ocr_msg}{visual_msg}"
            logger.info(f"Index merge complete: {final_msg}")
            
            _report("completed", 100, final_msg)

        except Exception as e:
            _report("error", 0, f"Indexing failed: {e}")
            raise
        finally:
            if doc:
                doc.close()

    def search(self, queries: List[str], k_total: int = 5, prioritize_visuals: bool = False) -> List[Dict]:
        """
        Agentic search across multiple sub-queries.
        Returns a deduplicated list of {"filename": str, "page_num": int, "score": float}
        """
        with self._lock:
            if not self._index_ready or not self._bm25 or not queries:
                return []
            
            page_scores = defaultdict(float)
            
            for query in queries:
                tokenized_query = query.lower().split()
                if not tokenized_query:
                    continue
                    
                scores = self._bm25.get_scores(tokenized_query)
                for i, score in enumerate(scores):
                    if score > 0:
                        # Normalize and boost
                        page_id = f"{self._pages_info[i]['filename']}_{self._pages_info[i]['page_num']}"
                        
                        boost = 1.0
                        if prioritize_visuals and page_id in self._visual_pages:
                            # Massive boost to visual pages if the agent detected a visual intent
                            boost = 2.0 
                            
                        page_scores[i] += (score * boost)

            top_indices = []
            if not page_scores:
                # Semantic fallback: if zero keyword matches (e.g. "what is his name"), 
                # return the most recently added pages up to k_total as assumed context.
                start_idx = len(self._pages_info) - 1
                end_idx = max(-1, start_idx - k_total)
                top_indices = list(range(start_idx, end_idx, -1))
            else:
                # Sort globally by combined score
                top_indices = sorted(page_scores.keys(), key=lambda idx: page_scores[idx], reverse=True)[:k_total]
            
            results = []
            for i in top_indices:
                info = self._pages_info[i]
                results.append({
                    "filename": info["filename"],
                    "page_num": info["page_num"]
                })
                
            return results

    def get_page_images(self, pages_to_fetch: List[Dict]) -> Dict[str, str]:
        """Fetch images. Input is list of dicts with 'filename' and 'page_num'."""
        with self._lock:
            result = {}
            for p in pages_to_fetch:
                page_id = f"{p['filename']}_{p['page_num']}"
                if page_id in self._page_images:
                    result[page_id] = self._page_images[page_id]
            return result

    def get_page_info(self, filename: str, page_num: int) -> Optional[Dict]:
        """Return stored info for a specific doc's page."""
        with self._lock:
            for info in self._pages_info:
                if info["filename"] == filename and info["page_num"] == page_num:
                    return info
            return None

    def rebuild_index(self, pdf_path: str, status_callback: Optional[Callable[[str, int, str], None]] = None):
        import uuid
        # Invalidate the current run FIRST — this causes any running background thread to abort
        self._current_run_id = str(uuid.uuid4())
        # Add to the unified index
        self.load_or_create_index(pdf_path, status_callback=status_callback)

    @property
    def is_ready(self) -> bool:
        return self._index_ready
        
    @property
    def ocr_page_count(self) -> int:
        return len(self._ocr_pages)

    @property
    def visual_page_count(self) -> int:
        return len(self._visual_pages)