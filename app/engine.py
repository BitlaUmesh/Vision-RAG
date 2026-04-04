import logging
import time
from typing import Dict, List, Optional

import pymupdf as fitz
from rank_bm25 import BM25Okapi

from app.ocr import extract_text_hybrid, extract_visual_description, page_to_base64

logger = logging.getLogger(__name__)


class VisionEngine:
    def __init__(self, index_root: str = ".byaldi", index_name: str = "vision_rag_index"):
        self._bm25 = None
        self._pages_info: List[Dict] = []
        self._index_ready = False
        self._ocr_pages: List[int] = []
        self._visual_pages: List[int] = []    # pages with visual content
        self._page_images: Dict[int, str] = {}  # page_num -> base64 image
        import uuid
        self._current_run_id = str(uuid.uuid4())

    def load_or_create_index(self, pdf_path: str):
        import uuid
        run_id = str(uuid.uuid4())
        self._current_run_id = run_id
        
        logger.info(f"Extracting text and analyzing visuals from {pdf_path}")
        self._pages_info = []
        self._ocr_pages = []
        self._visual_pages = []
        self._page_images = {}
        tokenized_corpus = []

        doc = None
        try:
            doc = fitz.open(pdf_path)
            if len(doc) == 0:
                raise ValueError(f"PDF has 0 pages: {pdf_path}")

            total = len(doc)
            for i, page in enumerate(doc):
                page_num = i + 1
                
                # Check for race conditions before processing page
                if self._current_run_id != run_id:
                    logger.warning("Indexing aborted: a new upload replaced this indexing run.")
                    return

                logger.info(f"Processing page {page_num}/{total}...")

                # Pace API calls to avoid rate limits
                if i > 0:
                    time.sleep(3)

                # --- Step 1: Text extraction (hybrid: embedded + OCR fallback) ---
                embedded = page.get_text() or ""
                text = extract_text_hybrid(page)

                # Track if OCR was used
                if len(embedded.strip()) < 50 and len(text.strip()) > len(embedded.strip()):
                    self._ocr_pages.append(page_num)

                # --- Step 2: Visual content analysis ---
                visual_desc = None
                try:
                    visual_desc = extract_visual_description(page)
                    if visual_desc:
                        self._visual_pages.append(page_num)
                        logger.info(f"Page {page_num}: visual content analyzed "
                                    f"({len(visual_desc)} chars)")
                except Exception as e:
                    logger.warning(f"Page {page_num}: visual analysis failed: {e}")

                # --- Step 3: Store page image for query-time multimodal calls ---
                try:
                    img_b64 = page_to_base64(page, dpi=150)  # lower DPI for storage
                    self._page_images[page_num] = img_b64
                except Exception as e:
                    logger.warning(f"Page {page_num}: failed to store image: {e}")

                # --- Step 4: Store page info ---
                self._pages_info.append({
                    "page_num": page_num,
                    "text": text,
                    "visual_description": visual_desc or "",
                })

                # --- Step 5: Build combined corpus for BM25 ---
                # Combine text + visual description so visual content is searchable
                combined = text
                if visual_desc:
                    combined += "\n\n[VISUAL CONTENT]\n" + visual_desc

                tokens = combined.lower().split()
                if tokens:
                    tokenized_corpus.append(tokens)
                else:
                    tokenized_corpus.append([""])  # keep alignment with page index

            if tokenized_corpus:
                self._bm25 = BM25Okapi(tokenized_corpus)
            self._index_ready = True

            ocr_msg = ""
            if self._ocr_pages:
                ocr_msg = f" (OCR used on pages: {self._ocr_pages})"
            visual_msg = ""
            if self._visual_pages:
                visual_msg = f" (Visual content on pages: {self._visual_pages})"
            logger.info(f"Index built: {len(self._pages_info)} pages indexed"
                        f"{ocr_msg}{visual_msg}")

        except Exception:
            self._index_ready = False
            raise
        finally:
            if doc:
                doc.close()

    def search(self, query: str, k: int = 3) -> List[int]:
        if not self._index_ready or not self._bm25:
            return []
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        # get top k indices
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._pages_info[i]["page_num"] for i in top_n]

    def get_page_images(self, page_numbers: List[int]) -> Dict[int, str]:
        """Return base64-encoded page images for the given page numbers."""
        return {p: self._page_images[p] for p in page_numbers if p in self._page_images}

    def get_page_info(self, page_num: int) -> Optional[Dict]:
        """Return stored info (text + visual_description) for a page."""
        for info in self._pages_info:
            if info["page_num"] == page_num:
                return info
        return None

    def rebuild_index(self, pdf_path: str):
        self._bm25 = None
        self._pages_info = []
        self._ocr_pages = []
        self._visual_pages = []
        self._page_images = {}
        self._index_ready = False
        self.load_or_create_index(pdf_path)

    @property
    def is_ready(self) -> bool:
        return self._index_ready

    @property
    def ocr_page_count(self) -> int:
        return len(self._ocr_pages)

    @property
    def visual_page_count(self) -> int:
        return len(self._visual_pages)