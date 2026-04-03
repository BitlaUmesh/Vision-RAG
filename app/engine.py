import logging
from typing import List

import pymupdf as fitz
from rank_bm25 import BM25Okapi

from app.ocr import extract_text_hybrid

logger = logging.getLogger(__name__)

class VisionEngine:
    def __init__(self, index_root: str = ".byaldi", index_name: str = "vision_rag_index"):
        self._bm25 = None
        self._pages_info = []
        self._index_ready = False
        self._ocr_pages = []  # track which pages needed OCR

    def load_or_create_index(self, pdf_path: str):
        logger.info(f"Extracting text from {pdf_path}")
        self._pages_info = []
        self._ocr_pages = []
        tokenized_corpus = []

        doc = None
        try:
            doc = fitz.open(pdf_path)
            if len(doc) == 0:
                raise ValueError(f"PDF has 0 pages: {pdf_path}")

            total = len(doc)
            for i, page in enumerate(doc):
                page_num = i + 1
                logger.info(f"Processing page {page_num}/{total}...")

                # Hybrid extraction: embedded text first, OCR fallback
                embedded = page.get_text() or ""
                text = extract_text_hybrid(page)

                # Track if OCR was used (text differs from embedded)
                if len(embedded.strip()) < 50 and len(text.strip()) > len(embedded.strip()):
                    self._ocr_pages.append(page_num)

                self._pages_info.append({"page_num": page_num, "text": text})
                tokens = text.lower().split()
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
            logger.info(f"Index built: {len(self._pages_info)} pages indexed{ocr_msg}")

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

    def rebuild_index(self, pdf_path: str):
        self._bm25 = None
        self._pages_info = []
        self._ocr_pages = []
        self._index_ready = False
        self.load_or_create_index(pdf_path)

    @property
    def is_ready(self) -> bool:
        return self._index_ready

    @property
    def ocr_page_count(self) -> int:
        return len(self._ocr_pages)