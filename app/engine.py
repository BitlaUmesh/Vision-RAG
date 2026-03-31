import logging
from typing import List

import pymupdf as fitz
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

class VisionEngine:
    def __init__(self, index_root: str = ".byaldi", index_name: str = "vision_rag_index"):
        self._bm25 = None
        self._pages_info = []
        self._index_ready = False

    def load_or_create_index(self, pdf_path: str):
        logger.info(f"Extracting text from {pdf_path}")
        self._pages_info = []
        tokenized_corpus = []

        doc = None
        try:
            doc = fitz.open(pdf_path)
            if len(doc) == 0:
                raise ValueError(f"PDF has 0 pages: {pdf_path}")

            for i, page in enumerate(doc):
                text = page.get_text()
                self._pages_info.append({"page_num": i + 1, "text": text})
                tokens = text.lower().split()
                if tokens:
                    tokenized_corpus.append(tokens)
                else:
                    tokenized_corpus.append([""]) # keep alignment with page index

            if tokenized_corpus:
                self._bm25 = BM25Okapi(tokenized_corpus)
            self._index_ready = True
            logger.info(f"Index built: {len(self._pages_info)} pages indexed")
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
        self._index_ready = False
        self.load_or_create_index(pdf_path)

    @property
    def is_ready(self) -> bool:
        return self._index_ready