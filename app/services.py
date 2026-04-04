import logging
import os
import time
from typing import Dict, List, Optional

import requests as http_requests

from app.ocr import extract_text_hybrid

logger = logging.getLogger(__name__)

_API_TIMEOUT = 90


class VisionService:
    """Generates answers using Groq API — text + multimodal vision."""

    TEXT_MODELS = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
    ]

    VISION_MODELS = [
        os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
    ]

    API_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY not set in .env")
        configured = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.text_models = [configured] + [m for m in self.TEXT_MODELS if m != configured]
        logger.info(f"Text models: {self.text_models}")
        logger.info(f"Vision models: {self.VISION_MODELS}")

    # ── public ────────────────────────────────────────────────────────────

    def generate(
        self,
        query: str,
        page_numbers: List[int],
        pdf_path: Optional[str],
        chat_history: str,
        needs_rag: bool = True,
        engine=None,
    ) -> str:
        context_text = self._extract_context(page_numbers, pdf_path, engine)
        system_prompt = self._build_system_prompt(context_text, chat_history, needs_rag)

        # Get page images for multimodal vision call
        page_images: Dict[int, str] = {}
        if needs_rag and engine and page_numbers:
            page_images = engine.get_page_images(page_numbers)

        if page_images:
            return self._call_vision(system_prompt, query, page_images)
        else:
            return self._call_text(system_prompt, query)

    # ── context extraction ────────────────────────────────────────────────

    @staticmethod
    def _extract_context(page_numbers, pdf_path, engine=None):
        if not page_numbers or not pdf_path:
            return ""
        parts = []
        if engine:
            for p in page_numbers:
                info = engine.get_page_info(p)
                if info:
                    part = f"\n--- Page {p} ---\n{info['text']}\n"
                    if info.get("visual_description"):
                        part += f"\n[VISUAL CONTENT on Page {p}]\n{info['visual_description']}\n"
                    parts.append(part)
            if parts:
                return "".join(parts)

        doc = None
        try:
            import pymupdf as fitz
            doc = fitz.open(pdf_path)
            for p in page_numbers:
                if 1 <= p <= len(doc):
                    page = doc[p - 1]
                    text = extract_text_hybrid(page)
                    parts.append(f"\n--- Page {p} ---\n{text}\n")
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
        finally:
            if doc:
                doc.close()
        return "".join(parts)

    # ── prompt builder ────────────────────────────────────────────────────

    @staticmethod
    def _build_system_prompt(context: str, history: str, needs_rag: bool) -> str:
        prompt = ""
        if history and history != "No previous conversation.":
            prompt += f"Previous conversation:\n{history}\n\n"
        if context:
            prompt += f"Retrieved document context:\n{context}\n\n"
        if needs_rag:
            prompt += (
                "You are a precise document assistant with VISUAL understanding. "
                "You have access to the actual document pages — both extracted text and "
                "page images. You can see and interpret diagrams, tables, charts, "
                "flowcharts, and any visual elements.\n\n"
                "CRITICAL RULES:\n"
                "1. ALWAYS answer from the retrieved document context provided above. "
                "The context contains the relevant pages from the user's uploaded document.\n"
                "2. NEVER say 'I don't have information' or 'I can't find' when document "
                "context is provided above — the answer IS in the context, find it.\n"
                "3. Be ACCURATE and FAITHFUL to the document content. Quote or paraphrase "
                "directly from the text. Do not make up information.\n"
                "4. When the user refers to 'questions', 'points', 'items', or numbered "
                "content, look for them in the document context and reproduce them exactly.\n"
                "5. If the user asks about diagrams, charts, or tables, describe what you "
                "see in the page images directly.\n"
                "6. For tables, extract and present data in a clear markdown table format.\n"
                "7. If you are shown page images, READ THE TEXT DIRECTLY from the images "
                "for maximum accuracy — do not rely solely on the OCR-extracted text.\n\n"
                "IMPORTANT: The document text was extracted via OCR from handwritten pages. "
                "There may be minor OCR errors. If the page images are provided, read the "
                "actual handwriting in the images for the most accurate answer."
            )
        else:
            prompt += (
                "You are a helpful, friendly, and conversational AI assistant. "
                "Respond naturally and politely."
            )
        return prompt

    # ── Text-only call ────────────────────────────────────────────────────

    def _call_text(self, system_prompt: str, user_message: str) -> str:
        last_err = None
        for model in self.text_models:
            for attempt in range(2):
                try:
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    }
                    payload = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        "temperature": 0.5,
                        "max_tokens": 4096,
                    }
                    logger.info(f"→ TEXT {model}")
                    resp = http_requests.post(self.API_URL, headers=headers, json=payload, timeout=_API_TIMEOUT)
                    resp.raise_for_status()
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content", "No content.")
                    return "No response generated."
                except http_requests.exceptions.Timeout:
                    last_err = "timeout"
                    break
                except http_requests.exceptions.HTTPError as e:
                    code = e.response.status_code if e.response else 0
                    if code == 429:
                        last_err = "rate-limited"
                        time.sleep(3 * (attempt + 1))
                        continue
                    elif code == 404:
                        last_err = f"{model} not found"
                        break
                    else:
                        body = e.response.text[:200] if e.response else str(e)
                        return f"⚠️ API error ({code}): {body}"
                except Exception as e:
                    return f"⚠️ Error: {e}"
        return f"⚠️ All models failed ({last_err}). Please wait and try again."

    # ── Multimodal vision call ────────────────────────────────────────────

    def _call_vision(self, system_prompt: str, user_message: str, page_images: Dict[int, str]) -> str:
        last_err = None

        # Build multimodal content
        user_content = [{"type": "text", "text": user_message}]
        sorted_pages = sorted(page_images.keys())[:4]
        for page_num in sorted_pages:
            user_content.append({"type": "text", "text": f"\n[Page {page_num} image:]"})
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{page_images[page_num]}"},
            })

        logger.info(f"→ VISION with {len(sorted_pages)} pages: {sorted_pages}")

        for model in self.VISION_MODELS:
            for attempt in range(3):
                try:
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    }
                    payload = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content},
                        ],
                        "temperature": 0.3,
                        "max_tokens": 4096,
                    }
                    logger.info(f"→ VISION {model} (attempt {attempt + 1})")
                    resp = http_requests.post(self.API_URL, headers=headers, json=payload, timeout=_API_TIMEOUT)

                    if resp.status_code == 429:
                        wait = 5 * (attempt + 1)
                        logger.warning(f"{model} rate-limited, waiting {wait}s...")
                        last_err = "rate-limited"
                        time.sleep(wait)
                        continue
                    if resp.status_code == 404:
                        last_err = f"{model} not found"
                        break
                    if resp.status_code == 400:
                        body = resp.text[:300]
                        last_err = f"400: {body[:100]}"
                        break

                    resp.raise_for_status()
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                        if content.strip():
                            return content.strip()
                    last_err = "empty response"
                    break
                except http_requests.exceptions.Timeout:
                    last_err = "timeout"
                    break
                except Exception as e:
                    last_err = str(e)
                    break

        # Fallback to text-only
        logger.warning(f"Vision failed ({last_err}), falling back to text-only")
        return self._call_text(system_prompt, user_message)

    # ── Streaming public API ──────────────────────────────────────────────

    def generate_stream(
        self,
        query: str,
        page_numbers: List[int],
        pdf_path: Optional[str],
        chat_history: str,
        needs_rag: bool = True,
        engine=None,
    ):
        """Generator that yields text chunks as they arrive from Groq API."""
        context_text = self._extract_context(page_numbers, pdf_path, engine)
        system_prompt = self._build_system_prompt(context_text, chat_history, needs_rag)

        page_images: Dict[int, str] = {}
        if needs_rag and engine and page_numbers:
            page_images = engine.get_page_images(page_numbers)

        if page_images:
            yield from self._call_vision_stream(system_prompt, query, page_images)
        else:
            yield from self._call_text_stream(system_prompt, query)

    # ── Streaming text call ───────────────────────────────────────────────

    def _call_text_stream(self, system_prompt: str, user_message: str):
        """Generator: yields text chunks from Groq text model with streaming."""
        last_err = None
        for model in self.text_models:
            for attempt in range(2):
                try:
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    }
                    payload = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        "temperature": 0.5,
                        "max_tokens": 4096,
                        "stream": True,
                    }
                    logger.info(f"→ TEXT STREAM {model}")
                    resp = http_requests.post(
                        self.API_URL, headers=headers, json=payload,
                        timeout=_API_TIMEOUT, stream=True,
                    )
                    resp.raise_for_status()

                    for chunk_text in self._parse_sse_stream(resp):
                        yield chunk_text
                    return  # success

                except http_requests.exceptions.Timeout:
                    last_err = "timeout"
                    break
                except http_requests.exceptions.HTTPError as e:
                    code = e.response.status_code if e.response else 0
                    if code == 429:
                        last_err = "rate-limited"
                        time.sleep(3 * (attempt + 1))
                        continue
                    elif code == 404:
                        last_err = f"{model} not found"
                        break
                    else:
                        body = e.response.text[:200] if e.response else str(e)
                        yield f"⚠️ API error ({code}): {body}"
                        return
                except Exception as e:
                    yield f"⚠️ Error: {e}"
                    return
        yield f"⚠️ All models failed ({last_err}). Please wait and try again."

    # ── Streaming vision call ─────────────────────────────────────────────

    def _call_vision_stream(self, system_prompt: str, user_message: str, page_images: Dict[int, str]):
        """Generator: yields text chunks from Groq vision model with streaming."""
        last_err = None

        user_content = [{"type": "text", "text": user_message}]
        sorted_pages = sorted(page_images.keys())[:4]
        for page_num in sorted_pages:
            user_content.append({"type": "text", "text": f"\n[Page {page_num} image:]"})
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{page_images[page_num]}"},
            })

        logger.info(f"→ VISION STREAM with {len(sorted_pages)} pages: {sorted_pages}")

        for model in self.VISION_MODELS:
            for attempt in range(3):
                try:
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    }
                    payload = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content},
                        ],
                        "temperature": 0.3,
                        "max_tokens": 4096,
                        "stream": True,
                    }
                    logger.info(f"→ VISION STREAM {model} (attempt {attempt + 1})")
                    resp = http_requests.post(
                        self.API_URL, headers=headers, json=payload,
                        timeout=_API_TIMEOUT, stream=True,
                    )

                    if resp.status_code == 429:
                        wait = 5 * (attempt + 1)
                        logger.warning(f"{model} rate-limited, waiting {wait}s...")
                        last_err = "rate-limited"
                        time.sleep(wait)
                        continue
                    if resp.status_code == 404:
                        last_err = f"{model} not found"
                        break
                    if resp.status_code == 400:
                        body = resp.text[:300]
                        last_err = f"400: {body[:100]}"
                        break

                    resp.raise_for_status()

                    has_content = False
                    for chunk_text in self._parse_sse_stream(resp):
                        has_content = True
                        yield chunk_text

                    if has_content:
                        return  # success

                    last_err = "empty response"
                    break
                except http_requests.exceptions.Timeout:
                    last_err = "timeout"
                    break
                except Exception as e:
                    last_err = str(e)
                    break

        # Fallback to text-only streaming
        logger.warning(f"Vision stream failed ({last_err}), falling back to text-only stream")
        yield from self._call_text_stream(system_prompt, user_message)

    # ── SSE parser ────────────────────────────────────────────────────────

    @staticmethod
    def _parse_sse_stream(resp):
        """Parse Groq SSE stream and yield content delta strings."""
        import json
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    return
                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                except json.JSONDecodeError:
                    continue