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
        page_numbers: List[Dict],
        chat_history: str,
        needs_rag: bool = True,
        engine=None,
    ) -> str:
        context_text = self._extract_context(page_numbers, engine)
        system_prompt = self._build_system_prompt(context_text, chat_history, needs_rag)

        # Get page images for multimodal vision call
        page_images: Dict[str, str] = {}
        if needs_rag and engine and page_numbers:
            page_images = engine.get_page_images(page_numbers)

        if page_images:
            return self._call_vision(system_prompt, query, page_images)
        else:
            return self._call_text(system_prompt, query)

    # ── context extraction ────────────────────────────────────────────────

    @staticmethod
    def _extract_context(page_numbers: List[Dict], engine=None):
        if not page_numbers or not engine:
            return ""
        parts = []
        seen_pages = set()
        for p in page_numbers:
            fname = p["filename"]
            pnum = p["page_num"]
            page_key = f"{fname}_{pnum}"

            if page_key in seen_pages:
                continue
            seen_pages.add(page_key)

            # Use retrieved chunks directly if available (from ChromaDB search)
            retrieved_chunks = p.get("retrieved_chunks", [])
            if retrieved_chunks:
                chunk_text = "\n".join(retrieved_chunks)
                part = f"\n--- {fname} (Page {pnum}) ---\n{chunk_text}\n"
                parts.append(part)
            else:
                # Fallback: fetch full page info from engine
                info = engine.get_page_info(fname, pnum)
                if info:
                    part = f"\n--- {fname} (Page {pnum}) ---\n{info['text']}\n"
                    if info.get("visual_description"):
                        part += f"\n[VISUAL CONTENT on {fname} Page {pnum}]\n{info['visual_description']}\n"
                    parts.append(part)
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
                "== VISUAL CITATIONS (CRITICAL) ==\n"
                "Every time you state a fact from the context, you MUST provide a visual citation "
                "so the user can click it to see the exact text in the source document.\n"
                "Format: [[Page X: \"exact continuous quote string from text\"]]\n"
                "Example: The battery life is 10 hours [[Page 4: \"average continuous battery life of 10 hours\"]].\n"
                "The quote string MUST be an exact substring from the text payload provided. Do not alter the letters or punctuation. Keep the quote brief (4-8 words).\n\n"
                "IMPORTANT: The document text was extracted via OCR. "
                "There may be minor OCR errors. Try to quote exactly as extracted."
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

    def _call_vision(self, system_prompt: str, user_message: str, page_images: Dict[str, str]) -> str:
        last_err = None

        # Build multimodal content
        user_content = [{"type": "text", "text": user_message}]
        sorted_pages = sorted(page_images.keys())[:4]
        for page_id in sorted_pages:
            user_content.append({"type": "text", "text": f"\n[Page {page_id} image:]"})
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{page_images[page_id]}"},
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
        page_numbers: List[Dict],
        chat_history: str,
        needs_rag: bool = True,
        engine=None,
    ):
        """Generator that yields text chunks as they arrive from Groq API."""
        context_text = self._extract_context(page_numbers, engine)
        system_prompt = self._build_system_prompt(context_text, chat_history, needs_rag)

        page_images: Dict[str, str] = {}
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

    def _call_vision_stream(self, system_prompt: str, user_message: str, page_images: Dict[str, str]):
        """Generator: yields text chunks from Groq vision model with streaming."""
        last_err = None

        user_content = [{"type": "text", "text": user_message}]
        sorted_pages = sorted(page_images.keys())[:4]
        for page_id in sorted_pages:
            user_content.append({"type": "text", "text": f"\n[Page {page_id} image:]"})
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{page_images[page_id]}"},
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

class AgenticPlanner:
    """Agent that intercepts user query and outputs JSON search plan."""

    API_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self):
        import os
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY not set in .env")
        self.model = "llama-3.1-8b-instant"

    def analyze_query(self, query: str, history: str) -> dict:
        prompt = (
            "You are a sophisticated JSON query planner for a multi-document research assistant.\n"
            "Given the user's query and conversation history, you must output ONLY valid JSON.\n\n"
            "Decision logic:\n"
            '1. "needs_rag" (boolean): True if the query asks for factual info, comparisons, document contents, visual elements, logic, answers, etc. False ONLY for purely casual greetings (like "hi" or "who are you").\n\n'
            '2. "search_queries" (list of strings): If needs_rag is True, decompose the user\'s complex question into 1 to 3 optimized, highly specific keyword search queries. If the query asks to compare two things, create one query for each thing. Put the most important keywords first. If needs_rag is False, return [].\n\n'
            '3. "prioritize_visuals" (boolean): True if the user specifically asks for diagrams, images, tables, figures, charts, or visual architecture. False otherwise.\n\n'
            "User Query: " + query + "\n"
        )
        if history and history != "No previous conversation.":
            prompt += f"\nRecent History:\n{history}"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 512,
        }

        try:
            import json
            logger.info(f"→ AGENTIC PLANNER ({self.model}) analyzing query...")
            resp = http_requests.post(self.API_URL, headers=headers, json=payload, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            result = json.loads(content)
            
            # Ensure defaults
            return {
                "needs_rag": bool(result.get("needs_rag", True)),
                "search_queries": result.get("search_queries", [query])[:3],
                "prioritize_visuals": bool(result.get("prioritize_visuals", False))
            }
        except Exception as e:
            logger.warning(f"Agentic Planner failed: {e}. Falling back to default routing.")
            return {
                "needs_rag": True,
                "search_queries": [query],
                "prioritize_visuals": False
            }