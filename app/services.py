import logging
import os
import time
from typing import List, Optional

import requests as http_requests

logger = logging.getLogger(__name__)

_API_TIMEOUT = 60  # Groq is fast, but give enough time for long prompts


class VisionService:
    """Generates answers using the Groq API (OpenAI-compatible)."""

    FALLBACK_MODELS = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
    ]
    API_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY not set in .env — get one free at https://console.groq.com/keys")
        configured = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.models = [configured] + [m for m in self.FALLBACK_MODELS if m != configured]
        logger.info(f"Groq models (priority): {self.models}")

    # ── public ────────────────────────────────────────────────────────────

    def generate(
        self,
        query: str,
        page_numbers: List[int],
        pdf_path: Optional[str],
        chat_history: str,
        needs_rag: bool = True,
    ) -> str:
        context_text = self._extract_context(page_numbers, pdf_path)
        system_prompt = self._build_system_prompt(context_text, chat_history, needs_rag)
        return self._call_groq(system_prompt, query)

    # ── context extraction ────────────────────────────────────────────────

    @staticmethod
    def _extract_context(page_numbers: List[int], pdf_path: Optional[str]) -> str:
        if not page_numbers or not pdf_path:
            return ""
        doc = None
        parts = []
        try:
            import pymupdf as fitz
            doc = fitz.open(pdf_path)
            for p in page_numbers:
                if 1 <= p <= len(doc):
                    parts.append(f"\n--- Page {p} ---\n{doc[p - 1].get_text()}\n")
        except Exception as e:
            logger.error(f"PDF text extraction error: {e}")
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
                "You are a helpful document assistant. If the user asks about the document, "
                "answer based on the retrieved context above. If the user's message is casual "
                "small-talk or a greeting, respond naturally and conversationally."
            )
        else:
            prompt += (
                "You are a helpful, friendly, and conversational AI assistant. "
                "Respond naturally and politely."
            )
        return prompt

    # ── Groq API calls ────────────────────────────────────────────────────

    def _call_groq(self, system_prompt: str, user_message: str) -> str:
        """Try each model with retry on rate-limit."""
        last_err = None

        for model in self.models:
            for attempt in range(2):
                try:
                    return self._single_call(model, system_prompt, user_message)
                except http_requests.exceptions.Timeout:
                    logger.warning(f"{model} timed out (attempt {attempt + 1})")
                    last_err = "Request timed out"
                    break
                except http_requests.exceptions.HTTPError as e:
                    code = e.response.status_code if e.response is not None else 0
                    if code == 429:
                        logger.warning(f"{model} rate-limited, waiting...")
                        last_err = "Rate limited"
                        time.sleep(3 * (attempt + 1))
                        continue
                    elif code == 404:
                        logger.warning(f"{model} not found, skipping")
                        last_err = f"Model {model} not available"
                        break
                    else:
                        body = e.response.text[:200] if e.response is not None else str(e)
                        logger.error(f"Groq HTTP {code}: {body}")
                        return f"⚠️ API error ({code}): {body}"
                except Exception as e:
                    logger.error(f"Groq call failed: {e}", exc_info=True)
                    return f"⚠️ Generation error: {e}"

        return (
            f"⚠️ All models are currently rate-limited ({last_err}). "
            "Please wait a moment and try again."
        )

    def _single_call(self, model: str, system_prompt: str, user_message: str) -> str:
        """One request to the Groq chat completions API."""
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
            "temperature": 0.7,
            "max_tokens": 2048,
        }
        logger.info(f"→ Groq {model}")

        resp = http_requests.post(self.API_URL, headers=headers, json=payload, timeout=_API_TIMEOUT)
        resp.raise_for_status()

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return "No response generated."
        return choices[0].get("message", {}).get("content", "No content in response.")