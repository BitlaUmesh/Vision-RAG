"""
ocr.py — Hybrid text extraction with Vision LLM fallback for handwritten PDFs.
================================================================================
Strategy:
  1. Try pymupdf's get_text() first (fast, works for printed/typed PDFs).
  2. If the page yields < MIN_TEXT_CHARS meaningful characters, assume it's
     a scanned/handwritten page and fall back to Groq Vision LLM.
  3. The vision model reads the page image directly — far superior to
     traditional OCR for messy handwriting.
"""

import base64
import io
import logging
import os
import re
import time
from typing import Optional

import requests as http_requests
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Pages with fewer than this many "real" characters trigger vision fallback.
MIN_TEXT_CHARS = 50

# DPI for rendering PDF pages to images.
OCR_DPI = int(os.getenv("PDF_DPI", "300"))

# Groq API config (reuses the same key as the main chat service).
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Vision models — try these in order.
VISION_MODELS = [
    os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
    "llama-3.2-90b-vision-preview",
    "llama-3.2-11b-vision-preview",
]

VISION_PROMPT = (
    "You are an expert handwriting reader. Extract ALL the text from this "
    "handwritten page image. Reproduce the text exactly as written — preserve "
    "paragraphs, line breaks, headings, numbering, and bullet points. "
    "If a word is unclear, make your best guess based on context. "
    "Do NOT describe the image or add commentary — output ONLY the extracted text."
)


# ---------------------------------------------------------------------------
# Image Helpers
# ---------------------------------------------------------------------------


def _page_to_base64(page, dpi: int) -> str:
    """Render a pymupdf page to a base64-encoded JPEG string."""
    import pymupdf

    zoom = dpi / 72
    mat = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    # Convert pixmap → PIL → JPEG bytes → base64
    pil_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    # Limit max dimension to avoid exceeding API payload limits
    max_dim = 2048
    w, h = pil_img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    size_kb = len(buf.getvalue()) / 1024
    logger.info(f"Page image: {pil_img.size[0]}x{pil_img.size[1]}, {size_kb:.0f}KB")
    return b64


# ---------------------------------------------------------------------------
# Groq Vision API
# ---------------------------------------------------------------------------


def _call_vision(image_b64: str, page_num: int) -> Optional[str]:
    """
    Send a page image to a Groq vision model and get back extracted text.
    Tries each vision model in VISION_MODELS until one succeeds.
    """
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY not set — cannot use vision OCR")
        return None

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": VISION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}",
                    },
                },
            ],
        }
    ]

    last_err = None

    for model in VISION_MODELS:
        for attempt in range(4):  # More retries for rate limits
            try:
                logger.info(f"Page {page_num}: trying vision model {model} "
                            f"(attempt {attempt + 1})")

                payload = {
                    "model": model,
                    "messages": payload_messages,
                    "temperature": 0.1,  # Low temp for faithful extraction
                    "max_tokens": 4096,
                }

                resp = http_requests.post(
                    GROQ_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=60,
                )

                if resp.status_code == 429:
                    wait = 5 * (attempt + 1)
                    logger.warning(f"Page {page_num}: rate-limited on {model}, "
                                   f"waiting {wait}s...")
                    last_err = "rate-limited"
                    time.sleep(wait)
                    continue

                if resp.status_code == 404:
                    logger.warning(f"Model {model} not available, trying next")
                    last_err = f"model {model} not found"
                    break  # try next model

                if resp.status_code == 400:
                    body = resp.text[:300]
                    logger.warning(f"Page {page_num}: 400 error from {model}: {body}")
                    last_err = f"400: {body[:100]}"
                    break  # try next model

                resp.raise_for_status()

                data = resp.json()
                choices = data.get("choices", [])
                if choices:
                    text = choices[0].get("message", {}).get("content", "")
                    if text.strip():
                        return text.strip()

                logger.warning(f"Page {page_num}: empty response from {model}")
                last_err = "empty response"
                break  # try next model

            except http_requests.exceptions.Timeout:
                logger.warning(f"Page {page_num}: {model} timed out")
                last_err = "timeout"
                break
            except Exception as e:
                logger.error(f"Page {page_num}: vision call error: {e}")
                last_err = str(e)
                break

    logger.error(f"Page {page_num}: all vision models failed ({last_err})")
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _meaningful_length(text: str) -> int:
    """Count non-whitespace, non-control characters."""
    return len(re.sub(r"\s+", "", text))


def extract_text_hybrid(page, dpi: int | None = None) -> str:
    """
    Extract text from a pymupdf page, falling back to vision LLM if needed.

    Parameters
    ----------
    page : fitz.Page
        An open pymupdf page object.
    dpi : int, optional
        Render DPI for the page image. Defaults to OCR_DPI from env / 300.

    Returns
    -------
    str
        The extracted text (from embedded text or vision LLM).
    """
    if dpi is None:
        dpi = OCR_DPI

    # --- Attempt 1: embedded text (fast path) ---
    embedded = page.get_text() or ""
    if _meaningful_length(embedded) >= MIN_TEXT_CHARS:
        return embedded

    # --- Attempt 2: vision LLM fallback ---
    page_num = page.number + 1
    logger.info(
        f"Page {page_num}: only {_meaningful_length(embedded)} chars "
        f"of embedded text — using vision LLM at {dpi} DPI"
    )

    try:
        image_b64 = _page_to_base64(page, dpi)
        vision_text = _call_vision(image_b64, page_num)

        if vision_text:
            logger.info(f"Page {page_num}: vision LLM extracted "
                        f"{len(vision_text)} chars")
            return vision_text
        else:
            logger.warning(f"Page {page_num}: vision LLM returned no text")
            return embedded

    except Exception as e:
        logger.error(f"Page {page_num}: vision pipeline error: {e}",
                     exc_info=True)
        return embedded
