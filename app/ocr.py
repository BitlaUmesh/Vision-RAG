"""
ocr.py — Hybrid text extraction with Vision LLM for handwritten PDFs.
================================================================================
Strategy:
  1. Try pymupdf's get_text() first (fast, works for printed/typed PDFs).
  2. If the page yields < MIN_TEXT_CHARS meaningful characters, assume it's
     a scanned/handwritten page and fall back to Groq Vision LLM.
  3. The vision model reads the page image directly.
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

MIN_TEXT_CHARS = 50
OCR_DPI = int(os.getenv("PDF_DPI", "300"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

VISION_MODELS = [
    os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
]

# Highly specific OCR prompt — forces word-by-word accuracy
VISION_PROMPT = (
    "You are an expert handwriting transcription specialist. Your task is to "
    "extract ALL text from this handwritten page image with MAXIMUM ACCURACY.\n\n"
    "RULES:\n"
    "1. Read each word LETTER BY LETTER. Do not guess or substitute words.\n"
    "2. Reproduce the text EXACTLY as written on the page.\n"
    "3. Preserve the EXACT question numbers (e.g., 1(A), 2(B), 3(A)) as they appear.\n"
    "4. Preserve paragraphs, line breaks, headings, numbering, bullet points, "
    "and underlines.\n"
    "5. If there is a heading at the top of the page (like 'ASSIGNMENT'), include it.\n"
    "6. If there is any institution name or logo text, include it.\n"
    "7. Do NOT paraphrase or summarize — output the EXACT handwritten text.\n"
    "8. Do NOT add any commentary, descriptions, or annotations.\n"
    "9. If a word is genuinely unreadable, write [illegible].\n\n"
    "Output ONLY the transcribed text from the page, nothing else."
)

VISUAL_ANALYSIS_PROMPT = (
    "You are an expert document analyst. Analyze this page image and describe "
    "ALL visual elements in detail. For each visual element found, provide:\n\n"
    "## Diagrams & Flowcharts\n"
    "Describe any diagrams, flowcharts, architecture drawings, or visual workflows. "
    "Explain the components, connections, and flow direction.\n\n"
    "## Tables\n"
    "Extract any tables as markdown tables with proper headers and rows. "
    "Preserve all data values accurately.\n\n"
    "## Charts & Graphs\n"
    "Describe any charts, graphs, or plots — type (bar, line, pie, etc.), "
    "axes labels, data trends, and key values.\n\n"
    "## Images & Figures\n"
    "Describe any embedded images, photos, screenshots, or figures. "
    "Note captions and figure numbers.\n\n"
    "## Annotations & Labels\n"
    "Note any callouts, arrows, highlighted text, or margin notes.\n\n"
    "If a section has no content, skip it. Be thorough and precise. "
    "Output ONLY the structured analysis — no preamble."
)


# ---------------------------------------------------------------------------
# Image Helpers
# ---------------------------------------------------------------------------


def page_to_base64(page, dpi: int) -> str:
    """Render a pymupdf page to a base64-encoded JPEG string."""
    import pymupdf

    zoom = dpi / 72
    mat = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    pil_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    max_dim = 2048
    w, h = pil_img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    size_kb = len(buf.getvalue()) / 1024
    logger.info(f"Page image: {pil_img.size[0]}x{pil_img.size[1]}, {size_kb:.0f}KB")
    return b64


_page_to_base64 = page_to_base64


# ---------------------------------------------------------------------------
# Groq Vision API
# ---------------------------------------------------------------------------


def _call_vision(image_b64: str, prompt: str, page_num: int) -> Optional[str]:
    """
    Send a page image to a Groq vision model and get back text.
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
                {"type": "text", "text": prompt},
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
        for attempt in range(4):
            try:
                logger.info(f"Page {page_num}: {model} (attempt {attempt + 1})")

                payload = {
                    "model": model,
                    "messages": payload_messages,
                    "temperature": 0.05,  # Very low for faithful transcription
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
                    break

                if resp.status_code == 400:
                    body = resp.text[:300]
                    logger.warning(f"Page {page_num}: 400 from {model}: {body}")
                    last_err = f"400: {body[:100]}"
                    break

                resp.raise_for_status()

                data = resp.json()
                choices = data.get("choices", [])
                if choices:
                    text = choices[0].get("message", {}).get("content", "")
                    if text.strip():
                        return text.strip()

                logger.warning(f"Page {page_num}: empty response from {model}")
                last_err = "empty response"
                break

            except http_requests.exceptions.Timeout:
                logger.warning(f"Page {page_num}: {model} timed out")
                last_err = "timeout"
                break
            except Exception as e:
                logger.error(f"Page {page_num}: vision error: {e}")
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
    if dpi is None:
        dpi = OCR_DPI

    embedded = page.get_text() or ""
    if _meaningful_length(embedded) >= MIN_TEXT_CHARS:
        return embedded

    page_num = page.number + 1
    logger.info(
        f"Page {page_num}: only {_meaningful_length(embedded)} chars "
        f"of embedded text — using vision LLM at {dpi} DPI"
    )

    try:
        image_b64 = page_to_base64(page, dpi)
        vision_text = _call_vision(image_b64, VISION_PROMPT, page_num)

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


def extract_visual_description(page, dpi: int | None = None) -> Optional[str]:
    if dpi is None:
        dpi = OCR_DPI

    page_num = page.number + 1
    logger.info(f"Page {page_num}: running visual content analysis...")

    try:
        image_b64 = page_to_base64(page, dpi)
        description = _call_vision(image_b64, VISUAL_ANALYSIS_PROMPT, page_num)

        if description:
            logger.info(f"Page {page_num}: visual analysis produced "
                        f"{len(description)} chars")
            return description
        else:
            logger.info(f"Page {page_num}: no visual content detected")
            return None

    except Exception as e:
        logger.error(f"Page {page_num}: visual analysis error: {e}",
                     exc_info=True)
        return None
