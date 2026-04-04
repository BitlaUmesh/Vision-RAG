import re
from typing import Dict

# Only match genuinely casual greetings/farewells — be very strict.
# These must be the ENTIRE message (or very close) to count as casual.
CASUAL = [re.compile(p, re.I) for p in [
    r"^(hi|hello|hey|howdy|yo)[\s!.,?]*$",       # standalone greetings only
    r"^(bye|goodbye|see you|good night)[\s!.,?]*$",
    r"^(thanks|thank you|thx)[\s!.,?]*$",
    r"^who are you[\s?]*$",
    r"^how are you[\s?]*$",
    r"^what'?s up[\s?]*$",
    r"^good (morning|afternoon|evening)[\s!.,?]*$",
    r"^what is your name[\s?]*$",
    r"^your name[\s?]*$",
]]

# If ANY of these words appear, definitely use RAG
DOC_KEYWORDS = re.compile(
    r"\b(question|answer|explain|describe|summarize|summary|list|give|show|"
    r"tell|find|what|how|why|where|which|define|definition|"
    r"page|chapter|section|paragraph|point|number|"
    r"phase|diagram|table|figure|chart|graph|image|picture|"
    r"architecture|code|review|bug|document|pdf|"
    r"formula|equation|algorithm|method|model|"
    r"compare|difference|example|step|process|concept|topic)\b",
    re.I
)


def route_query(query: str) -> Dict:
    q = query.strip().lower()
    q_clean = re.sub(r'[^\w\s]', '', q)

    # Short messages (< 4 words) that match casual patterns → casual
    word_count = len(q_clean.split())

    # Check for document-related keywords FIRST — they always win
    if DOC_KEYWORDS.search(q_clean):
        return {"needs_rag": True, "reason": "Document keyword matched"}

    # Only treat as casual if it's a short, standalone greeting/farewell
    if word_count <= 5 and any(p.search(q_clean) for p in CASUAL):
        return {"needs_rag": False, "reason": "Casual greeting"}

    # Default: always use RAG — the user uploaded a document for a reason
    return {"needs_rag": True, "reason": "Default to RAG"}