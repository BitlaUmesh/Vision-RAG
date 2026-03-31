import re
from typing import Dict

CASUAL = [re.compile(p, re.I) for p in [
    r"^(hi|hello|hey|howdy)", r"^(bye|goodbye)", r"^(thanks|thank you)", 
    r"who are you", r"how are you", r"what'?s up", r"good (morning|afternoon|evening|night)",
    r"what are you doing", r"^tell me", r"^can you", r"your name"
]]
DOC_KEYWORDS = re.compile(r"\b(phase|diagram|table|figure|architecture|code|review|bug|document|pdf)\b", re.I)

def route_query(query: str) -> Dict:
    q = query.strip().lower()
    q_clean = re.sub(r'[^\w\s]', '', q)
    if any(p.search(q_clean) for p in CASUAL):
        return {"needs_rag": False, "reason": "Casual query - no retrieval"}
    if DOC_KEYWORDS.search(q_clean):
        return {"needs_rag": True, "reason": "Document keyword matched"}
    return {"needs_rag": True, "reason": "Default to RAG"}