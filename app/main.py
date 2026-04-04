import logging
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from app.engine import VisionEngine
from app.memory import ChatMemory
from app.router import route_query
from app.services import VisionService

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "data/uploads"))
INDEX_ROOT = os.getenv("BYALDI_INDEX_ROOT", ".byaldi")
INDEX_NAME = os.getenv("BYALDI_INDEX_NAME", "vision_rag_index")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

engine = VisionEngine(index_root=INDEX_ROOT, index_name=INDEX_NAME)
memory = ChatMemory()
service = VisionService()
current_pdf_path: Optional[str] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global current_pdf_path
    logger.info("=" * 60)
    logger.info("Vision RAG - Phase 1 Ready")
    logger.info("=" * 60)

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing PDF + index IN BACKGROUND (non-blocking)
    existing = list(UPLOAD_DIR.glob("*.pdf"))
    if existing:
        latest = max(existing, key=lambda p: p.stat().st_mtime)
        current_pdf_path = str(latest)
        logger.info(f"Found existing PDF: {current_pdf_path}")
        import threading
        def _bg_index():
            try:
                engine.load_or_create_index(current_pdf_path)
            except Exception as e:
                logger.warning(f"Background index failed: {e}")
        threading.Thread(target=_bg_index, daemon=True).start()
        logger.info("Indexing started in background — server is ready")

    yield
    logger.info("Shutdown complete.")

app = FastAPI(title="Vision RAG - Phase 1", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# No-cache middleware — prevents browsers from caching static files
class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path.endswith(('.html', '.css', '.js', '/')):
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
        return response

app.add_middleware(NoCacheMiddleware)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    needs_rag: bool
    retrieved_pages: list[int]
    routing_reason: str

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "index_ready": engine.is_ready,
        "current_pdf": current_pdf_path,
        "conversation_turns": len(memory)
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global current_pdf_path
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files allowed")

    save_path = UPLOAD_DIR / file.filename

    # Save the uploaded file to disk
    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(400, "Uploaded file is empty")
        with open(save_path, "wb") as f:
            f.write(contents)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(500, f"Failed to save file: {e}")

    current_pdf_path = str(save_path)
    memory.clear()

    try:
        engine.rebuild_index(current_pdf_path)
        page_count = len(engine._pages_info)
        ocr_count = engine.ocr_page_count
        visual_count = engine.visual_page_count
        index_msg = f"Indexed {page_count} pages"
        if ocr_count > 0:
            index_msg += f" ({ocr_count} handwritten pages via OCR)"
        if visual_count > 0:
            index_msg += f" · {visual_count} pages with visual content analyzed"
        return {
            "filename": file.filename,
            "index_status": index_msg,
            "status": "success",
            "ocr_pages": ocr_count,
            "visual_pages": visual_count,
        }
    except Exception as e:
        logger.error(f"Indexing failed: {e}", exc_info=True)
        raise HTTPException(500, f"Indexing failed: {e}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global current_pdf_path
    user_msg = request.message.strip()

    routing = route_query(user_msg)
    needs_rag = routing["needs_rag"]

    retrieved_pages = []
    if needs_rag and engine.is_ready and current_pdf_path:
        retrieved_pages = engine.search(user_msg, k=RAG_TOP_K)

    history = memory.get_formatted_history()
    answer = service.generate(
        query=user_msg,
        page_numbers=retrieved_pages,
        pdf_path=current_pdf_path,
        chat_history=history,
        needs_rag=needs_rag,
        engine=engine,
    )

    memory.add(user_msg, answer)

    return ChatResponse(
        response=answer,
        needs_rag=needs_rag,
        retrieved_pages=retrieved_pages,
        routing_reason=routing["reason"]
    )

@app.delete("/memory")
async def clear_memory():
    memory.clear()
    return {"message": "Memory cleared"}

# MUST be last - static files catch-all
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")