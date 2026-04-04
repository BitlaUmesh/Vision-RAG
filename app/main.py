import logging
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
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

# Global state to track background indexing jobs
job_store: dict[str, dict] = {}

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
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
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

    # Create a new job ID for tracking
    job_id = str(uuid.uuid4())
    job_store[job_id] = {
        "status": "upload",
        "progress": 5,
        "message": "Upload complete. Submitting to queue...",
        "filename": file.filename,
    }

    # Define the background worker
    def _run_indexing(j_id: str, pdf_p: str):
        def _cb(step_id: str, progress: int, msg: str):
            if j_id in job_store:
                job_store[j_id]["status"] = step_id
                job_store[j_id]["progress"] = progress
                job_store[j_id]["message"] = msg
                # If completed with success, save extra metadata
                if step_id == "completed":
                    job_store[j_id]["ocr_pages"] = engine.ocr_page_count
                    job_store[j_id]["visual_pages"] = engine.visual_page_count

        try:
            engine.rebuild_index(pdf_p, status_callback=_cb)
        except Exception as e:
            logger.error(f"Background indexing `{j_id}` failed: {e}")

    # Kick off the background task
    background_tasks.add_task(_run_indexing, job_id, current_pdf_path)

    # Return immediately
    return {
        "job_id": job_id,
        "filename": file.filename,
        "message": "Indexing started in background"
    }

@app.get("/indexing/status/{job_id}")
async def get_indexing_status(job_id: str):
    if job_id not in job_store:
        raise HTTPException(404, "Job ID not found")
    return job_store[job_id]

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

# ── Streaming SSE chat endpoint ─────────────────────────────────────────
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    import json as _json

    global current_pdf_path
    user_msg = request.message.strip()

    routing = route_query(user_msg)
    needs_rag = routing["needs_rag"]

    retrieved_pages = []
    if needs_rag and engine.is_ready and current_pdf_path:
        retrieved_pages = engine.search(user_msg, k=RAG_TOP_K)

    history = memory.get_formatted_history()

    def event_generator():
        full_text = []
        for chunk in service.generate_stream(
            query=user_msg,
            page_numbers=retrieved_pages,
            pdf_path=current_pdf_path,
            chat_history=history,
            needs_rag=needs_rag,
            engine=engine,
        ):
            full_text.append(chunk)
            yield f"data: {_json.dumps({'type': 'token', 'content': chunk})}\n\n"

        # Send final metadata event
        complete_text = "".join(full_text)
        memory.add(user_msg, complete_text)

        yield f"data: {_json.dumps({'type': 'done', 'needs_rag': needs_rag, 'retrieved_pages': retrieved_pages, 'routing_reason': routing['reason']})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

# ── Serve the currently uploaded PDF ────────────────────────────────────
@app.get("/pdf/current")
async def get_current_pdf():
    if not current_pdf_path:
        raise HTTPException(404, "No PDF uploaded yet")
    pdf_file = Path(current_pdf_path)
    if not pdf_file.exists():
        raise HTTPException(404, "PDF file not found on disk")
    return FileResponse(
        path=str(pdf_file),
        media_type="application/pdf",
        filename=pdf_file.name,
    )

# MUST be last - static files catch-all
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")