from __future__ import annotations
import os
from collections import OrderedDict
from pathlib import Path
from typing import List
import json
import time


from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from multi_doc_chat.src.document_ingestion.data_ingestion import ChatIngestor
from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG
from langchain_core.messages import HumanMessage, AIMessage
from multi_doc_chat.exceptions.custom_exception import DocumentPortalException
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.utils.session_store import get_session_store
from multi_doc_chat.utils.config_loader import load_config


# ----------------------------
# FastAPI initialization
# ----------------------------
app = FastAPI(title="MultiDocChat", version="0.1.0")

# CORS (optional for local dev). Wildcard origins are only valid with
# allow_credentials=False per the CORS spec; this API uses no cookies (the
# session id travels in the request body), so credentials stay disabled.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static and templates
BASE_DIR = Path(__file__).resolve().parent
static_dir = BASE_DIR / "static"
templates_dir = BASE_DIR / "templates"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))


# ----------------------------
# Chat history store
# ----------------------------
# In-memory by default; Redis-backed (shared across workers) if REDIS_URL is set.
SESSIONS = get_session_store()

# Per-process LRU of ConversationalRAG objects (a live FAISS retriever isn't
# serializable, so this stays process-local; it's rebuilt from the on-disk index on a
# miss). Bounded so it can't grow without limit.
RAG_CACHE_MAX = int(os.getenv("RAG_CACHE_SIZE", "32"))
RAG_CACHE: "OrderedDict[str, ConversationalRAG]" = OrderedDict()


def _get_rag(session_id: str) -> "ConversationalRAG":
    """Return the session's ConversationalRAG, building + caching it on first use.

    Rebuilding per request re-reads .env/config, re-inits embeddings, and reloads the
    FAISS index every time. Caching keeps the retriever + LLM warm for the session,
    cutting setup latency on every /chat and /chat/stream call (incl. TTFT). The cache
    is a bounded LRU — least-recently-used sessions are evicted past RAG_CACHE_MAX.
    """
    rag = RAG_CACHE.get(session_id)
    if rag is not None:
        RAG_CACHE.move_to_end(session_id)  # mark most-recently-used
        return rag

    engine = (load_config().get("rag",{}) or {}).get("engine","standard")
    if engine == "corrective":
        from multi_doc_chat.src.document_chat.agentic_rag import CorrectiveRAG
        rag = CorrectiveRAG(session_id=session_id)
    else:
        rag = ConversationalRAG(session_id=session_id)
        
    rag.load_retriever_from_faiss(
        index_path=f"faiss_index/{session_id}",
        search_type="mmr", fetch_k=20, lambda_mult=0.5,
    )
    RAG_CACHE[session_id] = rag
    RAG_CACHE.move_to_end(session_id)
    while len(RAG_CACHE) > RAG_CACHE_MAX:
        RAG_CACHE.popitem(last=False)  # evict least-recently-used
    return rag


# ----------------------------
# Adapters
# ----------------------------
class FastAPIFileAdapter:
    """Adapt FastAPI UploadFile to a simple object with .name and .getbuffer()."""
    def __init__(self, uf: UploadFile):
        self._uf = uf
        self.name = uf.filename or "file"

    def getbuffer(self) -> bytes:
        self._uf.file.seek(0)
        return self._uf.file.read()


# ----------------------------
# Models
# ----------------------------
class UploadResponse(BaseModel):
    session_id: str
    indexed: bool
    message: str | None = None


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


@app.post("/upload", response_model=UploadResponse)
async def upload(files: List[UploadFile] = File(...)) -> UploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    try:
        # Wrap FastAPI files to preserve filename/ext and provide a read buffer
        wrapped_files = [FastAPIFileAdapter(f) for f in files]

        ingestor = ChatIngestor(use_session_dirs=True)
        session_id = ingestor.session_id

        # Save, load, split, embed, and write FAISS index with MMR
        ingestor.build_retriever(
            uploaded_files=wrapped_files,
            search_type="mmr",
            fetch_k=20,
            lambda_mult=0.5
        )

        # Initialize empty history for this session
        SESSIONS.create(session_id)

        return UploadResponse(session_id=session_id, indexed=True, message="Indexing complete with MMR")
    except DocumentPortalException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id
    message = req.message.strip()
    if not session_id or not SESSIONS.exists(session_id):
        raise HTTPException(status_code=400, detail="Invalid or expired session_id. Re-upload documents.")
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        # Reuse the session's cached RAG (built once on first request)
        rag = _get_rag(session_id)

        # Convert stored history to LangChain messages
        lc_history = _lc_history(session_id)

        answer = rag.invoke(message, chat_history=lc_history)

        # Update history
        SESSIONS.append(session_id, "user", message)
        SESSIONS.append(session_id, "assistant", answer)

        return ChatResponse(answer=answer)
    except DocumentPortalException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")


def _lc_history(session_id:str) -> List:
    """Convert stored {role, content} history into LangChain messages."""
    msgs =[]
    for m in SESSIONS.history(session_id):
        role, content = m.get('role'), m.get('content','')
        if role =='user':
            msgs.append(HumanMessage(content=content))
        elif role =='assistant':
            msgs.append(AIMessage(content=content))
    return msgs

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Stream the answer token-by-token as Server-Sent Events (SSE)."""
    session_id = req.session_id
    message = req.message.strip()

    # validate BEFORE opening the stream
    if not session_id or not SESSIONS.exists(session_id):
        raise HTTPException(status_code=400, detail="Invalid or expired session_id. Re-upload documents")
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    # Reuse the session's cached RAG; build BEFORE the response starts so setup
    # failures are clean 500s rather than mid-stream errors.
    try:
        rag = _get_rag(session_id)
    except DocumentPortalException as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    lc_history = _lc_history(session_id)

    async def event_stream():
        chunks: List[str] = []
        start = time.monotonic()
        ttft = None  # seconds to first token
        try:
            async for token in rag.astream_answer(message, lc_history):
                if ttft is None:
                    ttft = time.monotonic() - start
                chunks.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"
            answer = "".join(chunks)
            SESSIONS.append(session_id, "user", message)
            SESSIONS.append(session_id, "assistant", answer)
            log.info(
                "chat_stream complete",
                session_id=session_id,
                ttft_s=round(ttft, 3) if ttft is not None else None,
                total_s=round(time.monotonic() - start, 3),
                tokens=len(chunks),
            )
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            log.error("chat_stream failed", session_id=session_id, error=str(e))
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"}
    )

# Uvicorn entrypoint for `python main.py` (optional)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)