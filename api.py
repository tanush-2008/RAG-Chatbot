"""Optional advanced feature: FastAPI backend for mobile/web clients.

Exposes the same RAGPipeline used by the Streamlit app over a small REST
API, so a mobile app or a separate web frontend can integrate without going
through Streamlit. Run with:

    uvicorn api:app --reload --port 8000

Then see the interactive docs at http://localhost:8000/docs

Auth: if API_KEY is set in the environment, every request (except /health)
must include a matching `X-API-Key` header - this mirrors auth.py's opt-in
pattern (no key configured = no gate, for simple local/dev use).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import Depends, FastAPI, Form, Header, HTTPException, Request, UploadFile
from filelock import Timeout
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from document_loader import DocumentValidationError
from logging_config import get_logger
from rag_pipeline import RAGPipeline

log = get_logger(__name__)

# Rate limits are configurable via env vars so a real deployment can tune
# them without a code change; defaults are generous for local/dev use.
# /ask and /documents/upload get their own (tighter) limits since they're
# the expensive ones - an LLM call costs real money per request, and
# uploads consume disk/CPU - while everything else falls under the
# DEFAULT_RATE_LIMIT blanket via Limiter's default_limits.
DEFAULT_RATE_LIMIT = os.getenv("DEFAULT_RATE_LIMIT", "100/minute")
ASK_RATE_LIMIT = os.getenv("ASK_RATE_LIMIT", "20/minute")
UPLOAD_RATE_LIMIT = os.getenv("UPLOAD_RATE_LIMIT", "10/minute")

limiter = Limiter(key_func=get_remote_address, default_limits=[DEFAULT_RATE_LIMIT])


def _save_pipeline() -> None:
    try:
        pipeline.save(str(VECTOR_STORE_DIR))
    except Timeout as exc:
        raise HTTPException(
            status_code=503,
            detail="The document store is busy (another save is in progress). Please retry shortly.",
        ) from exc

VECTOR_STORE_DIR = Path("vector_store/saved_index/api")

app = FastAPI(
    title="Domain-Specific RAG Chatbot API",
    description="REST API for uploading PDFs and asking grounded questions about them.",
    version="1.2.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

pipeline = RAGPipeline()
if (VECTOR_STORE_DIR / "index.faiss").exists():
    try:
        pipeline.load(str(VECTOR_STORE_DIR))
        log.info("Loaded persisted index: %d document(s).", len(pipeline.vector_store.document_names))
    except Exception as exc:
        log.warning("Could not load persisted index at %s: %s", VECTOR_STORE_DIR, exc)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None


class SourceOut(BaseModel):
    document: str
    page: int
    score: float
    via_ocr: bool = False


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    grounded: bool
    response_time_seconds: float


class ProcessResponse(BaseModel):
    files_processed: int
    chunks_added: int
    indexed_documents: list[str]


@app.get("/health")
@limiter.exempt
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def root() -> dict:
    return {
        "name": "Domain-Specific RAG Chatbot API",
        "docs": "/docs",
        "endpoints": ["/health", "/documents", "/documents/upload", "/ask"],
    }


@app.get("/documents", dependencies=[Depends(require_api_key)])
def list_documents() -> dict:
    return {"documents": pipeline.vector_store.document_names}


@app.delete("/documents", dependencies=[Depends(require_api_key)])
def clear_documents() -> dict:
    pipeline.clear()
    _save_pipeline()
    return {"status": "cleared"}


@app.post("/documents/upload", response_model=ProcessResponse, dependencies=[Depends(require_api_key)])
@limiter.limit(UPLOAD_RATE_LIMIT)
async def upload_documents(
    request: Request, files: list[UploadFile], enable_ocr: bool = Form(False)
) -> ProcessResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    payload = []
    for f in files:
        content = await f.read()
        payload.append((f.filename, content, len(content)))

    try:
        num_chunks = pipeline.process_documents(payload, enable_ocr=enable_ocr)
    except DocumentValidationError as exc:
        log.warning("Upload rejected: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _save_pipeline()
    log.info("Uploaded %d file(s), %d chunk(s) added.", len(files), num_chunks)
    return ProcessResponse(
        files_processed=len(files),
        chunks_added=num_chunks,
        indexed_documents=pipeline.vector_store.document_names,
    )


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
@limiter.limit(ASK_RATE_LIMIT)
def ask(request: Request, body: AskRequest) -> AskResponse:
    original_top_k = pipeline.top_k
    if body.top_k:
        pipeline.top_k = body.top_k
    try:
        answer = pipeline.ask(body.question)
    finally:
        pipeline.top_k = original_top_k

    return AskResponse(
        answer=answer.text,
        sources=[
            SourceOut(document=s.document, page=s.page, score=s.score, via_ocr=s.via_ocr)
            for s in answer.sources
        ],
        grounded=answer.grounded,
        response_time_seconds=answer.response_time_seconds,
    )
