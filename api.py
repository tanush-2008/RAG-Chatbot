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

from fastapi import Depends, FastAPI, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel

from document_loader import DocumentValidationError
from logging_config import get_logger
from rag_pipeline import RAGPipeline

log = get_logger(__name__)

VECTOR_STORE_DIR = Path("vector_store/saved_index/api")

app = FastAPI(
    title="Domain-Specific RAG Chatbot API",
    description="REST API for uploading PDFs and asking grounded questions about them.",
    version="1.1.0",
)

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
    pipeline.save(str(VECTOR_STORE_DIR))
    return {"status": "cleared"}


@app.post("/documents/upload", response_model=ProcessResponse, dependencies=[Depends(require_api_key)])
async def upload_documents(
    files: list[UploadFile], enable_ocr: bool = Form(False)
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

    pipeline.save(str(VECTOR_STORE_DIR))
    log.info("Uploaded %d file(s), %d chunk(s) added.", len(files), num_chunks)
    return ProcessResponse(
        files_processed=len(files),
        chunks_added=num_chunks,
        indexed_documents=pipeline.vector_store.document_names,
    )


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
def ask(request: AskRequest) -> AskResponse:
    original_top_k = pipeline.top_k
    if request.top_k:
        pipeline.top_k = request.top_k
    try:
        answer = pipeline.ask(request.question)
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
