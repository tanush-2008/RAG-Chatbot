"""Tests for the optional FastAPI backend (api.py).

Uses a fresh in-memory pipeline and an isolated vector store directory so
these tests never touch real persisted data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import api
from rag_pipeline import RAGPipeline

DOCS_DIR = Path(__file__).resolve().parent.parent / "documents"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "pipeline", RAGPipeline())
    monkeypatch.setattr(api, "VECTOR_STORE_DIR", tmp_path / "api_store")
    monkeypatch.delenv("API_KEY", raising=False)
    # Rate-limit counters live in the module-level `limiter`'s storage, not
    # per-TestClient - reset them so tests don't pollute each other's counts
    # (the limit-enforcement tests below need a clean starting point too).
    api.limiter.reset()
    return TestClient(api.app)


def _sample_files():
    files = []
    for name in ["Policy.pdf", "Handbook.pdf"]:
        files.append(("files", (name, (DOCS_DIR / name).read_bytes(), "application/pdf")))
    return files


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_and_list_documents(client):
    response = client.post("/documents/upload", files=_sample_files())
    assert response.status_code == 200
    body = response.json()
    assert body["files_processed"] == 2
    assert set(body["indexed_documents"]) == {"Policy.pdf", "Handbook.pdf"}

    response = client.get("/documents")
    assert set(response.json()["documents"]) == {"Policy.pdf", "Handbook.pdf"}


def test_ask_answerable_question(client):
    client.post("/documents/upload", files=_sample_files())
    response = client.post("/ask", json={"question": "How is attendance calculated?"})
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert any(s["document"] == "Handbook.pdf" and s["page"] == 3 for s in body["sources"])


def test_ask_unanswerable_question_refuses(client):
    client.post("/documents/upload", files=_sample_files())
    response = client.post("/ask", json={"question": "Who is the company's Chief Executive Officer?"})
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is False
    assert body["sources"] == []


def test_clear_documents(client):
    client.post("/documents/upload", files=_sample_files())
    response = client.delete("/documents")
    assert response.status_code == 200
    assert client.get("/documents").json()["documents"] == []


def test_upload_rejects_non_pdf(client):
    response = client.post(
        "/documents/upload", files=[("files", ("notes.txt", b"hello", "text/plain"))]
    )
    assert response.status_code == 400


def test_upload_rejects_fake_pdf_content(client):
    # .pdf extension but not actually PDF content (magic-byte check).
    response = client.post(
        "/documents/upload",
        files=[("files", ("fake.pdf", b"not really a pdf", "application/pdf"))],
    )
    assert response.status_code == 400


def test_api_key_gate(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    response = client.get("/documents")
    assert response.status_code == 401

    response = client.get("/documents", headers={"X-API-Key": "secret123"})
    assert response.status_code == 200

    response = client.get("/documents", headers={"X-API-Key": "wrong"})
    assert response.status_code == 401


def test_health_bypasses_api_key(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    response = client.get("/health")
    assert response.status_code == 200


def test_ask_rate_limit_enforced(client):
    # ASK_RATE_LIMIT defaults to 20/minute (see api.py); the 21st request in
    # the same window must be rejected with 429, not silently accepted.
    client.post("/documents/upload", files=_sample_files())
    responses = [client.post("/ask", json={"question": "test"}) for _ in range(21)]
    assert responses[-1].status_code == 429
    assert all(r.status_code == 200 for r in responses[:20])


def test_upload_rate_limit_enforced(client):
    # UPLOAD_RATE_LIMIT defaults to 10/minute.
    responses = [client.post("/documents/upload", files=_sample_files()) for _ in range(11)]
    assert responses[-1].status_code == 429
    assert all(r.status_code == 200 for r in responses[:10])


def test_default_rate_limit_applies_to_undecorated_routes(client):
    # /documents has no explicit @limiter.limit() - it should still be
    # covered by DEFAULT_RATE_LIMIT (100/minute) via SlowAPIMiddleware.
    responses = [client.get("/documents") for _ in range(101)]
    assert responses[-1].status_code == 429
    assert all(r.status_code == 200 for r in responses[:100])


def test_health_is_exempt_from_rate_limiting(client):
    # Monitoring/liveness probes must never be throttled.
    responses = [client.get("/health") for _ in range(150)]
    assert all(r.status_code == 200 for r in responses)
