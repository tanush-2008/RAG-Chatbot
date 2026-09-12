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
