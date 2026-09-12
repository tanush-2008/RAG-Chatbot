"""Automated smoke tests for the RAG pipeline using the bundled sample PDFs.

Run with: python -m pytest tests/ -v
(Runs fully offline - no LLM API key needed, since retrieval and refusal
behavior don't depend on the LLM provider.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from document_loader import DocumentValidationError, load_documents, validate_file
from rag_pipeline import RAGPipeline

DOCS_DIR = Path(__file__).resolve().parent.parent / "documents"


def _sample_files() -> list[tuple[str, bytes, int]]:
    files = []
    for name in ["Policy.pdf", "Handbook.pdf"]:
        data = (DOCS_DIR / name).read_bytes()
        files.append((name, data, len(data)))
    return files


@pytest.fixture(scope="module")
def pipeline() -> RAGPipeline:
    p = RAGPipeline()
    p.process_documents(_sample_files())
    return p


def test_validate_file_rejects_non_pdf():
    with pytest.raises(DocumentValidationError):
        validate_file("notes.txt", 100)


def test_validate_file_rejects_oversized():
    with pytest.raises(DocumentValidationError):
        validate_file("big.pdf", 100 * 1024 * 1024)


def test_extraction_preserves_metadata():
    files = _sample_files()
    pages = load_documents(files)
    assert len(pages) > 0
    assert all(p.source in {"Policy.pdf", "Handbook.pdf"} for p in pages)
    assert all(p.page >= 1 for p in pages)


def test_pipeline_indexes_chunks(pipeline: RAGPipeline):
    assert not pipeline.vector_store.is_empty
    assert len(pipeline.vector_store.chunks) > 0


def test_answerable_question_returns_correct_source(pipeline: RAGPipeline):
    answer = pipeline.ask("How is attendance calculated?")
    assert answer.grounded
    assert any(s.document == "Handbook.pdf" and s.page == 3 for s in answer.sources)


def test_answerable_question_leave_types(pipeline: RAGPipeline):
    answer = pipeline.ask("How many days of Casual Leave are employees entitled to?")
    assert answer.grounded
    assert any(s.document == "Policy.pdf" for s in answer.sources)


def test_unanswerable_question_refuses(pipeline: RAGPipeline):
    answer = pipeline.ask("Who is the company's Chief Executive Officer?")
    assert not answer.grounded
    assert "could not find this information" in answer.text.lower()
    assert answer.sources == []


def test_empty_store_prompts_for_upload():
    empty_pipeline = RAGPipeline()
    answer = empty_pipeline.ask("Any question")
    assert "upload" in answer.text.lower()
