"""Tests for VectorStore persistence: round-tripping, atomic writes, and
cross-process locking (vector_store.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from filelock import FileLock, Timeout

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vector_store import VectorStore, chunk_documents
from document_loader import PageDocument

DOCS_DIR = Path(__file__).resolve().parent.parent / "documents"


def _sample_store() -> VectorStore:
    pages = [PageDocument(text="Casual Leave is 12 days per year.", source="Policy.pdf", page=2)]
    chunks = chunk_documents(pages)
    store = VectorStore()
    store.add(chunks)
    return store


def test_save_then_load_round_trips_chunks(tmp_path):
    store = _sample_store()
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    assert len(loaded.chunks) == len(store.chunks)
    assert loaded.chunks[0].text == store.chunks[0].text
    assert loaded.document_names == store.document_names
    assert not loaded.is_empty


def test_save_leaves_no_temp_files_behind(tmp_path):
    store = _sample_store()
    store.save(tmp_path)
    leftover_tmp = list(tmp_path.glob("*.tmp"))
    assert leftover_tmp == []
    assert (tmp_path / "index.faiss").exists()
    assert (tmp_path / "chunks.pkl").exists()


def test_save_is_atomic_on_writer_failure(tmp_path, monkeypatch):
    store = _sample_store()

    import vector_store as vs_module

    def failing_write_index(index, path):
        raise RuntimeError("disk full")

    monkeypatch.setattr(vs_module.faiss, "write_index", failing_write_index)

    with pytest.raises(RuntimeError):
        store.save(tmp_path)

    # No half-written index.faiss should exist, and no leftover temp files.
    assert not (tmp_path / "index.faiss").exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_raises_timeout_when_directory_locked(tmp_path):
    store = _sample_store()
    tmp_path.mkdir(exist_ok=True)
    external_lock = FileLock(str(tmp_path / ".lock"))
    external_lock.acquire()
    try:
        import vector_store as vs_module

        original_timeout = vs_module.LOCK_TIMEOUT_SECONDS
        vs_module.LOCK_TIMEOUT_SECONDS = 0.2
        try:
            with pytest.raises(Timeout):
                store.save(tmp_path)
        finally:
            vs_module.LOCK_TIMEOUT_SECONDS = original_timeout
    finally:
        external_lock.release()


def test_exists_reflects_saved_state(tmp_path):
    assert VectorStore.exists(tmp_path) is False
    _sample_store().save(tmp_path)
    assert VectorStore.exists(tmp_path) is True
