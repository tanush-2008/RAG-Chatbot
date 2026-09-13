"""Module 3 & 4: Chunking, embeddings, and the FAISS-backed vector store."""

from __future__ import annotations

import os
import pickle
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np
from filelock import FileLock

from document_loader import PageDocument
from embeddings import BaseEmbedder, load_embedder
from logging_config import get_logger
from text_splitter import RecursiveCharacterTextSplitter

log = get_logger(__name__)

DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LOCK_TIMEOUT_SECONDS = 30


def _pickle_dump(obj, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


@dataclass
class Chunk:
    """A chunk of document text with the metadata needed to cite it back."""

    text: str
    source: str
    page: int
    chunk_id: int = field(default=-1)
    via_ocr: bool = field(default=False)


def chunk_documents(
    pages: list[PageDocument],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split page text into overlapping chunks while preserving source/page metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    for page in pages:
        for piece in splitter.split_text(page.text):
            piece = piece.strip()
            if piece:
                chunks.append(
                    Chunk(text=piece, source=page.source, page=page.page, via_ocr=page.via_ocr)
                )

    for idx, chunk in enumerate(chunks):
        chunk.chunk_id = idx

    return chunks


class EmbeddingModel:
    """Thin wrapper around the embedding backend, loaded once and cached.

    Delegates to embeddings.load_embedder(), which uses the real Sentence
    Transformers model when available and falls back to an offline hashing
    embedder if torch cannot be loaded in the current environment.
    """

    _instance: "EmbeddingModel | None" = None

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self._embedder: BaseEmbedder = load_embedder(model_name)

    @classmethod
    def get(cls, model_name: str = DEFAULT_EMBEDDING_MODEL) -> "EmbeddingModel":
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = cls(model_name)
        return cls._instance

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._embedder.encode(texts)

    @property
    def dimension(self) -> int:
        return self._embedder.dimension

    @property
    def backend_name(self) -> str:
        return self._embedder.name


class VectorStore:
    """FAISS index over document chunks, with cosine-similarity search via inner product
    on normalized vectors, and save/load so the index can be reused across sessions."""

    def __init__(self, embedding_model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.embedding_model_name = embedding_model_name
        self._embedder = EmbeddingModel.get(embedding_model_name)
        self.index: faiss.Index | None = None
        self.chunks: list[Chunk] = []

    @property
    def is_empty(self) -> bool:
        return self.index is None or self.index.ntotal == 0

    @property
    def embedding_backend(self) -> str:
        return self._embedder.backend_name

    def build(self, chunks: list[Chunk]) -> None:
        """Embed chunks and (re)build the FAISS index from scratch."""
        self.chunks = chunks
        if not chunks:
            self.index = None
            return
        vectors = self._embedder.encode([c.text for c in chunks])
        index = faiss.IndexFlatIP(self._embedder.dimension)
        index.add(vectors)
        self.index = index

    def add(self, chunks: list[Chunk]) -> None:
        """Add more chunks to an existing index (used when uploading additional PDFs)."""
        if not chunks:
            return
        vectors = self._embedder.encode([c.text for c in chunks])
        if self.index is None:
            self.index = faiss.IndexFlatIP(self._embedder.dimension)
        offset = len(self.chunks)
        for i, chunk in enumerate(chunks):
            chunk.chunk_id = offset + i
        self.index.add(vectors)
        self.chunks.extend(chunks)

    @property
    def document_names(self) -> list[str]:
        """Distinct source document names currently indexed, in first-seen order."""
        seen: dict[str, None] = {}
        for chunk in self.chunks:
            seen.setdefault(chunk.source, None)
        return list(seen.keys())

    def search(
        self,
        query: str,
        top_k: int = 4,
        allowed_documents: set[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Return the top_k most similar chunks to the query with similarity scores.

        `allowed_documents`, if given, restricts results to chunks from those
        source document names only (Module 7's "document filters" feature) -
        useful when several unrelated PDFs are indexed together and a question
        should only be answered from a subset of them.
        """
        if self.is_empty:
            return []
        query_vector = self._embedder.encode([query])

        if allowed_documents is None:
            k = min(top_k, len(self.chunks))
            scores, indices = self.index.search(query_vector, k)
            return [
                (self.chunks[idx], float(score))
                for score, idx in zip(scores[0], indices[0])
                if idx != -1
            ]

        # Metadata filtering isn't native to a flat FAISS index, so over-fetch
        # and filter client-side, widening the search until we have enough
        # matches (or have covered the whole index).
        k = min(max(top_k * 4, top_k), len(self.chunks))
        while True:
            scores, indices = self.index.search(query_vector, k)
            results = [
                (self.chunks[idx], float(score))
                for score, idx in zip(scores[0], indices[0])
                if idx != -1 and self.chunks[idx].source in allowed_documents
            ]
            if len(results) >= top_k or k >= len(self.chunks):
                return results[:top_k]
            k = min(k * 4, len(self.chunks))

    def save(self, directory: str | Path) -> None:
        """Persist the index and chunk metadata, safely under concurrent access.

        Two failure modes matter for a store that real people hit daily:
        a crash or power-loss mid-write leaving a half-written file, and two
        processes (or two browser tabs for the same user) saving at the same
        time. Both are handled by writing to a temp file in the same
        directory and atomically renaming it into place (`os.replace` is
        atomic on both POSIX and Windows for same-volume renames), guarded by
        a cross-process file lock so the write and rename happen as one step.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(directory / ".lock"), timeout=LOCK_TIMEOUT_SECONDS)
        with lock:
            if self.index is not None:
                self._atomic_write(directory / "index.faiss", faiss.write_index, self.index)
            self._atomic_write(directory / "chunks.pkl", _pickle_dump, {
                "chunks": self.chunks,
                "embedding_model_name": self.embedding_model_name,
            })
        log.info("Saved vector store to %s (%d chunks).", directory, len(self.chunks))

    @staticmethod
    def _atomic_write(path: Path, writer, obj) -> None:
        fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
        os.close(fd)
        try:
            writer(obj, tmp_path)
            # os.replace is atomic on POSIX and Windows for same-volume
            # renames - but on Windows it also requires the destination
            # (and the source) not be open elsewhere, which is why every
            # writer below closes its file handle before returning.
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        directory = Path(directory)
        lock = FileLock(str(directory / ".lock"), timeout=LOCK_TIMEOUT_SECONDS)
        with lock:
            with open(directory / "chunks.pkl", "rb") as f:
                data = pickle.load(f)
            store = cls(embedding_model_name=data["embedding_model_name"])
            store.chunks = data["chunks"]
            index_path = directory / "index.faiss"
            if index_path.exists():
                store.index = faiss.read_index(str(index_path))
        return store

    @staticmethod
    def exists(directory: str | Path) -> bool:
        directory = Path(directory)
        return (directory / "index.faiss").exists() and (directory / "chunks.pkl").exists()
