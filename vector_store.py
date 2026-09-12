"""Module 3 & 4: Chunking, embeddings, and the FAISS-backed vector store."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np

from document_loader import PageDocument
from embeddings import BaseEmbedder, load_embedder
from text_splitter import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@dataclass
class Chunk:
    """A chunk of document text with the metadata needed to cite it back."""

    text: str
    source: str
    page: int
    chunk_id: int = field(default=-1)


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
                chunks.append(Chunk(text=piece, source=page.source, page=page.page))

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

    def search(self, query: str, top_k: int = 4) -> list[tuple[Chunk, float]]:
        """Return the top_k most similar chunks to the query with similarity scores."""
        if self.is_empty:
            return []
        query_vector = self._embedder.encode([query])
        scores, indices = self.index.search(query_vector, min(top_k, len(self.chunks)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(directory / "index.faiss"))
        with open(directory / "chunks.pkl", "wb") as f:
            pickle.dump({"chunks": self.chunks, "embedding_model_name": self.embedding_model_name}, f)

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        directory = Path(directory)
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
