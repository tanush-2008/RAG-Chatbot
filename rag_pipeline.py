"""Orchestrates Modules 2-6: extraction -> chunking -> embedding -> retrieval -> answer."""

from __future__ import annotations

import time
from dataclasses import dataclass

from document_loader import load_documents
from llm_provider import LLMError, call_llm
from prompt import NOT_FOUND_MESSAGE, build_messages
from vector_store import Chunk, VectorStore, chunk_documents

DEFAULT_TOP_K = 4
MIN_RELEVANCE_SCORE = 0.15  # cosine similarity floor below which we treat retrieval as "no match"


@dataclass
class Source:
    document: str
    page: int
    score: float


@dataclass
class Answer:
    text: str
    sources: list[Source]
    response_time_seconds: float
    grounded: bool


class RAGPipeline:
    """Stateful pipeline holding the vector store for the current session's documents."""

    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 150, top_k: int = DEFAULT_TOP_K):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.vector_store = VectorStore()
        self.processed_files: list[str] = []

    def process_documents(self, files: list[tuple[str, bytes, int]]) -> int:
        """Validate, extract, chunk, embed, and index a batch of uploaded PDFs.

        Returns the number of chunks added. Raises DocumentValidationError on bad input.
        """
        pages = load_documents(files)
        chunks = chunk_documents(pages, self.chunk_size, self.chunk_overlap)
        self.vector_store.add(chunks)
        self.processed_files.extend(name for name, _, _ in files)
        return len(chunks)

    def clear(self) -> None:
        self.vector_store = VectorStore()
        self.processed_files = []

    def save(self, directory: str) -> None:
        self.vector_store.save(directory)

    def load(self, directory: str) -> None:
        self.vector_store = VectorStore.load(directory)

    def ask(self, question: str) -> Answer:
        start = time.perf_counter()

        if self.vector_store.is_empty:
            return Answer(
                text="Please upload and process at least one PDF document before asking questions.",
                sources=[],
                response_time_seconds=time.perf_counter() - start,
                grounded=False,
            )

        results = self.vector_store.search(question, top_k=self.top_k)
        relevant: list[tuple[Chunk, float]] = [
            (c, s) for c, s in results if s >= MIN_RELEVANCE_SCORE
        ]

        if not relevant:
            return Answer(
                text=NOT_FOUND_MESSAGE,
                sources=[],
                response_time_seconds=time.perf_counter() - start,
                grounded=False,
            )

        chunks = [c for c, _ in relevant]
        messages = build_messages(question, chunks)

        try:
            answer_text = call_llm(messages)
        except LLMError as exc:
            answer_text = f"The language model could not be reached ({exc}). Please try again."

        sources = [Source(document=c.source, page=c.page, score=s) for c, s in relevant]
        grounded = NOT_FOUND_MESSAGE not in answer_text

        return Answer(
            text=answer_text,
            sources=sources if grounded else [],
            response_time_seconds=time.perf_counter() - start,
            grounded=grounded,
        )
