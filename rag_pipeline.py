"""Orchestrates Modules 2-6: extraction -> chunking -> embedding -> retrieval -> answer."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from document_loader import load_documents
from llm_provider import LLMError, call_llm, condense_question
from prompt import NOT_FOUND_MESSAGE, build_messages
from vector_store import Chunk, VectorStore, chunk_documents

DEFAULT_TOP_K = 4
MAX_QUESTIONS_PER_MESSAGE = 10

# Relevance thresholds are calibrated per embedding backend, since raw cosine
# similarity scores from a real semantic model and the offline hashing
# fallback (see embeddings.py) live in different ranges. Each entry is
# (gate, support): `gate` is the minimum score the *top* retrieved chunk must
# clear for the question to be considered in-scope at all (empirically, real
# answerable questions score >=0.3 while off-topic ones top out around 0.23 -
# see tests/generate_sample_pdfs.py-based calibration in the project notes);
# `support` is the lower floor used to keep secondary chunks once the gate
# has passed, so weaker-but-genuine supporting context isn't dropped.
RELEVANCE_THRESHOLDS: dict[str, tuple[float, float]] = {
    "sentence-transformers": (0.28, 0.20),
    "hashing-fallback": (0.20, 0.12),
}
DEFAULT_RELEVANCE_THRESHOLD: tuple[float, float] = (0.25, 0.15)

_LEADING_NUMBERING_RE = re.compile(r"^\s*(?:\d+[\.\)]|[-*])\s*")


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


def split_into_questions(text: str) -> list[str]:
    """Split a chat message into multiple questions if it looks like a batch.

    Handles two common shapes: one question per line, and several questions
    run together in one paragraph separated by '?'. Falls back to treating the
    whole message as a single question if the split doesn't look meaningful
    (e.g. a single sentence that just happens to end in '?').
    """
    stripped = text.strip()
    if not stripped:
        return [stripped]

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    candidates = lines if len(lines) > 1 else [
        p.strip() for p in re.split(r"(?<=\?)\s+", stripped) if p.strip()
    ]

    cleaned = []
    for candidate in candidates:
        candidate = _LEADING_NUMBERING_RE.sub("", candidate).strip()
        if len(candidate.split()) >= 3:
            cleaned.append(candidate)

    if len(cleaned) <= 1:
        return [stripped]
    return cleaned[:MAX_QUESTIONS_PER_MESSAGE]


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

    def ask(
        self,
        question: str,
        history: list[tuple[str, str]] | None = None,
        document_filter: set[str] | None = None,
    ) -> Answer:
        """Answer a chat message, transparently handling batches of questions.

        `history` is prior (question, answer_text) turns from this session,
        most recent last - used to resolve follow-up questions ("what about
        paternity leave instead?") into a better retrieval query and to give
        the LLM conversational continuity.

        `document_filter`, if given, restricts retrieval to that subset of
        indexed document names (the "document filters" feature) - useful when
        several unrelated PDFs are indexed and a question should only draw
        from some of them.
        """
        questions = split_into_questions(question)
        if len(questions) == 1:
            return self._ask_one(questions[0], history, document_filter)
        return self._ask_batch(questions, history, document_filter)

    def _ask_batch(
        self,
        questions: list[str],
        history: list[tuple[str, str]] | None,
        document_filter: set[str] | None,
    ) -> Answer:
        start = time.perf_counter()
        sub_answers = [self._ask_one(q, history, document_filter) for q in questions]

        text_parts = []
        combined_sources: dict[tuple[str, int], Source] = {}
        any_grounded = False
        for q, a in zip(questions, sub_answers):
            text_parts.append(f"**Q: {q}**\n{a.text}")
            any_grounded = any_grounded or a.grounded
            for s in a.sources:
                key = (s.document, s.page)
                if key not in combined_sources or s.score > combined_sources[key].score:
                    combined_sources[key] = s

        ordered_sources = sorted(combined_sources.values(), key=lambda s: s.score, reverse=True)
        return Answer(
            text="\n\n".join(text_parts),
            sources=ordered_sources,
            response_time_seconds=time.perf_counter() - start,
            grounded=any_grounded,
        )

    def _ask_one(
        self,
        question: str,
        history: list[tuple[str, str]] | None = None,
        document_filter: set[str] | None = None,
    ) -> Answer:
        start = time.perf_counter()

        if self.vector_store.is_empty:
            return Answer(
                text="Please upload and process at least one PDF document before asking questions.",
                sources=[],
                response_time_seconds=time.perf_counter() - start,
                grounded=False,
            )

        if document_filter is not None and not document_filter:
            return Answer(
                text="No documents are selected to search. Choose at least one in the sidebar filter.",
                sources=[],
                response_time_seconds=time.perf_counter() - start,
                grounded=False,
            )

        retrieval_query = condense_question(question, history or [])
        results = self.vector_store.search(
            retrieval_query, top_k=self.top_k, allowed_documents=document_filter
        )

        gate, support = RELEVANCE_THRESHOLDS.get(
            self.vector_store.embedding_backend, DEFAULT_RELEVANCE_THRESHOLD
        )
        if not results or results[0][1] < gate:
            # The best match isn't confident enough for the question to be
            # in-scope at all - refuse without spending an LLM call on it.
            return Answer(
                text=NOT_FOUND_MESSAGE,
                sources=[],
                response_time_seconds=time.perf_counter() - start,
                grounded=False,
            )
        relevant: list[tuple[Chunk, float]] = [(c, s) for c, s in results if s >= support]

        chunks = [c for c, _ in relevant]
        messages = build_messages(question, chunks, history=history)

        try:
            answer_text = call_llm(messages)
        except LLMError as exc:
            answer_text = f"The language model could not be reached ({exc}). Please try again."

        grounded = NOT_FOUND_MESSAGE not in answer_text
        sources = self._dedupe_sources(relevant) if grounded else []

        return Answer(
            text=answer_text,
            sources=sources,
            response_time_seconds=time.perf_counter() - start,
            grounded=grounded,
        )

    @staticmethod
    def _dedupe_sources(relevant: list[tuple[Chunk, float]]) -> list[Source]:
        """Collapse multiple chunks from the same document/page into one source
        entry (keeping the best score), so overlapping chunks don't show up as
        duplicate citations."""
        best: dict[tuple[str, int], Source] = {}
        for chunk, score in relevant:
            key = (chunk.source, chunk.page)
            if key not in best or score > best[key].score:
                best[key] = Source(document=chunk.source, page=chunk.page, score=score)
        return sorted(best.values(), key=lambda s: s.score, reverse=True)
