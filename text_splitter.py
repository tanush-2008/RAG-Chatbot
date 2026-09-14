"""Module 3: Recursive character text splitter (fallback implementation).

vector_store.py uses the real `langchain-text-splitters` package by default
(the brief's suggested tool). This is a minimal, dependency-free
reimplementation of the same class (same strategy: try each separator in
order, falling back to the next, merging pieces back up to chunk_size with
overlap), kept as a safety net for environments where that package can't be
imported - it transitively pulls in `langchain-core` and a native
`uuid_utils` extension that some locked-down environments (e.g. under an
Application Control / DLL allowlist policy) refuse to load, as happened
during this project's own development.
"""

from __future__ import annotations


class RecursiveCharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 900,
        chunk_overlap: int = 150,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> list[str]:
        pieces = self._split(text, self.separators)
        return self._merge(pieces)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        separator = separators[-1]
        remaining_separators = separators[1:]
        for i, sep in enumerate(separators):
            if sep == "" or sep in text:
                separator = sep
                remaining_separators = separators[i + 1 :]
                break

        if separator == "":
            splits = list(text)
        else:
            splits = text.split(separator)

        results: list[str] = []
        for split in splits:
            if len(split) <= self.chunk_size:
                if split:
                    results.append(split + separator if separator and split != splits[-1] else split)
            elif remaining_separators:
                results.extend(self._split(split, remaining_separators))
            else:
                results.append(split)
        return results

    def _merge(self, pieces: list[str]) -> list[str]:
        if not pieces:
            return []

        chunks: list[str] = []
        current = ""
        for piece in pieces:
            if not current:
                current = piece
                continue
            if len(current) + len(piece) <= self.chunk_size:
                current += piece
            else:
                chunks.append(current)
                overlap_text = current[-self.chunk_overlap :] if self.chunk_overlap else ""
                current = overlap_text + piece
        if current:
            chunks.append(current)
        return chunks
