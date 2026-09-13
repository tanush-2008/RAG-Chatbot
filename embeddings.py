"""Module 4 (embeddings): Sentence Transformers wrapper with an offline fallback.

Primary path: `all-MiniLM-L6-v2` via sentence-transformers/torch, as specified
in the project brief. Some locked-down environments (corporate Windows
machines with an Application Control / WDAC policy) block torch's native
DLLs outright; rather than let the whole app crash on import, this module
falls back to a deterministic, dependency-free hashing embedder so retrieval
still works (with lower semantic quality) until the real model is usable.
"""

from __future__ import annotations

import hashlib
import re
import warnings

import numpy as np

from logging_config import get_logger

log = get_logger(__name__)

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FALLBACK_DIMENSION = 384  # matches all-MiniLM-L6-v2's output dimension

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset(
    """
    a an the is are was were be been being to of in on at for with and or
    but if then than so as by from into it its this that these those who
    whom whose what which when where why how do does did can could will
    would shall should may might must not no yes i you he she we they
    them his her our your their s t re ve ll d
    """.split()
)


class BaseEmbedder:
    name: str
    dimension: int

    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedder(BaseEmbedder):
    name = "sentence-transformers"

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        # Import torch directly first and let any failure (e.g. a blocked native
        # DLL under a locked-down Application Control policy) surface immediately,
        # cheaply, and without ever importing the much heavier `transformers`
        # package - which some environments' dev-mode file watchers then probe
        # module-by-module, each probe re-triggering (and re-failing) the same
        # slow torch import.
        import torch  # noqa: F401
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        # get_embedding_dimension() replaced get_sentence_embedding_dimension()
        # in newer sentence-transformers releases; support both.
        get_dim = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension"
        )
        self.dimension = get_dim()

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")
        vectors = self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.astype("float32")


class HashingFallbackEmbedder(BaseEmbedder):
    """Deterministic bag-of-words hashing embedder used only when
    sentence-transformers/torch cannot be loaded in the current environment."""

    name = "hashing-fallback"

    def __init__(self, dimension: int = FALLBACK_DIMENSION):
        self.dimension = dimension

    def _hash_index_and_sign(self, token: str) -> tuple[int, float]:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % self.dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        return index, sign

    def _embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype="float32")
        tokens = [
            t for t in _TOKEN_RE.findall(text.lower())
            if len(t) > 2 and t not in _STOPWORDS
        ]
        for token in tokens:
            index, sign = self._hash_index_and_sign(token)
            vector[index] += sign
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")
        return np.stack([self._embed_one(t) for t in texts]).astype("float32")


def load_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL) -> BaseEmbedder:
    """Try to load the real Sentence Transformers model; fall back if unavailable."""
    try:
        embedder = SentenceTransformerEmbedder(model_name)
        log.info("Loaded Sentence Transformers embedding model '%s'.", model_name)
        return embedder
    except Exception as exc:  # ImportError, OSError (blocked DLL), etc.
        message = (
            f"Could not load Sentence Transformers model '{model_name}' ({exc}). "
            "Falling back to a lightweight offline hashing embedder with reduced "
            "semantic accuracy. Install/enable torch to use the real model."
        )
        log.warning(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return HashingFallbackEmbedder()
