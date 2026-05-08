"""Pluggable reranker service.

The reranker is the second stage of the retrieval pipeline. It takes the
top-N candidates from the HybridRetriever (dense + sparse + RRF) and
reorders them using a cross-encoder, returning the top-K to the LangGraph
tool.

Two providers are supported:

- ``BGEReranker``: ``BAAI/bge-reranker-v2-m3`` via sentence-transformers.
  Self-hosted, free, ~80 ms for 50 pairs on CPU. Default.
- ``CohereReranker``: managed Cohere Rerank v4-fast / v4-pro. Selected when
  ``RERANKER_PROVIDER=cohere`` and ``COHERE_API_KEY`` is set.

Both providers are imported lazily so the heavy ``sentence-transformers`` /
``cohere`` dependencies do not delay app startup.
"""
from __future__ import annotations

import logging
import threading
from typing import List, Optional, Protocol

from langchain_core.documents import Document

from app.core.config import settings

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    """Protocol every reranker provider implements."""

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int,
    ) -> List[Document]:
        ...


# --------------------------------------------------------------------------- #
# BGE (default, self-hosted)
# --------------------------------------------------------------------------- #


class BGEReranker:
    """sentence-transformers cross-encoder. Free, ~80ms for 50 pairs."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self._model_name = (
            model_name
            or getattr(settings, "BGE_RERANKER_MODEL", None)
            or "BAAI/bge-reranker-v2-m3"
        )
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    # Lazy import: a 500 MB model + torch is too heavy to
                    # pay for at import time, especially in dev where the
                    # reranker may never be exercised.
                    try:
                        from sentence_transformers import CrossEncoder
                    except ImportError as exc:
                        raise RuntimeError(
                            "BGEReranker requires sentence-transformers. "
                            "Install it or set RERANKER_PROVIDER=cohere."
                        ) from exc
                    self._model = CrossEncoder(self._model_name, max_length=512)
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int,
    ) -> List[Document]:
        if not documents:
            return []
        if not query or not query.strip():
            return documents[:top_k]

        model = self._get_model()
        pairs = [(query, doc.page_content or "") for doc in documents]
        try:
            scores = model.predict(pairs)
        except Exception as exc:
            logger.warning(
                "BGE rerank failed (returning original order): %s", exc
            )
            return documents[:top_k]

        ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_k]]


# --------------------------------------------------------------------------- #
# Cohere (managed, opt-in)
# --------------------------------------------------------------------------- #


class CohereReranker:
    """Managed Cohere Rerank v4-fast/v4-pro. Requires COHERE_API_KEY."""

    def __init__(
        self,
        api_key: str,
        model_name: Optional[str] = None,
    ) -> None:
        if not api_key:
            raise ValueError(
                "CohereReranker requires a non-empty api_key (COHERE_API_KEY)."
            )
        self._api_key = api_key
        self._model_name = (
            model_name
            or getattr(settings, "COHERE_RERANK_MODEL", None)
            or "rerank-v4.0-fast"
        )
        self._client = None
        self._lock = threading.Lock()

    def _get_client(self):
        if self._client is None:
            with self._lock:
                if self._client is None:
                    try:
                        import cohere
                    except ImportError as exc:
                        raise RuntimeError(
                            "CohereReranker requires the cohere package. "
                            "Install it or set RERANKER_PROVIDER=bge."
                        ) from exc
                    self._client = cohere.Client(self._api_key)
        return self._client

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int,
    ) -> List[Document]:
        if not documents:
            return []
        if not query or not query.strip():
            return documents[:top_k]

        client = self._get_client()
        try:
            res = client.rerank(
                query=query,
                model=self._model_name,
                top_n=min(top_k, len(documents)),
                documents=[doc.page_content or "" for doc in documents],
            )
        except Exception as exc:
            logger.warning(
                "Cohere rerank failed (returning original order): %s", exc
            )
            return documents[:top_k]

        # Cohere returns results sorted by relevance; map indices back.
        return [documents[r.index] for r in res.results]


# --------------------------------------------------------------------------- #
# No-op fallback
# --------------------------------------------------------------------------- #


class NoopReranker:
    """Used when reranking is disabled or both providers fail to load."""

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int,
    ) -> List[Document]:
        return documents[:top_k]


# --------------------------------------------------------------------------- #
# Factory + singleton
# --------------------------------------------------------------------------- #


_reranker: Optional[Reranker] = None
_reranker_lock = threading.Lock()


def _build_reranker() -> Reranker:
    provider = (getattr(settings, "RERANKER_PROVIDER", "bge") or "bge").lower()

    if provider == "cohere":
        api_key = getattr(settings, "COHERE_API_KEY", None)
        if not api_key:
            logger.warning(
                "RERANKER_PROVIDER=cohere but COHERE_API_KEY is not set; "
                "falling back to BGE."
            )
            return BGEReranker()
        try:
            return CohereReranker(api_key)
        except Exception as exc:
            logger.warning(
                "Failed to initialize CohereReranker (%s); falling back to BGE.",
                exc,
            )
            return BGEReranker()

    if provider == "noop":
        return NoopReranker()

    return BGEReranker()


def get_reranker() -> Reranker:
    """Return the global Reranker instance (thread-safe, lazy)."""
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                _reranker = _build_reranker()
    return _reranker
