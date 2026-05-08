"""Vector store service for PGVector operations.

Owns:
- The langchain-postgres ``PGVectorStore`` instance (dense embeddings).
- Bootstrap of the full-text-search infrastructure (generated ``tsv`` column
  + GIN index + JSONB ``user_id`` index) used by the sparse arm of
  HybridRetriever. Idempotent — safe to run on every startup.
- The full retrieval pipeline factory: dense + sparse + RRF + cross-encoder
  reranker.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGEngine, PGVectorStore
from sqlalchemy import text

from app.core.config import settings
from app.core.database import engine as sync_engine
from app.services.hybrid_retriever import HybridRetriever
from app.services.reranker_service import get_reranker
from app.utils.db_uri import normalize_db_uri_for_asyncpg
from app.utils.retrieval_logger import get_retrieval_logger

logger = logging.getLogger(__name__)


class HybridReranKRetriever(BaseRetriever):
    """Two-stage retriever: HybridRetriever → cross-encoder reranker.

    This is what the LangGraph tool actually calls. Behaves like a normal
    ``BaseRetriever`` (synchronous ``_get_relevant_documents``) so it slots
    into the existing tool-creation code without changes.
    """

    class Config:
        arbitrary_types_allowed = True

    hybrid: HybridRetriever
    top_k: int = 5
    user_id: Optional[str] = None  # for logging only

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        # Stage 1: dense + sparse + RRF
        candidates = self.hybrid._get_relevant_documents(query, run_manager=run_manager)

        # Stage 2: cross-encoder rerank → top-K
        reranker = get_reranker()
        try:
            top_docs = reranker.rerank(query, candidates, top_k=self.top_k)
        except Exception as exc:
            logger.warning("Reranker raised %s; using hybrid order", exc)
            top_docs = candidates[: self.top_k]

        # Log the final retrieval (matches existing UserFilteredRetriever logging).
        try:
            log = get_retrieval_logger()
            log.log_retrieval(
                query=query,
                retrieved_docs=top_docs,
                user_id=int(self.user_id) if self.user_id else None,
                metadata={
                    "stage": "hybrid+rerank",
                    "candidates": len(candidates),
                    "top_k": self.top_k,
                    "rrf_k": self.hybrid.rrf_k,
                    "k_dense": self.hybrid.k_dense,
                    "k_sparse": self.hybrid.k_sparse,
                },
            )
        except Exception as exc:
            logger.warning("Failed to log retrieval: %s", exc)

        return top_docs


class VectorStoreService:
    """Service for managing PGVector operations + retrieval pipeline factory."""

    def __init__(self):
        self._engine: Optional[PGEngine] = None
        self._vector_store: Optional[PGVectorStore] = None
        self._initialized = False
        self._fts_initialized = False

    def _ensure_initialized(self):
        """Ensure vector store and FTS infrastructure are initialized."""
        if not self._initialized:
            normalized_db_uri = normalize_db_uri_for_asyncpg(settings.DATABASE_URL)
            self._engine = PGEngine.from_connection_string(url=normalized_db_uri)

            try:
                self._engine.init_vectorstore_table(
                    table_name=settings.VECTOR_STORE_TABLE_NAME,
                    vector_size=settings.VECTOR_SIZE,
                )
            except Exception:
                pass  # Table already exists

            if not os.environ.get("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

            self._vector_store = PGVectorStore.create_sync(
                engine=self._engine,
                table_name=settings.VECTOR_STORE_TABLE_NAME,
                embedding_service=OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY),
            )
            self._initialized = True

        if not self._fts_initialized:
            self._ensure_fts_infrastructure()
            self._fts_initialized = True

    def _ensure_fts_infrastructure(self) -> None:
        """Idempotently install the tsvector column + GIN + user_id index.

        The Alembic migration ``8b3d2e0a4c5f`` no-ops on a fresh DB because
        the langchain-postgres ``data`` table is created lazily on first
        ingestion. So we also run the same DDL here, after the table is
        guaranteed to exist. ``IF NOT EXISTS`` makes this safe on every call.
        """
        table_name = settings.VECTOR_STORE_TABLE_NAME
        ddl = [
            f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS tsv tsvector "
            f"GENERATED ALWAYS AS (to_tsvector('english', content)) STORED",
            f"CREATE INDEX IF NOT EXISTS data_tsv_gin_idx "
            f"ON {table_name} USING GIN (tsv)",
            f"CREATE INDEX IF NOT EXISTS data_user_id_idx "
            f"ON {table_name} ((langchain_metadata->>'user_id'))",
        ]
        try:
            with sync_engine.begin() as conn:
                for stmt in ddl:
                    conn.execute(text(stmt))
        except Exception as exc:
            # Don't crash startup if the DDL fails (e.g. permissions);
            # sparse search will just return empty and hybrid retrieval
            # falls back to dense-only.
            logger.warning(
                "Failed to install FTS infrastructure (sparse arm will be a no-op): %s",
                exc,
            )

    def get_vector_store(self) -> PGVectorStore:
        """Get the vector store instance."""
        self._ensure_initialized()
        return self._vector_store

    def get_retriever(
        self,
        user_id: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> BaseRetriever:
        """Get the full retrieval pipeline: hybrid + RRF + reranker.

        Args:
            user_id: If set, applied as a post-filter on the dense arm and as
                a SQL ``WHERE`` clause on the sparse arm.
            top_k: Number of documents returned to the LangGraph tool. Defaults
                to ``settings.RERANK_TOP_K`` (5).

        Returns:
            A ``BaseRetriever`` that synchronously runs:
              1. Dense (PGVector cosine, k=HYBRID_DENSE_K).
              2. Sparse (Postgres ts_rank_cd, k=HYBRID_SPARSE_K).
              3. RRF fuse (k_rrf=RRF_K) → HYBRID_FINAL_K candidates.
              4. Cross-encoder rerank (BGE default, Cohere opt-in) → top_k.
        """
        self._ensure_initialized()

        if top_k is None:
            top_k = getattr(settings, "RERANK_TOP_K", 5)

        hybrid = HybridRetriever(
            vectorstore=self._vector_store,
            user_id=str(user_id) if user_id is not None else None,
            k_dense=getattr(settings, "HYBRID_DENSE_K", 20),
            k_sparse=getattr(settings, "HYBRID_SPARSE_K", 20),
            rrf_k=getattr(settings, "RRF_K", 60),
            k_final=getattr(settings, "HYBRID_FINAL_K", 20),
        )

        return HybridReranKRetriever(
            hybrid=hybrid,
            top_k=top_k,
            user_id=str(user_id) if user_id is not None else None,
        )

    def add_documents(self, documents: List, user_id: Optional[int] = None) -> List[str]:
        """Add documents to vector store with optional user_id metadata."""
        self._ensure_initialized()

        if user_id is not None:
            for doc in documents:
                if not hasattr(doc, "metadata") or doc.metadata is None:
                    doc.metadata = {}
                doc.metadata["user_id"] = str(user_id)

        return self._vector_store.add_documents(documents=documents)

    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from vector store."""
        if not document_ids:
            raise ValueError("document_ids list cannot be empty")
        self._ensure_initialized()
        self._vector_store.delete(ids=document_ids)


_vector_store_service: Optional[VectorStoreService] = None
_vector_store_service_lock = threading.Lock()


def get_vector_store_service() -> VectorStoreService:
    """Get the global vector store service instance (thread-safe)."""
    global _vector_store_service
    if _vector_store_service is None:
        with _vector_store_service_lock:
            if _vector_store_service is None:
                _vector_store_service = VectorStoreService()
    return _vector_store_service
