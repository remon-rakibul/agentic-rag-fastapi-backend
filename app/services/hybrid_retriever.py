"""Hybrid retriever: dense + Postgres full-text search fused with RRF.

This is the first stage of the retrieval pipeline. It runs two searches in
parallel and fuses their ranked outputs with Reciprocal Rank Fusion:

    score(d) = Σ 1 / (k_rrf + rank_i(d))     with k_rrf = 60

The dense arm uses the existing PGVector ``similarity_search`` (cosine).
The sparse arm uses Postgres ``ts_rank_cd`` over the ``tsv`` generated
column added by the ``8b3d2e0a4c5f`` migration. Both arms enforce the
``user_id`` post-filter / WHERE clause so a user only ever sees their own
chunks.

A reranker (see ``app/services/reranker_service.py``) is applied AFTER
this stage to produce the final top-K passed into the LangGraph workflow.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from sqlalchemy import text

from app.core.config import settings
from app.core.database import engine

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: List[List[Document]],
    k_rrf: int = 60,
    top_k: Optional[int] = None,
) -> List[Document]:
    """Fuse multiple ranked lists of Documents with Reciprocal Rank Fusion.

    The same Document can appear in multiple lists; its scores are summed.
    Documents are identified by ``langchain_id`` from metadata when present,
    otherwise by ``page_content`` (used as a fallback hash key).
    """
    scores: Dict[str, float] = {}
    docs: Dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k_rrf + rank + 1)
            # Prefer the first occurrence for the canonical Document object.
            docs.setdefault(key, doc)

    fused = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    if top_k is not None:
        fused = fused[:top_k]
    return [docs[k] for k in fused]


def _doc_key(doc: Document) -> str:
    """Stable identifier for a Document in fusion. Prefers ``langchain_id``."""
    md = doc.metadata or {}
    lid = md.get("langchain_id")
    if lid:
        return f"id:{lid}"
    # Fallback: hash the content + a small slice of metadata to disambiguate.
    return f"c:{hash(doc.page_content)}"


class HybridRetriever(BaseRetriever):
    """Dense (PGVector cosine) + sparse (Postgres tsvector) + RRF.

    Implements the LangChain ``BaseRetriever`` interface so it slots in
    wherever the existing ``UserFilteredRetriever`` was used. The reranker
    is intentionally NOT applied here so callers can compose freely; see
    ``vector_store_service.get_retriever`` for the wired pipeline.
    """

    # Pydantic config: allow arbitrary types for the vectorstore.
    class Config:
        arbitrary_types_allowed = True

    vectorstore: Any
    user_id: Optional[str] = None
    k_dense: int = 20
    k_sparse: int = 20
    rrf_k: int = 60
    k_final: int = 20  # before reranker

    # Multiplier for over-fetching on the dense arm so user_id post-filter
    # still leaves enough survivors. Mirrors the existing
    # UserFilteredRetriever pattern.
    dense_fetch_multiplier: int = 3

    def _dense_search(self, query: str) -> List[Document]:
        """Cosine similarity search with user_id post-filter."""
        if self.user_id is None:
            try:
                return self.vectorstore.similarity_search(query, k=self.k_dense)
            except Exception as exc:
                logger.warning("Dense search failed: %s", exc)
                return []

        # Over-fetch then post-filter, exactly like UserFilteredRetriever did.
        fetch_k = self.k_dense * self.dense_fetch_multiplier
        try:
            raw = self.vectorstore.similarity_search(query, k=fetch_k)
        except Exception as exc:
            logger.warning("Dense search failed: %s", exc)
            return []
        filtered = [
            d for d in raw if (d.metadata or {}).get("user_id") == self.user_id
        ]
        return filtered[: self.k_dense]

    def _sparse_search(self, query: str) -> List[Document]:
        """Postgres full-text search via ``ts_rank_cd`` on the ``tsv`` column."""
        if not query or not query.strip():
            return []

        table_name = settings.VECTOR_STORE_TABLE_NAME

        # Use plainto_tsquery so users can paste arbitrary strings without
        # worrying about tsquery operators. We restrict by user_id at the SQL
        # layer for tenant isolation.
        sql_parts = [
            f"SELECT langchain_id, content, langchain_metadata, ",
            f"  ts_rank_cd(tsv, plainto_tsquery('english', :q)) AS score ",
            f"FROM {table_name} ",
            f"WHERE tsv @@ plainto_tsquery('english', :q) ",
        ]
        params: Dict[str, Any] = {"q": query, "k": self.k_sparse}
        if self.user_id is not None:
            sql_parts.append(f"  AND langchain_metadata->>'user_id' = :uid ")
            params["uid"] = self.user_id
        sql_parts.append("ORDER BY score DESC LIMIT :k")
        sql = "".join(sql_parts)

        try:
            with engine.connect() as conn:
                rows = conn.execute(text(sql), params).fetchall()
        except Exception as exc:
            # If the tsv column / index hasn't been created yet (fresh DB
            # before first ingestion), the sparse arm just returns empty
            # and hybrid retrieval falls back to dense-only. No crash.
            logger.warning("Sparse search failed (falling back to dense-only): %s", exc)
            return []

        docs: List[Document] = []
        for row in rows:
            mapping = row._mapping if hasattr(row, "_mapping") else dict(row)
            md = dict(mapping.get("langchain_metadata") or {})
            md["langchain_id"] = str(mapping["langchain_id"])
            md["_sparse_score"] = float(mapping["score"])
            docs.append(
                Document(
                    page_content=mapping["content"] or "",
                    metadata=md,
                )
            )
        return docs

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        dense = self._dense_search(query)
        sparse = self._sparse_search(query)

        if not dense and not sparse:
            return []

        fused = reciprocal_rank_fusion(
            [dense, sparse],
            k_rrf=self.rrf_k,
            top_k=self.k_final,
        )
        return fused
