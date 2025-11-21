"""Vector store service for PGVector operations."""
from langchain_postgres import PGEngine, PGVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List, Optional
import os
from app.core.config import settings
from app.utils.db_uri import normalize_db_uri_for_asyncpg
from app.utils.retrieval_logger import get_retrieval_logger


class UserFilteredRetriever(VectorStoreRetriever):
    """Custom retriever that post-filters results by user_id.
    
    Since PGVector's built-in metadata filtering has issues with JSONB columns,
    we retrieve more documents than needed (fetch_k) and filter client-side.
    """
    
    user_id: Optional[str] = None
    fetch_k_multiplier: int = 3  # Fetch 3x more docs than needed for filtering
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to a query, filtered by user_id."""
        
        # Get k from search_kwargs
        k = self.search_kwargs.get('k', 5)
        
        if self.user_id is None:
            # No filtering needed
            docs = self.vectorstore.similarity_search(query, k=k)
        else:
            # Fetch more docs than needed (for post-filtering)
            fetch_k = k * self.fetch_k_multiplier
            
            # Remove filter from search_kwargs to avoid SQL errors
            search_kwargs_no_filter = {k_: v for k_, v in self.search_kwargs.items() if k_ != 'filter'}
            search_kwargs_no_filter['k'] = fetch_k
            
            # Get raw results without filter
            docs = self.vectorstore.similarity_search(query, **search_kwargs_no_filter)
            
            # Post-filter by user_id
            filtered_docs = [
                doc for doc in docs
                if doc.metadata.get('user_id') == self.user_id
            ]
            
            docs = filtered_docs[:k]
        
        # Log the retrieval
        try:
            logger = get_retrieval_logger()
            logger.log_retrieval(
                query=query,
                retrieved_docs=docs,
                user_id=int(self.user_id) if self.user_id else None,
                metadata={
                    "k": k,
                    "fetch_k_multiplier": self.fetch_k_multiplier if self.user_id else None
                }
            )
        except Exception as e:
            # Don't fail retrieval if logging fails
            import logging
            logging.warning(f"Failed to log retrieval: {e}")
        
        return docs


class VectorStoreService:
    """Service for managing PGVector operations."""
    
    def __init__(self):
        """Initialize vector store service."""
        self._engine: Optional[PGEngine] = None
        self._vector_store: Optional[PGVectorStore] = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """Ensure vector store is initialized."""
        if not self._initialized:
            normalized_db_uri = normalize_db_uri_for_asyncpg(settings.DATABASE_URL)
            self._engine = PGEngine.from_connection_string(url=normalized_db_uri)
            
            # Initialize table if needed
            try:
                self._engine.init_vectorstore_table(
                    table_name=settings.VECTOR_STORE_TABLE_NAME,
                    vector_size=settings.VECTOR_SIZE
                )
            except Exception:
                pass  # Table already exists
            
            # Ensure OPENAI_API_KEY is set as environment variable
            # OpenAI client checks os.environ directly even if we have it in settings
            if not os.environ.get("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
            
            self._vector_store = PGVectorStore.create_sync(
                engine=self._engine,
                table_name=settings.VECTOR_STORE_TABLE_NAME,
                embedding_service=OpenAIEmbeddings(
                    api_key=settings.OPENAI_API_KEY  # Pass API key explicitly
                )
            )
            self._initialized = True
    
    def get_vector_store(self) -> PGVectorStore:
        """Get the vector store instance."""
        self._ensure_initialized()
        return self._vector_store
    
    def get_retriever(
        self,
        user_id: Optional[int] = None,
        search_type: str = "similarity",
        search_kwargs: Optional[dict] = None
    ) -> VectorStoreRetriever:
        """Get a retriever with user isolation via metadata filtering.
        
        Retrieves top-k most similar documents from the user's filtered document set.
        
        Args:
            user_id: If provided, only retrieve documents belonging to this user
            search_type: Type of search (similarity, mmr, etc.)
            search_kwargs: Additional search parameters including:
                - k: Number of documents to return (default: 5)
                - score_threshold: Minimum similarity score (optional)
                - fetch_k: For MMR, number of docs to fetch before reranking (default: 20)
            
        Returns:
            VectorStoreRetriever configured with user filtering and top-k retrieval
            
        Example:
            >>> retriever = service.get_retriever(user_id=16, search_kwargs={"k": 5})
            >>> # Returns top 5 most relevant docs from user 16's documents only
        """
        self._ensure_initialized()
        
        # Initialize search_kwargs with defaults if not provided
        if search_kwargs is None:
            search_kwargs = {}
        
        # Ensure 'k' is set (number of documents to retrieve)
        # This determines how many top results to return from the filtered set
        if 'k' not in search_kwargs:
            search_kwargs['k'] = 5  # Default: retrieve top 5 most similar documents
        
        # CRITICAL FIX: Implement user isolation via POST-filtering
        # This prevents data leakage between users and ensures correct retrieval
        # 
        # Note: PGVector's built-in metadata filtering has issues with JSONB columns
        # generating incorrect SQL (WHERE user_id = X instead of WHERE langchain_metadata->>'user_id' = X)
        # So we use post-filtering: fetch more docs than needed, filter client-side, return top-k
        
        if user_id is not None:
            # Use custom retriever with post-filtering
            retriever = UserFilteredRetriever(
                vectorstore=self._vector_store,
                search_type=search_type,
                search_kwargs=search_kwargs,
                user_id=str(user_id)
            )
        else:
            # No filtering needed, use standard retriever
            retriever = self._vector_store.as_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs
            )
        
        return retriever
    
    def add_documents(self, documents: List, user_id: Optional[int] = None) -> List[str]:
        """Add documents to vector store with optional user_id metadata."""
        self._ensure_initialized()
        
        # Add user_id to metadata if provided
        if user_id is not None:
            for doc in documents:
                # Ensure metadata exists
                if not hasattr(doc, 'metadata') or doc.metadata is None:
                    doc.metadata = {}
                # Add user_id for filtering
                doc.metadata['user_id'] = str(user_id)
        
        return self._vector_store.add_documents(documents=documents)
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from vector store.
        
        Args:
            document_ids: List of document IDs to delete
            
        Raises:
            ValueError: If document_ids is empty
        """
        if not document_ids:
            raise ValueError("document_ids list cannot be empty")
        
        self._ensure_initialized()
        self._vector_store.delete(ids=document_ids)


# Global instance with thread-safe initialization
import threading
_vector_store_service = None
_vector_store_service_lock = threading.Lock()


def get_vector_store_service() -> VectorStoreService:
    """Get the global vector store service instance (thread-safe)."""
    global _vector_store_service
    if _vector_store_service is None:
        with _vector_store_service_lock:
            # Double-check pattern
            if _vector_store_service is None:
                _vector_store_service = VectorStoreService()
    return _vector_store_service

