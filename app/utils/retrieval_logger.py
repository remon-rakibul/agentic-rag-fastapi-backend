"""Retrieval logging utility for tracking queries and retrieved documents."""
import json
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
from contextvars import ContextVar
from langchain_core.documents import Document

# Context variables for passing thread_id and original_question to retriever
_retrieval_context_thread_id: ContextVar[Optional[str]] = ContextVar('retrieval_thread_id', default=None)
_retrieval_context_original_question: ContextVar[Optional[str]] = ContextVar('retrieval_original_question', default=None)


class RetrievalLogger:
    """Logs retrieval queries and their corresponding document chunks.
    
    Uses JSONL format (one JSON object per line) for easy appending and analysis.
    """
    
    def __init__(self, log_file: Optional[str] = None):
        """Initialize retrieval logger.
        
        Args:
            log_file: Path to log file. Defaults to 'retrieval_logs.jsonl' in project root.
        """
        if log_file is None:
            # Default to project root (refactored/)
            project_root = Path(__file__).parent.parent.parent
            log_file = str(project_root / "retrieval_logs.jsonl")
        
        self.log_file = log_file
        self._ensure_log_file()
    
    @staticmethod
    def set_context(thread_id: Optional[str] = None, original_question: Optional[str] = None):
        """Set context variables for current retrieval operation.
        
        Args:
            thread_id: Thread/conversation ID
            original_question: Original user question
        """
        _retrieval_context_thread_id.set(thread_id)
        _retrieval_context_original_question.set(original_question)
    
    @staticmethod
    def get_context() -> tuple[Optional[str], Optional[str]]:
        """Get current context variables.
        
        Returns:
            Tuple of (thread_id, original_question)
        """
        return (
            _retrieval_context_thread_id.get(),
            _retrieval_context_original_question.get()
        )
    
    def _ensure_log_file(self):
        """Ensure log file exists."""
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Create file if it doesn't exist (don't write anything yet)
        if not log_path.exists():
            log_path.touch()
    
    def log_retrieval(
        self,
        query: str,
        retrieved_docs: List[Document],
        user_id: Optional[int] = None,
        thread_id: Optional[str] = None,
        original_question: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log a retrieval operation.
        
        Args:
            query: The search query used for retrieval
            retrieved_docs: List of Document objects retrieved
            user_id: User ID who made the query
            thread_id: Thread/conversation ID
            original_question: Original user question (before query rewriting)
            metadata: Additional metadata to store
        """
        # Extract document information
        doc_chunks = []
        for i, doc in enumerate(retrieved_docs):
            doc_info = {
                "index": i,
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,  # Truncate for readability
                "content_length": len(doc.page_content),
                "metadata": {
                    k: v for k, v in doc.metadata.items()
                    if k not in ['user_id']  # Exclude sensitive or redundant fields
                }
            }
            # Include document ID if available
            if hasattr(doc, 'id') and doc.id:
                doc_info["doc_id"] = doc.id
            elif 'id' in doc.metadata:
                doc_info["doc_id"] = doc.metadata['id']
            
            doc_chunks.append(doc_info)
        
        # Get context from context variables if not provided
        if thread_id is None:
            thread_id = _retrieval_context_thread_id.get()
        if original_question is None:
            original_question = _retrieval_context_original_question.get()
        
        # Build log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "original_question": original_question,
            "user_id": user_id,
            "thread_id": thread_id,
            "num_documents_retrieved": len(retrieved_docs),
            "documents": doc_chunks,
            "metadata": metadata or {}
        }
        
        # Append to JSONL file
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            # Don't fail the retrieval if logging fails
            import logging
            logging.error(f"Failed to log retrieval: {e}")
    
    def read_logs(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Read retrieval logs.
        
        Args:
            limit: Maximum number of entries to return (most recent first)
            
        Returns:
            List of log entries
        """
        if not os.path.exists(self.log_file):
            return []
        
        entries = []
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))
        except Exception as e:
            import logging
            logging.error(f"Failed to read logs: {e}")
            return []
        
        # Return most recent first
        entries.reverse()
        if limit:
            entries = entries[:limit]
        
        return entries
    
    def get_logs_by_query(self, query: str) -> List[Dict[str, Any]]:
        """Get all logs for a specific query.
        
        Args:
            query: Query string to search for
            
        Returns:
            List of log entries matching the query
        """
        all_logs = self.read_logs()
        return [log for log in all_logs if log.get('query', '').lower() == query.lower()]
    
    def get_logs_by_user(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all logs for a specific user.
        
        Args:
            user_id: User ID to filter by
            
        Returns:
            List of log entries for the user
        """
        all_logs = self.read_logs()
        return [log for log in all_logs if log.get('user_id') == user_id]


# Global logger instance
_retrieval_logger: Optional[RetrievalLogger] = None


def get_retrieval_logger() -> RetrievalLogger:
    """Get the global retrieval logger instance."""
    global _retrieval_logger
    if _retrieval_logger is None:
        _retrieval_logger = RetrievalLogger()
    return _retrieval_logger

