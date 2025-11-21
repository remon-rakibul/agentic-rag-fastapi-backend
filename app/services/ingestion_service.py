"""Document ingestion service."""
from typing import List, Optional
from langchain_core.documents import Document
from app.services.vector_store_service import get_vector_store_service
from app.utils.loaders import (
    load_documents_from_urls,
    load_documents_from_file,
    save_uploaded_file,
    get_file_type,
    cleanup_file
)
from app.utils.text_splitter import split_documents
from app.core.config import settings
import json


class IngestionService:
    """Service for ingesting documents into the vector store."""
    
    def __init__(self):
        self.vector_store_service = get_vector_store_service()
    
    async def ingest_urls(
        self,
        urls: List[str],
        user_id: Optional[int] = None,
        metadata: Optional[dict] = None
    ) -> tuple[List[str], int]:
        """Ingest documents from URLs.
        
        Returns:
            Tuple of (document_ids, chunk_count)
        """
        # Load documents from URLs
        documents = await load_documents_from_urls(urls)
        
        # Validate that we got some documents
        if not documents:
            raise ValueError("No documents could be loaded from the provided URLs")
        
        # Add metadata
        for doc in documents:
            # Ensure metadata exists
            if not hasattr(doc, 'metadata') or doc.metadata is None:
                doc.metadata = {}
            
            # Add provided metadata
            if metadata:
                doc.metadata.update(metadata)
            
            # Add source information
            doc.metadata['source_type'] = 'url'
        
        # Split documents
        split_docs = split_documents(documents)
        
        # Add to vector store
        document_ids = self.vector_store_service.add_documents(split_docs, user_id=user_id)
        
        return document_ids, len(split_docs)
    
    async def ingest_files(
        self,
        files: List[tuple[bytes, str]],  # List of (file_content, filename) tuples
        user_id: Optional[int] = None,
        metadata: Optional[dict] = None
    ) -> tuple[List[str], int]:
        """Ingest documents from uploaded files.
        
        Returns:
            Tuple of (document_ids, chunk_count)
        """
        all_document_ids = []
        total_chunks = 0
        saved_files = []
        
        try:
            for file_content, filename in files:
                # Determine file type
                file_type = get_file_type(filename)
                
                # Save file temporarily
                file_path = await save_uploaded_file(
                    file_content,
                    filename,
                    settings.UPLOAD_DIR
                )
                saved_files.append(file_path)
                
                # Load documents
                documents = await load_documents_from_file(file_path, file_type)
                
                # Validate that we got some documents
                if not documents:
                    raise ValueError(f"No documents could be loaded from file {filename}")
                
                # Add metadata
                for doc in documents:
                    # Ensure metadata exists
                    if not hasattr(doc, 'metadata') or doc.metadata is None:
                        doc.metadata = {}
                    
                    # Add provided metadata
                    if metadata:
                        doc.metadata.update(metadata)
                    
                    # Add source information
                    doc.metadata['source_type'] = file_type
                    doc.metadata['source_path'] = filename
                
                # Split documents
                split_docs = split_documents(documents)
                
                # Add to vector store
                doc_ids = self.vector_store_service.add_documents(split_docs, user_id=user_id)
                all_document_ids.extend(doc_ids)
                total_chunks += len(split_docs)
        finally:
            # Cleanup temporary files
            for file_path in saved_files:
                await cleanup_file(file_path)
        
        return all_document_ids, total_chunks


def get_ingestion_service() -> IngestionService:
    """Get ingestion service instance."""
    return IngestionService()

