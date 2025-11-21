"""Document removal endpoint."""
from fastapi import APIRouter, Depends, HTTPException, status, Path
from sqlalchemy.orm import Session
import json
from typing import List
from app.api.dependencies import get_current_user
from app.core.database import get_db
from app.models.database import User, Document
from app.models.schemas import RemoveRequest, RemoveResponse
from app.services.vector_store_service import get_vector_store_service

router = APIRouter(prefix="/remove", tags=["documents"])


@router.delete("", response_model=RemoveResponse)
def remove_documents(
    request: RemoveRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Remove documents from vector store."""
    # Validate request
    if not request.document_ids or len(request.document_ids) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="document_ids list cannot be empty"
        )
    
    vector_store_service = get_vector_store_service()
    
    # Verify documents belong to user and get their IDs
    user_docs = db.query(Document).filter(
        Document.user_id == current_user.id
    ).all()
    
    # Collect all document IDs from user's documents
    all_user_doc_ids = set()
    for doc in user_docs:
        if doc.document_ids:
            try:
                doc_ids = json.loads(doc.document_ids)
                all_user_doc_ids.update(doc_ids)
            except json.JSONDecodeError:
                continue
    
    # Filter requested IDs to only include user's documents
    valid_ids = [doc_id for doc_id in request.document_ids if doc_id in all_user_doc_ids]
    
    if not valid_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No valid documents found to remove"
        )
    
    # Remove from vector store (with error handling)
    try:
        vector_store_service.delete_documents(valid_ids)
    except Exception as e:
        # If vector store deletion fails, rollback DB changes
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove documents from vector store: {str(e)}"
        )
    
    # Update document metadata entries - remove deleted IDs
    # Only delete Document records if ALL their document_ids were removed
    docs_to_delete = []
    for doc in user_docs:
        if doc.document_ids:
            try:
                doc_ids = json.loads(doc.document_ids)
                # Remove deleted IDs from the list
                remaining_ids = [doc_id for doc_id in doc_ids if doc_id not in valid_ids]
                
                if remaining_ids:
                    # Update with remaining IDs
                    doc.document_ids = json.dumps(remaining_ids)
                    doc.chunk_count = len(remaining_ids)  # Update chunk count
                else:
                    # All IDs were deleted, mark for deletion
                    docs_to_delete.append(doc)
            except json.JSONDecodeError:
                # Invalid JSON, mark for deletion
                docs_to_delete.append(doc)
    
    # Delete documents that have no remaining IDs
    for doc in docs_to_delete:
        db.delete(doc)
    
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove documents: {str(e)}"
        )
    
    return RemoveResponse(
        removed_count=len(valid_ids),
        status="success"
    )


@router.delete("/by-record/{record_id}", response_model=RemoveResponse)
def remove_document_by_record_id(
    record_id: int = Path(..., description="Database record ID from /data endpoint"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Remove a document by its database record ID (simpler alternative).
    
    This endpoint accepts the `id` field from the `/data` endpoint response,
    making it easier to remove documents without needing to extract document_ids.
    
    **Example:**
    - Get the document `id` from `/api/v1/data` (e.g., `"id": 2`)
    - Call DELETE `/api/v1/remove/by-record/2`
    """
    vector_store_service = get_vector_store_service()
    
    # Find the document record
    doc = db.query(Document).filter(
        Document.id == record_id,
        Document.user_id == current_user.id
    ).first()
    
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id {record_id} not found or doesn't belong to you"
        )
    
    # Get document IDs from the record
    doc_ids = []
    if doc.document_ids:
        try:
            doc_ids = json.loads(doc.document_ids)
        except json.JSONDecodeError:
            pass
    
    if not doc_ids:
        # No vector store IDs, just delete the database record
        db.delete(doc)
        try:
            db.commit()
        except Exception as e:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to remove document: {str(e)}"
            )
        
        return RemoveResponse(
            removed_count=0,
            status="success"
        )
    
    # Remove from vector store
    try:
        vector_store_service.delete_documents(doc_ids)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove documents from vector store: {str(e)}"
        )
    
    # Delete the database record
    db.delete(doc)
    
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove document: {str(e)}"
        )
    
    return RemoveResponse(
        removed_count=len(doc_ids),
        status="success"
    )

