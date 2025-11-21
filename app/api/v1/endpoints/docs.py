"""Document listing endpoint."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
import json
from app.api.dependencies import get_current_user
from app.core.database import get_db
from app.models.database import User, Document
from app.models.schemas import DocumentsListResponse, DocumentResponse

router = APIRouter(prefix="/data", tags=["documents"])


@router.get("", response_model=DocumentsListResponse)
def list_documents(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List all documents for the current user.
    
    Returns document metadata including vector store IDs needed for removal.
    Use the `document_ids` field values when calling the `/remove` endpoint.
    """
    offset = (page - 1) * limit
    
    # Get user's documents
    documents = db.query(Document).filter(
        Document.user_id == current_user.id
    ).order_by(Document.created_at.desc()).offset(offset).limit(limit).all()
    
    total = db.query(Document).filter(
        Document.user_id == current_user.id
    ).count()
    
    document_responses = []
    for doc in documents:
        # Parse document_ids from JSON
        doc_ids = None
        if doc.document_ids:
            try:
                doc_ids = json.loads(doc.document_ids)
            except json.JSONDecodeError:
                doc_ids = None
        
        document_responses.append(
            DocumentResponse(
                id=doc.id,
                source_type=doc.source_type,
                source_path=doc.source_path,
                chunk_count=doc.chunk_count,
                document_ids=doc_ids,
                created_at=doc.created_at
            )
        )
    
    return DocumentsListResponse(
        documents=document_responses,
        total=total,
        page=page,
        limit=limit
    )

