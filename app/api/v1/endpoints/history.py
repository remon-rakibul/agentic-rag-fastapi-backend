"""Chat history endpoint."""
from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy.orm import Session
from typing import Optional
from app.api.dependencies import get_current_user
from app.core.database import get_db
from app.models.database import User
from app.models.schemas import ChatHistoryResponse, ThreadMessagesResponse
from app.services.history_service import get_history_service

router = APIRouter(prefix="/history", tags=["chat"])


@router.get("", response_model=ChatHistoryResponse)
def get_history(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get chat history (list of threads) for the current user."""
    history_service = get_history_service(db)
    
    threads, total = history_service.get_user_threads(
        user_id=current_user.id,
        page=page,
        limit=limit
    )
    
    return ChatHistoryResponse(
        threads=threads,
        total=total,
        page=page,
        limit=limit
    )


@router.get("/{thread_id}", response_model=ThreadMessagesResponse)
def get_thread_messages(
    thread_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all messages for a specific thread."""
    # Validate thread_id
    if not thread_id or not thread_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="thread_id cannot be empty"
        )
    
    history_service = get_history_service(db)
    
    messages, total = history_service.get_thread_messages(
        thread_id=thread_id,
        user_id=current_user.id
    )
    
    return ThreadMessagesResponse(
        thread_id=thread_id,
        messages=messages,
        total=total
    )

