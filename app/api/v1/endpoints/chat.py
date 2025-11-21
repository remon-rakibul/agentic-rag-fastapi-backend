"""Chat endpoint with streaming."""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import json
import uuid
from app.api.dependencies import get_current_user
from app.core.database import get_db, SessionLocal
from app.models.database import User
from app.models.schemas import ChatRequest
from app.services.chat_service import get_chat_service
from app.services.history_service import get_history_service

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_class=StreamingResponse)
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Stream chat response from RAG workflow."""
    chat_service = get_chat_service()
    history_service = get_history_service(db)
    
    # Validate message
    if not request.message or not request.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty"
        )
    
    # Extract user_id BEFORE the async generator to avoid DetachedInstanceError
    # The database session will close after this function returns, but we need user_id in the generator
    user_id = current_user.id
    
    # Create or get thread
    thread_id = request.thread_id
    if not thread_id or not thread_id.strip():
        thread_id = str(uuid.uuid4())
    
    # Generate title from message (first 100 chars)
    title = request.message[:100].strip() if request.message else None
    
    thread = history_service.get_or_create_thread(
        user_id=user_id,  # Use extracted user_id
        thread_id=thread_id,
        title=title
    )
    
    # Save user message
    history_service.add_message(
        thread_id=thread_id,
        role="user",
        content=request.message.strip()
    )
    
    # Create a new session for the async generator
    # The original db session will be closed when the dependency context ends
    async def generate():
        """Generate streaming response (fully async)."""
        # Create a new database session for the async generator
        # This ensures the session stays alive during streaming
        async_db = SessionLocal()
        try:
            async_history_service = get_history_service(async_db)
            full_response = ""
            
            async for chunk in chat_service.stream_chat(
                message=request.message,
                user_id=user_id,  # Use extracted user_id instead of current_user.id
                thread_id=thread_id
            ):
                if chunk["type"] == "token":
                    full_response += chunk.get("content", "")
                    yield f"data: {json.dumps(chunk)}\n\n"
                elif chunk["type"] == "done":
                    # Save assistant response using the async session
                    async_history_service.add_message(
                        thread_id=thread_id,
                        role="assistant",
                        content=chunk.get("content", full_response)
                    )
                    yield f"data: {json.dumps(chunk)}\n\n"
                elif chunk["type"] == "error":
                    yield f"data: {json.dumps(chunk)}\n\n"
        finally:
            async_db.close()
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

