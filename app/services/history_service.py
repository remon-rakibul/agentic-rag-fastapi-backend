"""History service for managing chat threads and messages."""
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Optional
from app.models.database import ChatThread, ChatMessage, User
from app.models.schemas import ChatThreadResponse, ChatMessageHistory
from datetime import datetime, timezone


class HistoryService:
    """Service for managing chat history."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_thread(
        self,
        user_id: int,
        thread_id: str,
        title: Optional[str] = None
    ) -> ChatThread:
        """Create a new chat thread."""
        try:
            thread = ChatThread(
                user_id=user_id,
                thread_id=thread_id,
                title=title
            )
            self.db.add(thread)
            self.db.commit()
            self.db.refresh(thread)
            return thread
        except Exception:
            self.db.rollback()
            raise
    
    def get_or_create_thread(
        self,
        user_id: int,
        thread_id: str,
        title: Optional[str] = None
    ) -> ChatThread:
        """Get existing thread or create new one."""
        thread = self.db.query(ChatThread).filter(
            ChatThread.thread_id == thread_id,
            ChatThread.user_id == user_id
        ).first()
        
        if not thread:
            thread = self.create_thread(user_id, thread_id, title)
        
        return thread
    
    def add_message(
        self,
        thread_id: str,
        role: str,
        content: str
    ) -> ChatMessage:
        """Add a message to a thread."""
        try:
            message = ChatMessage(
                thread_id=thread_id,
                role=role,
                content=content
            )
            self.db.add(message)
            
            # Update thread updated_at
            thread = self.db.query(ChatThread).filter(
                ChatThread.thread_id == thread_id
            ).first()
            if thread:
                thread.updated_at = datetime.now(timezone.utc)
            
            self.db.commit()
            self.db.refresh(message)
            return message
        except Exception:
            self.db.rollback()
            raise
    
    def get_user_threads(
        self,
        user_id: int,
        page: int = 1,
        limit: int = 20
    ) -> tuple[List[ChatThreadResponse], int]:
        """Get all threads for a user."""
        offset = (page - 1) * limit
        
        threads = self.db.query(ChatThread).filter(
            ChatThread.user_id == user_id
        ).order_by(desc(ChatThread.updated_at)).offset(offset).limit(limit).all()
        
        total = self.db.query(ChatThread).filter(
            ChatThread.user_id == user_id
        ).count()
        
        thread_responses = []
        for thread in threads:
            message_count = self.db.query(ChatMessage).filter(
                ChatMessage.thread_id == thread.thread_id
            ).count()
            
            thread_responses.append(ChatThreadResponse(
                thread_id=thread.thread_id,
                title=thread.title,
                message_count=message_count,
                updated_at=thread.updated_at
            ))
        
        return thread_responses, total
    
    def get_thread_messages(
        self,
        thread_id: str,
        user_id: int
    ) -> tuple[List[ChatMessageHistory], int]:
        """Get all messages for a thread."""
        # Verify thread belongs to user
        thread = self.db.query(ChatThread).filter(
            ChatThread.thread_id == thread_id,
            ChatThread.user_id == user_id
        ).first()
        
        if not thread:
            return [], 0
        
        messages = self.db.query(ChatMessage).filter(
            ChatMessage.thread_id == thread_id
        ).order_by(ChatMessage.created_at).all()
        
        message_responses = [
            ChatMessageHistory(
                role=msg.role,
                content=msg.content,
                created_at=msg.created_at
            )
            for msg in messages
        ]
        
        return message_responses, len(message_responses)


def get_history_service(db: Session) -> HistoryService:
    """Get history service instance."""
    return HistoryService(db)

