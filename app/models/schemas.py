"""Pydantic schemas for request/response validation."""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime


# Auth Schemas
class UserCreate(BaseModel):
    """User registration schema."""
    email: EmailStr
    password: str = Field(..., min_length=8)


class UserResponse(BaseModel):
    """User response schema."""
    id: int
    email: str
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class Token(BaseModel):
    """Token response schema."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenRefresh(BaseModel):
    """Token refresh request schema."""
    refresh_token: str


class LogoutResponse(BaseModel):
    """Logout response schema."""
    message: str


class TokenData(BaseModel):
    """Token data schema."""
    user_id: Optional[int] = None


# Document Schemas
class DigestRequest(BaseModel):
    """Document ingestion request schema."""
    urls: Optional[List[str]] = Field(default=None, description="List of URLs to ingest")
    metadata: Optional[dict] = Field(default=None, description="Additional metadata")


class DigestResponse(BaseModel):
    """Document ingestion response schema."""
    document_ids: List[str]
    chunk_count: int
    status: str = "success"


class RemoveRequest(BaseModel):
    """Document removal request schema."""
    document_ids: List[str] = Field(..., description="List of document IDs to remove")


class RemoveResponse(BaseModel):
    """Document removal response schema."""
    removed_count: int
    status: str = "success"


class DocumentResponse(BaseModel):
    """Document response schema."""
    id: int
    source_type: str
    source_path: str
    chunk_count: int
    document_ids: Optional[List[str]] = Field(default=None, description="Vector store document IDs (needed for removal)")
    created_at: datetime
    
    class Config:
        from_attributes = True


class DocumentsListResponse(BaseModel):
    """Documents list response schema."""
    documents: List[DocumentResponse]
    total: int
    page: int
    limit: int


# Chat Schemas
class ChatRequest(BaseModel):
    """Chat request schema."""
    message: str = Field(..., description="User message")
    thread_id: Optional[str] = Field(default=None, description="Existing thread ID or None for new thread")


class ChatMessageResponse(BaseModel):
    """Chat message response schema."""
    type: str  # 'token', 'done', 'error'
    content: Optional[str] = None
    thread_id: Optional[str] = None


class ChatThreadResponse(BaseModel):
    """Chat thread response schema."""
    thread_id: str
    title: Optional[str]
    message_count: int
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ChatHistoryResponse(BaseModel):
    """Chat history response schema."""
    threads: List[ChatThreadResponse]
    total: int
    page: int
    limit: int


class ChatMessageHistory(BaseModel):
    """Chat message history schema."""
    role: str
    content: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class ThreadMessagesResponse(BaseModel):
    """Thread messages response schema."""
    thread_id: str
    messages: List[ChatMessageHistory]
    total: int

