"""SQLAlchemy database models."""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean, LargeBinary
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class User(Base):
    """User model for authentication."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")
    chat_threads = relationship("ChatThread", back_populates="user", cascade="all, delete-orphan")
    image_assets = relationship("ImageAsset", back_populates="user", cascade="all, delete-orphan")
    table_elements = relationship("TableElement", back_populates="user", cascade="all, delete-orphan")


class Document(Base):
    """Document metadata model."""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    source_type = Column(String, nullable=False)  # 'url', 'pdf', 'docx', etc.
    source_path = Column(String, nullable=False)  # URL or file path
    chunk_count = Column(Integer, default=0)
    document_ids = Column(Text)  # JSON array of PGVector document IDs
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="documents")
    image_assets = relationship("ImageAsset", back_populates="document", cascade="all, delete-orphan")
    table_elements = relationship("TableElement", back_populates="document", cascade="all, delete-orphan")


class ChatThread(Base):
    """Chat thread model."""
    __tablename__ = "chat_threads"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    thread_id = Column(String, unique=True, nullable=False, index=True)  # LangGraph thread_id
    title = Column(String, nullable=True)  # First message or user-defined title
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="chat_threads")
    messages = relationship("ChatMessage", back_populates="thread", cascade="all, delete-orphan")


class ChatMessage(Base):
    """Chat message model."""
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String, ForeignKey("chat_threads.thread_id"), nullable=False, index=True)
    role = Column(String, nullable=False)  # 'user', 'assistant', 'tool'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    thread = relationship("ChatThread", back_populates="messages")


class TokenBlacklist(Base):
    """Token blacklist model for logout functionality."""
    __tablename__ = "token_blacklist"
    
    id = Column(Integer, primary_key=True, index=True)
    token = Column(String, unique=True, nullable=False, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ImageAsset(Base):
    """Raw image bytes extracted from PDFs or uploaded directly.

    Linked from PGVector chunk metadata via ``image_asset_id``. The chunk in
    PGVector holds the gpt-4o-mini caption (which is what gets embedded and
    searched); this row holds the original bytes that are passed back to the
    multimodal LLM at answer-generation time.
    """
    __tablename__ = "image_assets"

    id = Column(String, primary_key=True)  # uuid; mirrored in PGVector metadata
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=True, index=True)
    mime_type = Column(String, nullable=False)
    image_bytes = Column(LargeBinary, nullable=False)
    summary = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="image_assets")
    document = relationship("Document", back_populates="image_assets")


class TableElement(Base):
    """Raw markdown of a table extracted from a PDF.

    Linked from PGVector chunk metadata via ``table_id``. The chunk in PGVector
    holds the table summary (embedded for retrieval); this row holds the raw
    markdown table that is appended to the prompt at answer-generation time.
    """
    __tablename__ = "table_elements"

    id = Column(String, primary_key=True)  # uuid; mirrored in PGVector metadata
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=True, index=True)
    raw_markdown = Column(Text, nullable=False)
    summary = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="table_elements")
    document = relationship("Document", back_populates="table_elements")

