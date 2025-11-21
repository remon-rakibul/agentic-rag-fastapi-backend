"""Document ingestion endpoint."""
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from typing import List, Optional, Union
from sqlalchemy.orm import Session
import json
from app.api.dependencies import get_current_user
from app.core.database import get_db
from app.core.config import settings
from app.models.database import User, Document
from app.models.schemas import DigestRequest, DigestResponse
from app.services.ingestion_service import get_ingestion_service

router = APIRouter(prefix="/digest", tags=["documents"])


@router.post("/urls", response_model=DigestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_urls_only(
    urls: List[str],
    metadata: Optional[dict] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Ingest documents from URLs only (simpler alternative to the main digest endpoint).
    
    **Example request body:**
    ```json
    {
      "urls": ["https://example.com", "https://recombd.com/"],
      "metadata": {"source": "web", "category": "documentation"}
    }
    ```
    
    **Parameters:**
    - **urls**: List of URLs to scrape and ingest
    - **metadata**: Optional metadata dictionary to attach to documents
    """
    ingestion_service = get_ingestion_service()
    
    # Validate URLs
    if not urls or len(urls) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one URL must be provided"
        )
    
    MAX_URLS = 50
    if len(urls) > MAX_URLS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many URLs. Maximum {MAX_URLS} URLs allowed per request"
        )
    
    # Validate metadata size
    metadata_dict = metadata or {}
    if len(str(metadata_dict)) > 10000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Metadata too large (max 10KB)"
        )
    
    try:
        doc_ids, chunk_count = await ingestion_service.ingest_urls(
            urls=urls,
            user_id=current_user.id,
            metadata=metadata_dict
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest URLs: {str(e)}"
        )
    
    # Store document metadata
    doc = Document(
        user_id=current_user.id,
        source_type="url",
        source_path=", ".join(urls),
        chunk_count=chunk_count,
        document_ids=json.dumps(doc_ids)
    )
    db.add(doc)
    
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save document metadata: {str(e)}"
        )
    
    return DigestResponse(
        document_ids=doc_ids,
        chunk_count=chunk_count,
        status="success"
    )


@router.post("", response_model=DigestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_documents(
    files: Union[List[UploadFile], None] = File(default=None),
    urls: Optional[str] = Form(default=None, description='JSON array of URLs, e.g., ["https://example.com"]. Leave empty if not using.'),
    metadata: Optional[str] = Form(default=None, description='JSON object with metadata, e.g., {"key": "value"}. Leave empty if not using.'),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Ingest documents from files and/or URLs.
    
    **⚠️ NOTE**: For URL-only ingestion, it's easier to use the `/digest/urls` endpoint instead!
    
    **Parameters:**
    - **files**: Upload one or more files (PDF, DOCX, TXT)
    - **urls**: JSON array of URLs (e.g., `["https://example.com"]`) - Leave empty or omit if not using
    - **metadata**: JSON object with additional metadata (e.g., `{"category": "docs"}`) - Leave empty or omit if not using
    
    **Example using only files:**
    - files: Upload your PDF/DOCX/TXT files
    - urls: (leave empty)
    - metadata: (leave empty)
    
    **Example using only URLs (IMPORTANT - Swagger UI quirk):**
    - files: **Check the "Send empty value" checkbox** or the request will fail
    - urls: `["https://example.com", "https://example.org"]`
    - metadata: `{"source": "web"}` or leave empty
    
    **Tip:** For cleaner URL ingestion, use POST `/api/v1/digest/urls` with JSON body instead.
    """
    ingestion_service = get_ingestion_service()
    all_document_ids = []
    total_chunks = 0
    
    # Normalize empty/invalid values to None
    # Swagger UI sends "string" as default, treat it as None
    if urls and urls.strip() in ["", "string"]:
        urls = None
    if metadata and metadata.strip() in ["", "string"]:
        metadata = None
    
    # Validate that at least one input is provided
    if not files and not urls:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either files or urls must be provided. Upload files or provide a JSON array of URLs."
        )
    
    # Parse metadata
    metadata_dict = {}
    if metadata:
        try:
            metadata_dict = json.loads(metadata)
            # Validate metadata is a dict
            if not isinstance(metadata_dict, dict):
                raise ValueError("Metadata must be a JSON object (e.g., {\"key\": \"value\"})")
            # Limit metadata size to prevent abuse
            if len(str(metadata_dict)) > 10000:  # ~10KB limit
                raise ValueError("Metadata too large (max 10KB)")
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid metadata: {str(e)}. Expected JSON object like {{\"key\": \"value\"}} or leave empty."
            )
    
    # Process URLs
    if urls:
        try:
            url_list = json.loads(urls)
            if not isinstance(url_list, list):
                raise ValueError("URLs must be a JSON array (e.g., [\"https://example.com\"])")
            if len(url_list) == 0:
                raise ValueError("URLs list cannot be empty")
            # Limit number of URLs
            MAX_URLS = 50
            if len(url_list) > MAX_URLS:
                raise ValueError(f"Too many URLs. Maximum {MAX_URLS} URLs allowed per request")
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid URLs format: {str(e)}. Expected JSON array like [\"https://example.com\"]"
            )
        
        doc_ids, chunk_count = await ingestion_service.ingest_urls(
            urls=url_list,
            user_id=current_user.id,
            metadata=metadata_dict
        )
        all_document_ids.extend(doc_ids)
        total_chunks += chunk_count
        
        # Store document metadata (one record per URL)
        # Note: All URLs share the same document_ids since they're processed together
        # For more granular tracking, process URLs individually
        doc = Document(
            user_id=current_user.id,
            source_type="url",
            source_path=", ".join(url_list),  # Store all URLs
            chunk_count=chunk_count,
            document_ids=json.dumps(doc_ids)
        )
        db.add(doc)
    
    # Process files
    if files:
        if len(files) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Files list cannot be empty"
            )
        
        file_data = []
        total_size = 0
        MAX_FILES = 20  # Limit number of files
        
        if len(files) > MAX_FILES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Too many files. Maximum {MAX_FILES} files allowed per request"
            )
        
        for file in files:
            # Validate filename
            if not file.filename or not file.filename.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Filename cannot be empty"
                )
            
            content = await file.read()
            file_size = len(content)
            total_size += file_size
            
            # Check file size limit
            if total_size > settings.MAX_UPLOAD_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Total file size exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE} bytes"
                )
            
            # Check individual file size
            if file_size > settings.MAX_UPLOAD_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File {file.filename} exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE} bytes"
                )
            
            file_data.append((content, file.filename))
        
        doc_ids, chunk_count = await ingestion_service.ingest_files(
            files=file_data,
            user_id=current_user.id,
            metadata=metadata_dict
        )
        all_document_ids.extend(doc_ids)
        total_chunks += chunk_count
        
        # Store document metadata (one record per file)
        for file_content, filename in file_data:
            # Estimate chunks per file (rough approximation)
            # In production, you might want to track this more accurately
            estimated_chunks = chunk_count // len(file_data) if file_data else 0
            doc = Document(
                user_id=current_user.id,
                source_type="file",
                source_path=filename,
                chunk_count=estimated_chunks,
                document_ids=json.dumps(doc_ids)  # All files share IDs (processed together)
            )
            db.add(doc)
    
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save document metadata: {str(e)}"
        )
    
    return DigestResponse(
        document_ids=all_document_ids,
        chunk_count=total_chunks,
        status="success"
    )

