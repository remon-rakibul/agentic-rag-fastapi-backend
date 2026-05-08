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
from app.utils.loaders import get_file_type

router = APIRouter(prefix="/digest", tags=["documents"])


# Source type stored on the Document row when a file is uploaded directly. PDF
# / DOCX / TXT keep their existing types; image uploads get ``image``.
_FILE_SOURCE_TYPES = {"pdf", "docx", "txt", "image"}


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

    Supported file types:
    - **PDF**: text + tables (extracted as markdown) + embedded images
      (captioned by the multimodal LLM)
    - **DOCX, TXT, MD**: text only
    - **PNG, JPG, JPEG, WEBP**: direct image uploads, captioned by the
      multimodal LLM and stored as a single image asset

    **Note**: For URL-only ingestion, prefer the `/digest/urls` endpoint.
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
        
        # Store document metadata (one record per URL group)
        url_doc = Document(
            user_id=current_user.id,
            source_type="url",
            source_path=", ".join(url_list),  # Store all URLs
            chunk_count=chunk_count,
            document_ids=json.dumps(doc_ids)
        )
        db.add(url_doc)
    
    # Process files (one Document row per uploaded file so image_assets /
    # table_elements can be linked to a single owning Document for cascade
    # deletes).
    if files:
        if len(files) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Files list cannot be empty"
            )

        MAX_FILES = 20  # Limit number of files
        if len(files) > MAX_FILES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Too many files. Maximum {MAX_FILES} files allowed per request"
            )

        total_size = 0
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

            if total_size > settings.MAX_UPLOAD_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Total file size exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE} bytes"
                )

            if file_size > settings.MAX_UPLOAD_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File {file.filename} exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE} bytes"
                )

            file_type = get_file_type(file.filename)
            if file_type not in _FILE_SOURCE_TYPES:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Unsupported file type for {file.filename}. "
                        f"Supported: PDF, DOCX, TXT, MD, PNG, JPG, JPEG, WEBP."
                    ),
                )

            # Create the Document row first so we have an id to link
            # image_assets / table_elements to (cascade delete on document
            # removal).
            doc_row = Document(
                user_id=current_user.id,
                source_type=file_type,
                source_path=file.filename,
                chunk_count=0,
                document_ids=json.dumps([]),
            )
            db.add(doc_row)
            db.flush()  # populate doc_row.id without committing

            try:
                doc_ids, chunk_count = await ingestion_service.ingest_files(
                    files=[(content, file.filename)],
                    user_id=current_user.id,
                    metadata=metadata_dict,
                    db=db,
                    document_id=doc_row.id,
                )
            except Exception as e:
                db.rollback()
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to ingest {file.filename}: {str(e)}",
                )

            doc_row.chunk_count = chunk_count
            doc_row.document_ids = json.dumps(doc_ids)

            all_document_ids.extend(doc_ids)
            total_chunks += chunk_count

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
