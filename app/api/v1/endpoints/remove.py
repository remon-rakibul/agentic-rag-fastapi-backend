"""Document removal endpoint."""
from fastapi import APIRouter, Depends, HTTPException, status, Path
from sqlalchemy import text
from sqlalchemy.orm import Session
import json
from typing import List, Set, Tuple

from app.api.dependencies import get_current_user
from app.core.config import settings
from app.core.database import get_db, engine as sync_engine
from app.models.database import User, Document, ImageAsset, TableElement
from app.models.schemas import RemoveRequest, RemoveResponse
from app.services.vector_store_service import get_vector_store_service

router = APIRouter(prefix="/remove", tags=["documents"])


def _collect_linked_asset_ids(pgvector_ids: List[str]) -> Tuple[Set[str], Set[str]]:
    """Look up langchain_metadata for the given chunks; return linked asset ids.

    Returns ``(image_asset_ids, table_ids)`` referenced by the chunks in
    ``pgvector_ids``. We do this BEFORE deleting from PGVector so we can
    cascade the cleanup into ``image_assets`` / ``table_elements`` even when
    callers remove individual chunks rather than whole documents.
    """
    if not pgvector_ids:
        return set(), set()

    image_ids: Set[str] = set()
    table_ids: Set[str] = set()
    table_name = settings.VECTOR_STORE_TABLE_NAME

    try:
        with sync_engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"SELECT langchain_metadata FROM {table_name} "
                    f"WHERE langchain_id = ANY(:ids)"
                ),
                {"ids": list(pgvector_ids)},
            ).fetchall()
            for row in rows:
                md = row[0] or {}
                if isinstance(md, str):
                    try:
                        md = json.loads(md)
                    except Exception:
                        md = {}
                aid = md.get("image_asset_id")
                tid = md.get("table_id")
                if aid:
                    image_ids.add(aid)
                if tid:
                    table_ids.add(tid)
    except Exception:
        # If lookup fails, return empty sets — the chunks themselves still
        # get deleted from PGVector below; we just don't cascade-clean.
        return set(), set()

    return image_ids, table_ids


def _delete_linked_assets(
    db: Session,
    user_id: int,
    image_asset_ids: Set[str],
    table_ids: Set[str],
) -> None:
    """Delete image_assets and table_elements referenced by removed chunks.

    Always scoped to ``user_id`` for tenant isolation.
    """
    if image_asset_ids:
        db.query(ImageAsset).filter(
            ImageAsset.user_id == user_id,
            ImageAsset.id.in_(image_asset_ids),
        ).delete(synchronize_session=False)
    if table_ids:
        db.query(TableElement).filter(
            TableElement.user_id == user_id,
            TableElement.id.in_(table_ids),
        ).delete(synchronize_session=False)


@router.delete("", response_model=RemoveResponse)
def remove_documents(
    request: RemoveRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Remove documents from vector store + linked image_assets / table_elements."""
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

    # Cascade-clean: figure out which raw assets these chunks reference
    # BEFORE deleting them from PGVector (or the lookup is gone).
    image_asset_ids, table_ids_set = _collect_linked_asset_ids(valid_ids)

    # Remove from vector store (with error handling)
    try:
        vector_store_service.delete_documents(valid_ids)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove documents from vector store: {str(e)}"
        )

    # Remove linked raw assets (always scoped to current_user.id).
    try:
        _delete_linked_assets(db, current_user.id, image_asset_ids, table_ids_set)
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove linked image/table assets: {str(e)}"
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

    # Delete documents that have no remaining IDs (cascade also drops any
    # remaining image_assets / table_elements via FK ON DELETE CASCADE).
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

    Cascades into PGVector chunks AND linked ``image_assets`` / ``table_elements``
    via the FK ON DELETE CASCADE on ``Document``.
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

    # Delete from PGVector first if there are any chunk ids.
    if doc_ids:
        try:
            vector_store_service.delete_documents(doc_ids)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to remove documents from vector store: {str(e)}"
            )

    # Delete the database record. SQLAlchemy ``cascade="all, delete-orphan"``
    # plus the FK ``ON DELETE CASCADE`` we added in the migration removes any
    # linked image_assets and table_elements rows automatically.
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
