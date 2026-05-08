"""Document ingestion service.

Implements Option 3 of the LangChain semi-structured / multi-modal RAG
cookbook: text chunks are embedded as-is (their own raw form); tables and
images are summarized by the multimodal LLM, summaries are embedded, and the
raw markdown / bytes are stored in dedicated docstores (``table_elements``,
``image_assets``) keyed by an id that is mirrored into the chunk's PGVector
metadata.

At answer time, ``generate_answer`` regex-extracts those ids from the tool
message, fetches the raw elements, and assembles a multimodal HumanMessage.
"""
from __future__ import annotations

import logging
import uuid
from typing import List, Optional

from langchain_core.documents import Document
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.database import ImageAsset, TableElement
from app.services.multimodal_summary_service import get_multimodal_summary_service
from app.services.vector_store_service import get_vector_store_service
from app.utils.loaders import (
    cleanup_file,
    extract_images_from_pdf,
    extract_tables_from_pdf,
    get_file_type,
    load_documents_from_file,
    load_documents_from_urls,
    load_image_file,
    save_uploaded_file,
)
from app.utils.text_splitter import split_documents

logger = logging.getLogger(__name__)


class IngestionService:
    """Service for ingesting documents into the vector store."""

    def __init__(self):
        self.vector_store_service = get_vector_store_service()
        self.summary_service = get_multimodal_summary_service()

    # ------------------------------------------------------------------ #
    # Multimodal helpers (Option 3)
    # ------------------------------------------------------------------ #

    def _ingest_pdf_tables(
        self,
        file_path: str,
        filename: str,
        user_id: Optional[int],
        document_id: Optional[int],
        metadata: Optional[dict],
        db: Optional[Session],
    ) -> List[Document]:
        """Extract tables from a PDF, summarize each, persist raw markdown.

        Returns a list of Documents whose ``page_content`` is the table
        summary and whose metadata carries ``table_id``, ``content_type``,
        and the standard source fields. These get appended to the chunk
        list passed to PGVector — they are NOT split further (tables are
        atomic units).
        """
        max_tables = getattr(settings, "MAX_TABLES_PER_DOC", 30)
        tables = extract_tables_from_pdf(file_path)
        if max_tables and len(tables) > max_tables:
            logger.warning(
                "PDF %s has %d tables; capping at %d (MAX_TABLES_PER_DOC)",
                filename,
                len(tables),
                max_tables,
            )
            tables = tables[:max_tables]

        summary_docs: List[Document] = []
        for table_md, page in tables:
            try:
                summary = self.summary_service.summarize_table(table_md)
            except Exception as exc:
                logger.warning(
                    "Failed to summarize table on page %d of %s: %s",
                    page,
                    filename,
                    exc,
                )
                continue

            table_id = str(uuid.uuid4())

            if db is not None and user_id is not None:
                db.add(
                    TableElement(
                        id=table_id,
                        user_id=user_id,
                        document_id=document_id,
                        raw_markdown=table_md,
                        summary=summary,
                        page_number=page,
                    )
                )

            doc_metadata = dict(metadata or {})
            doc_metadata.update(
                {
                    "table_id": table_id,
                    "content_type": "table_summary",
                    "page": page,
                    "source_type": "pdf",
                    "source_path": filename,
                }
            )
            summary_docs.append(
                Document(page_content=summary, metadata=doc_metadata)
            )

        return summary_docs

    def _ingest_pdf_images(
        self,
        file_path: str,
        filename: str,
        user_id: Optional[int],
        document_id: Optional[int],
        metadata: Optional[dict],
        db: Optional[Session],
    ) -> List[Document]:
        """Extract images from a PDF, caption each, persist raw bytes."""
        max_images = getattr(settings, "MAX_IMAGES_PER_DOC", 50)
        images = extract_images_from_pdf(file_path)
        if max_images and len(images) > max_images:
            logger.warning(
                "PDF %s has %d images; capping at %d (MAX_IMAGES_PER_DOC)",
                filename,
                len(images),
                max_images,
            )
            images = images[:max_images]

        summary_docs: List[Document] = []
        for img_bytes, mime, page in images:
            try:
                summary = self.summary_service.summarize_image(img_bytes, mime)
            except Exception as exc:
                logger.warning(
                    "Failed to summarize image on page %d of %s: %s",
                    page,
                    filename,
                    exc,
                )
                continue

            image_asset_id = str(uuid.uuid4())

            if db is not None and user_id is not None:
                db.add(
                    ImageAsset(
                        id=image_asset_id,
                        user_id=user_id,
                        document_id=document_id,
                        mime_type=mime,
                        image_bytes=img_bytes,
                        summary=summary,
                        page_number=page,
                    )
                )

            doc_metadata = dict(metadata or {})
            doc_metadata.update(
                {
                    "image_asset_id": image_asset_id,
                    "content_type": "image_summary",
                    "page": page,
                    "source_type": "pdf",
                    "source_path": filename,
                }
            )
            summary_docs.append(
                Document(page_content=summary, metadata=doc_metadata)
            )

        return summary_docs

    def _ingest_standalone_image(
        self,
        file_path: str,
        filename: str,
        user_id: Optional[int],
        document_id: Optional[int],
        metadata: Optional[dict],
        db: Optional[Session],
    ) -> List[Document]:
        """Caption a directly-uploaded image and persist raw bytes."""
        img_bytes, mime = load_image_file(file_path)
        try:
            summary = self.summary_service.summarize_image(img_bytes, mime)
        except Exception as exc:
            raise ValueError(
                f"Failed to summarize uploaded image {filename}: {exc}"
            ) from exc

        image_asset_id = str(uuid.uuid4())

        if db is not None and user_id is not None:
            db.add(
                ImageAsset(
                    id=image_asset_id,
                    user_id=user_id,
                    document_id=document_id,
                    mime_type=mime,
                    image_bytes=img_bytes,
                    summary=summary,
                    page_number=None,
                )
            )

        doc_metadata = dict(metadata or {})
        doc_metadata.update(
            {
                "image_asset_id": image_asset_id,
                "content_type": "image_summary",
                "source_type": "image",
                "source_path": filename,
            }
        )
        return [Document(page_content=summary, metadata=doc_metadata)]

    # ------------------------------------------------------------------ #
    # Public ingestion API
    # ------------------------------------------------------------------ #

    async def ingest_urls(
        self,
        urls: List[str],
        user_id: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> tuple[List[str], int]:
        """Ingest documents from URLs (text only).

        Returns:
            Tuple of (document_ids, chunk_count)
        """
        documents = await load_documents_from_urls(urls)

        if not documents:
            raise ValueError("No documents could be loaded from the provided URLs")

        for doc in documents:
            if not hasattr(doc, "metadata") or doc.metadata is None:
                doc.metadata = {}
            if metadata:
                doc.metadata.update(metadata)
            doc.metadata["source_type"] = "url"

        split_docs = split_documents(documents)
        document_ids = self.vector_store_service.add_documents(split_docs, user_id=user_id)
        return document_ids, len(split_docs)

    async def ingest_files(
        self,
        files: List[tuple[bytes, str]],  # List of (file_content, filename) tuples
        user_id: Optional[int] = None,
        metadata: Optional[dict] = None,
        db: Optional[Session] = None,
        document_id: Optional[int] = None,
    ) -> tuple[List[str], int]:
        """Ingest documents from uploaded files.

        For PDFs: extracts text chunks, tables, and embedded images. Tables
        and images are summarized and the originals are persisted into
        ``table_elements`` / ``image_assets``.

        For directly-uploaded images (PNG/JPG/WEBP): the image is captioned
        and stored as a single image asset.

        For DOCX/TXT: text-only ingestion (existing behavior).

        Returns:
            Tuple of (document_ids, chunk_count)
        """
        all_document_ids: List[str] = []
        total_chunks = 0
        saved_files: List[str] = []

        try:
            for file_content, filename in files:
                file_type = get_file_type(filename)

                file_path = await save_uploaded_file(
                    file_content,
                    filename,
                    settings.UPLOAD_DIR,
                )
                saved_files.append(file_path)

                if file_type == "image":
                    # Direct image upload — single asset, single summary chunk.
                    summary_docs = self._ingest_standalone_image(
                        file_path=file_path,
                        filename=filename,
                        user_id=user_id,
                        document_id=document_id,
                        metadata=metadata,
                        db=db,
                    )
                    chunks_to_index = summary_docs
                else:
                    # Text-bearing file: load text, then optionally extract
                    # tables and images (PDF only).
                    documents = await load_documents_from_file(file_path, file_type)
                    if not documents:
                        raise ValueError(
                            f"No documents could be loaded from file {filename}"
                        )

                    for doc in documents:
                        if not hasattr(doc, "metadata") or doc.metadata is None:
                            doc.metadata = {}
                        if metadata:
                            doc.metadata.update(metadata)
                        doc.metadata["source_type"] = file_type
                        doc.metadata["source_path"] = filename

                    text_chunks = split_documents(documents)

                    # PDFs additionally produce table and image summary chunks.
                    multimodal_chunks: List[Document] = []
                    if file_type == "pdf":
                        try:
                            multimodal_chunks.extend(
                                self._ingest_pdf_tables(
                                    file_path=file_path,
                                    filename=filename,
                                    user_id=user_id,
                                    document_id=document_id,
                                    metadata=metadata,
                                    db=db,
                                )
                            )
                        except Exception as exc:
                            logger.warning(
                                "Table extraction failed for %s (continuing without tables): %s",
                                filename,
                                exc,
                            )
                        try:
                            multimodal_chunks.extend(
                                self._ingest_pdf_images(
                                    file_path=file_path,
                                    filename=filename,
                                    user_id=user_id,
                                    document_id=document_id,
                                    metadata=metadata,
                                    db=db,
                                )
                            )
                        except Exception as exc:
                            logger.warning(
                                "Image extraction failed for %s (continuing without images): %s",
                                filename,
                                exc,
                            )

                    chunks_to_index = text_chunks + multimodal_chunks

                doc_ids = self.vector_store_service.add_documents(
                    chunks_to_index, user_id=user_id
                )
                all_document_ids.extend(doc_ids)
                total_chunks += len(chunks_to_index)
        finally:
            for file_path in saved_files:
                await cleanup_file(file_path)

        return all_document_ids, total_chunks


def get_ingestion_service() -> IngestionService:
    """Get ingestion service instance."""
    return IngestionService()
