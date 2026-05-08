"""Document loader factory for various file types.

Supports text-based loaders (PDF text via PyMuPDF4LLM, DOCX, TXT, URLs) plus
multimodal helpers used by the Option 3 multi-modal RAG pipeline:

- ``extract_images_from_pdf``: pulls embedded images out of a PDF as raw bytes
- ``extract_tables_from_pdf``: detects tables on each page and returns
  markdown for each one (PyMuPDF >= 1.23)
- ``load_image_file``: reads a directly-uploaded image file as raw bytes

All multimodal helpers are pure (no I/O outside the file system / PyMuPDF)
so they are easy to unit test.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import aiofiles
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_pymupdf4llm import PyMuPDF4LLMLoader


# Map common image extensions to their MIME types. Used by ``load_image_file``
# and by ``get_file_type`` to detect direct image uploads.
_IMAGE_EXT_TO_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


async def load_documents_from_urls(urls: List[str]) -> List[Document]:
    """Load documents from URLs.
    
    Args:
        urls: List of URLs to load
        
    Returns:
        List of loaded documents
        
    Raises:
        ValueError: If URLs list is invalid
    """
    if not urls:
        raise ValueError("URLs list cannot be empty")
    
    # Limit number of URLs to prevent abuse
    MAX_URLS = 50
    if len(urls) > MAX_URLS:
        raise ValueError(f"Too many URLs. Maximum {MAX_URLS} URLs allowed per request")
    
    all_docs = []
    for url in urls:
        # Basic URL validation
        url = url.strip()
        if not url:
            continue
        
        # Check for valid URL scheme
        if not (url.startswith('http://') or url.startswith('https://')):
            print(f"Warning: Skipping invalid URL (must start with http:// or https://): {url}")
            continue
        
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            # Log error but continue with other URLs
            # In production, you might want to log this properly
            print(f"Warning: Failed to load URL {url}: {str(e)}")
            continue
    return all_docs


async def load_documents_from_file(file_path: str, file_type: str) -> List[Document]:
    """Load documents from a file based on file type."""
    try:
        if file_type == "pdf":
            loader = PyMuPDF4LLMLoader(file_path)
            return loader.load()
        elif file_type == "docx":
            # Using langchain_community's UnstructuredWordDocumentLoader
            from langchain_community.document_loaders import UnstructuredWordDocumentLoader
            loader = UnstructuredWordDocumentLoader(file_path)
            return loader.load()
        elif file_type == "txt":
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(file_path)
            return loader.load()
        else:
            # Try unstructured loader as fallback
            from langchain_community.document_loaders import UnstructuredFileLoader
            loader = UnstructuredFileLoader(file_path)
            return loader.load()
    except Exception as e:
        # Re-raise with more context
        raise ValueError(f"Failed to load file {file_path} (type: {file_type}): {str(e)}") from e


def extract_images_from_pdf(file_path: str) -> List[Tuple[bytes, str, int]]:
    """Extract embedded images from a PDF.

    Uses PyMuPDF (``fitz``). For each page, ``page.get_images()`` lists every
    image XRef referenced on the page; ``doc.extract_image(xref)`` returns the
    raw bytes plus the original extension. We deduplicate by XRef so that a
    logo that appears on every page is only captioned and stored once.

    Returns a list of ``(image_bytes, mime_type, page_number)`` tuples. Pages
    are 1-indexed for human-friendly references in the UI/logs.
    """
    import fitz  # type: ignore[import-untyped]  # PyMuPDF

    results: List[Tuple[bytes, str, int]] = []
    seen_xrefs: set[int] = set()

    try:
        with fitz.open(file_path) as doc:
            for page_index, page in enumerate(doc):
                # Each entry: (xref, smask, w, h, bpc, colorspace, alt, name, filter, ...)
                for img_info in page.get_images(full=True):
                    xref = img_info[0]
                    if xref in seen_xrefs:
                        continue
                    seen_xrefs.add(xref)
                    try:
                        img = doc.extract_image(xref)
                    except Exception:
                        # Some entries are masks or unreadable; skip silently.
                        continue
                    image_bytes = img.get("image")
                    ext = (img.get("ext") or "png").lower()
                    if not image_bytes:
                        continue
                    mime = _IMAGE_EXT_TO_MIME.get(f".{ext}", f"image/{ext}")
                    results.append((image_bytes, mime, page_index + 1))
    except Exception as e:
        raise ValueError(f"Failed to extract images from {file_path}: {e}") from e

    return results


def extract_tables_from_pdf(file_path: str) -> List[Tuple[str, int]]:
    """Extract tables from a PDF as markdown.

    Uses PyMuPDF's ``page.find_tables()`` (>= 1.23). Each detected table is
    converted to markdown via ``Table.to_markdown()``. Pages without tables
    are skipped.

    Returns a list of ``(markdown, page_number)`` tuples (1-indexed).
    """
    import fitz  # type: ignore[import-untyped]

    results: List[Tuple[str, int]] = []

    try:
        with fitz.open(file_path) as doc:
            for page_index, page in enumerate(doc):
                if not hasattr(page, "find_tables"):
                    # PyMuPDF < 1.23 — skip silently rather than crash.
                    continue
                try:
                    tables = page.find_tables()
                except Exception:
                    continue
                # ``find_tables`` returns a TableFinder whose ``tables`` attr
                # is a list of Table objects. Older versions return a list
                # directly; handle both.
                table_iter = getattr(tables, "tables", tables)
                for tbl in table_iter:
                    try:
                        md = tbl.to_markdown()
                    except Exception:
                        continue
                    if not md or not md.strip():
                        continue
                    results.append((md.strip(), page_index + 1))
    except Exception as e:
        raise ValueError(f"Failed to extract tables from {file_path}: {e}") from e

    return results


def load_image_file(file_path: str) -> Tuple[bytes, str]:
    """Read a directly-uploaded image file as raw bytes.

    Returns ``(image_bytes, mime_type)``. The MIME type is inferred from the
    file extension; only PNG, JPEG, and WEBP are supported (matching what
    ``get_file_type`` returns ``'image'`` for).
    """
    ext = Path(file_path).suffix.lower()
    mime = _IMAGE_EXT_TO_MIME.get(ext)
    if mime is None:
        raise ValueError(
            f"Unsupported image extension: {ext}. "
            f"Supported: {sorted(_IMAGE_EXT_TO_MIME)}"
        )
    with open(file_path, "rb") as f:
        return f.read(), mime


async def save_uploaded_file(file_content: bytes, filename: str, upload_dir: str) -> str:
    """Save uploaded file to disk and return path.
    
    Args:
        file_content: File content as bytes
        filename: Original filename (will be sanitized)
        upload_dir: Directory to save file in
        
    Returns:
        Path to saved file
        
    Raises:
        IOError: If file save fails
        ValueError: If filename is invalid
    """
    # Sanitize filename to prevent path traversal attacks
    # Remove directory separators and dangerous characters
    sanitized = os.path.basename(filename)  # Remove any path components
    sanitized = sanitized.replace('/', '').replace('\\', '')
    sanitized = sanitized.replace('..', '')  # Remove parent directory references
    
    # Limit filename length
    if len(sanitized) > 255:
        # Keep extension, truncate name
        ext = Path(sanitized).suffix
        name = Path(sanitized).stem[:255 - len(ext)]
        sanitized = name + ext
    
    if not sanitized or sanitized.strip() == '':
        raise ValueError("Invalid filename: filename cannot be empty after sanitization")
    
    try:
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, sanitized)
        
        # Ensure we're still within upload_dir (prevent path traversal)
        real_upload_dir = os.path.realpath(upload_dir)
        real_file_path = os.path.realpath(file_path)
        if not real_file_path.startswith(real_upload_dir):
            raise ValueError(f"Invalid file path: {filename}")
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file_content)
        
        return file_path
    except Exception as e:
        raise IOError(f"Failed to save file {filename}: {str(e)}") from e


def get_file_type(filename: str) -> str:
    """Determine file type from filename.

    Returns one of: ``'pdf'``, ``'docx'``, ``'txt'``, ``'image'``, or
    ``'unknown'``.
    """
    ext = Path(filename).suffix.lower()
    type_map = {
        '.pdf': 'pdf',
        '.docx': 'docx',
        '.doc': 'docx',
        '.txt': 'txt',
        '.md': 'txt',
        '.png': 'image',
        '.jpg': 'image',
        '.jpeg': 'image',
        '.webp': 'image',
    }
    return type_map.get(ext, 'unknown')


async def cleanup_file(file_path: str):
    """Remove temporary file."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        pass  # Ignore cleanup errors
