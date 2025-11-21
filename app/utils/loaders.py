"""Document loader factory for various file types."""
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
import aiofiles
import os
from pathlib import Path


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
    """Determine file type from filename."""
    ext = Path(filename).suffix.lower()
    type_map = {
        '.pdf': 'pdf',
        '.docx': 'docx',
        '.doc': 'docx',
        '.txt': 'txt',
        '.md': 'txt',
    }
    return type_map.get(ext, 'unknown')


async def cleanup_file(file_path: str):
    """Remove temporary file."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        pass  # Ignore cleanup errors

