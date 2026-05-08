"""Multimodal summary service.

Wraps the multimodal LLM (default gpt-4o-mini) to produce text summaries of
images and tables. The summaries are what get embedded in PGVector for
retrieval. The original images and tables are stored separately in
``image_assets`` and ``table_elements`` and are fetched back at answer time.

This implements the summarization step of Option 3 from the LangChain
semi-structured / multi-modal RAG cookbook:
https://blog.langchain.com/semi-structured-multi-modal-rag
"""
from __future__ import annotations

import base64
import threading
from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from app.core.config import settings
from app.workflows.prompt_loader import get_prompt


class MultimodalSummaryService:
    """Generate retrieval-friendly summaries of images and tables.

    Uses one LLM instance (vision-capable) for both image and table
    summarization to keep memory and connection footprint small. The model
    is loaded lazily on first call.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        self._model_name = (
            model_name
            or getattr(settings, "MULTIMODAL_MODEL", None)
            or "gpt-4o-mini"
        )
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = init_chat_model(
                        self._model_name,
                        temperature=0,
                        streaming=False,
                    )
        return self._model

    def summarize_image(self, image_bytes: bytes, mime: str) -> str:
        """Caption an image for retrieval.

        Sends the image as a base64 data URL to the multimodal LLM. The
        returned text gets embedded in PGVector; the raw bytes stay in the
        ``image_assets`` table and are passed back to the LLM at answer time.
        """
        if not image_bytes:
            raise ValueError("image_bytes cannot be empty")
        if not mime:
            raise ValueError("mime cannot be empty")

        b64 = base64.b64encode(image_bytes).decode("utf-8")
        prompt = get_prompt("image_summary")

        msg = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                },
            ]
        )
        response = self._get_model().invoke([msg])
        return (response.content or "").strip()

    def summarize_table(self, table_markdown: str) -> str:
        """Summarize a table for retrieval.

        Tables are passed in as markdown (PyMuPDF's ``Table.to_markdown()``
        output). The summary is embedded; the raw markdown is stored in
        ``table_elements`` and appended to the prompt at answer time.
        """
        if not table_markdown or not table_markdown.strip():
            raise ValueError("table_markdown cannot be empty")

        prompt = get_prompt("table_summary", table=table_markdown.strip())
        response = self._get_model().invoke([HumanMessage(content=prompt)])
        return (response.content or "").strip()


_service: Optional[MultimodalSummaryService] = None
_service_lock = threading.Lock()


def get_multimodal_summary_service() -> MultimodalSummaryService:
    """Return the global MultimodalSummaryService instance (thread-safe)."""
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = MultimodalSummaryService()
    return _service
