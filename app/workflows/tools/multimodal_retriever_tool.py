"""Custom retriever tool that emits multi-modal markers in its output.

Replaces ``langchain_classic.tools.retriever.create_retriever_tool`` so the
returned tool message preserves type-aware markers (``image_asset_id=...``,
``table_id=...``). Those markers are parsed by ``generate_answer`` to fetch
the raw image bytes / table markdown from Postgres at answer-generation
time and assemble a multimodal ``HumanMessage``.

OpenAI tool messages must be plain strings, so we cannot put the original
images directly in the ``ToolMessage``; the marker pattern is the workaround.
"""
from __future__ import annotations

from typing import Annotated, List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool


def _format_doc(doc: Document, index: int) -> str:
    """Render a single retrieved Document with type-aware markers.

    Examples of output blocks:

        [doc 1] (page 4) <raw text content>

        [doc 2] (page 7, table_id=abc-123) Table summary: monthly revenue ...

        [doc 3] (page 12, image_asset_id=xyz-456) Image summary: bar chart ...
    """
    md = doc.metadata or {}
    content_type = md.get("content_type")
    page = md.get("page")
    extras: List[str] = []
    if page is not None:
        extras.append(f"page {page}")

    body_prefix = ""
    if content_type == "image_summary" and md.get("image_asset_id"):
        extras.append(f"image_asset_id={md['image_asset_id']}")
        body_prefix = "Image summary: "
    elif content_type == "table_summary" and md.get("table_id"):
        extras.append(f"table_id={md['table_id']}")
        body_prefix = "Table summary: "

    extras_str = f" ({', '.join(extras)})" if extras else ""
    return f"[doc {index}]{extras_str} {body_prefix}{doc.page_content}"


def render_documents(documents: List[Document]) -> str:
    """Render a list of Documents as a single tool-message string.

    Used by both the LangGraph tool and tests.
    """
    if not documents:
        return "No documents found."
    return "\n\n".join(_format_doc(d, i + 1) for i, d in enumerate(documents))


def create_multimodal_retriever_tool(
    retriever: BaseRetriever,
    name: str,
    description: str,
):
    """Build a LangChain tool that calls the retriever and emits markers.

    The returned tool has the given ``name`` / ``description`` (used by the
    LLM to decide when to call it) and accepts a single ``query`` string.
    """

    @tool(name, description=description)
    def _retrieve(
        query: Annotated[str, "Search query to find relevant documents"],
    ) -> str:
        docs = retriever.invoke(query)
        return render_documents(docs)

    return _retrieve
