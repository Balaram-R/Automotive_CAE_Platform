"""
app/loaders/markdown_loader.py
==============================
Markdown → UnstructuredMarkdownLoader with raw fallback.
"""

from __future__ import annotations

from pathlib import Path

from app.loaders.base import BaseLoader
from app.loaders.loader_factory import LoaderFactory
from app.models.schemas import FileMetadata, RawDocument
from app.utils.logging import get_logger

logger = get_logger("loaders.markdown")


class MarkdownLoader(BaseLoader):
    supported_extensions = [".md", ".markdown"]

    def load(self, filepath: str, metadata: FileMetadata, **kw) -> list[RawDocument]:
        try:
            from langchain_community.document_loaders import UnstructuredMarkdownLoader
            docs = UnstructuredMarkdownLoader(filepath, mode="elements").load()
            return [RawDocument(content=d.page_content, metadata=metadata, source_loader="UnstructuredMarkdownLoader") for d in docs]
        except (ImportError, Exception) as exc:
            logger.debug("UnstructuredMarkdownLoader failed (%s), using raw", exc)
        try:
            text = Path(filepath).read_text(encoding="utf-8", errors="replace")
            return [RawDocument(content=text, metadata=metadata, source_loader="MarkdownRawLoader")] if text.strip() else []
        except Exception as exc:
            logger.error("Markdown load failed %s: %s", filepath, exc)
            return []


LoaderFactory.register_many(MarkdownLoader.supported_extensions, MarkdownLoader())
