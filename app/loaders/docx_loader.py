"""
app/loaders/docx_loader.py
==========================
DOCX / DOC → docx2txt.
"""

from __future__ import annotations

from app.loaders.base import BaseLoader
from app.loaders.loader_factory import LoaderFactory
from app.models.schemas import FileMetadata, RawDocument
from app.utils.logging import get_logger

logger = get_logger("loaders.docx")


class DocxLoader(BaseLoader):
    supported_extensions = [".docx", ".doc"]

    def load(self, filepath: str, metadata: FileMetadata, **kw) -> list[RawDocument]:
        try:
            import docx2txt
        except ImportError:
            logger.error("pip install docx2txt")
            return []
        try:
            text = docx2txt.process(filepath)
            return [RawDocument(content=text, metadata=metadata, source_loader="DocxLoader")] if text.strip() else []
        except Exception as exc:
            logger.error("DOCX load failed %s: %s", filepath, exc)
            return []


LoaderFactory.register_many(DocxLoader.supported_extensions, DocxLoader())
