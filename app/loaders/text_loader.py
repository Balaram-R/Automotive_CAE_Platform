"""
app/loaders/text_loader.py
==========================
Plain text files.
"""

from __future__ import annotations

from pathlib import Path

from app.loaders.base import BaseLoader
from app.loaders.loader_factory import LoaderFactory
from app.models.schemas import FileMetadata, RawDocument
from app.utils.logging import get_logger

logger = get_logger("loaders.text")


class TextLoader(BaseLoader):
    supported_extensions = [".txt"]

    def load(self, filepath: str, metadata: FileMetadata, **kw) -> list[RawDocument]:
        try:
            text = Path(filepath).read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                return []
            return [RawDocument(content=text, metadata=metadata, source_loader="TextLoader")]
        except Exception as exc:
            logger.error("Text load failed %s: %s", filepath, exc)
            return []


LoaderFactory.register_many(TextLoader.supported_extensions, TextLoader())
