"""
app/loaders/html_loader.py
==========================
HTML → BeautifulSoup text extraction.
"""

from __future__ import annotations

from pathlib import Path

from app.loaders.base import BaseLoader
from app.loaders.loader_factory import LoaderFactory
from app.models.schemas import FileMetadata, RawDocument
from app.utils.logging import get_logger

logger = get_logger("loaders.html")


class HTMLLoader(BaseLoader):
    supported_extensions = [".html", ".htm"]

    def load(self, filepath: str, metadata: FileMetadata, **kw) -> list[RawDocument]:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.error("pip install beautifulsoup4")
            return []
        try:
            raw = Path(filepath).read_text(encoding="utf-8", errors="replace")
            soup = BeautifulSoup(raw, "html.parser")
            for t in soup(["script", "style"]):
                t.decompose()
            text = soup.get_text(separator="\n", strip=True)
            title = soup.find("title")
            if title:
                text = f"Title: {title.get_text(strip=True)}\n\n{text}"
            return [RawDocument(content=text, metadata=metadata, source_loader="HTMLLoader")] if text.strip() else []
        except Exception as exc:
            logger.error("HTML load failed %s: %s", filepath, exc)
            return []


LoaderFactory.register_many(HTMLLoader.supported_extensions, HTMLLoader())
