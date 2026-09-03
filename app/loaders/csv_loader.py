"""
app/loaders/csv_loader.py
=========================
CSV → LangChain CSVLoader with text fallback.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from app.loaders.base import BaseLoader
from app.loaders.loader_factory import LoaderFactory
from app.models.schemas import FileMetadata, RawDocument
from app.utils.logging import get_logger

logger = get_logger("loaders.csv")


class CSVLoader(BaseLoader):
    supported_extensions = [".csv"]

    def load(self, filepath: str, metadata: FileMetadata, **kw) -> list[RawDocument]:
        try:
            from langchain_community.document_loaders import CSVLoader as LCCSV
            docs = LCCSV(filepath).load()
            out = []
            for i, d in enumerate(docs):
                m = replace(metadata, doc_index=i)
                out.append(RawDocument(content=d.page_content, metadata=m, source_loader="LangChainCSVLoader"))
            return out
        except (ImportError, Exception) as exc:
            logger.debug("LangChain CSVLoader unavailable/failed (%s), using raw text", exc)
        try:
            text = Path(filepath).read_text(encoding="utf-8", errors="replace")
            return [RawDocument(content=text, metadata=metadata, source_loader="TextFallback")]
        except Exception as exc:
            logger.error("CSV load failed %s: %s", filepath, exc)
            return []


LoaderFactory.register_many(CSVLoader.supported_extensions, CSVLoader())
