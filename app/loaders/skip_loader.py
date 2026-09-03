"""
app/loaders/skip_loader.py
==========================
Fallback loader for unsupported file types.
"""

from __future__ import annotations

from app.loaders.base import BaseLoader
from app.models.schemas import FileMetadata, RawDocument


class SkipLoader(BaseLoader):
    supported_extensions: list[str] = []

    def load(self, filepath: str, metadata: FileMetadata, **kw) -> list[RawDocument]:
        return []
