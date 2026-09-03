"""
app/loaders/base.py
===================
Abstract base for all file loaders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.schemas import FileMetadata, RawDocument


class BaseLoader(ABC):
    """Every concrete loader inherits this and implements ``load()``."""

    supported_extensions: list[str] = []

    @abstractmethod
    def load(self, filepath: str, metadata: FileMetadata, **kw) -> list[RawDocument]:
        """Extract content from *filepath* and return RawDocuments."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} ext={self.supported_extensions}>"
