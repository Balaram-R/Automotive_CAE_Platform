"""
app/loaders/code_loader.py
==========================
Source code files → plain text with language hint.
"""

from __future__ import annotations

from pathlib import Path

from app.loaders.base import BaseLoader
from app.loaders.loader_factory import LoaderFactory
from app.models.schemas import FileMetadata, RawDocument
from app.utils.logging import get_logger

logger = get_logger("loaders.code")

_CODE_EXTS = [
    ".py", ".json", ".xml", ".yaml", ".yml",
    ".cpp", ".cc", ".cxx", ".h", ".hpp", ".c",
    ".java", ".sql", ".js", ".ts",
    ".rs", ".go", ".rb", ".sh", ".bat",
    ".r", ".m", ".toml", ".cfg", ".ini", ".cmake",
]


class CodeLoader(BaseLoader):
    supported_extensions = _CODE_EXTS

    def load(self, filepath: str, metadata: FileMetadata, **kw) -> list[RawDocument]:
        try:
            text = Path(filepath).read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                return []
            metadata.extra["language_hint"] = Path(filepath).suffix.lstrip(".")
            return [RawDocument(content=text, metadata=metadata, source_loader="CodeLoader")]
        except Exception as exc:
            logger.error("Code load failed %s: %s", filepath, exc)
            return []


LoaderFactory.register_many(CodeLoader.supported_extensions, CodeLoader())
