"""
app/utils/metadata.py
=====================
File extension → category mapping and metadata extraction.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from app.models.schemas import FileCategory, FileMetadata


# ── Extension → Category map ─────────────────────────────────────────────────

_EXT_MAP: dict[str, FileCategory] = {
    # Documents
    ".pdf": FileCategory.DOCUMENT, ".txt": FileCategory.DOCUMENT,
    ".docx": FileCategory.DOCUMENT, ".doc": FileCategory.DOCUMENT,
    ".md": FileCategory.DOCUMENT, ".markdown": FileCategory.DOCUMENT,
    ".html": FileCategory.DOCUMENT, ".htm": FileCategory.DOCUMENT,
    ".csv": FileCategory.DOCUMENT,
    ".xlsx": FileCategory.DOCUMENT, ".xls": FileCategory.DOCUMENT,
    ".pptx": FileCategory.DOCUMENT, ".ppt": FileCategory.DOCUMENT,
    # Code
    ".py": FileCategory.CODE, ".json": FileCategory.CODE,
    ".xml": FileCategory.CODE, ".yaml": FileCategory.CODE, ".yml": FileCategory.CODE,
    ".cpp": FileCategory.CODE, ".cc": FileCategory.CODE, ".cxx": FileCategory.CODE,
    ".h": FileCategory.CODE, ".hpp": FileCategory.CODE,
    ".java": FileCategory.CODE, ".sql": FileCategory.CODE,
    ".js": FileCategory.CODE, ".ts": FileCategory.CODE,
    ".rs": FileCategory.CODE, ".go": FileCategory.CODE,
    ".rb": FileCategory.CODE, ".sh": FileCategory.CODE,
    ".r": FileCategory.CODE, ".m": FileCategory.CODE,
    ".c": FileCategory.CODE, ".toml": FileCategory.CODE,
    # Images
    ".png": FileCategory.IMAGE, ".jpg": FileCategory.IMAGE,
    ".jpeg": FileCategory.IMAGE, ".tiff": FileCategory.IMAGE,
    ".tif": FileCategory.IMAGE, ".bmp": FileCategory.IMAGE,
    ".gif": FileCategory.IMAGE, ".webp": FileCategory.IMAGE,
    # Audio
    ".mp3": FileCategory.AUDIO, ".wav": FileCategory.AUDIO,
    ".m4a": FileCategory.AUDIO, ".aac": FileCategory.AUDIO,
    ".flac": FileCategory.AUDIO, ".ogg": FileCategory.AUDIO,
    # Video
    ".mp4": FileCategory.VIDEO, ".avi": FileCategory.VIDEO,
    ".mov": FileCategory.VIDEO, ".mkv": FileCategory.VIDEO,
    ".wmv": FileCategory.VIDEO, ".flv": FileCategory.VIDEO,
    ".webm": FileCategory.VIDEO,
    # Archives
    ".zip": FileCategory.ARCHIVE, ".gz": FileCategory.ARCHIVE,
    ".tar": FileCategory.ARCHIVE, ".rar": FileCategory.ARCHIVE,
    ".7z": FileCategory.ARCHIVE,
}


def resolve_file_category(extension: str) -> FileCategory:
    return _EXT_MAP.get(extension.lower(), FileCategory.UNKNOWN)


def _safe_stat(filepath: Path, attr: str) -> Optional[str]:
    try:
        ts = getattr(filepath.stat(), attr, None)
        return datetime.fromtimestamp(ts).isoformat() if ts else None
    except (OSError, ValueError):
        return None


def extract_file_metadata(
    filepath: str | Path,
    extra_extensions: Optional[list[str]] = None,
) -> FileMetadata:
    p = Path(filepath).resolve()
    ext = p.suffix.lower()

    effective = dict(_EXT_MAP)
    if extra_extensions:
        for e in extra_extensions:
            key = e if e.startswith(".") else f".{e}"
            effective[key.lower()] = FileCategory.DOCUMENT

    category = effective.get(ext, FileCategory.UNKNOWN)

    try:
        size = p.stat().st_size
    except OSError:
        size = 0

    return FileMetadata(
        filename=p.name,
        filepath=str(p),
        filetype=p.suffix,
        extension=ext,
        folder=p.parent.name,
        file_category=category,
        size_bytes=size,
        created_date=_safe_stat(p, "st_ctime"),
        modified_date=_safe_stat(p, "st_mtime"),
        accessed_date=_safe_stat(p, "st_atime"),
    )
