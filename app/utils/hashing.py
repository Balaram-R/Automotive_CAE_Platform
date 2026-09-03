"""
app/utils/hashing.py
====================
SHA-256 based file tracking for incremental processing.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.models.schemas import FileHashRecord


def compute_sha256(filepath: str | Path, chunk_size: int = 8192) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


class FileHashTracker:
    def __init__(self, tracking_file: str | Path):
        self.tracking_file = Path(tracking_file)
        self.records: dict[str, FileHashRecord] = {}
        self._load()

    def _load(self) -> None:
        if self.tracking_file.exists():
            raw = json.loads(self.tracking_file.read_text(encoding="utf-8"))
            for fp, d in raw.items():
                self.records[fp] = FileHashRecord(**d)

    def save(self) -> None:
        self.tracking_file.parent.mkdir(parents=True, exist_ok=True)
        self.tracking_file.write_text(
            json.dumps({fp: r.to_dict() for fp, r in self.records.items()}, indent=2),
            encoding="utf-8",
        )

    def is_new_or_modified(self, filepath: str | Path) -> bool:
        fp = str(Path(filepath).resolve())
        current = compute_sha256(filepath)
        if fp not in self.records:
            return True
        return self.records[fp].sha256_hash != current

    def record_processing(self, filepath: str | Path, chunk_count: int = 0) -> None:
        fp = str(Path(filepath).resolve())
        self.records[fp] = FileHashRecord(
            filepath=fp,
            sha256_hash=compute_sha256(filepath),
            last_processed=datetime.utcnow().isoformat(),
            chunk_count=chunk_count,
        )

    def find_deleted(self, current_files: list[str | Path]) -> list[str]:
        current = {str(Path(f).resolve()) for f in current_files}
        return [fp for fp in self.records if fp not in current]
