"""
app/utils/logging.py
====================
Structured logging with optional JSON output.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.utils.config import LoggingConfig


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            entry["exception"] = self.formatException(record.exc_info)
        for k in ("filepath", "loader", "chunks", "elapsed", "error"):
            v = getattr(record, k, None)
            if v is not None:
                entry[k] = v
        return json.dumps(entry, default=str)


class _TextFormatter(logging.Formatter):
    def __init__(self, fmt: str | None = None) -> None:
        super().__init__(fmt or "%(asctime)s | %(name)-24s | %(levelname)-7s | %(message)s")


def setup_logging(config: LoggingConfig) -> logging.Logger:
    root = logging.getLogger("automotive_cae")
    root.setLevel(getattr(logging, config.level.upper(), logging.INFO))

    fmt = _JSONFormatter() if config.json_logs else _TextFormatter(config.format)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    if config.log_dir:
        log_path = Path(config.log_dir) / config.log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"automotive_cae.{name}")
