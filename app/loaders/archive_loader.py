"""
app/loaders/archive_loader.py
=============================
ZIP → extract and recursively load inner files.
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

from app.loaders.base import BaseLoader
from app.loaders.loader_factory import LoaderFactory
from app.models.schemas import FileMetadata, RawDocument
from app.utils.logging import get_logger

logger = get_logger("loaders.archive")


class ArchiveLoader(BaseLoader):
    supported_extensions = [".zip"]

    def load(self, filepath: str, metadata: FileMetadata, **kw) -> list[RawDocument]:
        try:
            extract = Path(tempfile.mkdtemp(prefix="zip_"))
            with zipfile.ZipFile(filepath, "r") as zf:
                zf.extractall(extract)

            docs: list[RawDocument] = []
            for inner in extract.rglob("*"):
                if not inner.is_file():
                    continue
                loader = LoaderFactory.get_loader(inner.suffix)
                if isinstance(loader, type) and loader.__name__ == "SkipLoader":
                    continue
                if loader.__class__.__name__ == "SkipLoader":
                    continue
                meta = metadata
                meta.extra["archive_source"] = filepath
                meta.extra["archive_relative"] = str(inner.relative_to(extract))
                try:
                    docs.extend(loader.load(str(inner), meta, **kw))
                except Exception as exc:
                    logger.warning("Inner file failed %s: %s", inner, exc)

            logger.info("Extracted %d docs from %s", len(docs), filepath)
            return docs
        except zipfile.BadZipFile:
            logger.error("Bad zip: %s", filepath)
            return []
        except Exception as exc:
            logger.error("Archive failed %s: %s", filepath, exc)
            return []


LoaderFactory.register_many(ArchiveLoader.supported_extensions, ArchiveLoader())
