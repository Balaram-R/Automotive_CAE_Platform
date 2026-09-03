"""
app/loaders/loader_factory.py
=============================
Registry-based factory – zero if-else chains.

Loaders register themselves at import time via
``LoaderFactory.register(ext, loader)`` or the ``register_many`` helper.
"""

from __future__ import annotations

from pathlib import Path

from app.loaders.base import BaseLoader
from app.loaders.skip_loader import SkipLoader
from app.utils.logging import get_logger

logger = get_logger("loaders.factory")


class LoaderFactory:
    _registry: dict[str, BaseLoader] = {}
    _skip = SkipLoader()

    # ── Registration ─────────────────────────────────────────────────────
    @classmethod
    def register(cls, ext: str, loader: BaseLoader) -> None:
        e = ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        cls._registry[e] = loader

    @classmethod
    def register_many(cls, extensions: list[str], loader: BaseLoader) -> None:
        for ext in extensions:
            cls.register(ext, loader)

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()

    # ── Lookup ───────────────────────────────────────────────────────────
    @classmethod
    def get_loader(cls, extension: str) -> BaseLoader:
        e = extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        loader = cls._registry.get(e)
        if loader is None:
            logger.debug("No loader for '%s' → SkipLoader", e)
            return cls._skip
        return loader

    @classmethod
    def get_loader_for_file(cls, filepath: str) -> BaseLoader:
        return cls.get_loader(Path(filepath).suffix)

    # ── Introspection ────────────────────────────────────────────────────
    @classmethod
    def list_extensions(cls) -> dict[str, str]:
        return {ext: loader.__class__.__name__ for ext, loader in cls._registry.items()}
