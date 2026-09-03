"""
app/processors/ocr_processor.py
===============================
Pluggable OCR backends (Tesseract, EasyOCR).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.utils.logging import get_logger

logger = get_logger("processors.ocr")


class BaseOCR(ABC):
    @abstractmethod
    def extract_text(self, image_path: str, language: str = "eng") -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...


class TesseractOCR(BaseOCR):
    def is_available(self) -> bool:
        try:
            import pytesseract  # noqa: F401
            return True
        except ImportError:
            return False

    def extract_text(self, image_path: str, language: str = "eng") -> str:
        try:
            import pytesseract
            from PIL import Image
            return pytesseract.image_to_string(Image.open(image_path), lang=language).strip()
        except Exception as exc:
            logger.error("Tesseract failed %s: %s", image_path, exc)
            return ""


class EasyOCRProcessor(BaseOCR):
    def __init__(self) -> None:
        self._reader = None

    def is_available(self) -> bool:
        try:
            import easyocr  # noqa: F401
            return True
        except ImportError:
            return False

    def extract_text(self, image_path: str, language: str = "en") -> str:
        try:
            import easyocr
            if self._reader is None:
                self._reader = easyocr.Reader([language])
            results = self._reader.readtext(image_path)
            return "\n".join(r[1] for r in results).strip()
        except Exception as exc:
            logger.error("EasyOCR failed %s: %s", image_path, exc)
            return ""


class _NullOCR(BaseOCR):
    def is_available(self) -> bool:
        return True
    def extract_text(self, image_path: str, language: str = "eng") -> str:
        return ""


class OCRProcessorFactory:
    _processors: list[BaseOCR] = [TesseractOCR(), EasyOCRProcessor()]

    @classmethod
    def get_processor(cls) -> BaseOCR:
        for p in cls._processors:
            if p.is_available():
                return p
        return _NullOCR()
