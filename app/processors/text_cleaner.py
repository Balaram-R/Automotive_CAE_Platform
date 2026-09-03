"""
app/processors/text_cleaner.py
==============================
Text normalisation and cleaning pipeline.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

from app.utils.logging import get_logger

logger = get_logger("processors.cleaner")


class TextCleaner:
    def __init__(
        self,
        normalize_unicode: bool = True,
        remove_extra_whitespace: bool = True,
        remove_control_chars: bool = True,
        normalize_linebreaks: bool = True,
        strip_html: bool = True,
        lowercase: bool = False,
        min_line_length: int = 0,
        custom_patterns: Optional[list[tuple[str, str]]] = None,
    ):
        self.normalize_unicode = normalize_unicode
        self.remove_extra_whitespace = remove_extra_whitespace
        self.remove_control_chars = remove_control_chars
        self.normalize_linebreaks = normalize_linebreaks
        self.strip_html = strip_html
        self.lowercase = lowercase
        self.min_line_length = min_line_length
        self.custom_patterns = custom_patterns or []

    def clean(self, text: str) -> str:
        if not text:
            return ""
        if self.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)
        if self.strip_html:
            text = self._strip_html(text)
        if self.remove_control_chars:
            text = "".join(c for c in text if unicodedata.category(c)[0] != "C" or c in ("\n", "\t", "\r"))
        if self.normalize_linebreaks:
            text = re.sub(r'\r\n|\r', '\n', text)
        if self.remove_extra_whitespace:
            text = re.sub(r'[ \t]+', ' ', text)
            text = re.sub(r'\n{3,}', '\n\n', text)
        if self.lowercase:
            text = text.lower()
        if self.min_line_length > 0:
            text = "\n".join(l for l in text.split("\n") if len(l.strip()) >= self.min_line_length)
        for pat, rep in self.custom_patterns:
            text = re.sub(pat, rep, text)
        return text.strip()

    @staticmethod
    def _strip_html(text: str) -> str:
        clean = re.sub(r'<[^>]+>', ' ', text)
        for ent, ch in {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#39;": "'", "&nbsp;": " "}.items():
            clean = clean.replace(ent, ch)
        return clean


class CleanerPresets:
    @staticmethod
    def document() -> TextCleaner:
        return TextCleaner()

    @staticmethod
    def code() -> TextCleaner:
        return TextCleaner(remove_extra_whitespace=False, strip_html=False)

    @staticmethod
    def ocr() -> TextCleaner:
        return TextCleaner(custom_patterns=[(r'\|+', '|'), (r'[^\S\n]+', ' ')])

    @staticmethod
    def minimal() -> TextCleaner:
        return TextCleaner(remove_extra_whitespace=False, strip_html=False, remove_control_chars=False)
