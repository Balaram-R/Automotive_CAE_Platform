"""
app/processors/chunk_processor.py
=================================
5 chunking strategies: RecursiveCharacter, Markdown, Sentence, Token, Semantic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import re

from app.models.schemas import Chunk, RawDocument
from app.utils.logging import get_logger

logger = get_logger("processors.chunker")


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, document: RawDocument) -> list[Chunk]: ...


class RecursiveCharacterChunker(BaseChunker):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.cs, self.co = chunk_size, chunk_overlap

    def chunk(self, doc: RawDocument) -> list[Chunk]:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            texts = RecursiveCharacterTextSplitter(chunk_size=self.cs, chunk_overlap=self.co).split_text(doc.content)
        except ImportError:
            texts = _naive_split(doc.content, self.cs, self.co)
        return [Chunk.from_document(doc, t, i, len(texts)) for i, t in enumerate(texts)]


class MarkdownChunker(BaseChunker):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.cs, self.co = chunk_size, chunk_overlap

    def chunk(self, doc: RawDocument) -> list[Chunk]:
        try:
            from langchain.text_splitter import MarkdownTextSplitter
            texts = MarkdownTextSplitter(chunk_size=self.cs, chunk_overlap=self.co).split_text(doc.content)
        except ImportError:
            texts = _naive_split(doc.content, self.cs, self.co)
        return [Chunk.from_document(doc, t, i, len(texts)) for i, t in enumerate(texts)]


class SentenceChunker(BaseChunker):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.cs, self.co = chunk_size, chunk_overlap

    def chunk(self, doc: RawDocument) -> list[Chunk]:
        sentences = re.split(r'(?<=[.!?])\s+', doc.content)
        chunks: list[str] = []
        cur = ""
        for s in sentences:
            if len(cur) + len(s) + 1 > self.cs and cur:
                chunks.append(cur.strip())
                overlap_w = cur.split()[-(self.co // 5):] if self.co else []
                cur = (" ".join(overlap_w) + " " + s) if overlap_w else s
            else:
                cur = f"{cur} {s}".strip() if cur else s
        if cur.strip():
            chunks.append(cur.strip())
        return [Chunk.from_document(doc, t, i, len(chunks)) for i, t in enumerate(chunks)]


class TokenChunker(BaseChunker):
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.cs, self.co = chunk_size, chunk_overlap

    def chunk(self, doc: RawDocument) -> list[Chunk]:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(doc.content)
            chunks: list[str] = []
            i = 0
            while i < len(tokens):
                chunks.append(enc.decode(tokens[i:i + self.cs]))
                i += self.cs - self.co
        except ImportError:
            chunks = _naive_split(doc.content, self.cs, self.co)
        return [Chunk.from_document(doc, t, i, len(chunks)) for i, t in enumerate(chunks)]


class SemanticChunker(BaseChunker):
    def __init__(self, chunk_size: int = 1000, **_kw):
        self.cs = chunk_size

    def chunk(self, doc: RawDocument) -> list[Chunk]:
        try:
            from langchain_experimental.text_splitter import SemanticChunker
            from langchain_openai import OpenAIEmbeddings
            texts = SemanticChunker(OpenAIEmbeddings(), breakpoint_threshold_type="percentile").split_text(doc.content)
        except ImportError:
            logger.warning("SemanticChunker needs langchain_experimental, falling back")
            texts = _naive_split(doc.content, self.cs, 200)
        return [Chunk.from_document(doc, t, i, len(texts)) for i, t in enumerate(texts)]


def _naive_split(text: str, cs: int, co: int) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + cs])
        start += cs - co
    return chunks


class ChunkerFactory:
    _map = {
        "recursive_character": RecursiveCharacterChunker,
        "markdown": MarkdownChunker,
        "sentence": SentenceChunker,
        "token": TokenChunker,
        "semantic": SemanticChunker,
    }

    @classmethod
    def create(cls, strategy: str, chunk_size: int = 1000, chunk_overlap: int = 200, **kw):
        klass = cls._map.get(strategy, RecursiveCharacterChunker)
        return klass(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kw)
