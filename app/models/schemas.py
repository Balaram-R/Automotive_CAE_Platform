"""
app/models/schemas.py
=====================
Core domain types for the Automotive CAE Knowledge Platform.

All dataclasses, enums, and TypedDicts live here to avoid circular
imports across the rest of the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


# ── Enums ────────────────────────────────────────────────────────────────────

class FileCategory(str, Enum):
    DOCUMENT = "document"
    CODE     = "code"
    IMAGE    = "image"
    AUDIO    = "audio"
    VIDEO    = "video"
    ARCHIVE  = "archive"
    UNKNOWN  = "unknown"


class ChunkingStrategy(str, Enum):
    RECURSIVE_CHARACTER = "recursive_character"
    MARKDOWN            = "markdown"
    SENTENCE            = "sentence"
    SEMANTIC            = "semantic"
    TOKEN               = "token"


class VectorStoreType(str, Enum):
    CHROMA   = "chroma"
    FAISS    = "faiss"
    QDRANT   = "qdrant"
    PINECONE = "pinecone"
    MILVUS   = "milvus"


class EmbeddingProvider(str, Enum):
    OPENAI              = "openai"
    GROQ                = "groq"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    BGE                 = "bge"
    NOMIC               = "nomic"
    JINA                = "jina"


class LLMProvider(str, Enum):
    GROQ    = "groq"
    OPENAI  = "openai"


class ProcessingStatus(str, Enum):
    PENDING    = "pending"
    SCANNING   = "scanning"
    LOADING    = "loading"
    PROCESSING = "processing"
    EMBEDDING  = "embedding"
    STORING    = "storing"
    COMPLETED  = "completed"
    FAILED     = "failed"
    SKIPPED    = "skipped"


# ── File metadata ────────────────────────────────────────────────────────────

@dataclass
class FileMetadata:
    filename:     str
    filepath:     str
    filetype:     str
    extension:    str
    folder:       str
    file_category: FileCategory
    size_bytes:   int = 0
    created_date: Optional[str] = None
    modified_date: Optional[str] = None
    accessed_date: Optional[str] = None
    author:       Optional[str] = None
    language:     Optional[str] = None
    page_count:   Optional[int] = None
    duration_seconds: Optional[float] = None
    frame_count:  Optional[int] = None
    sha256_hash:  Optional[str] = None
    last_processed: Optional[str] = None
    doc_index:    int = 0
    extra:        dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # file_category may arrive as a plain string (e.g. after a to_dict()
        # round-trip through state / JSON). Normalize it back to the enum so
        # downstream code can safely call .value on it.
        if isinstance(self.file_category, str):
            try:
                self.file_category = FileCategory(self.file_category)
            except ValueError:
                self.file_category = FileCategory.UNKNOWN

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FileMetadata":
        """Reconstruct a FileMetadata from a to_dict() output (lossless)."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "filename": self.filename,
            "filepath": self.filepath,
            "filetype": self.filetype,
            "extension": self.extension,
            "folder": self.folder,
            "file_category": self.file_category.value,
            "size_bytes": self.size_bytes,
            "sha256_hash": self.sha256_hash,
        }
        for k in ("created_date", "modified_date", "accessed_date",
                   "author", "language", "page_count",
                   "duration_seconds", "frame_count", "last_processed"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        if self.doc_index:
            d["doc_index"] = self.doc_index
        if self.extra:
            d["extra"] = self.extra
        return d


# ── Document / Chunk ─────────────────────────────────────────────────────────

@dataclass
class RawDocument:
    """A single extracted document (pre-chunk)."""
    content:        str
    metadata:       FileMetadata
    source_loader:  str
    extracted_at:   datetime = field(default_factory=datetime.utcnow)

    @property
    def doc_id(self) -> str:
        base = self.metadata.sha256_hash or self.metadata.filepath
        # A single file (e.g. a multi-page PDF) can yield several RawDocuments.
        # Disambiguate them so chunk IDs never collide across pages.
        if self.metadata.doc_index:
            return f"{base}#{self.metadata.doc_index}"
        return base


@dataclass
class Chunk:
    """A chunk ready for embedding."""
    text:          str
    chunk_id:      str
    document_id:   str
    chunk_index:   int
    metadata:      dict[str, Any] = field(default_factory=dict)
    embedding:     Optional[list[float]] = None

    @classmethod
    def from_document(cls, doc: RawDocument, text: str,
                      chunk_index: int, total_chunks: int) -> Chunk:
        cid = f"{doc.doc_id}_chunk_{chunk_index:06d}"
        meta = doc.metadata.to_dict()
        meta["chunk_id"]      = cid
        meta["document_id"]   = doc.doc_id
        meta["chunk_index"]   = chunk_index
        meta["total_chunks"]  = total_chunks
        meta["source_loader"] = doc.source_loader
        meta["extracted_at"]  = doc.extracted_at.isoformat()
        return cls(text=text, chunk_id=cid, document_id=doc.doc_id,
                   chunk_index=chunk_index, metadata=meta)


# ── Pipeline result ──────────────────────────────────────────────────────────

@dataclass
class ProcessingResult:
    filepath:                str
    status:                  ProcessingStatus
    chunks_stored:           int = 0
    error:                   Optional[str] = None
    processing_time_seconds: float = 0.0
    embedding_time_seconds:  float = 0.0
    loader_used:             Optional[str] = None
    chunking_strategy_used:  Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "filepath": self.filepath,
            "status": self.status.value,
            "chunks_stored": self.chunks_stored,
            "error": self.error,
            "processing_time_seconds": self.processing_time_seconds,
            "embedding_time_seconds": self.embedding_time_seconds,
            "loader_used": self.loader_used,
            "chunking_strategy_used": self.chunking_strategy_used,
        }


# ── Incremental tracking ────────────────────────────────────────────────────

@dataclass
class FileHashRecord:
    filepath:       str
    sha256_hash:    str
    last_processed: str          # ISO timestamp
    chunk_count:    int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "filepath": self.filepath,
            "sha256_hash": self.sha256_hash,
            "last_processed": self.last_processed,
            "chunk_count": self.chunk_count,
        }
