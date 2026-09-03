"""
app/utils/config.py
===================
YAML-based configuration with .env secret resolution.

Secrets in the YAML are referenced as ``${ENV_VAR_NAME}`` and
resolved from OS environment or a ``.env`` file at project root.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

# ── Load .env ────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ── Sub-configs ──────────────────────────────────────────────────────────────

@dataclass
class KnowledgeBaseConfig:
    root_dir:           str  = "knowledge_base"
    scan_subdirs:       bool = True
    hash_tracking_file: str  = ".file_hashes.json"


@dataclass
class EmbeddingConfig:
    provider:       str           = "sentence_transformers"
    model_name:     str           = "all-MiniLM-L6-v2"
    dimension:      int           = 384
    batch_size:     int           = 100
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    device:         Optional[str] = None


@dataclass
class LLMConfig:
    provider:       str           = "groq"
    model_name:     str           = "llama-3.3-70b-versatile"
    temperature:    float         = 0.1
    max_tokens:     int           = 4096
    api_key:        Optional[str] = None
    # Multimodal (vision) model used to caption image files at ingest time.
    # Must be a Groq model that accepts image_url content, e.g. qwen/qwen3.6-27b.
    vision_model:   Optional[str] = "qwen/qwen3.6-27b"
    vision_enabled: bool          = True


@dataclass
class VectorStoreConfig:
    backend:          str           = "chroma"
    collection_name:  str           = "automotive_cae"
    persist_directory: str          = "vectordb/chroma"
    host:             Optional[str] = None
    port:             Optional[int] = None
    api_key:          Optional[str] = None
    index_name:       Optional[str] = None


@dataclass
class ChunkingConfig:
    strategy:      str = "recursive_character"
    chunk_size:    int = 1000
    chunk_overlap: int = 200


@dataclass
class OCRConfig:
    enabled:   bool     = True
    model:     str      = "tesseract"
    languages: list[str] = field(default_factory=lambda: ["eng"])


@dataclass
class WhisperConfig:
    model_size: str           = "base"
    language:   Optional[str] = None
    device:     Optional[str] = None


@dataclass
class VideoConfig:
    extract_audio:          bool  = True
    extract_frames:         bool  = True
    frame_interval_seconds: float = 5.0
    max_frames:             int   = 100


@dataclass
class LoggingConfig:
    level:    str  = "INFO"
    format:   str  = "%(asctime)s | %(name)-24s | %(levelname)-7s | %(message)s"
    log_dir:  str  = "logs"
    log_file: str  = "pipeline.log"
    json_logs: bool = False


@dataclass
class AppConfig:
    knowledge_base:    KnowledgeBaseConfig = field(default_factory=KnowledgeBaseConfig)
    embedding:         EmbeddingConfig      = field(default_factory=EmbeddingConfig)
    llm:               LLMConfig            = field(default_factory=LLMConfig)
    vector_store:      VectorStoreConfig    = field(default_factory=VectorStoreConfig)
    chunking:          ChunkingConfig       = field(default_factory=ChunkingConfig)
    ocr:               OCRConfig            = field(default_factory=OCRConfig)
    whisper:           WhisperConfig        = field(default_factory=WhisperConfig)
    video:             VideoConfig          = field(default_factory=VideoConfig)
    logging:           LoggingConfig        = field(default_factory=LoggingConfig)
    extra_document_extensions: list[str]    = field(default_factory=list)


# ── Loader ───────────────────────────────────────────────────────────────────

_ENV_RE = re.compile(r'\$\{(\w+)\}')


def _resolve_env(value: Any) -> Any:
    """Recursively resolve ${VAR} placeholders."""
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(i) for i in value]
    return value


def _from_dict(cls: type, d: dict[str, Any]) -> Any:
    if hasattr(cls, "__dataclass_fields__"):
        types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
        return cls(**{
            k: _from_dict(eval(types[k]), v) if k in types else v
            for k, v in d.items() if k in types
        })
    return d


def config_to_dict(cfg: AppConfig) -> dict[str, Any]:
    """Deep-serialize an AppConfig to a plain dict (safe for TypedDict state)."""
    import dataclasses
    if dataclasses.is_dataclass(cfg):
        return {k: config_to_dict(v) if dataclasses.is_dataclass(v) else v
                for k, v in dataclasses.asdict(cfg).items()}
    return cfg


def load_config(path: str | Path = "configs/config.yaml") -> AppConfig:
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        _write_default(p)
        return AppConfig()
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw = _resolve_env(raw)
    return _from_dict(AppConfig, raw)


def _write_default(path: Path) -> None:
    """Create a starter config.yaml."""
    data = {
        "knowledge_base": {"root_dir": "knowledge_base", "scan_subdirs": True},
        "embedding": {
            "provider": "sentence_transformers",
            "model_name": "all-MiniLM-L6-v2",
            "dimension": 384,
            "batch_size": 100,
        },
        "llm": {
            "provider": "groq",
            "model_name": "llama-3.3-70b-versatile",
            "temperature": 0.1,
            "max_tokens": 4096,
            "vision_model": "qwen/qwen3.6-27b",
            "vision_enabled": True,
        },
        "vector_store": {
            "backend": "chroma",
            "collection_name": "automotive_cae",
            "persist_directory": "vectordb/chroma",
        },
        "chunking": {"strategy": "recursive_character", "chunk_size": 1000, "chunk_overlap": 200},
        "ocr": {"enabled": True, "model": "tesseract", "languages": ["eng"]},
        "whisper": {"model_size": "base"},
        "video": {"extract_audio": True, "extract_frames": True},
        "logging": {"level": "INFO", "log_dir": "logs", "log_file": "pipeline.log"},
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
