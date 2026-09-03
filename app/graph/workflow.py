"""
app/graph/workflow.py
====================
LangGraph ingestion pipeline with TypedDict state.

Flow:
  FolderScanner → FileDetector → LoaderFactory → Loader → Cleaner →
  Chunker → EmbeddingGenerator → VectorStoreWriter → SummaryGenerator →
  (loop back or Finish)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.embeddings.base import EmbeddingFactory
from app.loaders.loader_factory import LoaderFactory
from app.models.schemas import (
    FileMetadata, ProcessingResult, ProcessingStatus, RawDocument, Chunk,
)
from app.processors.chunk_processor import ChunkerFactory
from app.processors.text_cleaner import CleanerPresets
from app.utils.config import AppConfig
from app.utils.hashing import FileHashTracker
from app.utils.logging import get_logger
from app.utils.metadata import extract_file_metadata
from app.vectorstore.base import VectorStoreFactory

# ── Force loader registration ────────────────────────────────────────────────
import app.loaders.pdf_loader      # noqa: F401
import app.loaders.text_loader     # noqa: F401
import app.loaders.docx_loader     # noqa: F401
import app.loaders.csv_loader      # noqa: F401
import app.loaders.html_loader     # noqa: F401
import app.loaders.markdown_loader # noqa: F401
import app.loaders.code_loader     # noqa: F401
import app.loaders.office_loader   # noqa: F401
import app.loaders.image_loader    # noqa: F401
import app.loaders.audio_loader    # noqa: F401
import app.loaders.video_loader    # noqa: F401
import app.loaders.archive_loader  # noqa: F401

logger = get_logger("graph.workflow")


# ═══════════════════════════════════════════════════════════════════════════════
# State
# ═══════════════════════════════════════════════════════════════════════════════

class IngestionState(TypedDict):
    config:              dict[str, Any]
    all_files:           list[str]
    current_file:        str
    current_metadata:    dict[str, Any] | None
    raw_documents:       list[dict[str, Any]]
    cleaned_documents:   list[dict[str, Any]]
    chunks:              list[dict[str, Any]]
    embeddings:          list[list[float]]
    chunks_stored:       int
    embedding_time:      float
    processed_count:     int
    skipped_count:       int
    error_count:         int
    results:             list[dict[str, Any]]
    status:              str
    current_file_status: str
    error_message:       str | None


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _cfg(state: IngestionState) -> AppConfig:
    from app.utils.config import _from_dict, load_config
    raw = state.get("config", {})
    if isinstance(raw, dict):
        return _from_dict(AppConfig, raw)
    # Already an AppConfig (shouldn't happen through LangGraph, but be safe)
    return raw


def _ser(doc: RawDocument) -> dict[str, Any]:
    return {"content": doc.content, "metadata": doc.metadata.to_dict(),
            "source_loader": doc.source_loader, "doc_id": doc.doc_id}


def _deser(d: dict[str, Any]) -> RawDocument:
    return RawDocument(content=d["content"], metadata=FileMetadata.from_dict(d["metadata"]),
                       source_loader=d["source_loader"])


# ═══════════════════════════════════════════════════════════════════════════════
# Nodes
# ═══════════════════════════════════════════════════════════════════════════════

def node_folder_scanner(state: IngestionState) -> dict:
    """Recursively scan knowledge_base for all files."""
    cfg = _cfg(state)
    root = Path(cfg.knowledge_base.root_dir)
    root.mkdir(parents=True, exist_ok=True)
    files = [str(p.resolve()) for p in root.rglob("*") if p.is_file() and not p.name.startswith(".")]
    logger.info("Scanned %d files from %s", len(files), root)
    return {"all_files": files, "status": "scanned"}


def node_file_detector(state: IngestionState) -> dict:
    """Pop next file, check hash, decide to load or skip."""
    files = state.get("all_files", [])
    idx = state.get("processed_count", 0) + state.get("skipped_count", 0) + state.get("error_count", 0)

    if idx >= len(files):
        return {"status": "queue_empty"}

    fp = files[idx]
    cfg = _cfg(state)

    ht = FileHashTracker(Path(cfg.knowledge_base.root_dir) / cfg.knowledge_base.hash_tracking_file)
    if not ht.is_new_or_modified(fp):
        logger.info("Unchanged, skipping: %s", Path(fp).name)
        # Do NOT increment skipped_count or append to results here; node_summary
        # handles counting + result recording so the count stays in sync with the
        # file index (otherwise the loop terminates early after half the files).
        return {"current_file": fp, "current_file_status": "skipped"}

    meta = extract_file_metadata(fp, cfg.extra_document_extensions)
    return {"current_file": fp, "current_metadata": meta.to_dict(), "current_file_status": "detected"}


def node_loader(state: IngestionState) -> dict:
    """Select loader via factory and extract content."""
    fp = state.get("current_file", "")
    md = state.get("current_metadata")
    if not fp or not md:
        return {"raw_documents": [], "current_file_status": "no_file"}

    metadata = FileMetadata.from_dict(md)
    loader = LoaderFactory.get_loader_for_file(fp)
    cfg = _cfg(state)

    t0 = time.time()
    try:
        docs = loader.load(fp, metadata, config=cfg) or []
        elapsed = time.time() - t0
        logger.info("Loaded %d doc(s) from %s via %s (%.2fs)",
                     len(docs), Path(fp).name, loader.__class__.__name__, elapsed)
        return {"raw_documents": [_ser(d) for d in docs], "current_file_status": "loaded"}
    except Exception as exc:
        logger.error("Loader failed %s: %s", fp, exc)
        return {"raw_documents": [], "current_file_status": "load_error", "error_message": str(exc)}


def node_cleaner(state: IngestionState) -> dict:
    """Clean extracted text based on file category."""
    raw = state.get("raw_documents", [])
    if not raw:
        return {"cleaned_documents": [], "current_file_status": "empty"}

    cat = raw[0].get("metadata", {}).get("file_category", "document")
    cleaner = CleanerPresets.code() if cat == "code" else CleanerPresets.document()

    cleaned = []
    for d in raw:
        doc = _deser(d)
        doc.content = cleaner.clean(doc.content)
        cleaned.append(_ser(doc))
    return {"cleaned_documents": cleaned, "current_file_status": "cleaned"}


def node_chunker(state: IngestionState) -> dict:
    """Split documents into chunks."""
    docs = state.get("cleaned_documents", [])
    if not docs:
        return {"chunks": [], "current_file_status": "no_clean_docs"}

    cfg = _cfg(state)
    chunker = ChunkerFactory.create(cfg.chunking.strategy, cfg.chunking.chunk_size, cfg.chunking.chunk_overlap)

    all_chunks: list[dict] = []
    for d in docs:
        doc = _deser(d)
        for c in chunker.chunk(doc):
            all_chunks.append({"text": c.text, "chunk_id": c.chunk_id, "document_id": c.document_id,
                               "chunk_index": c.chunk_index, "metadata": c.metadata})
    logger.info("Produced %d chunks from %d doc(s)", len(all_chunks), len(docs))
    return {"chunks": all_chunks, "current_file_status": "chunked"}


def node_embedder(state: IngestionState) -> dict:
    """Generate embeddings for all chunks."""
    chunks = state.get("chunks", [])
    if not chunks:
        return {"embeddings": [], "current_file_status": "no_chunks"}

    cfg = _cfg(state)
    provider = EmbeddingFactory.create(cfg.embedding.provider, cfg.embedding.model_name,
                                       dimension=cfg.embedding.dimension, batch_size=cfg.embedding.batch_size)

    t0 = time.time()
    try:
        embeddings = provider.embed_texts([c["text"] for c in chunks])
        elapsed = time.time() - t0
        logger.info("Embedded %d chunks in %.2fs via %s", len(embeddings), elapsed, provider.provider_name)
        return {"embeddings": embeddings, "embedding_time": elapsed, "current_file_status": "embedded"}
    except Exception as exc:
        logger.error("Embedding failed: %s", exc)
        return {"embeddings": [], "current_file_status": "embed_error", "error_message": str(exc)}


def node_writer(state: IngestionState) -> dict:
    """Write chunks + embeddings to vector store."""
    chunks_d = state.get("chunks", [])
    embeddings = state.get("embeddings", [])
    if not chunks_d or not embeddings:
        return {"chunks_stored": 0, "current_file_status": "nothing_to_store"}

    cfg = _cfg(state)
    vs = VectorStoreFactory.create(cfg.vector_store.backend,
                                   collection_name=cfg.vector_store.collection_name,
                                   persist_directory=cfg.vector_store.persist_directory,
                                   host=cfg.vector_store.host, port=cfg.vector_store.port,
                                   api_key=cfg.vector_store.api_key, index_name=cfg.vector_store.index_name,
                                   dimension=cfg.embedding.dimension)

    chunks = [Chunk(text=c["text"], chunk_id=c["chunk_id"], document_id=c["document_id"],
                    chunk_index=c["chunk_index"], metadata=c["metadata"]) for c in chunks_d]

    t0 = time.time()
    try:
        count = vs.add_chunks(chunks, embeddings)
        elapsed = time.time() - t0
        logger.info("Stored %d chunks in %.2fs", count, elapsed)

        # Update hash tracker
        fp = state.get("current_file", "")
        if fp:
            ht = FileHashTracker(Path(cfg.knowledge_base.root_dir) / cfg.knowledge_base.hash_tracking_file)
            ht.record_processing(fp, count)
            ht.save()

        return {"chunks_stored": count, "current_file_status": "stored"}
    except Exception as exc:
        logger.error("Vector store write failed: %s", exc)
        return {"chunks_stored": 0, "current_file_status": "store_error", "error_message": str(exc)}


def node_summary(state: IngestionState) -> dict:
    """Record result and advance to next file."""
    fp = state.get("current_file", "")
    st = state.get("current_file_status", "unknown")
    err = state.get("error_message")

    status_map = {"loaded": ProcessingStatus.COMPLETED, "cleaned": ProcessingStatus.COMPLETED,
                  "chunked": ProcessingStatus.COMPLETED, "embedded": ProcessingStatus.COMPLETED,
                  "stored": ProcessingStatus.COMPLETED,
                  "load_error": ProcessingStatus.FAILED, "embed_error": ProcessingStatus.FAILED,
                  "store_error": ProcessingStatus.FAILED,
                  "empty": ProcessingStatus.SKIPPED, "no_file": ProcessingStatus.SKIPPED,
                  "skipped": ProcessingStatus.SKIPPED}
    ps = status_map.get(st, ProcessingStatus.COMPLETED)

    result = ProcessingResult(filepath=fp, status=ps, chunks_stored=state.get("chunks_stored", 0),
                              error=err, embedding_time_seconds=state.get("embedding_time", 0),
                              loader_used=(state.get("raw_documents") or [{}])[0].get("source_loader") if state.get("raw_documents") else None,
                              chunking_strategy_used=state.get("config", {}).get("chunking", {}).get("strategy"))

    new_results = state.get("results", []) + [result.to_dict()]

    updates: dict[str, Any] = {"results": new_results, "current_file_status": "recorded", "error_message": None}
    if ps == ProcessingStatus.FAILED:
        updates["error_count"] = state.get("error_count", 0) + 1
    elif ps == ProcessingStatus.SKIPPED:
        updates["skipped_count"] = state.get("skipped_count", 0) + 1
    else:
        updates["processed_count"] = state.get("processed_count", 0) + 1
        updates["chunks_stored"] = 0
    return updates


# ═══════════════════════════════════════════════════════════════════════════════
# Routing
# ═══════════════════════════════════════════════════════════════════════════════

def _after_scan(s):       return "finish" if s.get("status") == "queue_empty" else "file_detector"
def _after_detect(s):
    if s.get("status") == "queue_empty": return "finish"
    return "summary" if s.get("current_file_status") == "skipped" else "loader"
def _after_load(s):       return "summary" if not s.get("raw_documents") else "cleaner"
def _after_clean(s):      return "summary" if not s.get("cleaned_documents") else "chunker"
def _after_chunk(s):      return "summary" if not s.get("chunks") else "embedder"
def _after_embed(s):      return "summary" if s.get("current_file_status") == "embed_error" or not s.get("embeddings") else "writer"
def _after_write(s):      return "summary"

def _after_summary(s):
    total = len(s.get("all_files", []))
    done = s.get("processed_count", 0) + s.get("skipped_count", 0) + s.get("error_count", 0)
    return "finish" if done >= total else "file_detector"


# ═══════════════════════════════════════════════════════════════════════════════
# Graph
# ═══════════════════════════════════════════════════════════════════════════════

def build_ingestion_graph():
    g = StateGraph(IngestionState)

    g.add_node("folder_scanner", node_folder_scanner)
    g.add_node("file_detector",  node_file_detector)
    g.add_node("loader",         node_loader)
    g.add_node("cleaner",        node_cleaner)
    g.add_node("chunker",        node_chunker)
    g.add_node("embedder",       node_embedder)
    g.add_node("writer",         node_writer)
    g.add_node("summary",        node_summary)

    g.set_entry_point("folder_scanner")
    g.add_edge("folder_scanner", "file_detector")

    g.add_conditional_edges("file_detector", _after_detect,
                            {"loader": "loader", "summary": "summary", "finish": END})
    g.add_conditional_edges("loader", _after_load,
                            {"cleaner": "cleaner", "summary": "summary"})
    g.add_conditional_edges("cleaner", _after_clean,
                            {"chunker": "chunker", "summary": "summary"})
    g.add_conditional_edges("chunker", _after_chunk,
                            {"embedder": "embedder", "summary": "summary"})
    g.add_conditional_edges("embedder", _after_embed,
                            {"writer": "writer", "summary": "summary"})
    g.add_edge("writer", "summary")
    g.add_conditional_edges("summary", _after_summary,
                            {"file_detector": "file_detector", "finish": END})

    return g.compile()
