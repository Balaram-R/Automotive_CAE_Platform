"""
scripts/reingest_image.py
=========================
Re-ingest a single image file through the full pipeline (loader → cleaner →
chunker → embedder → writer), deleting any existing chunks for that file first.

Usage:
    venv\\Scripts\\python.exe scripts\\reingest_image.py "<image_path>"
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.embeddings.base import EmbeddingFactory
from app.loaders.loader_factory import LoaderFactory
from app.models.schemas import Chunk, FileMetadata
from app.processors.chunk_processor import ChunkerFactory
from app.processors.text_cleaner import CleanerPresets
from app.utils.config import load_config
from app.utils.logging import get_logger
from app.utils.metadata import extract_file_metadata
from app.vectorstore.base import VectorStoreFactory

# Force loader registration
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

logger = get_logger("scripts.reingest_image")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/reingest_image.py <image_path>")
        sys.exit(1)

    fp = sys.argv[1]
    fp = str(Path(fp).resolve())
    if not Path(fp).exists():
        print(f"File not found: {fp}")
        sys.exit(1)

    cfg = load_config()

    # 1) Vector store
    vs = VectorStoreFactory.create(
        cfg.vector_store.backend,
        collection_name=cfg.vector_store.collection_name,
        persist_directory=cfg.vector_store.persist_directory,
        host=cfg.vector_store.host, port=cfg.vector_store.port,
        api_key=cfg.vector_store.api_key, index_name=cfg.vector_store.index_name,
        dimension=cfg.embedding.dimension,
    )

    # 2) Delete existing chunks for this file
    fname = Path(fp).name
    deleted = vs.delete_by_filename(fname)
    print(f"Deleted {deleted} existing chunk(s) for {fname}")

    # 3) Load
    metadata = extract_file_metadata(fp, cfg.extra_document_extensions)
    loader = LoaderFactory.get_loader_for_file(fp)
    docs = loader.load(fp, metadata, config=cfg) or []
    print(f"Loaded {len(docs)} doc(s) via {loader.__class__.__name__}")

    # 4) Clean
    cat = docs[0].metadata.file_category if docs else "document"
    cleaner = CleanerPresets.code() if cat == "code" else CleanerPresets.document()
    for d in docs:
        d.content = cleaner.clean(d.content)

    # 5) Chunk
    chunker = ChunkerFactory.create(cfg.chunking.strategy, cfg.chunking.chunk_size, cfg.chunking.chunk_overlap)
    chunks = []
    for d in docs:
        for c in chunker.chunk(d):
            chunks.append(c)
    print(f"Produced {len(chunks)} chunk(s)")

    # 6) Embed
    provider = EmbeddingFactory.create(cfg.embedding.provider, cfg.embedding.model_name,
                                       dimension=cfg.embedding.dimension, batch_size=cfg.embedding.batch_size)
    t0 = time.time()
    embeddings = provider.embed_texts([c.text for c in chunks])
    print(f"Embedded {len(embeddings)} chunk(s) in {time.time()-t0:.2f}s")

    # 7) Store
    count = vs.add_chunks(chunks, embeddings)
    print(f"Stored {count} chunk(s)")

    # 8) Update hash tracker
    from app.utils.hashing import FileHashTracker
    ht = FileHashTracker(Path(cfg.knowledge_base.root_dir) / cfg.knowledge_base.hash_tracking_file)
    ht.record_processing(fp, count)
    ht.save()
    print("Hash tracker updated")

    print(f"\nDone. {fname} re-ingested with {count} chunk(s).")


if __name__ == "__main__":
    main()