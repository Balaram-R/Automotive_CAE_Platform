"""
app/retrieval/retriever.py
==========================
Semantic retrieval over the vector store.
"""

from __future__ import annotations

from typing import Any, Optional

from app.embeddings.base import EmbeddingFactory
from app.utils.config import AppConfig
from app.utils.logging import get_logger
from app.vectorstore.base import VectorStoreFactory

logger = get_logger("retrieval")


class Retriever:
    """Query interface over the ingested knowledge base."""

    def __init__(self, embedding_provider, vector_store, top_k: int = 5):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.top_k = top_k

    @classmethod
    def from_config(cls, config: AppConfig) -> Retriever:
        emb = EmbeddingFactory.create(config.embedding.provider, config.embedding.model_name,
                                      dimension=config.embedding.dimension)
        vs = VectorStoreFactory.create(config.vector_store.backend,
                                      collection_name=config.vector_store.collection_name,
                                      persist_directory=config.vector_store.persist_directory,
                                      host=config.vector_store.host, port=config.vector_store.port,
                                      api_key=config.vector_store.api_key, index_name=config.vector_store.index_name,
                                      dimension=config.embedding.dimension)
        return cls(emb, vs)

    def retrieve(self, query: str, k: Optional[int] = None,
                 filter_dict: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        k = k or self.top_k
        qe = self.embedding_provider.embed_query(query)
        results = self.vector_store.similarity_search(qe, k=k, filter_dict=filter_dict)
        logger.info("Retrieved %d results for: '%s...'", len(results), query[:60])
        return results

    def retrieve_as_context(self, query: str, k: Optional[int] = None,
                            filter_dict: Optional[dict[str, Any]] = None) -> str:
        results = self.retrieve(query, k=k, filter_dict=filter_dict)
        parts = []
        for i, r in enumerate(results):
            src = r["metadata"].get("filename", "unknown")
            score = r.get("distance", "N/A")
            parts.append(f"[Source {i+1}: {src} (score: {score})]\n{r['document']}")
        return "\n\n---\n\n".join(parts)

    def delete_document(self, document_id: str) -> int:
        count = self.vector_store.delete_by_document_id(document_id)
        logger.info("Deleted %d chunks for doc %s", count, document_id)
        return count

    def stats(self) -> dict[str, Any]:
        return {
            "total_vectors": self.vector_store.count(),
            "embedding_provider": self.embedding_provider.provider_name,
            "embedding_dimension": self.embedding_provider.dimension,
            "healthy": self.vector_store.health_check(),
        }
