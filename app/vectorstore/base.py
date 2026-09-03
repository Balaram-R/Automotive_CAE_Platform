"""
app/vectorstore/base.py
=======================
Pluggable vector store backends: Chroma, FAISS, Qdrant, Pinecone, Milvus.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from app.models.schemas import Chunk, VectorStoreType
from app.utils.logging import get_logger

logger = get_logger("vectorstore")


# ── Abstract ─────────────────────────────────────────────────────────────────

class BaseVectorStore(ABC):
    @abstractmethod
    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int: ...
    @abstractmethod
    def similarity_search(self, query_embedding: list[float], k: int = 5,
                          filter_dict: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]: ...
    @abstractmethod
    def delete_by_document_id(self, document_id: str) -> int: ...
    @abstractmethod
    def count(self) -> int: ...
    @abstractmethod
    def health_check(self) -> bool: ...


# ── Chroma ───────────────────────────────────────────────────────────────────

def _sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Reduce metadata to values Chroma can persist.

    Chroma only accepts str / int / float / bool metadata values. Drop None
    values and flatten/stringify anything else (nested dicts, lists, datetimes).
    """
    out: dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = ",".join(str(i) for i in v)
        elif isinstance(v, dict):
            out[k] = ",".join(f"{kk}={vv}" for kk, vv in v.items())
        else:
            out[k] = str(v)
    return out


class ChromaVectorStore(BaseVectorStore):
    def __init__(self, collection_name: str = "automotive_cae",
                 persist_directory: str = "vectordb/chroma", **_kw):
        self._name = collection_name
        self._dir = persist_directory
        self._col = None

    def _get(self):
        if self._col is None:
            import chromadb
            from chromadb.config import Settings
            Path(self._dir).mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(
                path=self._dir
            )

            self._col = client.get_or_create_collection(
                self._name,
                metadata={"hnsw:space": "cosine"}
            )
        return self._col

    def add_chunks(self, chunks, embeddings):
        if not chunks: return 0
        self._get().add(ids=[c.chunk_id for c in chunks], embeddings=embeddings,
                        documents=[c.text for c in chunks],
                        metadatas=[_sanitize_metadata(c.metadata) for c in chunks])
        return len(chunks)

    def similarity_search(self, query_embedding, k=5, filter_dict=None):
        kw: dict[str, Any] = {"query_embeddings": [query_embedding], "n_results": k}
        if filter_dict: kw["where"] = filter_dict
        r = self._get().query(**kw)
        return [{"id": r["ids"][0][i], "document": r["documents"][0][i],
                 "metadata": r["metadatas"][0][i], "distance": (r.get("distances") or [None])[0][i]}
                for i in range(len(r["ids"][0]))]

    def delete_by_document_id(self, did):
        r = self._get().get(where={"document_id": did})
        if r["ids"]: self._get().delete(ids=r["ids"])
        return len(r["ids"])

    def delete_by_filename(self, filename: str) -> int:
        """Delete every chunk whose metadata filename matches (Chroma)."""
        r = self._get().get(where={"filename": filename})
        if r["ids"]: self._get().delete(ids=r["ids"])
        return len(r["ids"])

    def count(self): return self._get().count()
    def health_check(self):
        try: self._get(); return True
        except: return False

    def get_documents(self) -> dict[str, dict]:
        """filename -> {"chunks": int, "document_ids": list[str]}.

        Used by the UI Knowledge Base page to show per-document stats and
        support deletion. Chroma-specific; other backends return {}.
        """
        try:
            data = self._get().get(include=["metadatas"])
            out: dict[str, dict] = {}
            for m in data.get("metadatas") or []:
                fn = m.get("filename", "unknown")
                did = m.get("document_id", "")
                entry = out.setdefault(fn, {"chunks": 0, "document_ids": []})
                entry["chunks"] += 1
                if did and did not in entry["document_ids"]:
                    entry["document_ids"].append(did)
            return out
        except Exception:  # noqa: BLE001
            return {}


# ── FAISS ────────────────────────────────────────────────────────────────────

class FAISSVectorStore(BaseVectorStore):
    def __init__(self, dimension: int = 1536, persist_directory: str = "vectordb/faiss",
                 index_name: str = "index", **_kw):
        self._dim = dimension
        self._dir = persist_directory
        self._name = index_name
        self._index = None
        self._chunks: list[Chunk] = []

    def _get(self):
        if self._index is None:
            import faiss
            ip = Path(self._dir) / f"{self._name}.faiss"
            cp = Path(self._dir) / f"{self._name}_chunks.pkl"
            if ip.exists() and cp.exists():
                import pickle
                self._index = faiss.read_index(str(ip))
                self._chunks = pickle.loads(cp.read_bytes())
            else:
                self._index = faiss.IndexFlatIP(self._dim)
        return self._index

    def add_chunks(self, chunks, embeddings):
        import numpy as np, faiss
        if not chunks: return 0
        idx = self._get()
        vecs = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vecs)
        start = idx.ntotal
        idx.add(vecs)
        self._chunks.extend(chunks)
        self._save()
        return len(chunks)

    def _save(self):
        import faiss, pickle
        Path(self._dir).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(Path(self._dir) / f"{self._name}.faiss"))
        (Path(self._dir) / f"{self._name}_chunks.pkl").write_bytes(pickle.dumps(self._chunks))

    def similarity_search(self, query_embedding, k=5, filter_dict=None):
        import numpy as np, faiss
        idx = self._get()
        if idx.ntotal == 0: return []
        q = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(q)
        k = min(k, idx.ntotal)
        dists, ids = idx.search(q, k)
        return [{"id": self._chunks[i].chunk_id, "document": self._chunks[i].text,
                 "metadata": self._chunks[i].metadata, "distance": float(d)}
                for d, i in zip(dists[0], ids[0]) if 0 <= i < len(self._chunks)]

    def delete_by_document_id(self, did):
        n = sum(1 for c in self._chunks if c.metadata.get("document_id") == did)
        if n: self._chunks = [c for c in self._chunks if c.metadata.get("document_id") != did]
        return n

    def count(self): return self._get().ntotal
    def health_check(self):
        try: self._get(); return True
        except: return False


# ── Qdrant ───────────────────────────────────────────────────────────────────

class QdrantVectorStore(BaseVectorStore):
    def __init__(self, collection_name: str = "automotive_cae", host: str = "localhost",
                 port: int = 6333, api_key: Optional[str] = None, dimension: int = 1536, **_kw):
        self._name, self._host, self._port, self._key, self._dim = collection_name, host, port, api_key, dimension
        self._client = None

    def _get(self):
        if self._client is None:
            from qdrant_client import QdrantClient
            if self._key:
                self._client = QdrantClient(url=f"https://{self._host}:{self._port}", api_key=self._key)
            else:
                self._client = QdrantClient(host=self._host, port=self._port)
        return self._client

    def add_chunks(self, chunks, embeddings):
        from qdrant_client.models import PointStruct, Distance, VectorParams
        if not chunks: return 0
        c = self._get()
        try: c.get_collection(self._name)
        except: c.create_collection(self._name, vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE))
        pts = [PointStruct(id=abs(hash(ch.chunk_id)) % (2**63), vector=em,
                           payload={"text": ch.text, "chunk_id": ch.chunk_id, "document_id": ch.document_id, **ch.metadata})
               for ch, em in zip(chunks, embeddings)]
        c.upsert(collection_name=self._name, points=pts)
        return len(chunks)

    def similarity_search(self, query_embedding, k=5, filter_dict=None):
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        sf = Filter(must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filter_dict.items()]) if filter_dict else None
        return [{"id": r.id, "document": r.payload.get("text", ""),
                 "metadata": {k: v for k, v in r.payload.items() if k != "text"}, "distance": r.score}
                for r in self._get().search(self._name, query_embedding, limit=k, query_filter=sf)]

    def delete_by_document_id(self, did):
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        pts, _ = self._get().scroll(self._name, scroll_filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=did))]), limit=10000)
        ids = [p.id for p in pts]
        if ids: self._get().delete(self._name, ids)
        return len(ids)

    def count(self): return self._get().get_collection(self._name).points_count or 0
    def health_check(self):
        try: self._get().get_collections(); return True
        except: return False


# ── Pinecone ─────────────────────────────────────────────────────────────────

class PineconeVectorStore(BaseVectorStore):
    def __init__(self, index_name: str = "automotive-cae", api_key: Optional[str] = None,
                 dimension: int = 1536, **_kw):
        self._name = index_name
        self._key = api_key or os.environ.get("PINECONE_API_KEY", "")
        self._dim = dimension
        self._index = None

    def _get(self):
        if self._index is None:
            from pinecone import Pinecone
            self._index = Pinecone(api_key=self._key).Index(self._name)
        return self._index

    def add_chunks(self, chunks, embeddings):
        if not chunks: return 0
        idx = self._get()
        vecs = [{"id": ch.chunk_id, "values": em, "metadata": {"text": ch.text, "document_id": ch.document_id, **{k: str(v) for k, v in ch.metadata.items()}}}
                for ch, em in zip(chunks, embeddings)]
        for i in range(0, len(vecs), 100):
            idx.upsert(vectors=vecs[i:i + 100])
        return len(chunks)

    def similarity_search(self, query_embedding, k=5, filter_dict=None):
        return [{"id": m["id"], "document": m["metadata"].get("text", ""),
                 "metadata": {k: v for k, v in m["metadata"].items() if k != "text"}, "distance": m.get("score")}
                for m in self._get().query(vector=query_embedding, top_k=k, include_metadata=True, filter=filter_dict)["matches"]]

    def delete_by_document_id(self, did):
        self._get().delete(filter={"document_id": did}); return -1

    def count(self): return self._get().describe_index_stats().get("total_vector_count", 0)
    def health_check(self):
        try: self._get(); return True
        except: return False


# ── Milvus ───────────────────────────────────────────────────────────────────

class MilvusVectorStore(BaseVectorStore):
    def __init__(self, collection_name: str = "automotive_cae", host: str = "localhost",
                 port: int = 19530, dimension: int = 1536, **_kw):
        self._name, self._host, self._port, self._dim = collection_name, host, port, dimension
        self._col = None

    def _get(self):
        if self._col is None:
            from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
            connections.connect("default", host=self._host, port=self._port)
            if utility.has_collection(self._name):
                self._col = Collection(self._name); self._col.load()
            else:
                fields = [
                    FieldSchema("id", DataType.VARCHAR, is_primary=True, max_length=128),
                    FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=self._dim),
                    FieldSchema("text", DataType.VARCHAR, max_length=65535),
                    FieldSchema("document_id", DataType.VARCHAR, max_length=256),
                ]
                self._col = Collection(self._name, CollectionSchema(fields))
                self._col.create_index("embedding", {"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 128}})
                self._col.load()
        return self._col

    def add_chunks(self, chunks, embeddings):
        if not chunks: return 0
        col = self._get()
        col.insert([[c.chunk_id for c in chunks], embeddings, [c.text[:65535] for c in chunks], [c.document_id for c in chunks]])
        col.flush(); return len(chunks)

    def similarity_search(self, query_embedding, k=5, filter_dict=None):
        expr = None
        if filter_dict:
            expr = " and ".join(f'{k} == "{v}"' for k, v in filter_dict.items())
        results = self._get().search([query_embedding], "embedding", {"metric_type": "COSINE", "params": {"nprobe": 16}},
                                     limit=k, expr=expr, output_fields=["text", "document_id"])
        return [{"id": h.id, "document": h.entity.get("text", ""),
                 "metadata": {"document_id": h.entity.get("document_id", "")}, "distance": h.score}
                for hits in results for h in hits]

    def delete_by_document_id(self, did):
        self._get().delete(f'document_id == "{did}"'); return -1

    def count(self): return self._get().num_entities
    def health_check(self):
        try: self._get(); return True
        except: return False


import os  # needed for Pinecone env fallback


# ── Factory ──────────────────────────────────────────────────────────────────

class VectorStoreFactory:
    _backends = {
        VectorStoreType.CHROMA.value:   ChromaVectorStore,
        VectorStoreType.FAISS.value:    FAISSVectorStore,
        VectorStoreType.QDRANT.value:   QdrantVectorStore,
        VectorStoreType.PINECONE.value: PineconeVectorStore,
        VectorStoreType.MILVUS.value:   MilvusVectorStore,
    }

    @classmethod
    def create(cls, backend: str, **kw) -> BaseVectorStore:
        klass = cls._backends.get(backend)
        if klass is None:
            logger.warning("Unknown backend '%s', defaulting to Chroma", backend)
            klass = ChromaVectorStore
        return klass(**kw)
