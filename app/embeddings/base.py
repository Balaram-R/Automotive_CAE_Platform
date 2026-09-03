"""
app/embeddings/base.py
======================
Configurable embedding providers via strategy pattern.

Providers: OpenAI, Groq, SentenceTransformers, BGE, Nomic, Jina.
All implement the same interface so the rest of the pipeline is agnostic.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional

from app.models.schemas import EmbeddingProvider
from app.utils.logging import get_logger

logger = get_logger("embeddings")


# ── Abstract ─────────────────────────────────────────────────────────────────

class BaseEmbeddings(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @property
    @abstractmethod
    def provider_name(self) -> str: ...


# ── OpenAI ───────────────────────────────────────────────────────────────────

class OpenAIEmbeddings(BaseEmbeddings):
    def __init__(self, model_name: str = "text-embedding-3-small",
                 api_key: Optional[str] = None, base_url: Optional[str] = None,
                 batch_size: int = 100, dimension: int = 1536, **_kw):
        self._model = model_name
        self._key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._url = base_url
        self._batch = batch_size
        self._dim = dimension
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            kw = {}
            if self._key: kw["api_key"] = self._key
            if self._url: kw["base_url"] = self._url
            self._client = OpenAI(**kw)
        return self._client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        c = self._client or self._get_client()
        out = []
        for i in range(0, len(texts), self._batch):
            batch = texts[i:i + self._batch]
            resp = c.embeddings.create(input=batch, model=self._model)
            out.extend(e.embedding for e in resp.data)
        return out

    def embed_query(self, query: str) -> list[float]:
        c = self._client or self._get_client()
        return c.embeddings.create(input=[query], model=self._model).data[0].embedding

    @property
    def dimension(self) -> int: return self._dim

    @property
    def provider_name(self) -> str: return f"OpenAI({self._model})"


# ── Groq (OpenAI-compatible, fast inference) ─────────────────────────────────

class GroqEmbeddings(BaseEmbeddings):
    """Groq embeddings — nomic-embed-text-v1.5 (768d), mxbai-embed-large-v1 (1024d)."""

    def __init__(self, model_name: str = "nomic-embed-text-v1.5",
                 api_key: Optional[str] = None,
                 batch_size: int = 100, dimension: int = 768, **_kw):
        self._model = model_name
        self._key = api_key or os.environ.get("GROQ_API_KEY", "")
        self._batch = batch_size
        self._dim = dimension
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self._key:
                raise ValueError("GROQ_API_KEY required — set it in .env or OS env")
            from openai import OpenAI
            self._client = OpenAI(api_key=self._key, base_url="https://api.groq.com/openai/v1")
        return self._client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        c = self._client or self._get_client()
        out = []
        for i in range(0, len(texts), self._batch):
            batch = texts[i:i + self._batch]
            resp = c.embeddings.create(input=batch, model=self._model)
            out.extend(e.embedding for e in resp.data)
        return out

    def embed_query(self, query: str) -> list[float]:
        c = self._client or self._get_client()
        return c.embeddings.create(input=[query], model=self._model).data[0].embedding

    @property
    def dimension(self) -> int: return self._dim

    @property
    def provider_name(self) -> str: return f"Groq({self._model})"


# ── SentenceTransformers (covers BGE, Nomic local) ──────────────────────────

class SentenceTransformerEmbeddings(BaseEmbeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 device: Optional[str] = None, batch_size: int = 100, **_kw):
        self._model_name = model_name
        self._device = device or "cpu"
        self._batch = batch_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name, device=self._device)
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        m = self._get_model()
        return m.encode(texts, batch_size=self._batch, normalize_embeddings=True).tolist()

    def embed_query(self, query: str) -> list[float]:
        return self._get_model().encode([query], normalize_embeddings=True)[0].tolist()

    @property
    def dimension(self) -> int:
        return self._get_model().get_sentence_embedding_dimension()

    @property
    def provider_name(self) -> str:
        return f"SentenceTransformer({self._model_name})"


class BGEEmbeddings(SentenceTransformerEmbeddings):
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", **kw):
        super().__init__(model_name=model_name, **kw)


class NomicEmbeddings(SentenceTransformerEmbeddings):
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1", **kw):
        super().__init__(model_name=model_name, **kw)


# ── Jina ─────────────────────────────────────────────────────────────────────

class JinaEmbeddings(BaseEmbeddings):
    def __init__(self, model_name: str = "jina-embeddings-v2-base-en",
                 api_key: Optional[str] = None, batch_size: int = 100,
                 dimension: int = 768, **_kw):
        self._model = model_name
        self._key = api_key or os.environ.get("JINA_API_KEY", "")
        self._batch = batch_size
        self._dim = dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        import requests
        out = []
        for i in range(0, len(texts), self._batch):
            batch = texts[i:i + self._batch]
            r = requests.post(
                "https://api.jina.ai/v1/embeddings",
                json={"model": self._model, "input": batch},
                headers={"Authorization": f"Bearer {self._key}"},
            )
            r.raise_for_status()
            out.extend(d["embedding"] for d in r.json()["data"])
        return out

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]

    @property
    def dimension(self) -> int: return self._dim

    @property
    def provider_name(self) -> str: return f"Jina({self._model})"


# ── Factory ──────────────────────────────────────────────────────────────────

class EmbeddingFactory:
    _providers: dict[str, type[BaseEmbeddings]] = {
        EmbeddingProvider.OPENAI.value:              OpenAIEmbeddings,
        EmbeddingProvider.GROQ.value:                GroqEmbeddings,
        EmbeddingProvider.SENTENCE_TRANSFORMERS.value: SentenceTransformerEmbeddings,
        EmbeddingProvider.BGE.value:                 BGEEmbeddings,
        EmbeddingProvider.NOMIC.value:               NomicEmbeddings,
        EmbeddingProvider.JINA.value:                JinaEmbeddings,
    }

    @classmethod
    def create(cls, provider: str, model_name: Optional[str] = None, **kw) -> BaseEmbeddings:
        klass = cls._providers.get(provider)
        if klass is None:
            logger.warning("Unknown provider '%s', defaulting to SentenceTransformer", provider)
            klass = SentenceTransformerEmbeddings
        if model_name:
            kw["model_name"] = model_name
        return klass(**kw)
