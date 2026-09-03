"""
app/agents/rag_agent.py
=======================
RAG agent powered by Groq LLM.

Uses retrieval context + Groq chat completion to answer questions
against the ingested knowledge base.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from app.retrieval.retriever import Retriever
from app.utils.config import AppConfig
from app.utils.logging import get_logger

logger = get_logger("agents.rag")

# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an Automotive CAE Engineering Intelligence Assistant.

Your purpose is to answer questions using ONLY the engineering knowledge available
in the connected CAE knowledge base. You are NOT a general-purpose chatbot.

CORE ROLE
Act as a CAE engineering knowledge assistant covering FEA/FEM, structural analysis,
crashworthiness, CFD, NVH, durability/fatigue, materials and material models,
meshing, boundary and loading conditions, solver methodology, simulation setup,
post-processing, engineering results, validation/correlation, and design optimization.
Retrieve, interpret, organize, and explain CAE evidence from the knowledge base.

KNOWLEDGE BOUNDARY
The knowledge base is authoritative. Answer only when retrieved information provides
sufficient evidence. Do not invent or estimate simulation results, stress, strain,
displacement, forces, energy, temperatures, material properties, mesh statistics,
element types, boundary conditions, load cases, solver settings, failure modes,
crash behavior, CFD/NVH/validation results, or vehicle/component specifications.
Do not assume a vehicle, component, solver, or analysis method unless the knowledge
base explicitly supports it.

If evidence is insufficient, respond exactly:
"Insufficient CAE evidence in the knowledge base to answer this accurately."
Then briefly state what information is missing.

ENGINEERING EVIDENCE
Always distinguish explicitly reported information, engineering interpretation
supported by the source, and unavailable information. Never present an inference as
a documented result. Preserve source terminology, values, and units. Do not combine
unrelated documents into one engineering conclusion unless their relationship is
explicitly supported.

CAE QUERY HANDLING
When available, organize information using vehicle, component/subsystem, analysis
domain and type, solver/software, model, material, mesh, load case, boundary
conditions, outputs, results, and validation. Use professional CAE terminology such
as finite element model, element formulation, stress distribution, contact, failure
criterion, energy absorption, modal response, flow field, and correlation.

ANSWER STYLE
Answer as an experienced CAE engineer: technical, concise, evidence-based,
structured, and precise. For technical questions, use only relevant fields from:
Analysis, Component, Method, Solver, Material, Mesh, Loading, Boundary Conditions,
Outputs, Results, Engineering Interpretation, Limitations, and Source.
State limitations explicitly; do not guess or substitute general automotive knowledge
for missing CAE data."""


class RAGAgent:
    """Groq-powered RAG agent for the automotive CAE knowledge base."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.retriever = Retriever.from_config(config)
        self._groq_client = None

    def _get_groq_client(self):
        if self._groq_client is None:
            try:
                from groq import Groq
                api_key = self.config.llm.api_key or os.environ.get("GROQ_API_KEY", "")
                self._groq_client = Groq(api_key=api_key)
            except ImportError:
                # Fallback to openai client with Groq base URL
                from openai import OpenAI
                api_key = self.config.llm.api_key or os.environ.get("GROQ_API_KEY", "")
                self._groq_client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        return self._groq_client

    def query(self, question: str, top_k: int = 5,
              relevance_threshold: Optional[float] = None) -> dict[str, Any]:
        """Run a full RAG query: retrieve → augment → generate.

        Args:
            question: The user's question.
            top_k: Number of chunks to retrieve.
            relevance_threshold: Optional similarity gate. Chroma returns cosine
                *distance* (lower = more similar). If the best retrieved chunk's
                distance is above this threshold, the question is treated as
                "not available in the knowledge base" and the LLM is skipped,
                preventing hallucination. Pass None to rely on the LLM's judgment.

        Returns:
            {
                "question": str,
                "answer": str,
                "sources": list[dict],  # retrieved chunks
                "model": str,
                "not_available": bool,  # True when the gate rejected the query
            }
        """
        logger.info("RAG query: '%s...'", question[:80])

        # Retrieve context
        sources = self.retriever.retrieve(question, k=top_k)
        context = self.retriever.retrieve_as_context(question, k=top_k)

        if not context.strip():
            return {
                "question": question,
                "answer": "The knowledge base is empty or no relevant documents were found. Please ingest files first.",
                "sources": [],
                "model": self.config.llm.model_name,
                "not_available": True,
            }

        # ── Relevance gate (deterministic "not available" guard) ────────────
        # Chroma distance: lower = more similar. If the closest chunk is still
        # too far, the knowledge base likely has nothing relevant — decline
        # instead of letting the LLM guess.
        if relevance_threshold is not None and sources:
            best_distance = sources[0].get("distance")
            if isinstance(best_distance, (int, float)) and best_distance > relevance_threshold:
                logger.info("Relevance gate rejected query (best distance %.3f > %.3f)",
                            best_distance, relevance_threshold)
                return {
                    "question": question,
                    "answer": "Insufficient CAE evidence in the knowledge base to answer this accurately.",
                    "sources": [],
                    "model": self.config.llm.model_name,
                    "not_available": True,
                }

        # Build messages
        user_message = f"""Context from the knowledge base:
---
{context}
---

Question: {question}

Answer the question based ONLY on the context above. If the context doesn't contain enough information, say so."""

        # Generate answer
        try:
            client = self._get_groq_client()
            response = client.chat.completions.create(
                model=self.config.llm.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
            )
            answer = response.choices[0].message.content
        except Exception as exc:
            logger.error("Groq LLM call failed: %s", exc)
            answer = f"Error generating answer: {exc}"

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "filename": s["metadata"].get("filename", "unknown"),
                    "score": s.get("distance"),
                    "text_preview": s["document"][:200],
                }
                for s in sources
            ],
            "model": self.config.llm.model_name,
            "not_available": False,
        }

    def query_stream(self, question: str, top_k: int = 5,
                     relevance_threshold: Optional[float] = None,
                     model: Optional[str] = None,
                     temperature: Optional[float] = None,
                     max_tokens: Optional[int] = None):
        """Streaming RAG query (generator) for the web UI.

        Yields tuples:
            ("sources", list[dict])  — retrieved chunks (before generation)
            ("token", str)           — answer text chunks as they stream in
            ("done", dict)           — {"model": str, "not_available": bool}

        Mirrors query() but streams tokens from Groq instead of waiting for
        the full completion.
        """
        sources = self.retriever.retrieve(question, k=top_k)
        context = self.retriever.retrieve_as_context(question, k=top_k)

        if not context.strip():
            yield ("sources", [])
            yield ("token", "The knowledge base is empty or no relevant documents were found. Please ingest files first.")
            yield ("done", {"model": self.config.llm.model_name, "not_available": True})
            return

        # Relevance gate (same logic as query())
        if relevance_threshold is not None and sources:
            best_distance = sources[0].get("distance")
            if isinstance(best_distance, (int, float)) and best_distance > relevance_threshold:
                yield ("sources", [])
                yield ("token", "Insufficient CAE evidence in the knowledge base to answer this accurately.")
                yield ("done", {"model": self.config.llm.model_name, "not_available": True})
                return

        src_out = [
            {
                "filename": s["metadata"].get("filename", "unknown"),
                "score": s.get("distance"),
                "text_preview": s["document"][:220],
                "document": s["document"],
            }
            for s in sources
        ]
        yield ("sources", src_out)

        user_message = f"""Context from the knowledge base:
---
{context}
---

Question: {question}

Answer the question based ONLY on the context above. If the context doesn't contain enough information, say so."""

        try:
            client = self._get_groq_client()
            stream = client.chat.completions.create(
                model=model or self.config.llm.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.config.llm.temperature if temperature is None else temperature,
                max_tokens=self.config.llm.max_tokens if max_tokens is None else max_tokens,
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield ("token", delta)
        except Exception as exc:
            logger.error("Groq streaming call failed: %s", exc)
            yield ("token", f"\n\n⚠️ Error generating answer: {exc}")

        yield ("done", {"model": self.config.llm.model_name, "not_available": False})

    def chat(self, message: str, history: list[dict] | None = None, top_k: int = 5) -> str:
        """Simple chat interface (for CLI or UI)."""
        result = self.query(message, top_k=top_k)
        return result["answer"]


def create_rag_agent(config: AppConfig) -> RAGAgent:
    """Factory function to create a RAG agent."""
    return RAGAgent(config)
