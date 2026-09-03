"""
Automotive CAE Knowledge Platform – CLI entry point.

Usage:
    python main.py ingest                      # Ingest all files from knowledge_base/
    python main.py query "FEA boundary conds"  # RAG query via Groq
    python main.py chat                        # Interactive chat mode
    python main.py status                      # Vector store stats
    python main.py list-loaders                # Show registered loaders
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from app.utils.config import load_config
from app.utils.logging import get_logger, setup_logging


# ── Ingest ───────────────────────────────────────────────────────────────────

def cmd_ingest(args):
    cfg = load_config(args.config)
    setup_logging(cfg.logging)
    log = get_logger("main.ingest")

    log.info("=" * 70)
    log.info("  Automotive CAE Knowledge Platform — Ingestion Pipeline")
    log.info("=" * 70)
    log.info("  Knowledge base : %s", cfg.knowledge_base.root_dir)
    log.info("  Embedding      : %s / %s", cfg.embedding.provider, cfg.embedding.model_name)
    log.info("  Vector store   : %s", cfg.vector_store.backend)
    log.info("  Chunking       : %s (%d / %d)", cfg.chunking.strategy, cfg.chunking.chunk_size, cfg.chunking.chunk_overlap)
    log.info("-" * 70)

    from app.utils.config import config_to_dict
    from app.graph.workflow import build_ingestion_graph, IngestionState

    graph = build_ingestion_graph()

    initial: IngestionState = {
        "config": config_to_dict(cfg), "all_files": [], "current_file": "",
        "current_metadata": None, "raw_documents": [], "cleaned_documents": [],
        "chunks": [], "embeddings": [], "chunks_stored": 0, "embedding_time": 0,
        "processed_count": 0, "skipped_count": 0, "error_count": 0,
        "results": [], "status": "initial", "current_file_status": "", "error_message": None,
    }

    t0 = time.time()
    final = graph.invoke(initial)
    elapsed = time.time() - t0

    log.info("=" * 70)
    log.info("  DONE in %.2fs", elapsed)
    log.info("  Processed: %d | Skipped: %d | Errors: %d | Total: %d",
             final.get("processed_count", 0), final.get("skipped_count", 0),
             final.get("error_count", 0), len(final.get("all_files", [])))
    log.info("-" * 70)

    for r in final.get("results", []):
        s = r["status"].upper()
        fp = r["filepath"]
        ch = r.get("chunks_stored", 0)
        err = r.get("error")
        log.info("  [%s] %s — %s", s, fp, err if err else f"{ch} chunks")

    log.info("=" * 70)


# ── Query (RAG) ─────────────────────────────────────────────────────────────

def cmd_query(args):
    cfg = load_config(args.config)
    setup_logging(cfg.logging)

    from app.agents.rag_agent import RAGAgent

    agent = RAGAgent(cfg)
    result = agent.query(
        args.question,
        top_k=args.top_k,
        relevance_threshold=args.relevance_threshold,
    )

    if args.json:
        print(json.dumps({
            "answer": result["answer"],
            "sources": result["sources"],
        }))
        return

    print(f"\n{'=' * 70}")
    print(f"  Question: {result['question']}")
    print(f"  Model:    {result['model']}")
    print(f"{'=' * 70}\n")
    print(f"  {result['answer']}\n")

    if result["sources"]:
        print(f"  {'-' * 60}")
        print("  Sources:")
        for s in result["sources"]:
            print(f"    - {s['filename']} (score: {s['score']})")
            print(f"      {s['text_preview']}...")
        print()


# ── Chat (Interactive Chatbot) ────────────────────────────────────────────────

_BANNER = r"""
  ╔══════════════════════════════════════════════════════════════════════╗
  ║                                                                    ║
  ║        🚗  Automotive CAE Knowledge Assistant  🏎️                  ║
  ║                                                                    ║
  ║   Powered by Groq LLM + LangGraph RAG                             ║
  ║                                                                    ║
  ╚══════════════════════════════════════════════════════════════════════╝
"""

_HELP_TEXT = """
  Commands:
    /help     — Show this help
    /status   — Show knowledge base stats
    /sources  — Show last query sources
    /clear    — Clear screen
    quit      — Exit the chatbot
"""


def cmd_chat(args):
    cfg = load_config(args.config)
    setup_logging(cfg.logging)

    from app.agents.rag_agent import RAGAgent
    from app.retrieval.retriever import Retriever

    agent = RAGAgent(cfg)
    retriever = Retriever.from_config(cfg)

    # ── Greeting ────────────────────────────────────────────────────────
    print(_BANNER)

    # Check knowledge base status
    try:
        stats = retriever.stats()
        n_vectors = stats["total_vectors"]
        healthy = stats["healthy"]
    except Exception:
        n_vectors = 0
        healthy = False

    if not healthy:
        print("  ⚠  Vector store not ready or empty.")
        print("     Run  python main.py ingest  first to load your knowledge base.\n")
    elif n_vectors == 0:
        print("  ⚠  Knowledge base is empty.")
        print("     Run  python main.py ingest  to load files.\n")
    else:
        print(f"  ✅  Knowledge base ready — {n_vectors:,} vectors loaded.\n")

    print("  Hi! I'm your Automotive CAE assistant.")
    print("  Ask me anything about your ingested documents —")
    print("  FEA, crash simulation, NVH, CFD, durability, materials, CAD/CAE workflows.\n")
    print(_HELP_TEXT)

    # ── Chat loop ───────────────────────────────────────────────────────
    history: list[dict] = []
    last_sources: list[dict] = []

    while True:
        try:
            q = input("  🔧 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye! 👋\n")
            break

        if not q:
            continue

        # ── Commands ────────────────────────────────────────────────────
        low = q.lower()

        if low in ("quit", "exit", "q", "bye"):
            print("\n  Goodbye! 👋\n")
            break

        if low == "/help":
            print(_HELP_TEXT)
            continue

        if low == "/clear":
            import os
            os.system("cls" if os.name == "nt" else "clear")
            print(_BANNER)
            continue

        if low == "/status":
            try:
                s = retriever.stats()
                print(f"\n  📊 Status:")
                print(f"     Vectors     : {s['total_vectors']:,}")
                print(f"     Embeddings  : {s['embedding_provider']}")
                print(f"     Dimension   : {s['embedding_dimension']}")
                print(f"     Healthy     : {s['healthy']}\n")
            except Exception as e:
                print(f"\n  ❌ Error getting status: {e}\n")
            continue

        if low == "/sources":
            if last_sources:
                print(f"\n  📄 Sources from last query:")
                for i, s in enumerate(last_sources):
                    print(f"     {i+1}. {s['filename']} (score: {s['score']})")
                print()
            else:
                print("\n  No sources yet. Ask a question first.\n")
            continue

        if low.startswith("/"):
            print(f"\n  Unknown command: {q}")
            print("  Type /help for available commands.\n")
            continue

        # ── RAG query ───────────────────────────────────────────────────
        print()  # blank line before answer

        try:
            result = agent.query(q)
            answer = result["answer"]
            sources = result["sources"]
            last_sources = sources

            # Print answer
            print(f"  🤖 AI: {answer}\n")

            # Print sources
            if sources:
                src_names = list(dict.fromkeys(s["filename"] for s in sources))
                print(f"  📄 Sources: {', '.join(src_names)}")
                print()

            # Track history
            history.append({"role": "user", "content": q})
            history.append({"role": "assistant", "content": answer})

        except Exception as exc:
            print(f"  ❌ Error: {exc}\n")

    # ── Session summary ─────────────────────────────────────────────────
    if history:
        q_count = sum(1 for m in history if m["role"] == "user")
        print(f"  Session complete — {q_count} question(s) asked.\n")


# ── Status ───────────────────────────────────────────────────────────────────

def cmd_status(args):
    cfg = load_config(args.config)
    setup_logging(cfg.logging)

    from app.retrieval.retriever import Retriever

    r = Retriever.from_config(cfg)
    s = r.stats()

    print(f"\n{'=' * 70}")
    print("  Vector Store Status")
    print(f"{'=' * 70}")
    print(f"  Backend     : {cfg.vector_store.backend}")
    print(f"  Collection  : {cfg.vector_store.collection_name}")
    print(f"  Vectors     : {s['total_vectors']}")
    print(f"  Embeddings  : {s['embedding_provider']}")
    print(f"  Dimension   : {s['embedding_dimension']}")
    print(f"  Healthy     : {s['healthy']}")
    print(f"{'=' * 70}\n")


# ── List Loaders ─────────────────────────────────────────────────────────────

def cmd_list_loaders(args):
    from app.loaders.loader_factory import LoaderFactory
    # Force registration
    import app.loaders.pdf_loader, app.loaders.text_loader, app.loaders.docx_loader  # noqa: F401
    import app.loaders.csv_loader, app.loaders.html_loader, app.loaders.markdown_loader  # noqa: F401
    import app.loaders.code_loader, app.loaders.office_loader, app.loaders.image_loader  # noqa: F401
    import app.loaders.audio_loader, app.loaders.video_loader, app.loaders.archive_loader  # noqa: F401

    exts = LoaderFactory.list_extensions()
    print(f"\n{'=' * 70}")
    print(f"  Registered Loaders ({len(exts)} extensions)")
    print(f"{'=' * 70}\n")

    by_class: dict[str, list[str]] = {}
    for ext, cls in sorted(exts.items()):
        by_class.setdefault(cls, []).append(ext)
    for cls, exts_list in sorted(by_class.items()):
        print(f"  {cls}")
        print(f"    {', '.join(sorted(exts_list))}\n")
    print(f"{'=' * 70}\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Windows consoles default to cp1252, which cannot encode many Unicode
    # characters (e.g. narrow no-break space \u202f) that LLM answers may
    # contain. Reconfigure stdout/stderr to UTF-8 so printing never crashes.
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    p = argparse.ArgumentParser(description="Automotive CAE Knowledge Platform")
    sub = p.add_subparsers(dest="command")

    s_ingest = sub.add_parser("ingest", help="Ingest files into knowledge base")
    s_ingest.add_argument("--config", default="configs/config.yaml")
    s_ingest.set_defaults(func=cmd_ingest)

    s_query = sub.add_parser("query", help="RAG query via Groq")
    s_query.add_argument("question")
    s_query.add_argument("--config", default="configs/config.yaml")
    s_query.add_argument("--top-k", type=int, default=5)
    s_query.add_argument("--relevance-threshold", type=float, default=None,
                         help="Maximum cosine distance accepted for a retrieved answer")
    s_query.add_argument("--json", action="store_true",
                         help="Return only the answer and sources as JSON")
    s_query.set_defaults(func=cmd_query)

    s_chat = sub.add_parser("chat", help="Interactive chat mode")
    s_chat.add_argument("--config", default="configs/config.yaml")
    s_chat.set_defaults(func=cmd_chat)

    s_status = sub.add_parser("status", help="Vector store status")
    s_status.add_argument("--config", default="configs/config.yaml")
    s_status.set_defaults(func=cmd_status)

    s_list = sub.add_parser("list-loaders", help="Show registered loaders")
    s_list.set_defaults(func=cmd_list_loaders)

    args = p.parse_args()
    if not args.command:
        p.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
