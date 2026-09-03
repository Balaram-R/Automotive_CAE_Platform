<div align="center">

# 🚗 Automotive CAE Intelligence Platform

### Multimodal RAG Knowledge Base for Engineering & Crash-Test Analysis

**Ingest documents, images & crash-test videos → Ask engineering questions → Get source-grounded CAE answers**

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Groq-LLM-F55036?style=for-the-badge&logo=groq&logoColor=white" alt="Groq"/>
  <img src="https://img.shields.io/badge/LangGraph-Agents-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangGraph"/>
  <img src="https://img.shields.io/badge/ChromaDB-VectorDB-4B8BBE?style=for-the-badge&logo=chromadb&logoColor=white" alt="ChromaDB"/>
  <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Embeddings-Local-FF6F00?style=for-the-badge&logo=huggingface&logoColor=white" alt="Sentence-Transformers"/>
</p>


**A production-style RAG pipeline that turns raw engineering documents, crash-test photos, and videos into an answerable CAE knowledge base — with source citations, a relevance gate against hallucination, and a full web UI.**

<a href="#-quick-start">Quick Start</a> •
<a href="#-features">Features</a> •
<a href="#️-architecture">Architecture</a> •
<a href="#️-web-ui">Web UI</a> •
<a href="#-configuration">Configuration</a> •
<a href="#-cli-reference">CLI</a>

</div>

<br/>

---

## 🚀 Quick Start

<table>
<tr><td width="40" align="center"><b>1</b></td><td>

**Install**

```bash
git clone https://github.com/your-org/automotive_cae_platform.git
cd automotive_cae_platform
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

</td></tr>
<tr><td align="center"><b>2</b></td><td>

**Add your Groq API key**

```bash
copy .env.example .env         # Windows
# cp .env.example .env         # macOS / Linux
```

</td></tr>
<tr><td align="center"><b>3</b></td><td>

**Drop your files into the knowledge base**

```
knowledge_base/
├── FEA_Report.pdf
├── crash_test.mp4
├── vehicle_photo.jpg
├── material_data.csv
└── ...
```

</td></tr>
<tr><td align="center"><b>4</b></td><td>

**Ingest**

```bash
python main.py ingest
```

</td></tr>
<tr><td align="center"><b>5</b></td><td>

**Ask questions**

```bash
# One-shot RAG query
python main.py query "What are the FEA boundary conditions?"

# Interactive chat
python main.py chat

# JSON output (for scripting / UI)
python main.py query "Describe the crash test" --json
```

</td></tr>
<tr><td align="center"><b>6</b></td><td>

**Launch the web UI**

```bash
streamlit run app.py
```

Open **http://localhost:8501** for a full dashboard with an ingest pipeline, chat interface, and system status.

</td></tr>
</table>

---

## ✨ Features

| | | |
|---|---|---|
| 🧠 | **Multimodal RAG** | Ingest PDFs, DOCX, PPTX, images, videos, audio, CSV, code & more — all into one vector store |
| 🎥 | **Vision-Captioned Videos** | Crash-test videos are frame-by-frame captioned by a **Groq vision model**, so RAG answers *what's happening* (deformation, test types, dummy kinematics) — not just brightness stats |
| 🖼️ | **Image Understanding** | Photos (crash tests, FEA screenshots) are described by a vision model at ingest time |
| 🛡️ | **Anti-Hallucination Gate** | A deterministic relevance threshold rejects queries with no supporting evidence instead of letting the LLM guess |
| 📚 | **Source-Grounded Answers** | Every answer cites the exact files and chunks it was built from |
| 🖥️ | **Full Web UI** | Streamlit dashboard: Home, Ingest Pipeline, Chat, and System pages |
| ⚙️ | **LangGraph Pipeline** | Clean, observable ingestion graph: load → clean → chunk → embed → store |
| 🔌 | **Pluggable Backends** | Swap vector stores (Chroma, FAISS, Qdrant, Pinecone, Milvus) and embedding providers via config |
| 🚀 | **Fast & Free** | Local embeddings (no API key) + Groq's fast free-tier LLM |

---

## 🏗️ Architecture

```
                    ┌──────────────────────────────────────────────────────┐
                    │                    KNOWLEDGE BASE                    │
                    │   PDF · DOCX · PPTX · Images · Videos · Audio · CSV  │
                    └────────────────────────┬─────────────────────────────┘
                                              │
                    ┌─────────────────────────▼─────────────────────────────┐
                    │              INGESTION PIPELINE (LangGraph)           │
                    │                                                       │
                    │   Load ──► Clean ──► Chunk ──► Embed ──► Store        │
                    │  (per-file   (recursive)         (local)  (Chroma)    │
                    │   loader)                                             │
                    │                                                       │
                    │   Vision captioning for images & video key-frames     │
                    └─────────────────────────┬─────────────────────────────┘
                                              │
                                              ▼
                    ┌───────────────────────────────────────────────────────┐
                    │                    VECTOR STORE                      │
                    │              ChromaDB (local, zero-config)           │
                    └────────────────────────┬──────────────────────────────┘
                                              │
                    ┌─────────────────────────▼─────────────────────────────┐
                    │               RETRIEVAL + GENERATION                 │
                    │                                                       │
                    │   Query ─► Retrieve top-k ─► Relevance Gate ─► Groq   │
                    │                                    │                  │
                    │                                    ▼                  │
                    │                          Source-cited answer          │
                    └───────────────────────────────────────────────────────┘
```

---

## 🖥️ Web UI

The Streamlit dashboard gives you everything in one place:

| Page | What it does |
|---|---|
| 🏠 **Home** | Overview of the platform and knowledge base |
| 📥 **Ingest Pipeline** | Discover files, see extraction quality (`TRACKED` / `WARNING` / `NEW`), and run ingestion with live status |
| 💬 **Chat** | Ask questions with streaming answers and source citations |
| ⚙️ **System** | Vector store stats, backend health, and configuration |

> 💡 **Extraction quality badges** — the UI flags files whose extraction produced only placeholders (e.g. image-only decks) as `WARNING`, so you know which files need OCR or vision captioning.

---

## 📄 Supported Formats

| Category | Extensions |
|---|---|
| 📄 **Documents** | `.pdf` `.doc` `.docx` `.txt` `.md` `.markdown` `.htm` `.html` |
| 📊 **Spreadsheets** | `.csv` `.xls` `.xlsx` |
| 📽️ **Presentations** | `.ppt` `.pptx` |
| 🖼️ **Images** | `.jpg` `.jpeg` `.png` `.gif` `.bmp` `.tif` `.tiff` `.webp` |
| 🎬 **Video** | `.mp4` `.avi` `.mkv` `.mov` `.webm` `.wmv` `.flv` |
| 🎵 **Audio** | `.mp3` `.wav` `.flac` `.m4a` `.ogg` `.aac` `.wma` |
| 💻 **Code** | `.py` `.js` `.ts` `.java` `.c` `.cpp` `.go` `.rs` `.rb` `.sql` `.json` `.yaml` `.xml` + more |
| 📦 **Archives** | `.zip` |

---

## 🧠 How It Works

### Ingestion
1. **Load** — each file type has a dedicated loader (registry-based, zero if-else)
2. **Vision captioning** — images and video key-frames are described by a Groq vision model (`qwen/qwen3.6-27b`), so the vector store contains *semantic* descriptions, not raw pixels
3. **Clean** — text is normalized and de-garbled
4. **Chunk** — recursive character chunking (configurable size/overlap)
5. **Embed** — local `all-MiniLM-L6-v2` (no API key needed)
6. **Store** — ChromaDB with per-file metadata and hash tracking (only new/modified files re-ingest)

### Query
1. **Retrieve** — top-k most similar chunks via cosine distance
2. **Relevance gate** — if the best match is still too far, the system declines to answer instead of hallucinating
3. **Generate** — Groq LLM (`openai/gpt-oss-20b`) builds a source-grounded answer from the retrieved context
4. **Cite** — every answer lists the exact source files and scores

---

## ⚙️ Configuration

Everything is driven by `configs/config.yaml`:

```yaml
embedding:
  provider: "sentence_transformers"   # groq | openai | sentence_transformers | bge | nomic | jina
  model_name: "all-MiniLM-L6-v2"

llm:
  provider: "groq"
  model_name: "openai/gpt-oss-20b"    # openai/gpt-oss-20b | qwen/qwen3.6-27b | openai/gpt-oss-120b
  vision_model: "qwen/qwen3.6-27b"    # multimodal model for image/video captioning

vector_store:
  backend: "chroma"                   # chroma | faiss | qdrant | pinecone | milvus
  collection_name: "automotive_cae"

chunking:
  strategy: "recursive_character"
  chunk_size: 1000
  chunk_overlap: 200
```

---

## 📁 Project Structure

```
automotive_cae_platform/
├── app.py                    # Streamlit web UI
├── main.py                   # CLI entry point (ingest / query / chat / status)
├── configs/
│   └── config.yaml           # All configuration
├── app/
│   ├── agents/                # RAG agent (Groq LLM + relevance gate)
│   ├── embeddings/            # Embedding providers (local, Groq, OpenAI, ...)
│   ├── graph/                 # LangGraph ingestion workflow
│   ├── loaders/                # Per-file-type loaders (registry-based)
│   ├── processors/             # Text cleaning + chunking
│   ├── retrieval/               # Vector store retrieval
│   ├── vectorstore/             # Chroma / FAISS / Qdrant / Pinecone / Milvus
│   └── utils/                   # Config, logging, hashing, metadata
├── knowledge_base/            # Drop your files here
├── scripts/                    # Utility scripts (re-ingest, OCR)
└── requirements.txt
```

---

## 🛠️ CLI Reference

| Command | Description |
|---|---|
| `python main.py ingest` | Ingest all files from `knowledge_base/` |
| `python main.py query "..."` | One-shot RAG query |
| `python main.py query "..." --json` | Query as JSON (answer + sources) |
| `python main.py query "..." --top-k 10` | Control number of retrieved chunks |
| `python main.py query "..." --relevance-threshold 0.7` | Set the anti-hallucination gate |
| `python main.py chat` | Interactive chat mode |
| `python main.py status` | Vector store stats |
| `python main.py list-loaders` | Show all registered file loaders |

---

## 🧪 Example

```bash
python main.py query "What happens in the crash test video?" --json
```

```json
{
  "answer": "The video shows a frontal offset deformable barrier crash test of a Maruti Suzuki Celerio at 64 km/h. The front-end structure (bumper, grille, windshield) exhibits typical frontal-impact deformation, the A-pillar is bowed inward, and the front driver-side door is deformed...",
  "sources": [
    {
      "filename": "Cel_crash_test.mp4",
      "score": 0.4449,
      "text_preview": "[Video Summary] This is a vehicle crash test video showing a frontal impact/collision test..."
    }
  ]
}
```
<img width="1915" height="957" alt="image" src="https://github.com/user-attachments/assets/35c33e90-0b7b-4876-89b0-cfd4f2970366" />

---

## 🛡️ Anti-Hallucination

The platform is built for **engineering trust**. It will never invent simulation results, stress values, or material properties. If the knowledge base lacks sufficient evidence, it responds:

> *"Insufficient CAE evidence in the knowledge base to answer this accurately."*

This is enforced two ways:

1. **Deterministic relevance gate** — rejects queries whose best match is too far in embedding space
2. **Strict system prompt** — the LLM is instructed to answer only from retrieved context and to state limitations explicitly


<div align="center">
<br/>

**Designed & Developed by : Balaram R**

⭐ Star this repo if you find it useful!

</div>
