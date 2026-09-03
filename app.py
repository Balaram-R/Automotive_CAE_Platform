from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# ============================================================
# AUTOMOTIVE CAE INTELLIGENCE
# Streamlit frontend for the existing backend.
# ============================================================

ROOT = Path(__file__).resolve().parent
KB = ROOT / "knowledge_base"
CONFIG = ROOT / "configs" / "config.yaml"
MAIN = ROOT / "main.py"
LOG = ROOT / "logs" / "pipeline.log"

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".flac"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

TYPE_MAP = {
    ".pdf": "PDF", ".txt": "TXT", ".docx": "DOCX", ".doc": "DOC",
    ".pptx": "PPTX", ".ppt": "PPT", ".xlsx": "XLSX", ".xls": "XLS",
    ".csv": "CSV", ".md": "Markdown", ".html": "HTML", ".htm": "HTML",
    ".json": "JSON", ".xml": "XML", ".yaml": "YAML", ".yml": "YAML",
    ".py": "Python", ".cpp": "C++", ".c": "C", ".h": "C/C++ Header",
    ".java": "Java", ".sql": "SQL",
}
for ext in IMAGE_EXT: TYPE_MAP[ext] = "IMAGE"
for ext in AUDIO_EXT: TYPE_MAP[ext] = "AUDIO"
for ext in VIDEO_EXT: TYPE_MAP[ext] = "VIDEO"
UPLOAD_TYPES = sorted(ext.lstrip(".") for ext in TYPE_MAP)

# The original 0.45 default rejected valid knowledge-base matches around 0.64.
# Migrate that initial value once without overriding a user's later choice.
if "threshold_default_migrated" not in st.session_state:
    if st.session_state.get("relevance_threshold") in (None, 0.45):
        st.session_state["relevance_threshold"] = 0.70
    st.session_state["threshold_default_migrated"] = True


def kind(path: Path) -> str:
    return TYPE_MAP.get(path.suffix.lower(), "OTHER")


def fmt_size(n: int) -> str:
    if n < 1024: return f"{n} B"
    if n < 1024**2: return f"{n/1024:.1f} KB"
    if n < 1024**3: return f"{n/1024**2:.2f} MB"
    return f"{n/1024**3:.2f} GB"


@st.cache_data(ttl=5, show_spinner=False)
def files():
    if not KB.exists():
        return []
    out = []
    for p in KB.rglob("*"):
        if p.is_file() and not p.name.startswith("."):
            try:
                s = p.stat()
                out.append({
                    "name": p.name,
                    "relative": str(p.relative_to(KB)),
                    "type": kind(p),
                    "size": s.st_size,
                    "mtime": s.st_mtime,
                    "path": p,
                })
            except OSError:
                pass
    return sorted(out, key=lambda x: x["mtime"], reverse=True)


@st.cache_data(ttl=5, show_spinner=False)
def read_hashes():
    p = KB / ".file_hashes.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        return {}


# Placeholder / low-quality markers that indicate extraction did NOT produce
# usable CAE content (e.g. image-only decks, OCR-required placeholders).
_PLACEHOLDER_MARKERS = (
    "Image-only slide deck",
    "OCR is required",
    "no extractable text",
)


@st.cache_data(ttl=30, show_spinner=False)
def extraction_quality(filename: str) -> str:
    """Return 'ok', 'placeholder', or 'missing' for a file's stored chunks.

    Queries the vector store for the file's chunks and inspects their text.
    A file whose chunks are all placeholders (image-only deck, OCR-required)
    is flagged so the UI can surface it as a warning instead of "TRACKED".
    """
    try:
        from app.utils.config import load_config
        from app.vectorstore.base import VectorStoreFactory
        cfg = load_config()
        vs = VectorStoreFactory.create(
            cfg.vector_store.backend,
            collection_name=cfg.vector_store.collection_name,
            persist_directory=cfg.vector_store.persist_directory,
            host=cfg.vector_store.host, port=cfg.vector_store.port,
            api_key=cfg.vector_store.api_key, index_name=cfg.vector_store.index_name,
        )
        res = vs._get().get(where={"filename": filename})
        docs = res.get("documents") or []
        if not docs:
            return "missing"
        # If every chunk is a placeholder marker, extraction produced no real data.
        if all(any(m in (d or "") for m in _PLACEHOLDER_MARKERS) for d in docs):
            return "placeholder"
        return "ok"
    except Exception:
        return "unknown"


def backend_cmd(*args, timeout=3600):
    if not MAIN.exists():
        return False, "main.py not found."
    cmd = [sys.executable, str(MAIN), *args]
    if CONFIG.exists():
        cmd += ["--config", str(CONFIG)]
    try:
        r = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout
        )
        text = ((r.stdout or "") + ("\n" + r.stderr if r.stderr else "")).strip()
        return r.returncode == 0, text
    except subprocess.TimeoutExpired:
        return False, "Operation timed out."
    except Exception as e:
        return False, str(e)


@st.cache_data(ttl=30, show_spinner=False)
def status_output():
    ok, text = backend_cmd("status", timeout=60)
    return ok, text


def parse_chat_result(output: str) -> tuple[str, list[dict]]:
    """Extract the structured query response, excluding command and model logs."""
    for line in output.splitlines():
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict) and "answer" in result:
            answer = str(result["answer"])
            no_evidence_phrases = (
                "don't have enough information",
                "do not have enough information",
                "not available in the knowledge base",
                "cannot answer this accurately",
                "insufficient cae evidence",
            )
            if any(phrase in answer.lower() for phrase in no_evidence_phrases):
                return answer, []
            sources = []
            for source in result.get("sources", []):
                sources.append({
                    "filename": str(source.get("filename", "")),
                    "score": source.get("score"),
                    "preview": str(source.get("text_preview", ""))[:200],
                })
            # Deduplicate by filename, keep first
            seen = set()
            unique = []
            for s in sources:
                if s["filename"] and s["filename"] not in seen:
                    seen.add(s["filename"])
                    unique.append(s)
            return answer, unique
    return "Unable to read the assistant response.", []


def parse_status(text, command_ok=False):
    data = {
        "backend": "ONLINE" if MAIN.exists() else "OFFLINE",
        "vector": "UNKNOWN",
        "vectors": 0,
        "embedding": "—",
        "dimension": "—",
        "collection": "—",
    }
    for line in text.splitlines():
        low = line.lower()
        if "backend" in low and ":" in line:
            data["vector"] = line.split(":", 1)[1].strip().upper()
        elif "collection" in low and ":" in line:
            data["collection"] = line.split(":", 1)[1].strip()
        elif "embedding" in low and ":" in line:
            data["embedding"] = line.split(":", 1)[1].strip()
        elif "dimension" in low and ":" in line:
            data["dimension"] = line.split(":", 1)[1].strip()
        elif "vectors" in low and ":" in line:
            m = re.search(r"([\d,]+)", line.split(":", 1)[1])
            if m: data["vectors"] = int(m.group(1).replace(",", ""))
    data["vector"] = "ONLINE" if command_ok else data["vector"]
    return data


def log_lines(limit=80):
    if not LOG.exists():
        return []
    try:
        return LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
    except Exception:
        return []


def activity():
    rows = []
    patterns = [
        ("File discovered", r"Scanned\s+(\d+)\s+files"),
        ("PDF indexed", r"Loaded\s+(\d+)\s+pages\s+from\s+(.+\.pdf)"),
        ("Video processing", r"(?:Video transcription failed|Frame extraction failed)\s+(.+\.mp4)"),
        ("OCR", r"OCR failed\s+(.+)"),
        ("Error", r"\|\s*(ERROR|WARNING)\s*\|"),
    ]
    for line in reversed(log_lines()):
        parts = line.split(" | ")
        stamp = parts[0] if parts else ""
        msg = parts[-1] if parts else line
        label = None
        detail = msg
        if "Scanned " in msg:
            label = "Folder scan"
        elif "Loaded " in msg and "pages from" in msg:
            label = "PDF loaded"
        elif "Video transcription failed" in msg:
            label = "Audio extraction"
        elif "Frame extraction failed" in msg:
            label = "Frame extraction"
        elif "OCR failed" in msg:
            label = "Image OCR"
        elif "DONE in" in msg:
            label = "Ingestion completed"
        elif "ERROR" in line:
            label = "Error"
        if label:
            rows.append((stamp[-8:] if len(stamp) >= 8 else stamp, label, detail))
        if len(rows) >= 7:
            break
    return rows


@st.cache_data(show_spinner=False)
def image_data(path: Path):
    try:
        ext = path.suffix.lower()
        mime = "image/png" if ext == ".png" else "image/webp" if ext == ".webp" else "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"
    except Exception:
        return None


st.set_page_config(
    page_title="Automotive CAE Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(r"""
<style>
.stApp {
  background:#050d14;
  color:#e7f0f5;
}
[data-testid="stHeader"] {background:rgba(5,13,20,.94);}
.block-container {max-width:1540px;padding-top:1.15rem;}
section[data-testid="stSidebar"] {
  background:#071019;
  border-right:1px solid #173142;
}
section[data-testid="stSidebar"] [data-testid="stElementContainer"] {
  text-align:left;
}
section[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.footer-note) {
  position:fixed;
  left:52px;
  bottom:24px;
  width:240px;
  z-index:10;
  text-align:left;
}
.brand {padding:10px 8px 28px;text-align:left;}
.brand-car {font-size:34px;color:#2bd3ff;line-height:1;}
.brand-name {font-weight:800;letter-spacing:.1em;font-size:20px;line-height:1.2;}
.brand-sub {font-size:13px;color:#2bd3ff;letter-spacing:.12em;line-height:1.45;margin-top:6px;}
.navtitle {font-size:15px;color:#647f8d;letter-spacing:.13em;text-transform:uppercase;margin:14px 0 11px;text-align:left;}
.footer-note {color:#58707d;font-size:15px;line-height:1.55;margin-top:28px;text-align:left;}
.footer-note b {color:#28c9ff;font-size:18px;}
section[data-testid="stSidebar"] [data-testid="stRadio"] label {padding:4px 0;}
section[data-testid="stSidebar"] [data-testid="stRadio"] label p {font-size:19px;}
section[data-testid="stSidebar"] [data-testid="stRadio"] {text-align:left;}
section[data-testid="stSidebar"] [data-testid="stRadio"] label {display:flex;justify-content:flex-start;}
section[data-testid="stSidebar"] .systemrow {font-size:16px;padding:12px 0;}
.kicker {color:#2bd3ff;font-size:10px;letter-spacing:.18em;text-transform:uppercase;font-weight:700;}
.subtitle {color:#78919e;font-size:13px;margin-top:-7px;}
.panel {
  background:linear-gradient(180deg,#0b1721,#08131c);
  border:1px solid #173142;border-radius:12px;padding:15px;
}
.panelhead {display:flex;justify-content:space-between;align-items:center;margin-bottom:9px;}
.panelhead b {font-size:13px;letter-spacing:.04em;}
.muted {color:#6f8794;font-size:11px;}
.metric {
  background:linear-gradient(180deg,#0c1923,#08131c);
  border:1px solid #173142;border-radius:10px;padding:13px 15px;
}
.metric-label {font-size:9px;color:#718b99;letter-spacing:.14em;text-transform:uppercase;}
.metric-value {font-size:23px;font-weight:800;margin-top:3px;}
.metric-hint {font-size:9px;color:#5f7886;margin-top:2px;}
.hero {
  position:relative;width:50%;margin:0 auto;aspect-ratio:1654 / 860;overflow:hidden;border:1px solid #173142;
  border-radius:12px;background:#06111a;
}
.hero img {width:100%;height:100%;object-fit:cover;opacity:.82;filter:saturate(.82) contrast(1.12);}
.hero-overlay {
  position:absolute;inset:0;
  background:linear-gradient(90deg,rgba(4,12,18,.92),rgba(4,12,18,.22) 60%,rgba(4,12,18,.35)),
             linear-gradient(0deg,rgba(4,12,18,.72),transparent 50%);
}
.scan {
  position:absolute;left:0;right:0;top:5%;height:2px;z-index:3;
  background:linear-gradient(90deg,transparent,#37e79a 25%,#eafff7 50%,#37e79a 75%,transparent);
  box-shadow:0 0 8px #37e79a,0 0 25px rgba(55,231,154,.55);
  animation:laser 7s ease-in-out infinite;
}
.scan::after {content:"";position:absolute;left:0;right:0;top:-25px;height:50px;
  background:linear-gradient(transparent,rgba(55,231,154,.10),transparent);}
@keyframes laser {0%,100%{top:6%}50%{top:93%}}
.scantext {position:absolute;left:17px;top:14px;color:#55efa9;font-size:9px;letter-spacing:.15em;z-index:4;}
.hero-copy {position:absolute;left:102px;bottom:22px;z-index:4;max-width:650px;}
.hero-copy .tiny {color:#2bd3ff;font-size:9px;letter-spacing:.16em;text-transform:uppercase;font-weight:700;}
.hero-copy h2 {font-size:26px;margin:5px 0 5px;}
.hero-copy p {font-size:12px;color:#b0c1ca;line-height:1.5;margin:0;}
.badge {font-size:8px;font-weight:800;letter-spacing:.08em;padding:3px 6px;border-radius:4px;border:1px solid currentColor;}
.green {color:#36e48d;background:rgba(54,228,141,.05);}
.amber {color:#ffb84e;background:rgba(255,184,78,.05);}
.red {color:#ff6370;background:rgba(255,99,112,.05);}
.blue {color:#42cfff;background:rgba(66,207,255,.05);}
.fileline {
  display:grid;grid-template-columns:34px minmax(0,1fr) 62px 86px;
  gap:9px;align-items:center;padding:9px 0;border-top:1px solid #142a38;
}
.ficon {width:29px;height:29px;border-radius:6px;background:#102432;color:#45d7ff;display:grid;place-items:center;font-weight:800;font-size:11px;}
.fname {font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.fmeta {font-size:9px;color:#617987;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.activity {display:grid;grid-template-columns:50px 18px minmax(0,1fr);gap:7px;align-items:start;padding:8px 0;border-bottom:1px solid #142a38;}
.activity-time {color:#607b89;font-size:9px;padding-top:2px;}
.activity-dot {width:8px;height:8px;border-radius:50%;background:#35e38d;margin-top:3px;box-shadow:0 0 8px rgba(53,227,141,.45);}
.activity-dot.problem {background:#ff6370;box-shadow:0 0 8px rgba(255,99,112,.45);}
.activity-title {font-size:10px;font-weight:700;}
.activity-detail {font-size:9px;color:#607b89;margin-top:2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.pipe {
  display:flex;gap:10px;align-items:center;padding:10px 0;border-bottom:1px solid #142a38;
}
.pipe-dot {width:22px;height:22px;border:1px solid #2d5267;border-radius:50%;display:grid;place-items:center;font-size:8px;color:#7b96a3;}
.pipe-dot.done {border-color:#36e48d;color:#36e48d;}
.pipe-dot.active {border-color:#ffb84e;color:#ffb84e;box-shadow:0 0 12px rgba(255,184,78,.18);}
.pipe-name {font-size:10px;font-weight:700;}
.pipe-desc {font-size:8px;color:#607987;margin-top:2px;}
.pipe-state {margin-left:auto;font-size:8px;color:#718994;}
.systemrow {display:flex;justify-content:space-between;padding:9px 0;border-bottom:1px solid #142a38;font-size:10px;}
.systemok {color:#36e48d;}
.systemwarn {color:#ffb84e;}
.copyright {border-top:1px solid #173142;margin-top:20px;padding-top:10px;color:#536c79;font-size:9px;display:flex;justify-content:space-between;}
.copyright b {color:#2bd3ff;}
.source {
  background:#0c1923;border:1px solid #173142;border-radius:8px;
  padding:8px 12px;margin:4px 0;font-size:12px;color:#9fb8c4;
}
.source b {color:#42cfff;font-size:11px;display:block;margin-bottom:3px;}
.source .src-score {color:#36e48d;font-size:10px;margin-left:6px;}
.source .src-preview {color:#78919e;font-size:11px;margin-top:3px;line-height:1.4;}
.media-preview {border:1px solid #173142;border-radius:10px;overflow:hidden;margin:8px 0;}
.media-preview img, .media-preview video {width:100%;max-height:320px;object-fit:contain;background:#06111a;}
.media-preview .media-caption {padding:6px 10px;font-size:10px;color:#78919e;background:#0c1923;}
div[data-testid="stMetric"] {background:transparent;}
button[kind="primary"] {background:#0c73ad;border-color:#1d9bd8;}
@media (prefers-reduced-motion: reduce) {.scan{animation:none;top:50%;}}
</style>
""", unsafe_allow_html=True)


def sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="brand">
          <div class="brand-car">⌁</div>
          <div class="brand-name">AUTOMOTIVE</div>
          <div class="brand-name" style="color:#2bd3ff">CAE INTELLIGENCE</div>
          <div class="brand-sub">ENGINEERING KNOWLEDGE PLATFORM</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="navtitle">Workspace</div>', unsafe_allow_html=True)
        page = st.radio("", ["Home", "Ingest Pipeline", "Chat", "System"], label_visibility="collapsed")
        st.markdown(
            '<div class="footer-note">Designed by<br><b>Balaram R</b></div>',
            unsafe_allow_html=True,
        )
    return page


def home():
    fs = files()
    status_ok, raw_status = status_output()
    stat = parse_status(raw_status, status_ok)

    st.markdown('<div class="kicker">Engineering Intelligence Platform</div>', unsafe_allow_html=True)
    st.title("Automotive CAE Intelligence")
    st.markdown('<div class="subtitle">A single workspace for engineering documents, media and retrieval.</div>', unsafe_allow_html=True)
    st.write("")

    processed = sum(1 for f in fs if f["type"] != "OTHER")
    types = {}
    for f in fs: types[f["type"]] = types.get(f["type"], 0) + 1

    m = st.columns(6)
    values = [
        ("TOTAL FILES", len(fs), "Auto-discovered"),
        ("PROCESSED", processed, "Supported files"),
        ("CHUNKS", stat["vectors"], "Vector records"),
        ("EMBEDDINGS", stat["vectors"], "Stored vectors"),
        ("IN PROGRESS", 0, "Current UI session"),
        ("FAILED", 0, "See pipeline/logs"),
    ]
    for c, (label, val, hint) in zip(m, values):
        with c:
            st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{val:,}</div><div class="metric-hint">{hint}</div></div>', unsafe_allow_html=True)

    st.write("")
    left = st.container()

    with left:
        preferred = ROOT / "assets" / "fea_vehicle_clean.png"
        img = preferred if preferred.exists() else next(
            (f["path"] for f in fs if f["type"] == "IMAGE"), None
        )
        if img:
            src = image_data(Path(img))
            st.markdown(f"""
            <div class="hero">
              <img src="{src}" alt="FEA vehicle structural visualization">
              <div class="hero-overlay"></div>
              <div class="scan"></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="panel" style="height:370px;display:grid;place-items:center"><span class="muted">Place a vehicle image inside knowledge_base/ to show it here.</span></div>', unsafe_allow_html=True)

    st.write("")
    a, b = st.columns([1.2, 1], gap="large")

    with a:
        st.markdown('<div class="panel"><div class="panelhead"><b>KNOWLEDGE BASE</b><span class="muted">Automatic discovery</span></div>', unsafe_allow_html=True)
        if not fs:
            st.info("No files detected.")
        for f in fs[:8]:
            ic = {"PDF":"PDF","TXT":"TXT","PPTX":"PPT","VIDEO":"▶","AUDIO":"♪","IMAGE":"IMG"}.get(f["type"], "·")
            st.markdown(f"""
            <div class="fileline">
              <div class="ficon">{ic}</div>
              <div><div class="fname">{f["name"]}</div><div class="fmeta">{f["relative"]} · {fmt_size(f["size"])}</div></div>
              <div class="muted">{f["type"]}</div>
              <div><span class="badge blue">DETECTED</span></div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with b:
        st.markdown('<div class="panel"><div class="panelhead"><b>FILE TYPES DISTRIBUTION</b><span class="muted">Current folder</span></div>', unsafe_allow_html=True)
        total = max(len(fs), 1)
        for t, n in sorted(types.items(), key=lambda x: -x[1]):
            pct = n / total * 100
            st.markdown(f"""
            <div style="margin:10px 0">
              <div style="display:flex;justify-content:space-between;font-size:9px"><span>{t}</span><span class="muted">{n} · {pct:.0f}%</span></div>
              <div style="height:5px;background:#122531;border-radius:3px;margin-top:5px"><div style="width:{pct:.1f}%;height:100%;background:#2bd3ff;border-radius:3px"></div></div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="copyright"><span>Automotive CAE Intelligence Platform</span><span>Designed &amp; Developed by <b>Balaram R</b></span></div>', unsafe_allow_html=True)


def ingest():
    fs = files()
    hashes = read_hashes()

    st.markdown('<div class="kicker">LangGraph Operations</div>', unsafe_allow_html=True)
    st.title("Ingest Pipeline")
    st.markdown('<div class="subtitle">Control the existing ingestion workflow and inspect its real backend output.</div>', unsafe_allow_html=True)
    if notice := st.session_state.pop("upload_notice", None):
        st.success(notice)
    st.write("")

    st.subheader("Upload documents")
    with st.form("knowledge_base_upload", clear_on_submit=True):
        uploads = st.file_uploader(
            "Select documents to add to the knowledge base",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            help="Uploaded files are saved to knowledge_base and can then be processed with Run Ingestion.",
        )
        save_uploads = st.form_submit_button("Save to Knowledge Base", type="primary")

    if save_uploads:
        if not uploads:
            st.warning("Select at least one document before saving.")
        else:
            KB.mkdir(parents=True, exist_ok=True)
            saved = []
            for upload in uploads:
                original = Path(upload.name).name
                destination = KB / original
                suffix = 1
                while destination.exists():
                    destination = KB / f"{Path(original).stem}_{suffix}{Path(original).suffix}"
                    suffix += 1
                destination.write_bytes(upload.getvalue())
                saved.append(destination.name)
            st.session_state["upload_notice"] = (
                f"Saved {len(saved)} file(s) to knowledge_base: {', '.join(saved)}"
            )
            files.clear()
            read_hashes.clear()
            st.rerun()

    st.write("")

    l, r = st.columns([1.15, 1], gap="large")
    with l:
        st.markdown('<div class="panel"><div class="panelhead"><b>INGESTION FLOW</b><span class="badge blue">EXISTING BACKEND</span></div>', unsafe_allow_html=True)
        steps = [
            ("01","Folder scanning","Discover files recursively"),
            ("02","File detection","Determine file type"),
            ("03","Loader selection","Route to registered loader"),
            ("04","Content extraction","PDF / text / media"),
            ("05","Chunking","Prepare retrieval units"),
            ("06","Embeddings","Generate vectors"),
            ("07","Vector store","Persist knowledge"),
        ]
        for n, name, desc in steps:
            st.markdown(f'<div class="pipe"><div class="pipe-dot">{n}</div><div><div class="pipe-name">{name}</div><div class="pipe-desc">{desc}</div></div><div class="pipe-state">BACKEND</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.write("")
        if st.button("▶  RUN INGESTION", type="primary", use_container_width=True):
            with st.status("Running the existing ingestion graph...", expanded=True) as s:
                ok, output = backend_cmd("ingest", timeout=10800)
                if output:
                    st.code(output[-15000:], language="text")
                if ok:
                    s.update(label="Ingestion completed", state="complete")
                else:
                    s.update(label="Ingestion failed — backend output shown above", state="error")
            files.clear()
            read_hashes.clear()
            status_output.clear()
            st.rerun()

    with r:
        st.markdown('<div class="panel"><div class="panelhead"><b>DISCOVERED FILES</b><span class="muted">{0} files</span></div>'.format(len(fs)), unsafe_allow_html=True)
        for f in fs:
            tracked = f["relative"] in hashes or f["name"] in hashes or str(f["path"]) in hashes
            if tracked:
                q = extraction_quality(f["name"])
                if q == "placeholder":
                    status, cls = "WARNING", "red"
                elif q == "missing":
                    status, cls = "NO DATA", "red"
                else:
                    status, cls = "TRACKED", "green"
            else:
                status, cls = "NEW", "amber"
            st.markdown(f'<div class="fileline"><div class="ficon">{f["type"][:3]}</div><div><div class="fname">{f["name"]}</div><div class="fmeta">{f["relative"]}</div></div><div class="muted">{f["type"]}</div><div><span class="badge {cls}">{status}</span></div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def render_media_preview(filename: str):
    """Render an inline media preview for a knowledge-base file if it exists."""
    p = KB / filename
    if not p.exists():
        return ""
    ext = p.suffix.lower()
    if ext in IMAGE_EXT:
        src = image_data(p)
        if src:
            return f'<div class="media-preview"><img src="{src}" alt="{filename}"><div class="media-caption">{filename}</div></div>'
    elif ext in VIDEO_EXT:
        return f'<div class="media-preview"><video controls src="data:video/mp4;base64,{base64.b64encode(p.read_bytes()).decode()}"><div class="media-caption">{filename}</div></video></div>'
    return ""


def chat():
    st.markdown('<div class="kicker">Retrieval Augmented Generation</div>', unsafe_allow_html=True)
    st.title("CAE Assistant")
    st.markdown('<div class="subtitle">Ask the existing RAG backend about the engineering knowledge base.</div>', unsafe_allow_html=True)

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.markdown("**Sources**")
                for src in msg["sources"]:
                    score = src.get("score")
                    score_html = f'<span class="src-score">score: {score:.3f}</span>' if isinstance(score, (int, float)) else ""
                    preview = src.get("preview", "")
                    preview_html = f'<div class="src-preview">{preview}</div>' if preview else ""
                    st.markdown(f'<div class="source"><b>{src.get("filename", "")}</b>{score_html}{preview_html}</div>', unsafe_allow_html=True)

    q = st.chat_input("Ask about FEA, crashworthiness, materials, NVH, CFD, durability...")
    if q:
        st.session_state.chat.append({"role":"user","content":q})
        with st.chat_message("user"):
            st.markdown(q)

        with st.chat_message("assistant"):
            with st.spinner("Querying the existing RAG system..."):
                threshold = st.session_state.get("relevance_threshold", 0.70)
                top_k = st.session_state.get("retrieval_top_k", 5)
                ok, output = backend_cmd(
                    "query", q,
                    "--relevance-threshold", str(threshold),
                    "--top-k", str(top_k),
                    "--json",
                    timeout=300,
                )
            if ok:
                answer, sources = parse_chat_result(output)
                st.markdown(answer)
                if sources:
                    st.markdown("**Sources**")
                    for src in sources:
                        score = src.get("score")
                        score_html = f'<span class="src-score">score: {score:.3f}</span>' if isinstance(score, (int, float)) else ""
                        preview = src.get("preview", "")
                        preview_html = f'<div class="src-preview">{preview}</div>' if preview else ""
                        st.markdown(f'<div class="source"><b>{src.get("filename", "")}</b>{score_html}{preview_html}</div>', unsafe_allow_html=True)
                    # Show media previews for any source that is an image/video
                    for src in sources:
                        media = render_media_preview(src.get("filename", ""))
                        if media:
                            st.markdown(media, unsafe_allow_html=True)
                st.session_state.chat.append({"role":"assistant","content":answer,"sources":sources})
            else:
                st.error(output or "Query failed.")


def system():
    st.markdown('<div class="kicker">Platform Controls</div>', unsafe_allow_html=True)
    st.title("System")
    st.markdown('<div class="subtitle">Connection health and retrieval controls.</div>', unsafe_allow_html=True)
    st.write("")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Connection")
        st.markdown(
            f'<div class="systemrow"><span>Backend</span><span class="{"systemok" if MAIN.exists() else "systemwarn"}">● {"Connected" if MAIN.exists() else "Unavailable"}</span></div>'
            f'<div class="systemrow"><span>Knowledge Base</span><span class="{"systemok" if KB.exists() else "systemwarn"}">● {"Available" if KB.exists() else "Unavailable"}</span></div>',
            unsafe_allow_html=True,
        )
    with right:
        st.subheader("Retrieval Controls")
        threshold = st.slider("Threshold", 0.00, 1.00, 0.70, 0.01,
                              key="relevance_threshold")
        top_k = st.slider("Top-K sources", 1, 10, 5, 1, key="retrieval_top_k",
                          help="Number of source chunks used for each Chat answer.")
        tolerance = st.slider("Tolerance", 0.00, 1.00, 0.10, 0.01,
                               key="numerical_tolerance")
        st.caption(f"Threshold: {threshold:.2f}  •  Top-K: {top_k}  •  Tolerance: {tolerance:.2f}")

    st.write("")
    with st.expander("User Notes — recommended settings", expanded=True):
        st.markdown(
            "**Recommended start:** Threshold **0.70**, Top-K **5**, Tolerance **0.10**.\n\n"
            "- **Threshold** controls how closely a retrieved source must match before Chat answers. "
            "Lower is stricter; use **0.55–0.65** for precise engineering questions and **0.70–0.80** "
            "when you want broader answers.\n"
            "- **Top-K sources** controls how many source chunks Chat considers. Use **3–5** for focused "
            "answers, or **6–8** for broad topics such as crashworthiness or NVH.\n"
            "- **Tolerance** is reserved for future numerical-result validation. Keep it at **0.10** for now; "
            "it does not yet change Chat responses."
        )


page = sidebar()
if page == "Home":
    home()
elif page == "Ingest Pipeline":
    ingest()
elif page == "Chat":
    chat()
else:
    system()
