"""
app/loaders/image_loader.py
===========================
Images → OCR (Tesseract / EasyOCR) with graceful fallbacks, plus an
optional Groq vision-model caption so RAG can answer questions about
photos (crash tests, FEA screenshots, etc.) that contain no readable text.
"""

from __future__ import annotations

import base64
import os

from app.loaders.base import BaseLoader
from app.loaders.loader_factory import LoaderFactory
from app.models.schemas import FileMetadata, RawDocument
from app.utils.logging import get_logger

logger = get_logger("loaders.image")


# ── Groq vision caption ──────────────────────────────────────────────────────

_VISION_SYSTEM_PROMPT = (
    "You are an Automotive CAE engineering assistant. Describe the image "
    "from a CAE / crash-test / structural-analysis engineering perspective. "
    "Note any visible vehicle deformation, structural damage, panel intrusion, "
    "energy-absorption behavior, test setup (barrier, impact direction), and "
    "any text, labels, ratings, or scorecards shown. Be factual and precise; "
    "do not invent measurements or ratings that are not visible in the image."
)


def _groq_vision_caption(filepath: str, model: str, api_key: str,
                         max_tokens: int = 800) -> str:
    """Caption *filepath* using a Groq multimodal model.

    Returns the model's text description, or "" on any failure (so the
    loader can fall back to OCR / metadata without crashing ingestion).
    """
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
    except ImportError:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        except Exception as exc:
            logger.debug("Vision client unavailable: %s", exc)
            return ""

    try:
        with open(filepath, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(filepath)[1].lower().lstrip(".") or "jpeg"
        if ext == "jpg":
            ext = "jpeg"
        data_url = f"data:image/{ext};base64,{b64}"

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _VISION_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": "Describe this image from an automotive CAE / crash-test engineering perspective."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]},
            ],
            max_tokens=max_tokens,
        )
        text = (response.choices[0].message.content or "").strip()
        return text
    except Exception as exc:
        logger.warning("Groq vision caption failed for %s: %s", filepath, exc)
        return ""


class ImageLoader(BaseLoader):
    supported_extensions = [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp"]

    def load(self, filepath: str, metadata: FileMetadata, **kw) -> list[RawDocument]:
        texts: list[str] = []

        # 1) Tesseract OCR (if the binary is installed)
        try:
            import pytesseract
            from PIL import Image
            text = pytesseract.image_to_string(Image.open(filepath)).strip()
            if text:
                texts.append(f"[OCR Text]\n{text}")
        except Exception as exc:
            logger.debug("Tesseract OCR unavailable for %s: %s", filepath, exc)

        # 2) EasyOCR fallback (if installed)
        if not texts:
            try:
                import easyocr
                reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                result = reader.readtext(filepath, detail=0)
                text = "\n".join(str(t).strip() for t in result if str(t).strip())
                if text:
                    texts.append(f"[OCR Text]\n{text}")
            except Exception as exc:
                logger.debug("EasyOCR unavailable for %s: %s", filepath, exc)

        # 3) Groq vision caption (multimodal) — describes the image content
        #    even when there is no readable text (photos, screenshots, etc.).
        cfg = kw.get("config")
        vision_enabled = True
        vision_model = "qwen/qwen3.6-27b"
        api_key = os.environ.get("GROQ_API_KEY", "")
        if cfg is not None:
            llm = getattr(cfg, "llm", None)
            if llm is not None:
                vision_enabled = getattr(llm, "vision_enabled", True)
                vision_model = getattr(llm, "vision_model", None) or vision_model
                api_key = getattr(llm, "api_key", None) or api_key

        if vision_enabled and api_key:
            caption = _groq_vision_caption(filepath, vision_model, api_key)
            if caption:
                texts.append(f"[Vision Description]\n{caption}")

        # 4) Image metadata fallback (dimensions, format, EXIF)
        if not texts:
            try:
                from PIL import Image
                from PIL.ExifTags import TAGS
                img = Image.open(filepath)
                meta_lines = [f"[Image: {metadata.filename}]",
                              f"Format: {img.format}",
                              f"Size: {img.width}x{img.height}px",
                              f"Mode: {img.mode}"]
                exif = img._getexif()
                if exif:
                    for tag_id, value in exif.items():
                        tag = TAGS.get(tag_id, tag_id)
                        if tag in ("ImageDescription", "Make", "Model", "Software", "DateTime"):
                            meta_lines.append(f"{tag}: {value}")
                texts.append("\n".join(meta_lines))
            except Exception as exc:
                logger.warning("Image metadata extraction failed %s: %s", filepath, exc)
                texts.append(f"[Image file: {metadata.filename}]")

        return [RawDocument(content="\n\n".join(texts), metadata=metadata, source_loader="ImageLoader")]


LoaderFactory.register_many(ImageLoader.supported_extensions, ImageLoader())