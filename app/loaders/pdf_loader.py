"""
app/loaders/pdf_loader.py
=========================
PDF → text extraction with OCR fallback.

Some PDFs (e.g. scanned books, or PDFs using Type3/custom font encodings)
do not yield readable text via normal extraction. For those, we render each
page to an image with PyMuPDF and run OCR (RapidOCR) to recover the content.

Strategy per page:
  1. Try PyMuPDF text extraction.
  2. If the extracted text looks garbled (low ratio of readable characters,
     or the classic Type3 "/G<digits>" glyph pattern), fall back to OCR.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from app.loaders.base import BaseLoader
from app.loaders.loader_factory import LoaderFactory
from app.models.schemas import FileMetadata, RawDocument
from app.utils.logging import get_logger

logger = get_logger("loaders.pdf")

# Classic Type3 / custom-encoding garbage: sequences like "/G84/G101/G120"
_GARBAGE_RE = re.compile(r"/[A-Za-z]\d{2,3}")


def _looks_garbled(text: str) -> bool:
    """Heuristic: is this extracted text actually readable?"""
    if not text or not text.strip():
        return True
    # Type3 glyph garbage pattern
    if _GARBAGE_RE.search(text):
        return True
    # Ratio of readable alphanumeric characters
    alnum = sum(c.isalnum() for c in text)
    total = len(text)
    if total == 0:
        return True
    return (alnum / total) < 0.35


def _ocr_page(page, ocr, dpi: int = 200) -> str:
    """Render a PyMuPDF page to an image and OCR it with a shared RapidOCR instance."""
    # Render page to a temp PNG
    with tempfile.TemporaryDirectory() as td:
        img_path = str(Path(td) / "page.png")
        pix = page.get_pixmap(dpi=dpi)
        pix.save(img_path)

        try:
            result, _ = ocr(img_path)
        except Exception as exc:
            logger.warning("OCR failed on a page: %s", exc)
            return ""

    if not result:
        return ""
    # result is a list of [box, text, confidence, ...]; join the text lines
    lines = [str(item[1]) for item in result if len(item) > 1 and item[1]]
    return "\n".join(lines)


class PDFLoader(BaseLoader):
    supported_extensions = [".pdf"]

    def load(self, filepath: str, metadata: FileMetadata, **kw) -> list[RawDocument]:
        try:
            import pymupdf
        except ImportError:
            logger.error("pip install pymupdf")
            return []

        ocr_fallback = kw.get("ocr_fallback", True)
        dpi = int(kw.get("ocr_dpi", 200))

        try:
            doc = pymupdf.open(filepath)
        except Exception as exc:
            logger.error("Failed to open PDF %s: %s", filepath, exc)
            return []

        metadata.page_count = doc.page_count
        result: list[RawDocument] = []
        ocr_pages = 0

        # Load the OCR model once (it is expensive to initialize per page).
        ocr = None
        if ocr_fallback:
            try:
                from rapidocr_onnxruntime import RapidOCR
                # Enable CUDA (GPU) for the OCR models. onnxruntime-gpu needs the
                # CUDA/cuDNN DLLs on the DLL search path; they live in the
                # onnxruntime capi folder (copied there from the CUDA runtime).
                try:
                    import onnxruntime as ort
                    capi = os.path.join(os.path.dirname(ort.__file__), "capi")
                    if os.path.isdir(capi):
                        os.add_dll_directory(capi)
                    use_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
                except Exception:
                    use_cuda = False
                if use_cuda:
                    ocr = RapidOCR(det_use_cuda=True, rec_use_cuda=True, cls_use_cuda=True)
                    logger.info("OCR using CUDA (GPU) provider")
                else:
                    ocr = RapidOCR()
            except ImportError:
                logger.error("rapidocr_onnxruntime not installed; OCR disabled")
                ocr = None

        for i in range(doc.page_count):
            page = doc[i]
            text = page.get_text() or ""

            used_ocr = False
            if ocr is not None and _looks_garbled(text):
                ocr_text = _ocr_page(page, ocr, dpi=dpi)
                if ocr_text.strip():
                    text = ocr_text
                    used_ocr = True
                    ocr_pages += 1
                else:
                    logger.warning("OCR produced no text for page %d of %s", i, filepath)

            page_meta = FileMetadata(**metadata.to_dict())
            page_meta.doc_index = i
            if used_ocr:
                page_meta.extra["extraction"] = "ocr"
            result.append(RawDocument(content=text, metadata=page_meta,
                                      source_loader="PyPDFLoader"))

        doc.close()
        logger.info("Loaded %d pages from %s (OCR used on %d pages)",
                    len(result), filepath, ocr_pages)
        return result


LoaderFactory.register_many(PDFLoader.supported_extensions, PDFLoader())