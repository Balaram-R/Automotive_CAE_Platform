"""
Manual GPU OCR for a scanned/Type3-encoded PDF.

Usage:
    python scripts/ocr_pdf_manual.py <pdf_path> [--dpi 150] [--out out_dir]

Renders each page with PyMuPDF and OCRs it with RapidOCR on the GPU (CUDA),
saving per-page text to <out_dir>/page_XXXX.txt so you can watch progress.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure the CUDA DLLs (in the onnxruntime capi folder) are on the DLL search path.
def _setup_cuda():
    try:
        import onnxruntime as ort
        capi = os.path.join(os.path.dirname(ort.__file__), "capi")
        if os.path.isdir(capi):
            os.add_dll_directory(capi)
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="Path to the PDF file")
    ap.add_argument("--dpi", type=int, default=150, help="Render DPI (default 150)")
    ap.add_argument("--out", default="ocr_output", help="Output directory")
    ap.add_argument("--start", type=int, default=0, help="Start page index")
    ap.add_argument("--end", type=int, default=None, help="End page index (exclusive)")
    args = ap.parse_args()

    use_cuda = _setup_cuda()
    print(f"CUDA available: {use_cuda}")

    import pymupdf
    from rapidocr_onnxruntime import RapidOCR

    doc = pymupdf.open(args.pdf)
    total = doc.page_count
    end = args.end if args.end is not None else total
    print(f"PDF: {args.pdf} | pages: {total} | OCR range: [{args.start}, {end})")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ocr = RapidOCR(det_use_cuda=True, rec_use_cuda=True, cls_use_cuda=True) if use_cuda else RapidOCR()

    t_start = time.time()
    for i in range(args.start, min(end, total)):
        page = doc[i]
        # Render to temp image
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            img = str(Path(td) / "page.png")
            page.get_pixmap(dpi=args.dpi).save(img)
            result, _ = ocr(img)

        lines = [str(item[1]) for item in result if len(item) > 1 and item[1]] if result else []
        text = "\n".join(lines)
        (out_dir / f"page_{i:04d}.txt").write_text(text, encoding="utf-8")

        elapsed = time.time() - t_start
        rate = elapsed / (i - args.start + 1)
        remaining = rate * (end - i - 1)
        print(f"[{i+1}/{end}] page {i}: {len(text)} chars | {rate:.1f}s/page | ETA {remaining/60:.1f} min")

    doc.close()
    print(f"Done. Output in {out_dir}")


if __name__ == "__main__":
    main()