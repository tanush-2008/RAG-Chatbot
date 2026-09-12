"""Module 2 optional extension: OCR fallback for scanned/image-only PDF pages.

The brief calls this out explicitly: "For scanned documents, add OCR only as
an optional extension." It's opt-in (a checkbox in the UI, off by default)
and degrades gracefully: if the Tesseract OCR engine isn't installed on the
host, `is_available()` returns False and document_loader.py falls back to
its normal behavior of skipping empty/image-only pages - the app never
crashes for lacking an external binary that pip can't install for you.

Setup (only needed to actually use OCR):
    - Install the Tesseract binary (not a pip package): apt-get install
      tesseract-ocr (Linux), brew install tesseract (Mac), or the installer
      at https://github.com/UB-Mannheim/tesseract/wiki (Windows).
    - pip install pytesseract pypdfium2 (already in requirements.txt).
"""

from __future__ import annotations

_tesseract_checked = False
_tesseract_available = False


def is_available() -> bool:
    """Whether the Tesseract binary is installed and callable, cached after first check."""
    global _tesseract_checked, _tesseract_available
    if not _tesseract_checked:
        try:
            import pytesseract

            pytesseract.get_tesseract_version()
            _tesseract_available = True
        except Exception:
            _tesseract_available = False
        _tesseract_checked = True
    return _tesseract_available


def ocr_pdf_page(pdf_bytes: bytes, page_number: int, dpi: int = 200) -> str:
    """OCR a single 1-indexed page of a PDF given as raw bytes.

    Returns the extracted text, or "" if OCR isn't available or fails for
    this page (e.g. a corrupt page) - callers should treat that the same as
    "no text on this page", not as an error.
    """
    if not is_available():
        return ""
    try:
        import pypdfium2 as pdfium
        import pytesseract

        pdf = pdfium.PdfDocument(pdf_bytes)
        try:
            page = pdf[page_number - 1]
            bitmap = page.render(scale=dpi / 72)
            pil_image = bitmap.to_pil()
            return pytesseract.image_to_string(pil_image).strip()
        finally:
            pdf.close()
    except Exception:
        return ""
