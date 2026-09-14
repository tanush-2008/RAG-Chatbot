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

import os
import shutil

from logging_config import get_logger

log = get_logger(__name__)

_tesseract_checked = False
_tesseract_available = False

# Common Windows install locations for Tesseract that the official installer
# does NOT reliably add to PATH - a well-known gotcha that otherwise makes a
# genuine local install look "not available" to this module.
_WINDOWS_FALLBACK_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)


def _locate_tesseract_binary() -> str | None:
    """Find the Tesseract executable even if it isn't on PATH.

    Checks PATH first (the normal case, e.g. Linux/Mac package managers),
    then an explicit TESSERACT_CMD override, then common Windows install
    locations the official installer leaves off PATH by default.
    """
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    explicit = os.getenv("TESSERACT_CMD")
    if explicit and os.path.isfile(explicit):
        return explicit
    for candidate in _WINDOWS_FALLBACK_PATHS:
        if os.path.isfile(candidate):
            return candidate
    return None


def is_available() -> bool:
    """Whether the Tesseract binary is installed and callable, cached after first check."""
    global _tesseract_checked, _tesseract_available
    if not _tesseract_checked:
        try:
            import pytesseract

            binary = _locate_tesseract_binary()
            if binary:
                pytesseract.pytesseract.tesseract_cmd = binary
            version = pytesseract.get_tesseract_version()
            _tesseract_available = True
            log.info("Tesseract OCR engine available (version %s, at %s).", version, binary or "PATH")
        except Exception as exc:
            _tesseract_available = False
            log.info("Tesseract OCR engine not available (%s); OCR will be skipped.", exc)
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
