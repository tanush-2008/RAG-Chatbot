"""Tests for the optional OCR extension (ocr.py, document_loader.py's enable_ocr).

Tesseract isn't installed in this dev environment, so these tests verify
the graceful-degradation path (no crash, pages skipped exactly as without
OCR) rather than actual text recognition - that path only differs once a
real Tesseract install makes ocr.is_available() return True, which the
image-only PDF fixture below is built to exercise if it ever is.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ocr
from document_loader import extract_pages, load_documents

DOCS_DIR = Path(__file__).resolve().parent.parent / "documents"


def _image_only_pdf_bytes() -> bytes:
    """Build a minimal PDF whose only page has an image, no text layer -
    i.e. what a scanned document looks like to a text extractor."""
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (400, 100), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 40), "Scanned Page Sample Text", fill="black")

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    c.drawInlineImage(img, 50, 700, width=300, height=75)
    c.showPage()
    c.save()
    return buf.getvalue()


def test_is_available_does_not_raise():
    # Whatever the environment, this must never crash - it's checked on
    # every upload if the user ticks the OCR checkbox.
    assert isinstance(ocr.is_available(), bool)


def test_ocr_pdf_page_returns_empty_string_when_unavailable(monkeypatch):
    monkeypatch.setattr(ocr, "is_available", lambda: False)
    assert ocr.ocr_pdf_page(b"not a real pdf", 1) == ""


def test_extract_pages_without_ocr_skips_image_only_page():
    pdf_bytes = _image_only_pdf_bytes()
    pages = extract_pages(pdf_bytes, "scanned.pdf", enable_ocr=False)
    assert pages == []


def test_extract_pages_with_ocr_degrades_gracefully_when_unavailable(monkeypatch):
    monkeypatch.setattr(ocr, "is_available", lambda: False)
    pdf_bytes = _image_only_pdf_bytes()
    # Should behave exactly like enable_ocr=False, not raise, when Tesseract
    # isn't installed - this is the realistic CI/most-users case.
    pages = extract_pages(pdf_bytes, "scanned.pdf", enable_ocr=True)
    assert pages == []


def test_extract_pages_normal_pdf_unaffected_by_ocr_flag():
    data = (DOCS_DIR / "Policy.pdf").read_bytes()
    without_ocr = load_documents([("Policy.pdf", data, len(data))], enable_ocr=False)
    with_ocr = load_documents([("Policy.pdf", data, len(data))], enable_ocr=True)
    assert len(without_ocr) == len(with_ocr)
    assert all(not p.via_ocr for p in without_ocr)
    assert all(not p.via_ocr for p in with_ocr)  # real text present, OCR never triggered


def test_ocr_pdf_page_marks_via_ocr_when_available(monkeypatch):
    """Simulate a working Tesseract install to verify the plumbing (via_ocr
    flag, text propagation) without depending on a real OCR engine."""
    monkeypatch.setattr(ocr, "is_available", lambda: True)
    monkeypatch.setattr(ocr, "ocr_pdf_page", lambda pdf_bytes, page_number, dpi=200: "Recognized text")

    pdf_bytes = _image_only_pdf_bytes()
    pages = extract_pages(pdf_bytes, "scanned.pdf", enable_ocr=True)
    assert len(pages) == 1
    assert pages[0].text == "Recognized text"
    assert pages[0].via_ocr is True
