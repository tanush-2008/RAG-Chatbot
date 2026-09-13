"""Module 1 & 2: Document upload validation and PDF text extraction."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import BinaryIO, Iterable, Union

from pypdf import PdfReader

import ocr

MAX_FILE_SIZE_MB = 25
ALLOWED_EXTENSIONS = (".pdf",)

_WORD_PER_LINE_THRESHOLD = 3.0  # avg words/line below this looks like a layout artifact


def _normalize_extracted_text(text: str) -> str:
    """Clean up whitespace artifacts from pypdf's text extraction.

    Some PDF layouts (tables, diagrams, text boxes with absolute positioning)
    make pypdf emit one word per line instead of natural sentences - the
    content is all there, but "Beginner\\nembedding\\nmodel\\nall-MiniLM-L6-v2"
    reads as noise to a sentence embedding model instead of the phrase it is.
    Detected via a simple heuristic (average words per non-blank line) so
    normally-formatted prose (which already reads fine) is left with its
    paragraph structure intact.
    """
    lines = [line for line in text.split("\n") if line.strip()]
    if not lines:
        return text.strip()

    avg_words_per_line = sum(len(line.split()) for line in lines) / len(lines)
    if avg_words_per_line < _WORD_PER_LINE_THRESHOLD:
        return re.sub(r"\s+", " ", text).strip()

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class DocumentValidationError(ValueError):
    """Raised when an uploaded file fails validation."""


@dataclass
class PageDocument:
    """A single non-empty page of text plus its source metadata."""

    text: str
    source: str
    page: int  # 1-indexed page number
    via_ocr: bool = field(default=False)


def validate_file(filename: str, size_bytes: int) -> None:
    """Validate file extension and size before processing.

    Raises DocumentValidationError with a user-facing message on failure.
    """
    if not filename.lower().endswith(ALLOWED_EXTENSIONS):
        raise DocumentValidationError(
            f"'{filename}' is not a PDF file. Only {ALLOWED_EXTENSIONS} files are accepted."
        )
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if size_bytes > max_bytes:
        raise DocumentValidationError(
            f"'{filename}' is {size_bytes / (1024 * 1024):.1f} MB, "
            f"which exceeds the {MAX_FILE_SIZE_MB} MB limit."
        )
    if size_bytes == 0:
        raise DocumentValidationError(f"'{filename}' is empty.")


def extract_pages(
    file_obj: Union[BinaryIO, bytes], source_name: str, enable_ocr: bool = False
) -> list[PageDocument]:
    """Read every page of a PDF, skipping empty pages, and keep page metadata.

    file_obj can be a file-like object (e.g. Streamlit's UploadedFile) or raw bytes.

    If `enable_ocr` is set and a page has no extractable text (a scanned or
    image-only page), falls back to OCR (ocr.py) - which itself degrades to a
    no-op if the Tesseract binary isn't installed, so this is always safe to
    pass even when OCR isn't set up.
    """
    if isinstance(file_obj, (bytes, bytearray)):
        raw_bytes = bytes(file_obj)
    else:
        raw_bytes = file_obj.read()
    file_obj = io.BytesIO(raw_bytes)

    pages: list[PageDocument] = []
    try:
        reader = PdfReader(file_obj)
        # pypdf parses the cross-reference/page tree lazily, so a malformed
        # PDF (e.g. a corrupted trailer) can raise only once .pages is
        # actually accessed, not at construction - force that resolution
        # here so the failure is caught as a friendly validation error
        # instead of an unhandled exception later in the loop below.
        num_pages = len(reader.pages)
    except Exception as exc:
        raise DocumentValidationError(f"Could not read '{source_name}': {exc}") from exc

    for page_number in range(1, num_pages + 1):
        try:
            page = reader.pages[page_number - 1]
            text = page.extract_text() or ""
        except Exception:
            # A single corrupt page shouldn't abort the whole file - skip it
            # safely, same as an empty page (falls through to OCR below).
            text = ""
        text = _normalize_extracted_text(text)

        via_ocr = False
        if not text and enable_ocr:
            text = ocr.ocr_pdf_page(raw_bytes, page_number).strip()
            via_ocr = bool(text)

        if not text:
            continue
        pages.append(PageDocument(text=text, source=source_name, page=page_number, via_ocr=via_ocr))

    return pages


def load_documents(
    files: Iterable[tuple[str, Union[BinaryIO, bytes], int]], enable_ocr: bool = False
) -> list[PageDocument]:
    """Validate and extract text from a batch of uploaded files.

    Each item in `files` is (filename, file_obj, size_bytes).
    Returns the combined list of PageDocument across all files.
    Raises DocumentValidationError on the first invalid file.
    """
    all_pages: list[PageDocument] = []
    for filename, file_obj, size_bytes in files:
        validate_file(filename, size_bytes)
        all_pages.extend(extract_pages(file_obj, filename, enable_ocr=enable_ocr))
    return all_pages
