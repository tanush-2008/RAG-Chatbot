"""Module 1 & 2: Document upload validation and PDF text extraction."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import BinaryIO, Iterable, Union

from pypdf import PdfReader

MAX_FILE_SIZE_MB = 25
ALLOWED_EXTENSIONS = (".pdf",)


class DocumentValidationError(ValueError):
    """Raised when an uploaded file fails validation."""


@dataclass
class PageDocument:
    """A single non-empty page of text plus its source metadata."""

    text: str
    source: str
    page: int  # 1-indexed page number


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


def extract_pages(file_obj: Union[BinaryIO, bytes], source_name: str) -> list[PageDocument]:
    """Read every page of a PDF, skipping empty pages, and keep page metadata.

    file_obj can be a file-like object (e.g. Streamlit's UploadedFile) or raw bytes.
    """
    if isinstance(file_obj, (bytes, bytearray)):
        file_obj = io.BytesIO(file_obj)

    pages: list[PageDocument] = []
    try:
        reader = PdfReader(file_obj)
    except Exception as exc:  # pragma: no cover - defensive, surfaced to UI
        raise DocumentValidationError(f"Could not read '{source_name}': {exc}") from exc

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            # Corrupt or image-only page - skip safely instead of crashing the whole upload.
            continue
        text = text.strip()
        if not text:
            continue
        pages.append(PageDocument(text=text, source=source_name, page=page_number))

    return pages


def load_documents(files: Iterable[tuple[str, Union[BinaryIO, bytes], int]]) -> list[PageDocument]:
    """Validate and extract text from a batch of uploaded files.

    Each item in `files` is (filename, file_obj, size_bytes).
    Returns the combined list of PageDocument across all files.
    Raises DocumentValidationError on the first invalid file.
    """
    all_pages: list[PageDocument] = []
    for filename, file_obj, size_bytes in files:
        validate_file(filename, size_bytes)
        all_pages.extend(extract_pages(file_obj, filename))
    return all_pages
