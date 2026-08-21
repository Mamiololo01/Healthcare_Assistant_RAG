"""Text extraction for uploaded .pdf / .txt documents."""
from __future__ import annotations

import io

from pypdf import PdfReader


def extract_text_from_pdf_bytes(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def extract_text(filename: str, data: bytes) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf_bytes(data)
    # default: treat as UTF-8 text
    return data.decode("utf-8", errors="replace")