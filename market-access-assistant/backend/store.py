"""
In-memory document store.

Kept deliberately simple (a dict) for this test - see ARCHITECTURE.md for what would
change for production (persistent storage, e.g. Postgres for metadata + a vector store
for chunks, so the corpus survives a restart and scales beyond memory).
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional

from backend.models import Document
from backend.retrieval import Retriever


def slugify(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    value = re.sub(r"[-\s]+", "-", value)
    return value or "document"


class DocumentStore:
    def __init__(self, retriever: Retriever) -> None:
        self._documents: Dict[str, Document] = {}
        self._retriever = retriever

    def add(self, title: str, text: str, country: Optional[str], source_type: str) -> Document:
        base_id = slugify(title)
        document_id = base_id
        suffix = 1
        while document_id in self._documents:
            suffix += 1
            document_id = f"{base_id}-{suffix}"

        doc = Document(document_id=document_id, title=title, country=country, source_type=source_type, text=text)
        self._documents[document_id] = doc
        self._retriever.add_document(document_id, text)
        return doc

    def get(self, document_id: str) -> Optional[Document]:
        return self._documents.get(document_id)

    def list(self) -> List[Document]:
        return list(self._documents.values())

    def exists_by_id(self, document_id: str) -> bool:
        return document_id in self._documents

    def __len__(self) -> int:
        return len(self._documents)