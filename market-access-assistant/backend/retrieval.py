"""
Chunking + retrieval layer.

Chunking strategy
------------------
Documents are split into paragraph-based chunks (blank-line-separated), then any chunk
longer than CHUNK_MAX_CHARS is further split on sentence boundaries with a small
overlap (CHUNK_OVERLAP_SENTENCES) so a fact that sits near a chunk boundary is not
stranded in only one chunk. Paragraph-based chunking was chosen over fixed-length
sliding windows because these source documents are already organised into short,
topically coherent sections ("Evidence Gaps", "Managed Entry Agreement", etc.) -
splitting on paragraphs keeps each chunk semantically self-contained, which both
improves retrieval precision and makes the returned "snippet" readable on its own.

Retrieval strategy
-------------------
TF-IDF + cosine similarity (scikit-learn) over the chunk corpus. This was chosen
instead of embeddings for this test because:
- It requires no external API call or model download, so it works fully offline and
  deterministically (important for a technical test that must run anywhere).
- For a small, fixed corpus of a handful of short documents, TF-IDF is a reasonable
  and interpretable baseline retrieval implementation.
- The `Retriever` class isolates this choice behind a small interface
  (`index`, `search`) so swapping in a vector store (FAISS/Chroma/pgvector) with
  embeddings is a contained change - see ARCHITECTURE.md, "What would need to change
  for production".
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CHUNK_MAX_CHARS = 700
CHUNK_OVERLAP_SENTENCES = 1


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    text: str


@dataclass
class RetrievedChunk(Chunk):
    score: float


def _split_sentences(paragraph: str) -> List[str]:
    # Simple sentence splitter - good enough for these structured, well-punctuated notes.
    parts = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    return [p for p in parts if p]


def chunk_document(document_id: str, text: str) -> List[Chunk]:
    """Split a document's text into paragraph-level chunks, sub-splitting long paragraphs."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: List[Chunk] = []
    idx = 0
    for para in paragraphs:
        if len(para) <= CHUNK_MAX_CHARS:
            chunks.append(Chunk(chunk_id=f"{document_id}::{idx}", document_id=document_id, text=para))
            idx += 1
            continue

        sentences = _split_sentences(para)
        current: List[str] = []
        current_len = 0
        for sentence in sentences:
            current.append(sentence)
            current_len += len(sentence) + 1
            if current_len >= CHUNK_MAX_CHARS:
                chunks.append(
                    Chunk(chunk_id=f"{document_id}::{idx}", document_id=document_id, text=" ".join(current))
                )
                idx += 1
                # keep a small overlap so context isn't lost at the boundary
                current = current[-CHUNK_OVERLAP_SENTENCES:] if CHUNK_OVERLAP_SENTENCES else []
                current_len = sum(len(s) + 1 for s in current)
        if current:
            chunks.append(
                Chunk(chunk_id=f"{document_id}::{idx}", document_id=document_id, text=" ".join(current))
            )
            idx += 1

    return chunks


class Retriever:
    """In-memory TF-IDF index over all chunks from all registered documents.

    Rebuilt lazily whenever the corpus changes (simple and fine at this scale - a
    production version would use an incremental/persistent vector index instead, see
    ARCHITECTURE.md).
    """

    def __init__(self) -> None:
        self._chunks: List[Chunk] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._dirty = True

    def add_document(self, document_id: str, text: str) -> int:
        new_chunks = chunk_document(document_id, text)
        self._chunks.extend(new_chunks)
        self._dirty = True
        return len(new_chunks)

    def remove_document(self, document_id: str) -> None:
        self._chunks = [c for c in self._chunks if c.document_id != document_id]
        self._dirty = True

    def _ensure_index(self) -> None:
        if not self._dirty:
            return
        if not self._chunks:
            self._vectorizer = None
            self._matrix = None
            self._dirty = False
            return
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._matrix = self._vectorizer.fit_transform([c.text for c in self._chunks])
        self._dirty = False

    def search(self, query: str, top_k: int = 4, document_id: str | None = None) -> List[RetrievedChunk]:
        self._ensure_index()
        if self._vectorizer is None or self._matrix is None:
            return []

        candidate_indices = range(len(self._chunks))
        if document_id:
            candidate_indices = [i for i in candidate_indices if self._chunks[i].document_id == document_id]
            if not candidate_indices:
                return []

        query_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._matrix)[0]

        scored = sorted(((sims[i], i) for i in candidate_indices), key=lambda x: x[0], reverse=True)
        results: List[RetrievedChunk] = []
        for score, i in scored[:top_k]:
            if score <= 0:
                continue
            c = self._chunks[i]
            results.append(RetrievedChunk(chunk_id=c.chunk_id, document_id=c.document_id, text=c.text, score=float(score)))
        return results

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)


def confidence_from_scores(results: List[RetrievedChunk]) -> str:
    """Confidence is derived purely from retrieval similarity, never from the LLM's own
    tone - this keeps an overconfident-sounding generation from being reported as
    high-confidence when the retrieved evidence was actually weak or absent."""
    if not results:
        return "low"
    top = results[0].score
    if top >= 0.35:
        return "high"
    if top >= 0.15:
        return "medium"
    return "low"