"""
Request/response data models.
 
Pydantic (the recommended stack in the brief) was not used here because the offline
test/CI environment for this submission has no package installer access, and pydantic
was not preinstalled. Validation is done by hand in `validation.py` instead, kept
narrow and explicit. In a real deployment with FastAPI + Pydantic available (as
recommended), these dataclasses would be replaced 1:1 with Pydantic `BaseModel`s -
the field shapes below are already written to map directly onto that migration.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional


@dataclass
class Document:
    document_id: str
    title: str
    country: Optional[str]
    source_type: str
    text: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


    def to_public_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "country": self.country,
            "source_type": self.source_type,
            "created_at": self.created_at,
            "char_count": len(self.text),
        }

@dataclass
class SourceSnippet:
    document_id: str
    snippet: str
    relevance_score: float

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "snippet": self.snippet,
            "relevance_score": round(self.relevance_score, 4)
        }


@dataclass
class AskResponse:
    answer: str
    sources: List[SourceSnippet]
    confidence: str
    limitations: str

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "confidence": self.confidence,
            "limitations": self.limitations,
        }
