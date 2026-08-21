"""
LLM provider abstraction.
 
Every provider implements the same narrow interface (`generate`) so the rest of the
application never depends on a specific vendor SDK. Swapping providers is a one-line
change in `backend/llm/factory.py` (or the LLM_PROVIDER environment variable) - nothing
in the retrieval or API layer needs to change.
 
Design notes:
- The provider is given the question *and* the retrieved, already-ranked context
  chunks. It must not use any knowledge outside those chunks.
- The provider returns a structured `LLMResult`, not a raw string, so the API layer can
  reliably attach confidence/limitations without re-parsing free text.
- Grounding/anti-hallucination is enforced in two places by design, not one:
    1. The prompt instructs the model to answer only from context and to say so when
       the context is insufficient.
    2. The confidence score is computed independently from retrieval similarity
       (see backend/retrieval.py), so a provider that "sounds confident" cannot
       override a low-relevance retrieval result - confidence is never taken purely
       from the LLM's own tone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Protocol

from backend.retrieval import RetrievedChunk


@dataclass
class LLMResult:
    answer: str
    used_sources: List[str] = field(default_factory=list)
    caveat: str = ""


SYSTEM_INSTRUCTIONS = (
    "You are an evidence assistant for healthcare market access consultants. "
    "Answer ONLY using the numbered context snippets provided below. "
    "If the answer is not fully contained in the snippets, say so explicitly rather than "
    "guessing or using outside knowledge. "
    "Always mention which document(s) the answer draws on. "
    "State clearly when evidence is incomplete or uncertain. "
    "Do not give medical, legal, or regulatory advice - describe what the source "
    "documents say, and nothing more."
)


class LLMProvider(Protocol):
    """Interface every LLM backend (mock, OpenAI, Anthropic, Azure OpenAI, ...) must implement."""

    def generate(self, question: str, context: List[RetrievedChunk]) -> LLMResult:
        ...

def build_prompt(question: str, contetxt: List[RetrievedChunk]) -> str:
    """Shared prompt builder so every real-LLM provider is grounded the same way."""
    numbered_context = "\n\n".join(
        f"[{i+i}] (document: {c.document_id})\n{c.text}" for i, c in enumerate(contetxt)
    )
    return(
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"CONTEXT SNIPPETS:\n{numbered_context}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer using only the snippets above, and reference snippet numbers where relevant."
    )




