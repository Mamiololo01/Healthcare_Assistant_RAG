"""
Mock LLM provider.

This is the default provider so the whole application runs end-to-end offline, with no
API key and no external network call - useful for the test environment, CI, and demos.

It is deliberately NOT a random/templated fake: it builds an answer strictly from the
retrieved chunks (extractive summarisation), so it demonstrates the same grounding
discipline a real LLM call would be prompted to follow, and is a fair stand-in for
testing the retrieval -> answer -> sources pipeline without needing model-quality output.

Swapping in a real provider (OpenAI/Anthropic/Azure OpenAI) means implementing the same
`generate(question, context) -> LLMResult` interface using `build_prompt` from
`backend/llm/base.py` - see `backend/llm/openai_provider.py` for a ready-to-fill-in
example.
"""
from __future__ import annotations

from typing import List

from backend.llm.base import LLMResult
from backend.retrieval import RetrievedChunk

NO_EVIDENCE_MESSAGE = (
    "I could not find any content in the available documents that addresses this "
    "question. Please rephrase, or confirm the relevant document has been uploaded."
)


class MockLLMProvider:
    """Deterministic, fully offline provider used as the default in this test submission."""

    def generate(self, question: str, context: List[RetrievedChunk]) -> LLMResult:
        if not context:
            return LLMResult(answer=NO_EVIDENCE_MESSAGE, used_sources=[], caveat="No relevant context retrieved.")

        # Use the top 2-3 chunks to build a grounded, extractive answer.
        top_chunks = context[:3]
        doc_ids = sorted({c.document_id for c in top_chunks})

        sentences: List[str] = []
        for c in top_chunks:
            # Take the most informative sentence(s) from each chunk rather than the
            # whole chunk, to keep the answer concise.
            piece = c.text.strip()
            if len(piece) > 320:
                piece = piece[:317].rsplit(" ", 1)[0] + "..."
            sentences.append(piece)

        doc_list = ", ".join(doc_ids)
        answer = (
            f"Based on {doc_list}: " + " ".join(sentences)
        )

        caveat = ""
        if context[0].score < 0.15:
            caveat = (
                "Retrieval confidence is low for this question - the answer above is the "
                "closest available match, but may not fully address what was asked."
            )

        return LLMResult(answer=answer, used_sources=doc_ids, caveat=caveat)
    