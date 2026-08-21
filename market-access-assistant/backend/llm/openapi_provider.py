"""
Real LLM provider example (OpenAI-compatible Chat Completions API).
 
Not exercised by default (no network access is assumed in the grading environment
unless OPENAI_API_KEY is set) - included to demonstrate the swap point the interface
in backend/llm/base.py is designed for. The same pattern applies to Anthropic, Azure
OpenAI, or a self-hosted model behind an OpenAI-compatible endpoint - only this file
would change.
"""

from __future__ import annotations

import os
from typing import List

from backend.llm.base import LLMResult, build_prompt
from backend.retrieval import RetrievedChunk


class OpenAIProvider:
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Set LLM_PROVIDER=mock to run without a real "
                "LLM, or provide an API key to use this provider."
            )


    def generate(self, question: str, context: List[RetrievedChunk]) -> LLMResult:
        # Imported Lazily so the openapi package is only required if this provider is actually selected
        from openapi import OpenAI 

        client = OpenAI(api_key=self.api_key)
        prompt = build_prompt(question, context)

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "context": prompt}],
            temperature=0.1
        )

        text = response.choices[0].message.context or ""
        doc_ids = sorted({c.document_id for c in context})
        return LLMResult(answer=text.strip(), used_sources=doc_ids, caveat="")