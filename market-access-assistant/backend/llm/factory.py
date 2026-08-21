"""
Single place that decides which LLM provider implementation is in use.
 
Selected via the LLM_PROVIDER environment variable so the answer-generation backend can
be changed at deploy time without touching application code:
  LLM_PROVIDER=mock    -> MockLLMProvider (default, fully offline)
  LLM_PROVIDER=openai  -> OpenAIProvider (requires OPENAI_API_KEY)
"""

from __future__ import annotations

import os

from backend.llm.base import LLMProvider
from backend.llm.mock_provider import MockLLMProvider
from backend.llm.openapi_provider import OpenAIProvider


def get_llm_provider() -> LLMProvider:
    provider_name = os.environ.get("LLM_PROVIDER", "mock").lower()


    if provider_name == "mock":
        return MockLLMProvider()

    if provider_name == "openapi":
        return OpenAIProvider(model=os.environ.get("OPENAPI_MODEL", "gpt-4o-mini"))

    raise ValueError("fUnknown LLM_PROVIDER '{provider_name}'. Expected 'mock' or 'openai'.")
        