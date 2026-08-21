# Architecture Note

## 1. High-level system design

Single Flask process serving both the JSON API and the static frontend, backed by an
in-memory document store and TF-IDF retrieval index. One process is enough at this
scale and keeps the whole thing runnable with zero infrastructure; the internal
boundaries (store / retrieval / LLM) are drawn so each can be swapped or pulled into
its own service without touching the others.

```
                         ┌───────────────────────────┐
   Browser  ── HTTP ──▶  │        Flask app          │
   (frontend/index.html) │        backend/app.py      │
                         │                             │
                         │  routes:                   │
                         │   GET  /health              │
                         │   GET  /documents           │
                         │   POST /documents ──────────┼──▶ pdf_utils.extract_text
                         │   POST /ask                 │        │
                         │   GET  /docs, /openapi.json │        ▼
                         └───────────┬─────────────────┘  DocumentStore (store.py)
                                     │                          │
                                     │ question                 │ text
                                     ▼                          ▼
                         ┌───────────────────────────┐  Retriever (retrieval.py)
                         │   Retriever.search()        │◀── chunk_document()
                         │   TF-IDF + cosine similarity│      (paragraph + sentence
                         └───────────┬─────────────────┘       chunking)
                                     │ ranked chunks
                                     ▼
                         ┌───────────────────────────┐
                         │  LLMProvider.generate()     │
                         │  (mock | openai, via         │
                         │   llm/factory.py)            │
                         └───────────┬─────────────────┘
                                     │ answer + used_sources
                                     ▼
                         AskResponse (answer, sources, confidence, limitations)
```

## 2. Major components

- **`backend/app.py`** - Flask routes, request validation, error handling, logging,
  optional API-key auth, and wiring between the store/retriever/LLM provider.
- **`backend/store.py`** - in-memory `DocumentStore`: registers documents, assigns
  stable slugged IDs, feeds them into the retriever.
- **`backend/retrieval.py`** - chunking (`chunk_document`) and retrieval
  (`Retriever`), plus `confidence_from_scores`.
- **`backend/llm/`** - the LLM abstraction (`base.py`), the default offline provider
  (`mock_provider.py`), a real-provider example (`openai_provider.py`), and the
  selection point (`factory.py`).
- **`backend/pdf_utils.py`** - text extraction for `.txt`/`.pdf` uploads (via `pypdf`).
- **`backend/openapi.py`** - hand-written OpenAPI 3.0 spec served at `/openapi.json`
  and rendered at `/docs` via Swagger UI.
- **`frontend/index.html`** - single-page vanilla HTML/CSS/JS UI (document list,
  upload form, question box with suggested questions, answer + sources + confidence
  display). No build step, no framework.

## 3. Chosen tech stack (and the one deviation, explained)

Flask + scikit-learn (TF-IDF) + pypdf + stdlib `unittest`, instead of the brief's
recommended FastAPI + Pydantic + pytest + React/Streamlit.

This was a constrained-environment decision, not a preference: the environment this
was built in had no package-installer/network access, so only already-installed
libraries could be used *and actually run*. FastAPI, Pydantic, pytest, and Streamlit
were unavailable; Flask, scikit-learn, and pypdf were. Given the choice between (a)
writing FastAPI/Pydantic code that had never once been executed, or (b) writing an
equivalent, fully tested application on an available stack, (b) was chosen -
un-run code is a worse deliverable for a role that explicitly values "engineering
discipline" and "production-readiness thinking" than an honest stack substitution.

The substitution is contained by design:
- `backend/models.py` uses plain dataclasses with the exact field shapes Pydantic
  `BaseModel`s would have - swapping is a mechanical 1:1 change.
- Route handlers, retrieval, and the LLM layer have no Flask-specific logic bleeding
  into them, so moving the routes into FastAPI path operations is straightforward.
- Tests are written with stdlib `unittest.TestCase`, which pytest can already run
  as-is (`pytest tests/` works unmodified) - the migration to pytest is zero-effort in
  the other direction too.

**Vanilla JS instead of React/Streamlit**, similarly: no `npm install` was possible
without network access, and Streamlit wasn't preinstalled. A dependency-free static
page calling the same JSON API was the option that could actually be run and tested in
this environment, and it satisfies the brief's "minimal UI ... does not need to be
polished" framing directly.

## 4. Retrieval approach

**Chunking**: paragraph-based (blank-line-separated), because the source documents are
already organised into short, topically coherent sections ("Evidence Gaps", "Managed
Entry Agreement", etc.). Splitting on paragraph boundaries keeps each chunk
semantically self-contained - both for retrieval precision and for producing a
readable standalone "snippet" in the API response. Paragraphs over ~700 characters are
further split on sentence boundaries with a 1-sentence overlap, so a fact near a
chunk boundary isn't stranded.

**Selecting relevant context**: TF-IDF (1- and 2-grams, English stopwords removed) +
cosine similarity over all chunks, scikit-learn. Chosen over embeddings because it
requires no external API call or model download - it runs fully offline and
deterministically, which matters both for this test environment and for keeping the
demo frictionless for a reviewer. For a small, fixed corpus of short documents, it's a
reasonable, interpretable baseline. The `Retriever` class exposes only `add_document` /
`remove_document` / `search`, so swapping in embeddings + a vector store (FAISS,
Chroma, pgvector) is a contained change behind the same interface - see "What would
change for production" below.

**Reducing hallucination risk**, enforced at two independent points so an
overconfident-sounding generation can't silently override a weak retrieval result:
1. *Prompting*: `backend/llm/base.py`'s `SYSTEM_INSTRUCTIONS` explicitly tells the
   model to answer only from the numbered context snippets, say so when the context is
   insufficient, cite which document(s) it used, and avoid medical/legal/regulatory
   advice.
2. *Confidence scoring*: computed independently from retrieval similarity
   (`confidence_from_scores`), not from the LLM's own tone or wording. A question with
   no good matching chunk is reported `low` confidence and, in the default mock
   provider, returns an explicit "I could not find..." message rather than a guess -
   verified in `tests/test_api.py::test_ask_out_of_scope_question_does_not_fabricate`.

**Returning source snippets**: every `/ask` response includes the actual retrieved
text (truncated to 400 chars) alongside its `document_id` and `relevance_score`, so a
consultant can verify the answer against the source rather than trusting it blind.

**What would need to change for production**: see section 9.

## 5. LLM abstraction

`backend/llm/base.py` defines a narrow `LLMProvider` protocol - one method,
`generate(question, context) -> LLMResult` - plus a shared prompt builder so every
real provider is grounded identically. `backend/llm/factory.py` is the single switch
point, selected via the `LLM_PROVIDER` env var (`mock` default, `openai` implemented as
an example). Nothing in the API layer or retrieval code knows which provider is
active. `backend/llm/mock_provider.py` is the shipped default: a deterministic,
extractive answer built strictly from the top retrieved chunks - it exercises the same
grounding discipline a real model would be prompted to follow, without needing an API
key or network access to run and test.

## 6. Security considerations

- **Optional API-key gate** (`API_KEY` env var) on the two write/compute endpoints
  (`POST /documents`, `POST /ask`), off by default for a frictionless demo. This is a
  shared-secret check, not per-user auth - see limitations below.
- **Upload limits**: 5 MB max file size (`MAX_CONTENT_LENGTH`, returns 413 over
  limit), extension allow-list (`.txt`/`.pdf` only) rejecting anything else with a 400.
- **Input length caps**: question capped at 2,000 chars, pasted document text capped at
  200,000 chars, to bound both memory use and prompt size if a real LLM is behind it.
- **Prompt-injection surface**: uploaded document text is untrusted and flows directly
  into the LLM prompt as context. The system prompt constrains the model to answering
  from context and refusing to give advice, which narrows (but does not eliminate) the
  blast radius of an adversarial document; a production system would add an
  instruction-defense layer and treat retrieved content as data, not instructions (see
  below).
- **No secrets in the repo**: the OpenAI provider reads its key from environment
  variables only, never hardcoded.
- **CORS**: not configured, because the frontend is served from the same origin as the
  API (single Flask process) - no cross-origin requests are needed for this
  submission.

## 7. Testing approach

35 tests across three files (`python -m unittest discover -s tests`, or `pytest
tests/` - the tests are plain `unittest.TestCase`, which pytest runs natively):

- **`test_retrieval.py`** - chunking correctness (paragraph splits, long-paragraph
  sub-splitting, empty input, unique chunk IDs) and retrieval behaviour (relevant
  document ranks first, document-scoped filtering, empty index, low scores for
  irrelevant queries, removal).
- **`test_llm.py`** - the mock provider's grounding behaviour specifically: no context
  → explicit no-evidence message rather than a guess; answer text only draws from
  supplied context; low top-score triggers a caveat; multiple source documents are all
  credited.
- **`test_api.py`** - full HTTP-level coverage: health check; document CRUD paths
  including JSON-text and file-upload variants, unsupported extensions, empty files,
  duplicate-title ID uniqueness; `/ask` validation (empty/missing question), the
  no-documents-yet path, response-contract shape, and the out-of-scope /
  no-hallucination case; API-key auth on and off; Swagger UI and OpenAPI spec
  availability; frontend served at `/`.

This is meant to demonstrate testing *discipline* (unit tests for the core
algorithmic pieces, integration tests for the HTTP contract, an explicit test for the
anti-hallucination behaviour) rather than exhaustive coverage.

## 8. Known limitations

- **In-memory storage only** - all documents and the retrieval index are lost on
  restart; no persistence layer.
- **TF-IDF retrieval is keyword-based, not semantic** - it will miss paraphrased or
  synonym-based matches an embedding model would catch. This is visible in the demo
  data itself: the German document is written in German (as the brief specifies), and
  an English-language question about it retrieves weaker matches than an
  English-language document would, because TF-IDF has no cross-lingual understanding.
  A production system would need either a multilingual embedding model or a
  translation step ahead of retrieval.
- **Single-process, single-node** - no horizontal scaling, no queueing for slow LLM
  calls, no rate limiting beyond the basic upload size cap.
- **Shared-secret auth only** - `API_KEY` is a single shared value, not per-user
  identity, scopes, or audit trail.
- **Mock LLM is extractive, not generative** - it demonstrates the grounding/retrieval
  pipeline faithfully, but produces answers by selecting and lightly trimming source
  text rather than synthesising across chunks the way a real LLM would; qualitative
  answer *fluency* would improve with `LLM_PROVIDER=openai` (or Anthropic/Azure OpenAI)
  connected to a real key.
- **No evaluation harness** - answer quality/groundedness isn't scored automatically;
  the suggested test questions were checked manually against the mock provider's
  output during development.
- **No streaming responses** - `/ask` returns the full answer in one response; a
  production UI would likely want a streamed answer for a real (slower) LLM call.

## 9. How this would be deployed in production, and what would change

- **Stack migration**: FastAPI + Pydantic replacing Flask + dataclasses (the models
  and route logic already map 1:1, per section 3); `pytest` for tests, no code change
  needed.
- **Persistent storage**: document metadata in Postgres; chunk embeddings in a vector
  store (pgvector, if minimising the number of moving pieces is a priority, or a
  managed service like Pinecone/Weaviate at larger scale) instead of the in-memory
  `Retriever`.
- **Embeddings-based retrieval**: replace TF-IDF with a multilingual embedding model
  (e.g. `text-embedding-3-large`, or an open-source multilingual model if data
  residency requires self-hosting) to fix the cross-lingual gap noted above and improve
  recall on paraphrased questions. Add a re-ranking step (cross-encoder) over the
  top-N candidates for better precision before they reach the LLM.
- **Real LLM by default**, with response caching for repeated questions, and a hard
  per-request token/cost ceiling.
- **Auth**: replace the shared `API_KEY` with proper per-user authentication (e.g.
  OAuth2/OIDC against the firm's identity provider) and role-based access if documents
  ever carry client-confidentiality restrictions - relevant given Kintiga's consulting
  context, where document access may need to be scoped per client engagement.
- **Observability**: structured logging (already started via the request logger in
  `app.py`) shipped to a log aggregator; request tracing; token/cost tracking per
  question; an evaluation pipeline that periodically replays a fixed question set
  against the live index and flags regressions in groundedness or retrieval recall.
- **Deployment**: containerised (Dockerfile/compose already in this repo) behind a
  managed container platform (e.g. ECS/Cloud Run/AKS depending on where the firm's
  other infrastructure sits), with the frontend either still served by the API or
  split out to a static host/CDN once it grows past a single page.
- **Data governance**: given the domain (payer evidence, pricing data, sometimes
  commercially sensitive), production would need document-level access control, an
  audit log of who asked what and which documents were used, and a clear data
  retention/deletion policy for uploaded content - none of which is needed for this
  synthetic-data test but all of which would be a priority before real client
  documents touch the system.

## 10. What would be prioritised next

1. Swap TF-IDF for multilingual embeddings - directly fixes the observed German-note
   retrieval weakness and would give the biggest quality jump for the least effort.
2. Persistent storage (documents + index) so the app survives a restart.
3. A small evaluation script running the brief's 10 suggested questions against the
   corpus and checking retrieved-document precision, to catch retrieval regressions
   automatically rather than by manual spot-check.
4. Real LLM as the default provider, with cost/token tracking from day one.