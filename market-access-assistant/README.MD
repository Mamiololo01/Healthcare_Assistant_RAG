# Market Access Evidence Assistant

A small retrieval-grounded Q&A application over healthcare market access documents (HTA
summaries, reimbursement notes, pricing pathway notes)

Ask a question, get an answer grounded in the uploaded source documents, with the
source snippets and a confidence level shown alongside it. Answers are explicitly not
medical, legal, or regulatory advice.

See [ARCHITECTURE.md](ARCHITECTURE.md) for system design, trade-offs, and what would
change for production.

## Stack, and why it differs from the "recommended" one

The brief's recommended stack is FastAPI + Pydantic + pytest + React/Streamlit. This
submission uses **Flask + dataclasses + unittest + a vanilla HTML/JS frontend**
instead. That's a deliberate substitution, not a shortcut: the build environment this
was developed in had no package-installer/network access, so anything not already
preinstalled couldn't be used or verified running. Flask, scikit-learn, and pypdf were
available and testable end-to-end; FastAPI, Pydantic, pytest, and Streamlit were not.
Rather than hand over code that had never actually been run, the equivalent
already-available tools were used, and the architecture keeps a 1:1 migration path
back to the recommended stack (see ARCHITECTURE.md).

If you have `pip` access, swapping in FastAPI/Pydantic/pytest is a mechanical
follow-up, not a rewrite - the route handlers, retrieval, and LLM layers are already
decoupled from the web framework.

## Requirements

- Python 3.10+
- No API key needed to run the default configuration (see "LLM provider" below)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python -m backend.app
# or: FLASK_APP=backend.app flask run --port 8000
```

The app starts on **http://localhost:8000** and preloads the 4 synthetic sample
documents from `data/` automatically.

- **UI**: http://localhost:8000/
- **API docs (Swagger UI)**: http://localhost:8000/docs
- **Raw OpenAPI spec**: http://localhost:8000/openapi.json
- **Health check**: http://localhost:8000/health

### Run with Docker

```bash
docker compose up --build
```

Same URLs as above, on port 8000.

## Run the tests

```bash
python -m unittest discover -s tests -v
```

35 tests covering chunking, retrieval ranking, the mock LLM's grounding behaviour, and
the full API (validation, uploads, auth, error paths, response contract). All tests use
the stdlib `unittest`, which is pytest-compatible - if you have `pytest` installed,
`pytest tests/` also works unmodified.

## Try it

With the sample documents preloaded, try asking (via the UI or `curl`):

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What were the main evidence gaps for the UK oncology medicine?"}'
```

```json
{
  "answer": "Based on uk_nice_oncology_drug_summary: ...",
  "sources": [
    {"document_id": "uk_nice_oncology_drug_summary", "snippet": "...", "relevance_score": 0.20}
  ],
  "confidence": "medium",
  "limitations": "This answer is based only on the documents currently registered ..."
}
```

Other questions from the brief to try: see the clickable suggestions in the UI, or the
full list in `ARCHITECTURE.md`.

## Sample dataset

`data/` contains 4 synthetic documents matching the brief's dataset description (the
brief did not include the actual zip file, so these were authored from scratch to match
each document's described content and are clearly marked "Synthetic Example"):

- `uk_nice_oncology_drug_summary.txt`
- `germany_amnog_digital_therapeutic_note_de.txt` (in German, as specified)
- `france_has_medtech_reimbursement_summary.pdf`
- `italy_pricing_reimbursement_pathway_note.pdf`

The two PDFs were generated from `scripts/france_source.txt` / `scripts/italy_source.txt`
via `scripts/generate_sample_pdfs.py` (not part of the running app - a one-off content
generation step).

## LLM provider

Set via the `LLM_PROVIDER` environment variable:

- `LLM_PROVIDER=mock` (**default**) - fully offline, deterministic, extractive
  grounding over the retrieved chunks. No API key, no network call. This is what ships
  and what the tests run against.
- `LLM_PROVIDER=openai` - routes through the OpenAI Chat Completions API using the same
  grounding prompt (`backend/llm/openai_provider.py`). Requires `OPENAI_API_KEY`. Not
  exercised in this submission's test environment (no network access), but implements
  the same `generate(question, context) -> LLMResult` interface, so it's a drop-in
  swap - see `backend/llm/base.py` and `backend/llm/factory.py`.

## Optional: application-level API key

Unset by default (frictionless demo). To require a shared secret on write endpoints
(`POST /documents`, `POST /ask`):

```bash
API_KEY=some-secret python -m backend.app
```

Then include `X-API-Key: some-secret` on those requests. `GET /health` and
`GET /documents` remain open.

## Project layout

```
backend/
  app.py            Flask app, routes, validation, error handling, security
  models.py         Document / request / response data shapes
  store.py          In-memory document store
  retrieval.py      Chunking + TF-IDF retrieval
  pdf_utils.py       PDF/text extraction
  openapi.py         Hand-written OpenAPI spec (served at /openapi.json)
  llm/
    base.py         Provider interface + shared grounding prompt
    mock_provider.py   Default offline provider
    openai_provider.py Real-LLM example provider
    factory.py         Picks a provider based on LLM_PROVIDER
frontend/
  index.html         Single-page vanilla HTML/CSS/JS UI, no build step
data/                 4 synthetic sample documents
tests/                 35 unit + API tests (unittest, pytest-compatible)
.github/workflows/ci.yml   Runs the test suite on push/PR
Dockerfile, docker-compose.yml
ARCHITECTURE.md
```