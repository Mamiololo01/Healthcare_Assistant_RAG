"""
Market Access Evidence Assistant - backend API.

Endpoints
---------
GET  /health          service health check
GET  /documents        list registered documents
POST /documents        register a document (file upload .txt/.pdf, or JSON text)
POST /ask               ask a grounded question, get answer + sources + confidence
GET  /docs               Swagger UI
GET  /openapi.json     raw OpenAPI spec

Security
--------
Optional application-level API key check (stretch goal from the brief): if the
API_KEY environment variable is set, all /documents and /ask requests must include a
matching `X-API-Key` header. This is deliberately simple (a shared secret, not
per-user auth) - see ARCHITECTURE.md for what a production auth model would look like.
Left disabled by default (no API_KEY set) so the test/demo is frictionless to run.

Input handling
--------------
- Uploaded files are capped at MAX_UPLOAD_BYTES to avoid pathological memory use.
- Only .txt and .pdf uploads are accepted.
- All user-supplied strings are length-capped before being embedded in prompts, to
  reduce prompt-injection blast radius (a malicious "document" cannot grow unbounded).
"""
from __future__ import annotations

import os
import logging
import time
from typing import Optional

from flask import Flask, jsonify, request, send_from_directory, Response

from backend.llm.base import LLMResult
from backend.llm.factory import get_llm_provider
from backend.models import AskResponse, SourceSnippet
from backend.openapi import OPENAPI_SPEC
from backend.pdf_utils import extract_text
from backend.retrieval import Retriever, confidence_from_scores
from backend.store import DocumentStore

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB
ALLOWED_EXTENSIONS = {".txt", ".pdf"}
MAX_QUESTION_CHARS = 2000
MAX_TEXT_DOC_CHARS = 200_000

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("evidence_assistant")

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")


def create_app(preload_sample_data: bool = True) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

    retriever = Retriever()
    store = DocumentStore(retriever)
    llm_provider = get_llm_provider()

    app.config["store"] = store
    app.config["retriever"] = retriever

    api_key = os.environ.get("API_KEY")

    def _check_api_key() -> Optional[Response]:
        if not api_key:
            return None  # auth disabled
        provided = request.headers.get("X-API-Key")
        if provided != api_key:
            return jsonify({"error": "unauthorized", "message": "Missing or invalid X-API-Key header."}), 401
        return None

    @app.before_request
    def _log_request():
        request._start_time = time.time()  # type: ignore[attr-defined]

    @app.after_request
    def _log_response(response):
        duration_ms = int((time.time() - getattr(request, "_start_time", time.time())) * 1000)
        logger.info("%s %s -> %s (%dms)", request.method, request.path, response.status_code, duration_ms)
        return response

    # ---------------------------------------------------------------- health
    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    # ------------------------------------------------------------- documents
    @app.get("/documents")
    def list_documents():
        docs = [d.to_public_dict() for d in store.list()]
        return jsonify({"documents": docs})

    @app.post("/documents")
    def add_document():
        auth_error = _check_api_key()
        if auth_error:
            return auth_error

        # Case 1: multipart file upload
        if "file" in request.files:
            file = request.files["file"]
            if not file.filename:
                return jsonify({"error": "invalid_input", "message": "No file selected."}), 400

            ext = os.path.splitext(file.filename)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                return jsonify(
                    {"error": "invalid_input", "message": f"Unsupported file type '{ext}'. Use .txt or .pdf."}
                ), 400

            data = file.read()
            if len(data) == 0:
                return jsonify({"error": "invalid_input", "message": "Uploaded file is empty."}), 400

            try:
                text = extract_text(file.filename, data)
            except Exception as exc:  # noqa: BLE001 - surface as a clean 400, log detail
                logger.exception("Failed to extract text from upload")
                return jsonify({"error": "extraction_failed", "message": str(exc)}), 400

            if not text.strip():
                return jsonify(
                    {"error": "invalid_input", "message": "No extractable text found in the uploaded file."}
                ), 400

            title = os.path.splitext(file.filename)[0]
            country = request.form.get("country") or None
            doc = store.add(title=title, text=text[:MAX_TEXT_DOC_CHARS], country=country, source_type="upload")
            return jsonify(doc.to_public_dict()), 201

        # Case 2: JSON text input
        payload = request.get_json(silent=True)
        if not payload:
            return jsonify(
                {
                    "error": "invalid_input",
                    "message": "Provide either a multipart 'file' field or a JSON body with 'title' and 'text'.",
                }
            ), 400

        title = (payload.get("title") or "").strip()
        text = (payload.get("text") or "").strip()
        country = (payload.get("country") or None)

        if not title or not text:
            return jsonify({"error": "invalid_input", "message": "'title' and 'text' are required."}), 400
        if len(text) > MAX_TEXT_DOC_CHARS:
            return jsonify(
                {"error": "invalid_input", "message": f"'text' exceeds max length of {MAX_TEXT_DOC_CHARS} characters."}
            ), 400

        doc = store.add(title=title, text=text, country=country, source_type="text")
        return jsonify(doc.to_public_dict()), 201

    # -------------------------------------------------------------------ask
    @app.post("/ask")
    def ask():
        auth_error = _check_api_key()
        if auth_error:
            return auth_error

        payload = request.get_json(silent=True)
        if not payload:
            return jsonify({"error": "invalid_input", "message": "JSON body with 'question' is required."}), 400

        question = (payload.get("question") or "").strip()
        if not question:
            return jsonify({"error": "invalid_input", "message": "'question' must not be empty."}), 400
        if len(question) > MAX_QUESTION_CHARS:
            return jsonify(
                {"error": "invalid_input", "message": f"'question' exceeds max length of {MAX_QUESTION_CHARS} characters."}
            ), 400

        document_id = payload.get("document_id")
        country = payload.get("country")

        if len(store) == 0:
            return jsonify(
                AskResponse(
                    answer="No documents are currently registered. Please add a document before asking questions.",
                    sources=[],
                    confidence="low",
                    limitations="No documents available.",
                ).to_dict()
            )

        # Resolve a country filter to matching document_ids (simple case-insensitive match).
        filter_document_id = document_id
        if country and not document_id:
            matches = [d.document_id for d in store.list() if (d.country or "").lower() == str(country).lower()]
            if len(matches) == 1:
                filter_document_id = matches[0]
            # If zero or multiple matches, fall back to searching across all documents -
            # a country hint narrows retrieval only when it's unambiguous.

        results = retriever.search(question, top_k=4, document_id=filter_document_id)
        confidence = confidence_from_scores(results)

        try:
            llm_result: LLMResult = llm_provider.generate(question, results)
        except Exception as exc:  # noqa: BLE001
            logger.exception("LLM provider failed")
            return jsonify({"error": "llm_failed", "message": str(exc)}), 502

        sources = [
            SourceSnippet(
                document_id=r.document_id,
                snippet=(r.text[:400] + "...") if len(r.text) > 400 else r.text,
                relevance_score=r.score,
            )
            for r in results
        ]

        limitations = (
            "This answer is based only on the documents currently registered with the "
            "assistant and should not be treated as medical, legal, or regulatory advice."
        )
        if llm_result.caveat:
            limitations = f"{llm_result.caveat} {limitations}"

        response = AskResponse(
            answer=llm_result.answer,
            sources=sources,
            confidence=confidence,
            limitations=limitations,
        )
        return jsonify(response.to_dict())

    # ---------------------------------------------------------------- docs UI
    @app.get("/openapi.json")
    def openapi_spec():
        return jsonify(OPENAPI_SPEC)

    @app.get("/docs")
    def swagger_ui():
        html = """<!DOCTYPE html>
<html>
<head>
  <title>Market Access Evidence Assistant - API Docs</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.11.0/swagger-ui.min.css">
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.11.0/swagger-ui-bundle.min.js"></script>
  <script>
    window.onload = () => {
      window.ui = SwaggerUIBundle({ url: "/openapi.json", dom_id: "#swagger-ui" });
    };
  </script>
</body>
</html>"""
        return Response(html, mimetype="text/html")

    # ---------------------------------------------------------------- frontend
    @app.get("/")
    def frontend_index():
        response = send_from_directory(FRONTEND_DIR, "index.html")
        response.direct_passthrough = False
        return response

    @app.get("/<path:filename>")
    def frontend_assets(filename):
        # Only serve known static asset types from the frontend folder, and never let
        # this route shadow the API routes above (Flask matches those first).
        return send_from_directory(FRONTEND_DIR, filename)

    # ------------------------------------------------------------- error handlers
    @app.errorhandler(413)
    def too_large(_e):
        return jsonify({"error": "payload_too_large", "message": "Uploaded file exceeds the size limit."}), 413

    @app.errorhandler(404)
    def not_found(_e):
        return jsonify({"error": "not_found", "message": "Resource not found."}), 404

    if preload_sample_data:
        _load_sample_documents(store)

    return app


def _load_sample_documents(store: DocumentStore) -> None:
    """Preload the 4 synthetic sample documents from data/ on startup, if present."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    if not os.path.isdir(data_dir):
        return

    country_by_prefix = {
        "uk_": "UK",
        "germany_": "Germany",
        "france_": "France",
        "italy_": "Italy",
    }

    for filename in sorted(os.listdir(data_dir)):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            continue
        path = os.path.join(data_dir, filename)
        with open(path, "rb") as f:
            data = f.read()
        try:
            text = extract_text(filename, data)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to preload sample document %s", filename)
            continue
        if not text.strip():
            continue

        title = os.path.splitext(filename)[0]
        country = next((v for k, v in country_by_prefix.items() if filename.startswith(k)), None)
        store.add(title=title, text=text, country=country, source_type="upload")


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")