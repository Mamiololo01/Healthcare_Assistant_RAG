"""Hand-written OpenAPI 3.0 spec for the API (served at GET /openapi.json).

Flask has no built-in OpenAPI generation (unlike FastAPI), so the spec is maintained
here manually and kept in sync with the routes in backend/app.py. It powers the
Swagger UI page at GET /docs (static HTML pulling this JSON via the swagger-ui CDN
bundle).
"""

OPENAPI_SPEC = {
    "openapi": "3.0.3",
    "info": {
        "title": "Market Access Evidence Assistant API",
        "version": "1.0.0",
        "description": (
            "Retrieval-grounded Q&A API over healthcare market access documents "
            "(HTA summaries, reimbursement notes, pricing pathway notes). "
            "Answers are generated only from retrieved source snippets and are "
            "NOT medical, legal, or regulatory advice."
        ),
    },
    "paths": {
        "/health": {
            "get": {
                "summary": "Service health check",
                "responses": {
                    "200": {
                        "description": "Service is healthy",
                        "content": {
                            "application/json": {
                                "schema": {"type": "object", "properties": {"status": {"type": "string"}}},
                                "example": {"status": "ok"},
                            }
                        },
                    }
                },
            }
        },
        "/documents": {
            "get": {
                "summary": "List available documents",
                "responses": {
                    "200": {
                        "description": "List of registered documents",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "documents": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/DocumentSummary"},
                                        }
                                    },
                                }
                            }
                        },
                    }
                },
            },
            "post": {
                "summary": "Register a document (file upload OR raw text)",
                "description": (
                    "Accepts either multipart/form-data with a `file` field (.txt or .pdf), "
                    "or application/json with `title` and `text` fields."
                ),
                "requestBody": {
                    "required": True,
                    "content": {
                        "multipart/form-data": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "file": {"type": "string", "format": "binary"},
                                    "country": {"type": "string"},
                                },
                            }
                        },
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/DocumentTextInput"}
                        },
                    },
                },
                "responses": {
                    "201": {
                        "description": "Document registered",
                        "content": {
                            "application/json": {"schema": {"$ref": "#/components/schemas/DocumentSummary"}}
                        },
                    },
                    "400": {"description": "Invalid input"},
                },
            },
        },
        "/ask": {
            "post": {
                "summary": "Ask a grounded question over the registered documents",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/AskRequest"},
                            "example": {
                                "question": "What evidence gaps were identified for the oncology drug in the UK?",
                                "country": "UK",
                            },
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Grounded answer with sources",
                        "content": {
                            "application/json": {"schema": {"$ref": "#/components/schemas/AskResponse"}}
                        },
                    },
                    "400": {"description": "Invalid input"},
                },
            }
        },
    },
    "components": {
        "schemas": {
            "DocumentSummary": {
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"},
                    "title": {"type": "string"},
                    "country": {"type": "string", "nullable": True},
                    "source_type": {"type": "string", "enum": ["upload", "text"]},
                    "created_at": {"type": "string", "format": "date-time"},
                    "char_count": {"type": "integer"},
                },
            },
            "DocumentTextInput": {
                "type": "object",
                "required": ["title", "text"],
                "properties": {
                    "title": {"type": "string"},
                    "text": {"type": "string"},
                    "country": {"type": "string"},
                },
            },
            "AskRequest": {
                "type": "object",
                "required": ["question"],
                "properties": {
                    "question": {"type": "string"},
                    "country": {"type": "string", "description": "Optional filter to a specific country's document(s)"},
                    "document_id": {"type": "string", "description": "Optional filter to a single document"},
                },
            },
            "SourceSnippet": {
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"},
                    "snippet": {"type": "string"},
                    "relevance_score": {"type": "number"},
                },
            },
            "AskResponse": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "sources": {"type": "array", "items": {"$ref": "#/components/schemas/SourceSnippet"}},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "limitations": {"type": "string"},
                },
            },
        },
        "securitySchemes": {
            "ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-API-Key"}
        },
    },
}