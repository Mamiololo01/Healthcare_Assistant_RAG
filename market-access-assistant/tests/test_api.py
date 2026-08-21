import io
import os
import unittest

from backend.app import create_app


class TestHealth(unittest.TestCase):
    def setUp(self):
        self.app = create_app(preload_sample_data=False)
        self.client = self.app.test_client()

    def test_health_check(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json(), {"status": "ok"})


class TestDocumentsEndpoint(unittest.TestCase):
    def setUp(self):
        self.app = create_app(preload_sample_data=False)
        self.client = self.app.test_client()

    def test_list_documents_empty(self):
        res = self.client.get("/documents")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json(), {"documents": []})

    def test_add_document_via_json_text(self):
        res = self.client.post(
            "/documents",
            json={"title": "Test Note", "text": "Some evidence about a drug.", "country": "UK"},
        )
        self.assertEqual(res.status_code, 201)
        body = res.get_json()
        self.assertEqual(body["title"], "Test Note")
        self.assertEqual(body["country"], "UK")
        self.assertEqual(body["source_type"], "text")

        res2 = self.client.get("/documents")
        self.assertEqual(len(res2.get_json()["documents"]), 1)

    def test_add_document_missing_fields_returns_400(self):
        res = self.client.post("/documents", json={"title": "Only title"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("error", res.get_json())

    def test_add_document_via_txt_upload(self):
        data = {"file": (io.BytesIO(b"Some uploaded document content."), "note.txt")}
        res = self.client.post("/documents", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 201)
        self.assertEqual(res.get_json()["source_type"], "upload")

    def test_add_document_rejects_unsupported_extension(self):
        data = {"file": (io.BytesIO(b"not really a doc"), "note.docx")}
        res = self.client.post("/documents", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)

    def test_add_document_rejects_empty_file(self):
        data = {"file": (io.BytesIO(b""), "empty.txt")}
        res = self.client.post("/documents", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)

    def test_document_ids_are_unique_for_duplicate_titles(self):
        self.client.post("/documents", json={"title": "Dup", "text": "First version."})
        res = self.client.post("/documents", json={"title": "Dup", "text": "Second version."})
        docs = self.client.get("/documents").get_json()["documents"]
        ids = [d["document_id"] for d in docs]
        self.assertEqual(len(ids), len(set(ids)))


class TestAskEndpoint(unittest.TestCase):
    def setUp(self):
        self.app = create_app(preload_sample_data=False)
        self.client = self.app.test_client()
        self.client.post(
            "/documents",
            json={
                "title": "UK Oncology Note",
                "text": (
                    "The committee identified evidence gaps including immature overall survival "
                    "data and limited real-world evidence for the oncology drug in the UK."
                ),
                "country": "UK",
            },
        )

    def test_ask_with_no_documents_returns_helpful_message(self):
        empty_app = create_app(preload_sample_data=False)
        client = empty_app.test_client()
        res = client.post("/ask", json={"question": "Anything?"})
        self.assertEqual(res.status_code, 200)
        self.assertIn("No documents", res.get_json()["answer"])

    def test_ask_empty_question_returns_400(self):
        res = self.client.post("/ask", json={"question": ""})
        self.assertEqual(res.status_code, 400)

    def test_ask_missing_body_returns_400(self):
        res = self.client.post("/ask", json={})
        self.assertEqual(res.status_code, 400)

    def test_ask_relevant_question_returns_grounded_answer_with_sources(self):
        res = self.client.post("/ask", json={"question": "What evidence gaps were identified?"})
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertIn("answer", body)
        self.assertIn("sources", body)
        self.assertIn("confidence", body)
        self.assertIn("limitations", body)
        self.assertGreater(len(body["sources"]), 0)
        self.assertEqual(body["sources"][0]["document_id"], "uk-oncology-note")

    def test_ask_response_shape_matches_contract(self):
        res = self.client.post("/ask", json={"question": "evidence gaps"})
        body = res.get_json()
        self.assertIsInstance(body["answer"], str)
        self.assertIsInstance(body["sources"], list)
        self.assertIn(body["confidence"], ["high", "medium", "low"])
        for s in body["sources"]:
            self.assertIn("document_id", s)
            self.assertIn("snippet", s)
            self.assertIn("relevance_score", s)

    def test_ask_out_of_scope_question_does_not_fabricate(self):
        res = self.client.post("/ask", json={"question": "What is the capital of France?"})
        body = res.get_json()
        self.assertEqual(body["confidence"], "low")
        self.assertEqual(body["sources"], [])


class TestApiKeyAuth(unittest.TestCase):
    def setUp(self):
        os.environ["API_KEY"] = "test-secret-key"
        self.app = create_app(preload_sample_data=False)
        self.client = self.app.test_client()

    def tearDown(self):
        del os.environ["API_KEY"]

    def test_ask_without_key_is_unauthorized(self):
        res = self.client.post("/ask", json={"question": "Anything?"})
        self.assertEqual(res.status_code, 401)

    def test_ask_with_correct_key_succeeds(self):
        res = self.client.post(
            "/ask", json={"question": "Anything?"}, headers={"X-API-Key": "test-secret-key"}
        )
        self.assertEqual(res.status_code, 200)

    def test_health_does_not_require_key(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)


class TestDocsAndSpec(unittest.TestCase):
    def setUp(self):
        self.app = create_app(preload_sample_data=False)
        self.client = self.app.test_client()

    def test_openapi_json_served(self):
        res = self.client.get("/openapi.json")
        self.assertEqual(res.status_code, 200)
        spec = res.get_json()
        self.assertIn("/ask", spec["paths"])
        self.assertIn("/documents", spec["paths"])
        self.assertIn("/health", spec["paths"])

    def test_swagger_ui_served(self):
        res = self.client.get("/docs")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"swagger-ui", res.data)

    def test_frontend_served_at_root(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Market Access Evidence Assistant", res.data)


if __name__ == "__main__":
    unittest.main()