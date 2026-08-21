import unittest

from backend.retrieval import Retriever, chunk_document, confidence_from_scores


class TestChunking(unittest.TestCase):
    def test_paragraph_split(self):
        text = "First paragraph about oncology.\n\nSecond paragraph about cardiology."
        chunks = chunk_document("doc1", text)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].document_id, "doc1")
        self.assertIn("oncology", chunks[0].text)
        self.assertIn("cardiology", chunks[1].text)

    def test_long_paragraph_is_split(self):
        sentence = "This is a sentence about reimbursement policy. "
        long_para = sentence * 40  # well over CHUNK_MAX_CHARS
        chunks = chunk_document("doc2", long_para)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c.text), 900)  # generous bound incl. overlap

    def test_empty_document_produces_no_chunks(self):
        self.assertEqual(chunk_document("doc3", "   \n\n  "), [])

    def test_chunk_ids_are_unique_within_document(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = chunk_document("doc4", text)
        ids = [c.chunk_id for c in chunks]
        self.assertEqual(len(ids), len(set(ids)))


class TestRetriever(unittest.TestCase):
    def setUp(self):
        self.retriever = Retriever()
        self.retriever.add_document(
            "uk_doc",
            "The committee noted uncertainty in overall survival data for the oncology drug. "
            "Evidence gaps included immature survival data and limited real-world evidence.",
        )
        self.retriever.add_document(
            "italy_doc",
            "AIFA proposed a managed entry agreement with an outcome-based rebate. "
            "The budget impact estimate was uncertain due to population assumptions.",
        )

    def test_search_returns_relevant_document_first(self):
        results = self.retriever.search("What are the evidence gaps for the oncology drug?", top_k=3)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].document_id, "uk_doc")

    def test_search_respects_document_id_filter(self):
        results = self.retriever.search("managed entry agreement", top_k=3, document_id="uk_doc")
        for r in results:
            self.assertEqual(r.document_id, "uk_doc")

    def test_search_on_empty_index_returns_empty(self):
        empty_retriever = Retriever()
        self.assertEqual(empty_retriever.search("anything"), [])

    def test_irrelevant_query_returns_no_or_low_score_results(self):
        results = self.retriever.search("weather forecast for tomorrow", top_k=3)
        for r in results:
            self.assertLess(r.score, 0.2)

    def test_remove_document_excludes_it_from_search(self):
        self.retriever.remove_document("uk_doc")
        results = self.retriever.search("oncology evidence gaps", top_k=3)
        for r in results:
            self.assertNotEqual(r.document_id, "uk_doc")


class TestConfidence(unittest.TestCase):
    def test_no_results_is_low(self):
        self.assertEqual(confidence_from_scores([]), "low")


if __name__ == "__main__":
    unittest.main()