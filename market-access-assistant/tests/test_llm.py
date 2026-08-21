import unittest

from backend.llm.mock_provider import MockLLMProvider, NO_EVIDENCE_MESSAGE
from backend.retrieval import RetrievedChunk


class TestMockLLMProvider(unittest.TestCase):
    def setUp(self):
        self.provider = MockLLMProvider()

    def test_no_context_returns_no_evidence_message(self):
        result = self.provider.generate("What is the evidence gap?", [])
        self.assertEqual(result.answer, NO_EVIDENCE_MESSAGE)
        self.assertEqual(result.used_sources, [])

    def test_answer_only_uses_provided_context(self):
        context = [
            RetrievedChunk(chunk_id="doc1::0", document_id="doc1", text="The sky is blue today.", score=0.5)
        ]
        result = self.provider.generate("What colour is the sky?", context)
        self.assertIn("sky is blue", result.answer)
        self.assertEqual(result.used_sources, ["doc1"])

    def test_low_score_top_result_adds_caveat(self):
        context = [
            RetrievedChunk(chunk_id="doc1::0", document_id="doc1", text="Some marginally related text.", score=0.05)
        ]
        result = self.provider.generate("Unrelated question", context)
        self.assertIn("low", result.caveat.lower())

    def test_high_score_top_result_has_no_caveat(self):
        context = [
            RetrievedChunk(chunk_id="doc1::0", document_id="doc1", text="Directly relevant text.", score=0.6)
        ]
        result = self.provider.generate("Relevant question", context)
        self.assertEqual(result.caveat, "")

    def test_multiple_documents_are_all_credited(self):
        context = [
            RetrievedChunk(chunk_id="doc1::0", document_id="doc1", text="Fact from doc1.", score=0.5),
            RetrievedChunk(chunk_id="doc2::0", document_id="doc2", text="Fact from doc2.", score=0.4),
        ]
        result = self.provider.generate("Combined question", context)
        self.assertIn("doc1", result.used_sources)
        self.assertIn("doc2", result.used_sources)


if __name__ == "__main__":
    unittest.main()