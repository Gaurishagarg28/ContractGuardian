import io
import unittest

from app import create_app


class FakeClauseAnalyzer:
    def analyze(self, clause):
        return {"clause_id": clause.clause_id, "text": clause.text}


class FakeContractAnalyzer:
    clause_analyzer = FakeClauseAnalyzer()

    def analyze_pdf(self, path):
        return {"document": {"pdf_path": str(path), "pages": 1}, "clauses": []}


class ApiTestCase(unittest.TestCase):
    def setUp(self):
        app = create_app(analyzer_factory=FakeContractAnalyzer)
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_health_check(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_analyze_text_requires_text(self):
        response = self.client.post("/api/analyze-text", json={})
        self.assertEqual(response.status_code, 400)

    def test_analyze_text_returns_analysis(self):
        response = self.client.post(
            "/api/analyze-text", json={"text": "A valid contract clause."}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["clause_id"], "TEXT-0001")

    def test_analyze_pdf_rejects_non_pdf(self):
        response = self.client.post(
            "/api/analyze-pdf",
            data={"file": (io.BytesIO(b"not a PDF"), "contract.txt")},
        )
        self.assertEqual(response.status_code, 400)
