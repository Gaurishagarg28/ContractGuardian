import unittest

from src.risk.text_risk_analyzer import analyze_text


class TextRiskAnalyzerTestCase(unittest.TestCase):
    def test_detects_automatic_renewal_wording(self):
        result = analyze_text("This agreement renews automatically each year.")

        self.assertEqual(result["risk_points"], 10)
        self.assertEqual(result["indicators"][0]["indicator"], "automatic_renewal")
