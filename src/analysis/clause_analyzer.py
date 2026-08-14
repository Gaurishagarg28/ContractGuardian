from src.risk.text_risk_analyzer import analyze_text

from src.ml.inference.prediction_resolver import (
    PredictionResolver
)


class ClauseAnalyzer:

    def __init__(
        self,
        embedding_service,
        predictor,
        prediction_resolver=None
    ):

        self.embedding_service = (
            embedding_service
        )

        self.predictor = predictor

        self.prediction_resolver = (
            prediction_resolver
            or PredictionResolver()
        )

    def analyze(self, clause):

        text = clause.text.strip()

        if not text:

            raise ValueError(
                "Clause text cannot be empty."
            )

        # =====================================================
        # 1. TEXT → BERT EMBEDDING
        # =====================================================

        embedding = (
            self.embedding_service.encode(
                text
            )
        )

        # =====================================================
        # 2. EMBEDDING → DNN CLASSIFICATION
        # =====================================================

        classification = (
            self.predictor.predict_embedding(
                embedding,
                top_k=5
            )
        )

        # =====================================================
        # 3. PREDICTION RESOLUTION
        # =====================================================

        resolved_prediction = (
            self.prediction_resolver.resolve(
                classification=classification,
                heading=getattr(
                    clause,
                    "heading",
                    None
                )
            )
        )

        # =====================================================
        # 4. RULE-BASED RISK
        # =====================================================

        risk = analyze_text(
            text
        )

        # =====================================================
        # 5. STRUCTURED RESULT
        # =====================================================

        return {

            "clause_id":
                getattr(
                    clause,
                    "clause_id",
                    None
                ),

            "section":
                getattr(
                    clause,
                    "section",
                    None
                ),

            "heading":
                getattr(
                    clause,
                    "heading",
                    None
                ),

            "page_start":
                getattr(
                    clause,
                    "page_start",
                    None
                ),

            "page_end":
                getattr(
                    clause,
                    "page_end",
                    None
                ),

            "text":
                text,

            "classification":
                classification,

            "resolved_classification":
                resolved_prediction,

            "risk":
                risk
        }