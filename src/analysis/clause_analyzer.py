from src.risk.text_risk_analyzer import analyze_text

from src.ml.inference.prediction_resolver import (
    PredictionResolver
)

from src.legal.indian_legal_rag import (
    IndianLegalRAG
)


class ClauseAnalyzer:

    def __init__(
        self,
        embedding_service,
        predictor,
        prediction_resolver=None,
        indian_legal_rag=None
    ):

        self.embedding_service = (
            embedding_service
        )

        self.predictor = predictor

        self.prediction_resolver = (
            prediction_resolver
            or PredictionResolver()
        )

        # -----------------------------------------------------
        # INDIAN LEGAL RAG
        # -----------------------------------------------------

        self.indian_legal_rag = (
            indian_legal_rag
            or IndianLegalRAG()
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
        # 4. FINAL CLAUSE TYPE
        # =====================================================

        clause_type = (
            resolved_prediction.get(
                "predicted_clause"
            )
            or classification.get(
                "predicted_clause"
            )
        )

        # =====================================================
        # 5. INDIAN LEGAL RAG
        # =====================================================

        indian_legal_references = (
            self.indian_legal_rag.retrieve(
                clause_text=text,
                clause_type=clause_type,
                top_k=3
            )
        )

        # =====================================================
        # 6. RULE-BASED RISK
        # =====================================================

        risk = analyze_text(
            text
        )

        # =====================================================
        # 7. STRUCTURED RESULT
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

            "indian_legal_references":
                indian_legal_references,

            "risk":
                risk
        }