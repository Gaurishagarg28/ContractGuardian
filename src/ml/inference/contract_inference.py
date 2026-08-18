from src.ml.inference.embedding_service import EmbeddingService
from src.ml.inference.predictor import ContractPredictor
from src.ml.inference.prediction_resolver import PredictionResolver


class ContractInference:
    """
    High-level compatibility interface for direct clause inference.

    Pipeline:
        text
          ↓
        BERT embedding
          ↓
        DNN classification
          ↓
        PredictionResolver
    """

    def __init__(
        self,
        embedding_service=None,
        predictor=None,
        prediction_resolver=None
    ):

        self.embedding_service = (
            embedding_service
            or EmbeddingService()
        )

        self.predictor = (
            predictor
            or ContractPredictor()
        )

        self.prediction_resolver = (
            prediction_resolver
            or PredictionResolver()
        )

    def predict(
        self,
        text,
        top_k=5
    ):

        if not isinstance(text, str):

            raise TypeError(
                "Text must be a string."
            )

        text = text.strip()

        if not text:

            raise ValueError(
                "Text cannot be empty."
            )

        # -----------------------------------------------------
        # 1. TEXT → BERT EMBEDDING
        # -----------------------------------------------------

        embedding = (
            self.embedding_service.encode(
                text
            )
        )

        # -----------------------------------------------------
        # 2. EMBEDDING → DNN CLASSIFICATION
        # -----------------------------------------------------

        classification = (
            self.predictor.predict_embedding(
                embedding,
                top_k=top_k
            )
        )

        # -----------------------------------------------------
        # 3. PREDICTION RESOLUTION
        # -----------------------------------------------------

        resolved = (
            self.prediction_resolver.resolve(
                classification=classification
            )
        )

        # -----------------------------------------------------
        # 4. COMPATIBILITY RESULT
        # -----------------------------------------------------

        prediction = (
            resolved.get(
                "final_prediction"
            )
            or classification.get(
                "predicted_clause"
            )
        )

        confidence = float(
            resolved.get(
                "confidence",
                classification.get(
                    "confidence",
                    0.0
                )
            )
        )

        status = resolved.get(
            "status",
            "REVIEW_REQUIRED"
        )

        if status == "HIGH_CONFIDENCE":

            confidence_level = "HIGH"

        elif status == "LOW_CONFIDENCE":

            confidence_level = "LOW"

        else:

            confidence_level = "MEDIUM"

        return {

            "prediction":
                prediction,

            "confidence":
                confidence,

            "confidence_level":
                confidence_level,

            "needs_review":
                bool(
                    resolved.get(
                        "ambiguous",
                        False
                    )
                    or status == "REVIEW_REQUIRED"
                ),

            "top_predictions": [
    {
        "index": item.get(
            "class_id"
        ),
        "label": item.get(
            "label"
        ),
        "confidence": item.get(
            "confidence"
        )
    }
    for item in classification.get(
        "top_predictions",
        []
    )
],

            "resolved_classification":
                resolved,

            "device":
                classification.get(
                    "device"
                )
        }