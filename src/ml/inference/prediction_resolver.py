from difflib import SequenceMatcher


class PredictionResolver:
    """
    Resolves the raw ML prediction using:
        1. DNN probabilities
        2. Prediction margin
        3. Clause heading evidence
        4. Ambiguity detection

    This does NOT retrain or modify the classifier.
    """

    def __init__(
        self,
        confidence_threshold=0.60,
        ambiguity_margin=0.10,
        heading_weight=0.25
    ):
        self.confidence_threshold = confidence_threshold
        self.ambiguity_margin = ambiguity_margin
        self.heading_weight = heading_weight

    # =========================================================
    # TEXT NORMALIZATION
    # =========================================================

    @staticmethod
    def normalize(text):

        if not text:
            return ""

        return (
            str(text)
            .lower()
            .replace("&", "and")
            .replace("/", " ")
            .replace("-", " ")
            .replace("_", " ")
            .strip()
        )

    # =========================================================
    # HEADING SIMILARITY
    # =========================================================

    def heading_similarity(
        self,
        heading,
        label
    ):

        if not heading or not label:
            return 0.0

        heading = self.normalize(
            heading
        )

        label = self.normalize(
            label
        )

        # Exact normalized match
        if heading == label:
            return 1.0

        # One contains the other
        if (
            heading in label
            or label in heading
        ):
            return 0.90

        # Word overlap
        heading_words = set(
            heading.split()
        )

        label_words = set(
            label.split()
        )

        if not heading_words or not label_words:
            return 0.0

        intersection = (
            heading_words
            & label_words
        )

        union = (
            heading_words
            | label_words
        )

        jaccard = (
            len(intersection)
            / len(union)
        )

        # Character-level similarity
        sequence_score = SequenceMatcher(
            None,
            heading,
            label
        ).ratio()

        return max(
            jaccard,
            sequence_score
        )

    # =========================================================
    # CONFIDENCE STATUS
    # =========================================================

    def confidence_status(
        self,
        confidence,
        margin
    ):

        if (
            confidence >= self.confidence_threshold
            and margin >= self.ambiguity_margin
        ):
            return "HIGH_CONFIDENCE"

        if confidence < 0.40:
            return "LOW_CONFIDENCE"

        return "REVIEW_REQUIRED"

    # =========================================================
    # RESOLVE
    # =========================================================

    def resolve(
        self,
        classification,
        heading=None
    ):

        predictions = classification.get(
            "top_predictions",
            []
        )

        if not predictions:

            return {
                "final_prediction": None,
                "confidence": 0.0,
                "status": "NO_PREDICTION",
                "margin": 0.0,
                "ambiguous": True,
                "heading": heading,
                "heading_match": None,
                "alternatives": []
            }

        # -----------------------------------------------------
        # Raw DNN winner
        # -----------------------------------------------------

        primary = predictions[0]

        primary_label = primary[
            "label"
        ]

        primary_confidence = float(
            primary["confidence"]
        )

        # -----------------------------------------------------
        # Second-best prediction
        # -----------------------------------------------------

        if len(predictions) > 1:

            second = predictions[1]

            second_confidence = float(
                second["confidence"]
            )

        else:

            second_confidence = 0.0

        margin = (
            primary_confidence
            - second_confidence
        )

        # -----------------------------------------------------
        # Heading evidence
        # -----------------------------------------------------

        heading_scores = []

        if heading:

            for prediction in predictions:

                label = prediction[
                    "label"
                ]

                score = self.heading_similarity(
                    heading,
                    label
                )

                heading_scores.append({

                    "label": label,

                    "similarity": round(
                        score,
                        4
                    )
                })

        # -----------------------------------------------------
        # Determine strongest heading match
        # -----------------------------------------------------

        best_heading = None

        if heading_scores:

            best_heading = max(
                heading_scores,
                key=lambda item:
                item["similarity"]
            )

        # -----------------------------------------------------
        # Decide final prediction
        # -----------------------------------------------------

        final_label = primary_label

        resolution_reason = (
            "DNN prediction"
        )

        heading_match = None

        if best_heading:

            heading_match = best_heading

            best_label = (
                best_heading["label"]
            )

            best_score = (
                best_heading["similarity"]
            )

            primary_heading_score = next(
                (
                    item["similarity"]
                    for item in heading_scores
                    if item["label"]
                    == primary_label
                ),
                0.0
            )

            # -------------------------------------------------
            # Heading strongly supports another class
            # -------------------------------------------------

            if (
                best_label != primary_label
                and best_score >= 0.70
                and primary_confidence < 0.75
            ):

                final_label = best_label

                resolution_reason = (
                    "Heading evidence overrides "
                    "low-confidence DNN prediction"
                )

            # -------------------------------------------------
            # Heading agrees with DNN
            # -------------------------------------------------

            elif (
                best_label == primary_label
                and primary_heading_score >= 0.70
            ):

                resolution_reason = (
                    "DNN prediction supported "
                    "by clause heading"
                )

        # -----------------------------------------------------
        # Confidence / ambiguity
        # -----------------------------------------------------

        status = self.confidence_status(
            primary_confidence,
            margin
        )

        ambiguous = (
            margin < self.ambiguity_margin
            or primary_confidence < 0.40
        )

        # Heading disagreement is important
        if (
            best_heading
            and best_heading["label"]
            != primary_label
            and best_heading["similarity"] >= 0.70
        ):

            ambiguous = True

            if status == "HIGH_CONFIDENCE":

                status = "REVIEW_REQUIRED"

        # -----------------------------------------------------
        # Alternatives
        # -----------------------------------------------------

        alternatives = []

        for prediction in predictions[1:]:

            alternatives.append({

                "label":
                    prediction["label"],

                "confidence":
                    float(
                        prediction["confidence"]
                    )
            })

        return {

            "final_prediction":
                final_label,

            "dnn_prediction":
                primary_label,

            "confidence":
                primary_confidence,

            "second_best_confidence":
                second_confidence,

            "margin":
                round(
                    margin,
                    4
                ),

            "status":
                status,

            "ambiguous":
                ambiguous,

            "resolution_reason":
                resolution_reason,

            "heading":
                heading,

            "heading_match":
                heading_match,

            "alternatives":
                alternatives
        }