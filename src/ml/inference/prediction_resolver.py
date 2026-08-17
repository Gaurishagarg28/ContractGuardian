class PredictionResolver:
    """
    Conservative resolver for clause classification.

    Uses:
        1. DNN prediction
        2. Prediction confidence
        3. Prediction margin
        4. Strong clause-heading evidence

    Important:
        Weak heading similarity does NOT override the DNN.
    """

    def __init__(
        self,
        confidence_threshold=0.60,
        ambiguity_margin=0.10,
        minimum_heading_tokens=2,
        strong_heading_overlap=0.80
    ):

        self.confidence_threshold = (
            confidence_threshold
        )

        self.ambiguity_margin = (
            ambiguity_margin
        )

        self.minimum_heading_tokens = (
            minimum_heading_tokens
        )

        self.strong_heading_overlap = (
            strong_heading_overlap
        )

    # =========================================================
    # NORMALIZE TEXT
    # =========================================================

    @staticmethod
    def normalize(text):

        if not text:
            return ""

        text = str(text).lower()

        replacements = {
            "&": " and ",
            "/": " ",
            "-": " ",
            "_": " ",
            ",": " ",
            ".": " ",
            ":": " ",
            "(": " ",
            ")": " "
        }

        for old, new in replacements.items():

            text = text.replace(
                old,
                new
            )

        return " ".join(
            text.split()
        )

    # =========================================================
    # GET TOKENS
    # =========================================================

    def tokens(self, text):

        normalized = self.normalize(
            text
        )

        if not normalized:
            return set()

        return set(
            normalized.split()
        )

    # =========================================================
    # HEADING MATCH
    # =========================================================

    def heading_match(
        self,
        heading,
        label
    ):

        if not heading or not label:

            return {
                "matched": False,
                "score": 0.0,
                "match_type": None
            }

        heading_normalized = (
            self.normalize(heading)
        )

        label_normalized = (
            self.normalize(label)
        )

        # -----------------------------------------------------
        # EXACT MATCH
        # -----------------------------------------------------

        if (
            heading_normalized
            == label_normalized
        ):

            return {
                "matched": True,
                "score": 1.0,
                "match_type": "exact"
            }

        # -----------------------------------------------------
        # TOKEN MATCH
        # -----------------------------------------------------

        heading_tokens = self.tokens(
            heading
        )

        label_tokens = self.tokens(
            label
        )

        if (
            not heading_tokens
            or not label_tokens
        ):

            return {
                "matched": False,
                "score": 0.0,
                "match_type": None
            }

        intersection = (
            heading_tokens
            & label_tokens
        )

        shared_count = len(
            intersection
        )

        # -----------------------------------------------------
        # Require at least two shared words
        # -----------------------------------------------------

        if (
            shared_count
            < self.minimum_heading_tokens
        ):

            return {
                "matched": False,
                "score": 0.0,
                "match_type": None
            }

        # -----------------------------------------------------
        # Dice similarity
        # -----------------------------------------------------

        score = (
            2 * shared_count
        ) / (
            len(heading_tokens)
            + len(label_tokens)
        )

        # -----------------------------------------------------
        # Strong match
        # -----------------------------------------------------

        if (
            score
            >= self.strong_heading_overlap
        ):

            return {
                "matched": True,
                "score": round(
                    score,
                    4
                ),
                "match_type":
                    "strong_token_overlap"
            }

        return {
            "matched": False,
            "score": round(
                score,
                4
            ),
            "match_type":
                "weak_token_overlap"
        }

    # =========================================================
    # FIND STRONG HEADING MATCH
    # =========================================================

    def find_heading_match(
        self,
        heading,
        predictions
    ):

        if not heading:
            return None

        matches = []

        for prediction in predictions:

            label = prediction[
                "label"
            ]

            evidence = (
                self.heading_match(
                    heading,
                    label
                )
            )

            matches.append({

                "label":
                    label,

                "score":
                    evidence["score"],

                "match_type":
                    evidence["match_type"],

                "matched":
                    evidence["matched"]
            })

        strong_matches = [

            item

            for item in matches

            if item["matched"]

        ]

        if not strong_matches:
            return None

        return max(
            strong_matches,
            key=lambda item:
                item["score"]
        )

    # =========================================================
    # CONFIDENCE STATUS
    # =========================================================

    def confidence_status(
        self,
        confidence,
        margin
    ):

        # Strong prediction with clear separation
        if (
            confidence
            >= self.confidence_threshold
            and
            margin
            >= self.ambiguity_margin
        ):

            return "HIGH_CONFIDENCE"

        # Very weak prediction
        if confidence < 0.40:

            return "LOW_CONFIDENCE"

        # Everything in between needs review
        return "REVIEW_REQUIRED"

    # =========================================================
    # MAIN RESOLUTION
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

        # -----------------------------------------------------
        # No prediction
        # -----------------------------------------------------

        if not predictions:

            return {

                "final_prediction":
                    None,

                "dnn_prediction":
                    None,

                "confidence":
                    0.0,

                "second_best_confidence":
                    0.0,

                "margin":
                    0.0,

                "status":
                    "NO_PREDICTION",

                "ambiguous":
                    True,

                "resolution_reason":
                    "No model prediction",

                "heading":
                    heading,

                "heading_match":
                    None,

                "alternatives":
                    []
            }

        # -----------------------------------------------------
        # PRIMARY DNN PREDICTION
        # -----------------------------------------------------

        primary = predictions[0]

        primary_label = primary[
            "label"
        ]

        primary_confidence = float(
            primary[
                "confidence"
            ]
        )

        # -----------------------------------------------------
        # SECOND PREDICTION
        # -----------------------------------------------------

        if len(predictions) > 1:

            second_confidence = float(
                predictions[1][
                    "confidence"
                ]
            )

        else:

            second_confidence = 0.0

        # -----------------------------------------------------
        # PREDICTION MARGIN
        # -----------------------------------------------------

        margin = (
            primary_confidence
            - second_confidence
        )

        # -----------------------------------------------------
        # DEFAULT RESULT
        # -----------------------------------------------------

        final_label = (
            primary_label
        )

        resolution_reason = (
            "DNN prediction"
        )

        # -----------------------------------------------------
        # HEADING EVIDENCE
        # -----------------------------------------------------

        heading_match = (
            self.find_heading_match(
                heading,
                predictions
            )
        )

        # -----------------------------------------------------
        # CONSERVATIVE HEADING OVERRIDE
        # -----------------------------------------------------

        if heading_match:

            heading_label = (
                heading_match[
                    "label"
                ]
            )

            # Only allow heading evidence to
            # override a weak/moderate DNN.
            #
            # We DO NOT override a strong DNN.

            if (
                heading_label
                != primary_label
                and
                primary_confidence
                < 0.75
            ):

                final_label = (
                    heading_label
                )

                resolution_reason = (
                    "Strong heading evidence "
                    "overrides low-confidence "
                    "DNN prediction"
                )

        # -----------------------------------------------------
        # CONFIDENCE
        # -----------------------------------------------------

        status = (
            self.confidence_status(
                primary_confidence,
                margin
            )
        )

        ambiguous = (

            primary_confidence < 0.40

            or

            margin < self.ambiguity_margin

        )

        # -----------------------------------------------------
        # MODEL / HEADING DISAGREEMENT
        # -----------------------------------------------------

        if (
            heading_match
            and
            heading_match[
                "label"
            ]
            != primary_label
        ):

            ambiguous = True

            status = (
                "REVIEW_REQUIRED"
            )

        # -----------------------------------------------------
        # ALTERNATIVE PREDICTIONS
        # -----------------------------------------------------

        alternatives = []

        for prediction in predictions[1:]:

            alternatives.append({

                "label":
                    prediction[
                        "label"
                    ],

                "confidence":
                    float(
                        prediction[
                            "confidence"
                        ]
                    )
            })

        # -----------------------------------------------------
        # FINAL RESULT
        # -----------------------------------------------------

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