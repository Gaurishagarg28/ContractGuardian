from risk_rules import RISK_RULES
from text_risk_analyzer import analyze_text


def calculate_risk(
    clause_type,
    confidence,
    clause_text
):

    # ========================================================
    # CLAUSE-LEVEL RISK
    # ========================================================

    rule = RISK_RULES.get(
        clause_type
    )


    if rule is None:

        base_score = 30

        severity = "Unknown"

        reason = (
            "No predefined risk rule "
            "is available for this clause."
        )

    else:

        base_score = rule[
            "base_score"
        ]

        severity = rule[
            "severity"
        ]

        reason = rule[
            "reason"
        ]


    # ========================================================
    # TEXT-LEVEL RISK
    # ========================================================

    text_analysis = analyze_text(
        clause_text
    )


    text_points = (
        text_analysis["risk_points"]
    )


    # ========================================================
    # COMBINED SCORE
    # ========================================================

    # Base clause risk contributes 60%
    # Textual indicators contribute 40%.

    normalized_text_score = (
        text_points / 60
    ) * 100


    risk_score = (
        0.60 * base_score
        +
        0.40 * normalized_text_score
    )


    risk_score = min(
        round(risk_score, 2),
        100
    )


    # ========================================================
    # DETERMINE SEVERITY
    # ========================================================

    if risk_score >= 75:

        final_severity = "Critical"

    elif risk_score >= 55:

        final_severity = "High"

    elif risk_score >= 30:

        final_severity = "Medium"

    else:

        final_severity = "Low"


    # ========================================================
    # RESULT
    # ========================================================

    return {

        "clause":
            clause_type,

        "confidence":
            round(
                confidence,
                4
            ),

        "base_score":
            base_score,

        "text_risk_points":
            text_points,

        "risk_score":
            risk_score,

        "severity":
            final_severity,

        "reason":
            reason,

        "risk_indicators":
            text_analysis[
                "indicators"
            ],

        "matched_text":
            text_analysis[
                "matched_text"
            ]
    }