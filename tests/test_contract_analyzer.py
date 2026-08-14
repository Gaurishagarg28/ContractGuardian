import sys
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(
    __file__
).resolve().parents[1]

if str(ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(ROOT)
    )


from src.analysis.contract_analyzer import (
    ContractAnalyzer
)


def main():

    print("=" * 70)
    print("CONTRACTGUARDIAN CONTRACT ANALYZER TEST")
    print("=" * 70)

    pdf_path = input(
        "\nEnter path to a contract PDF: "
    ).strip().strip('"')

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():

        print(
            f"\nERROR: File not found:\n"
            f"{pdf_path}"
        )

        return

    print(
        "\nInitializing analysis pipeline..."
    )

    analyzer = ContractAnalyzer()

    print(
        "\nStarting contract analysis..."
    )

    result = analyzer.analyze_pdf(
        pdf_path
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    document = result[
        "document"
    ]

    print("\n")
    print("=" * 70)
    print("CONTRACT ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Pages: "
        f"{document['pages']}"
    )

    print(
        f"Characters: "
        f"{document['characters']}"
    )

    print(
        f"Clauses detected: "
        f"{document['clauses_detected']}"
    )

    print(
        f"Clauses analyzed: "
        f"{document['clauses_analyzed']}"
    )

    print(
        f"Clauses failed: "
        f"{document['clauses_failed']}"
    )

    # ========================================================
    # CLAUSE RESULTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("CLAUSE ANALYSIS")
    print("=" * 70)

    for clause in result[
        "clauses"
    ]:

        print("\n")
        print(
            f"{clause['clause_id']}"
        )

        print(
            f"Section: "
            f"{clause['section']}"
        )

        print(
            f"Heading: "
            f"{clause['heading']}"
        )

        print(
            f"Pages: "
            f"{clause['page_start']}-"
            f"{clause['page_end']}"
        )

        print(
            f"Status: "
            f"{clause['status']}"
        )

        if clause[
            "status"
        ] == "failed":

            print(
                f"Error: "
                f"{clause['error']}"
            )

            continue

        classification = (
            clause[
                "classification"
            ]
        )

        risk = (
            clause[
                "risk"
            ]
        )

        resolved = clause[
            "resolved_classification"
        ]

        print(
            "\nDNN Prediction: "
            f"{resolved['dnn_prediction']}"
        )

        print(
            "DNN Confidence: "
            f"{resolved['confidence']:.4f}"
        )

        print(
            "Final Prediction: "
            f"{resolved['final_prediction']}"
        )

        print(
            "Status: "
            f"{resolved['status']}"
        )

        print(
            "Ambiguous: "
            f"{resolved['ambiguous']}"
        )

        print(
            "Margin: "
            f"{resolved['margin']:.4f}"
        )

        print(
            "Reason: "
            f"{resolved['resolution_reason']}"
        )

        if resolved["heading_match"]:

            print(
                "Heading Match: "
                f"{resolved['heading_match']}"
            )

        if resolved["alternatives"]:

            print("Alternatives:")

            for alternative in (
                resolved["alternatives"]
            ):

                print(
                    f"  - "
                    f"{alternative['label']}: "
                    f"{alternative['confidence']:.4f}"
                )

        print(
            "Risk Points: "
            f"{risk['risk_points']}"
        )

        print(
            "Risk Indicators: "
            f"{len(risk['indicators'])}"
        )

        if risk[
            "matched_text"
        ]:

            print(
                "Matched Text: "
                f"{risk['matched_text']}"
            )

        print(
            "-" * 70
        )


if __name__ == "__main__":

    main()