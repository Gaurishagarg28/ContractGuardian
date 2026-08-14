from src.ml.inference import ContractInference


def main():

    print("=" * 70)
    print("CONTRACTGUARDIAN REAL INFERENCE TEST")
    print("=" * 70)

    inference = ContractInference()

    test_clauses = [

        (
            "Renewal Term",
            """
            The initial term of this Agreement shall commence
            on 1 January 2027 and continue for three years.
            Thereafter, this Agreement shall automatically renew
            for successive one-year periods unless either party
            provides written notice of non-renewal at least
            ninety days before the end of the then-current term.
            """
        ),

        (
            "License Grant",
            """
            Licensor grants Licensee a non-exclusive,
            non-transferable license to use the Software solely
            for Licensee's internal business operations during
            the Term.
            """
        ),

        (
            "Minimum Commitment",
            """
            Licensee agrees to purchase at least 10,000
            software licenses during each contract year.
            """
        ),

        (
            "Revenue/Profit Sharing",
            """
            Licensee shall pay Licensor five percent of
            Net Revenue attributable to the Software during
            each calendar quarter.
            """
        ),

        (
            "Audit Rights",
            """
            Licensor may, upon at least fifteen business days'
            written notice, audit Licensee's records solely
            to verify fees, usage, and compliance with this
            Agreement.
            """
        ),

        (
            "Governing Law",
            """
            This Agreement shall be governed by and construed
            in accordance with the laws applicable in the
            State of Maharashtra, India.
            """
        )
    ]

    print()

    for expected, clause in test_clauses:

        print("-" * 70)

        print(
            "EXPECTED:",
            expected
        )

        result = inference.predict(
            clause,
            top_k=5
        )

        print(
            "PREDICTED:",
            result["prediction"]
        )

        print(
            "CONFIDENCE:",
            f"{result['confidence']:.4f}"
        )

        print(
            "LEVEL:",
            result["confidence_level"]
        )

        print(
            "REVIEW:",
            result["needs_review"]
        )

        print()

        print("TOP 5:")

        for item in result[
            "top_predictions"
        ]:

            print(
                f"{item['index']:2d} | "
                f"{item['label']:<40} | "
                f"{item['confidence']:.4f}"
            )

    print()

    print("=" * 70)
    print("REAL INFERENCE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()