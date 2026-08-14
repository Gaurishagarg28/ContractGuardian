from pathlib import Path

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from src.ml.inference.embedding_service import (
    EmbeddingService
)

from src.ml.inference.predictor import (
    ContractPredictor
)

from src.analysis.clause_analyzer import (
    ClauseAnalyzer
)


class TestClause:

    def __init__(
        self,
        text,
        clause_id="TEST-001",
        section="1",
        heading="TERM AND RENEWAL",
        page_start=1,
        page_end=1
    ):
        self.text = text
        self.clause_id = clause_id
        self.section = section
        self.heading = heading
        self.page_start = page_start
        self.page_end = page_end


def main():

    print("=" * 70)
    print("CONTRACTGUARDIAN CLAUSE ANALYZER TEST")
    print("=" * 70)

    print("\nInitializing services...")

    embedding_service = EmbeddingService()

    predictor = ContractPredictor()

    analyzer = ClauseAnalyzer(
        embedding_service=embedding_service,
        predictor=predictor
    )

    clause = TestClause(
        text=(
            "The initial term of this Agreement shall "
            "commence on 1 January 2027 and continue "
            "for three (3) years. Thereafter, this "
            "Agreement shall automatically renew for "
            "successive one-year periods unless either "
            "party provides written notice of "
            "non-renewal at least ninety (90) days "
            "before the end of the then-current term."
        )
    )

    print("\nAnalyzing clause...")

    result = analyzer.analyze(clause)

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    print("\nClause:")
    print(result["text"])

    print("\nClassification:")
    classification = result["classification"]

    print(
        "Prediction:",
        classification["predicted_clause"]
    )

    print(
        "Confidence:",
        round(
            classification["confidence"],
            4
        )
    )

    print("\nTop predictions:")

    for prediction in classification[
        "top_predictions"
    ]:

        print(
            f'{prediction["class_id"]:2} | '
            f'{prediction["label"]:<40} | '
            f'{prediction["confidence"]:.4f}'
        )

    print("\nRisk:")

    risk = result["risk"]

    print(
        "Risk Points:",
        risk["risk_points"]
    )

    print(
        "Matched Text:",
        risk["matched_text"]
    )

    print("\nRisk Indicators:")

    for indicator in risk["indicators"]:

        print(
            f'- {indicator["indicator"]}'
        )

        print(
            f'  Points: {indicator["points"]}'
        )

        print(
            f'  Description: '
            f'{indicator["description"]}'
        )

    print("\n" + "=" * 70)
    print("CLAUSE ANALYZER TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()