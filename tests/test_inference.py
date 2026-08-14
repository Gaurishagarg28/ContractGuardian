import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT)
)


from src.ml.inference.predictor import (
    ContractPredictor
)


# ============================================================
# TEST
# ============================================================

def main():

    print("=" * 70)
    print("CONTRACTGUARDIAN INFERENCE TEST")
    print("=" * 70)


    predictor = ContractPredictor()


    # --------------------------------------------------------
    # RANDOM EMBEDDING TEST
    # --------------------------------------------------------

    embedding = np.random.randn(
        768
    ).astype(
        np.float32
    )


    result = predictor.predict_embedding(
        embedding,
        top_k=5
    )


    print()
    print("Prediction:")
    print(
        result["predicted_clause"]
    )


    print()
    print("Confidence:")
    print(
        f"{result['confidence']:.4f}"
    )


    print()
    print("Top 5 predictions:")


    for item in result[
        "top_predictions"
    ]:

        print(
            f"{item['class_id']:>2} | "
            f"{item['label']:<40} | "
            f"{item['confidence']:.4f}"
        )


    print()
    print("Device:")
    print(
        result["device"]
    )


    print()
    print("=" * 70)
    print("INFERENCE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":

    main()