from pathlib import Path
import sys


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.clause_segmenter import ClauseSegmenter

from src.ml.inference import (
    EmbeddingService,
    ContractPredictor,
)


def main():

    print("=" * 70)
    print(
        "CONTRACTGUARDIAN DYNAMIC INFERENCE"
    )
    print("=" * 70)

    pdf_path = input(
        "\nEnter contract PDF path: "
    ).strip()

    pdf_path = Path(
        pdf_path.strip('"').strip("'")
    )

    if not pdf_path.is_file():

        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    print("\nInitializing pipeline...")

    loader = PDFLoader()

    segmenter = ClauseSegmenter()

    embedding_service = EmbeddingService()

    predictor = ContractPredictor()

    print("Pipeline ready.")

    # ---------------------------------------------------------
    # PDF EXTRACTION
    # ---------------------------------------------------------

    print("\nExtracting document...")

    pages = loader.load(
        pdf_path
    )

    print(
        f"Pages extracted: {len(pages)}"
    )

    # ---------------------------------------------------------
    # CLAUSE SEGMENTATION
    # ---------------------------------------------------------

    print("\nSegmenting clauses...")

    clauses = segmenter.segment(
        pages
    )

    print(
        f"Clauses detected: {len(clauses)}"
    )

    # ---------------------------------------------------------
    # INFERENCE
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL PREDICTIONS")
    print("=" * 70)

    successful = 0
    failed = 0

    for clause in clauses:

        print("\n" + "-" * 70)

        print(
            f"Clause: {clause.clause_id}"
        )

        if clause.section:
            print(
                f"Section: {clause.section}"
            )

        if clause.heading:
            print(
                f"Heading: {clause.heading}"
            )

        print(
            f"Pages: "
            f"{clause.page_start}-"
            f"{clause.page_end}"
        )

        try:

            embedding = (
                embedding_service.encode(
                    clause.text
                )
            )

            prediction = (
                predictor.predict_embedding(
                    embedding,
                    top_k=5
                )
            )

            successful += 1

            print(
                "\nPrediction:"
            )

            print(
                prediction[
                    "predicted_clause"
                ]
            )

            print(
                "Confidence:",
                f"{prediction['confidence']:.4f}"
            )

            print(
                "\nTop predictions:"
            )

            for item in prediction[
                "top_predictions"
            ]:

                print(
                    f"{item['class_id']:>2} | "
                    f"{item['label']:<45} | "
                    f"{item['confidence']:.4f}"
                )

        except Exception as error:

            failed += 1

            print(
                "\nInference failed:"
            )

            print(
                str(error)
            )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("INFERENCE COMPLETE")
    print("=" * 70)

    print(
        f"Total clauses: {len(clauses)}"
    )

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed: {failed}"
    )

    print(
        f"Device: {predictor.device}"
    )


if __name__ == "__main__":
    main()