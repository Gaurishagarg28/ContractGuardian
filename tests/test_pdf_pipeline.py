import sys
from pathlib import Path


# Project root
ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))


from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.clause_segmenter import ClauseSegmenter


def main():

    print("=" * 70)
    print("CONTRACTGUARDIAN PDF PIPELINE TEST")
    print("=" * 70)

    pdf_path = input(
        "\nEnter path to a contract PDF: "
    ).strip().strip('"')

    # ---------------------------------------------------------
    # PDF EXTRACTION
    # ---------------------------------------------------------

    loader = PDFLoader()

    print("\nExtracting PDF...")

    pages = loader.load(pdf_path)

    print(
        f"Pages extracted: {len(pages)}"
    )

    total_characters = sum(
        len(page.text)
        for page in pages
    )

    print(
        f"Characters extracted: {total_characters}"
    )

    if total_characters == 0:

        print(
            "\nWARNING: No text was extracted."
        )

        print(
            "This may be a scanned/image-only PDF."
        )

        return

    # ---------------------------------------------------------
    # CLAUSE SEGMENTATION
    # ---------------------------------------------------------

    segmenter = ClauseSegmenter()

    print("\nSegmenting clauses...")

    clauses = segmenter.segment(pages)

    print(
        f"Clauses detected: {len(clauses)}"
    )

    # ---------------------------------------------------------
    # DISPLAY RESULTS
    # ---------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("DETECTED CLAUSES")
    print("=" * 70)

    for clause in clauses:

        print(
            f"\n{clause.clause_id}"
        )

        print(
            f"Section: {clause.section}"
        )

        print(
            f"Heading: {clause.heading}"
        )

        print(
            f"Pages: "
            f"{clause.page_start}-"
            f"{clause.page_end}"
        )

        print(
            f"Text: {clause.text[:350]}"
        )

        print("-" * 70)

    # ---------------------------------------------------------
    # FINAL SUMMARY
    # ---------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("PDF PIPELINE TEST COMPLETE")
    print("=" * 70)

    print(
        f"Pages: {len(pages)}"
    )

    print(
        f"Characters: {total_characters}"
    )

    print(
        f"Clauses: {len(clauses)}"
    )


if __name__ == "__main__":
    main()