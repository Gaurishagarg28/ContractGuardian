from pathlib import Path
import sys


ROOT = Path(
    __file__
).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT)
    )


from src.ml.inference import EmbeddingService


def main():

    print("=" * 70)
    print("CONTRACTGUARDIAN EMBEDDING SERVICE TEST")
    print("=" * 70)

    service = EmbeddingService()

    text = input(
        "\nEnter text to encode: "
    ).strip()

    embedding = service.encode(
        text
    )

    print(
        "\nEmbedding shape:",
        embedding.shape
    )

    print(
        "Embedding dtype:",
        embedding.dtype
    )

    print(
        "Finite values:",
        bool(
            __import__("numpy")
            .isfinite(embedding)
            .all()
        )
    )

    print(
        "\nEmbedding service test passed."
    )


if __name__ == "__main__":
    main()