import json
import sys
from pathlib import Path


ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT)
)


from src.legal.chunker import LegalChunker
from src.legal.embeddings import LegalEmbeddingService
from src.legal.vector_store import LegalVectorStore


def main():

    print("=" * 70)
    print("CONTRACTGUARDIAN LEGAL RAG TEST")
    print("=" * 70)

    config_path = (
        ROOT
        / "config"
        / "legal_rag.json"
    )

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as file:

        config = json.load(
            file
        )

    knowledge_base = (
        ROOT
        / "src"
        / "legal"
        / "knowledge_base"
        / "indian_contract_law.json"
    )

    # --------------------------------------------------------
    # CHUNKS
    # --------------------------------------------------------

    chunker = LegalChunker(
        knowledge_base
    )

    chunks = (
        chunker.create_chunks()
    )

    print(
        f"\nLegal chunks: {len(chunks)}"
    )

    if not chunks:

        raise RuntimeError(
            "No legal chunks found."
        )

    # --------------------------------------------------------
    # EMBEDDING SERVICE
    # --------------------------------------------------------

    embedding_config = (
        config["embedding_model"]
    )

    service = LegalEmbeddingService(

        model_name=
            embedding_config["name"],

        cache_dir=
            embedding_config.get(
                "cache_dir"
            ),

        device=
            embedding_config.get(
                "device"
            )
    )

    # --------------------------------------------------------
    # EMBED QUERY
    # --------------------------------------------------------

    query = (
        "contract breach compensation "
        "for loss or damage"
    )

    print(
        f"\nQuery:\n{query}"
    )

    query_embedding = (
        service.encode_text(
            query
        )
    )

    print(
        f"Embedding dimension: "
        f"{len(query_embedding)}"
    )

    # --------------------------------------------------------
    # LOAD INDEX
    # --------------------------------------------------------

    vector_config = (
        config["vector_store"]
    )

    store = LegalVectorStore(

        index_path=(
            ROOT
            / vector_config["index_path"]
        ),

        metadata_path=(
            ROOT
            / vector_config["metadata_path"]
        )
    )

    store.load()

    # --------------------------------------------------------
    # RETRIEVE
    # --------------------------------------------------------

    results = store.search(
        query_embedding,
        top_k=config[
            "retrieval"
        ]["top_k"]
    )

    print()
    print("=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)

    for rank, result in enumerate(
        results,
        start=1
    ):

        metadata = result[
            "metadata"
        ]

        print(
            f"\n#{rank}"
        )

        print(
            f"Score: "
            f"{result['score']:.4f}"
        )

        print(
            f"Section: "
            f"{metadata.get('section')}"
        )

        print(
            f"Title: "
            f"{metadata.get('title')}"
        )

        print(
            f"Source: "
            f"{metadata.get('source_id')}"
        )

    print()
    print("=" * 70)
    print("LEGAL RAG TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()