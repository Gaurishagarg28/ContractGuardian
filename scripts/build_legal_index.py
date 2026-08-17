import json
from pathlib import Path

from src.legal.chunker import LegalChunker
from src.legal.embeddings import LegalEmbeddingService
from src.legal.vector_store import LegalVectorStore


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    ROOT / "config" / "legal_rag.json"
)

KNOWLEDGE_BASE = (
    ROOT
    / "src"
    / "legal"
    / "knowledge_base"
    / "indian_contract_law.json"
)


# ============================================================
# CONFIG
# ============================================================

with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8"
) as file:

    config = json.load(
        file
    )


embedding_config = (
    config["embedding_model"]
)

vector_config = (
    config["vector_store"]
)


# ============================================================
# SERVICES
# ============================================================

chunker = LegalChunker(
    KNOWLEDGE_BASE
)

chunks = chunker.create_chunks()

if not chunks:

    raise RuntimeError(
        "No legal chunks were created."
    )

print(
    f"Legal chunks: {len(chunks)}"
)


embedding_service = LegalEmbeddingService(
    model_name=embedding_config["name"],
    cache_dir=embedding_config.get(
        "cache_dir"
    ),
    device=embedding_config.get(
        "device"
    )
)


# ============================================================
# EMBEDDINGS
# ============================================================

texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = (
    embedding_service.encode_documents(
        texts
    )
)

print(
    f"Embedding shape: "
    f"{embeddings.shape}"
)


# ============================================================
# VECTOR STORE
# ============================================================

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

store.build(
    embeddings,
    chunks
)

store.save()


print()
print("=" * 70)
print("LEGAL RAG INDEX BUILD COMPLETE")
print("=" * 70)