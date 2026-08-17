from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


class LegalEmbeddingService:
    """
    Embedding service dedicated to Legal RAG.

    This is intentionally separate from the BERT model
    used by ContractGuardian's clause classifier.
    """

    def __init__(
        self,
        model_name,
        cache_dir=None,
        device=None
    ):

        self.model_name = model_name

        self.cache_dir = (
            Path(cache_dir)
            if cache_dir
            else None
        )

        self.device = device

        model_kwargs = {}

        if self.cache_dir:
            self.cache_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            model_kwargs[
                "cache_folder"
            ] = str(
                self.cache_dir
            )

        print(
            f"Loading legal embedding model: "
            f"{self.model_name}"
        )

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
            **model_kwargs
        )

        print(
            "Legal embedding model loaded."
        )

    # =========================================================
    # SINGLE TEXT
    # =========================================================

    def encode_text(
        self,
        text
    ):

        if not text or not text.strip():

            raise ValueError(
                "Cannot embed empty text."
            )

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding.astype(
            np.float32
        )

    # =========================================================
    # MULTIPLE TEXTS
    # =========================================================

    def encode_documents(
        self,
        texts,
        batch_size=16
    ):

        if not texts:

            raise ValueError(
                "No texts provided."
            )

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        return embeddings.astype(
            np.float32
        )

    # =========================================================
    # DIMENSION
    # =========================================================

    @property
    def dimension(self):

        return self.model.get_sentence_embedding_dimension()