import json
from pathlib import Path

import faiss
import numpy as np


class LegalVectorStore:

    def __init__(
        self,
        index_path,
        metadata_path
    ):

        self.index_path = Path(
            index_path
        )

        self.metadata_path = Path(
            metadata_path
        )

        self.index = None
        self.metadata = []

    # =========================================================
    # BUILD
    # =========================================================

    def build(
        self,
        embeddings,
        metadata
    ):

        if len(embeddings) != len(metadata):

            raise ValueError(
                "Embedding count and metadata "
                "count must match."
            )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        if embeddings.ndim != 2:

            raise ValueError(
                "Embeddings must be a 2D array."
            )

        dimension = embeddings.shape[1]

        # Because embeddings are normalized,
        # inner product == cosine similarity.
        index = faiss.IndexFlatIP(
            dimension
        )

        index.add(
            embeddings
        )

        self.index = index

        self.metadata = list(
            metadata
        )

    # =========================================================
    # SEARCH
    # =========================================================

    def search(
        self,
        query_embedding,
        top_k=5
    ):

        if self.index is None:

            raise RuntimeError(
                "Vector store has not been built "
                "or loaded."
            )

        query = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        if query.ndim == 1:

            query = query.reshape(
                1,
                -1
            )

        scores, indices = (
            self.index.search(
                query,
                min(
                    top_k,
                    self.index.ntotal
                )
            )
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index < 0:
                continue

            results.append({

                "score":
                    float(score),

                "metadata":
                    self.metadata[
                        int(index)
                    ]
            })

        return results

    # =========================================================
    # SAVE
    # =========================================================

    def save(self):

        if self.index is None:

            raise RuntimeError(
                "Nothing to save."
            )

        self.index_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            str(self.index_path)
        )

        with open(
            self.metadata_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.metadata,
                file,
                ensure_ascii=False,
                indent=2
            )

        print(
            f"Vector index saved: "
            f"{self.index_path}"
        )

        print(
            f"Metadata saved: "
            f"{self.metadata_path}"
        )

    # =========================================================
    # LOAD
    # =========================================================

    def load(self):

        if not self.index_path.exists():

            raise FileNotFoundError(
                f"Index not found: "
                f"{self.index_path}"
            )

        if not self.metadata_path.exists():

            raise FileNotFoundError(
                f"Metadata not found: "
                f"{self.metadata_path}"
            )

        self.index = faiss.read_index(
            str(self.index_path)
        )

        with open(
            self.metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.metadata = json.load(
                file
            )

        print(
            f"Loaded {self.index.ntotal} "
            f"legal vectors."
        )