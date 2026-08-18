import json
from pathlib import Path

from src.legal.embeddings import LegalEmbeddingService
from src.legal.vector_store import LegalVectorStore


class IndianLegalRAG:

    def __init__(self):

        # =====================================================
        # PROJECT ROOT
        # =====================================================

        self.root = (
            Path(__file__)
            .resolve()
            .parents[2]
        )

        # =====================================================
        # CONFIG
        # =====================================================

        self.config_path = (
            self.root
            / "config"
            / "legal_rag.json"
        )

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Legal RAG configuration not found: "
                f"{self.config_path}"
            )

        with open(
            self.config_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.config = json.load(file)

        # =====================================================
        # EMBEDDING SERVICE
        # =====================================================

        embedding_config = self.config[
            "embedding_model"
        ]

        self.embedding_service = (
            LegalEmbeddingService(
                model_name=embedding_config["name"],
                cache_dir=embedding_config.get(
                    "cache_dir"
                ),
                device=embedding_config.get(
                    "device"
                )
            )
        )

        # =====================================================
        # VECTOR STORE
        # =====================================================

        vector_config = self.config[
            "vector_store"
        ]

        self.vector_store = LegalVectorStore(

            index_path=(
                self.root
                / vector_config["index_path"]
            ),

            metadata_path=(
                self.root
                / vector_config["metadata_path"]
            )
        )

        self.vector_store.load()

        # =====================================================
        # RETRIEVAL CONFIG
        # =====================================================

        retrieval_config = self.config.get(
            "retrieval",
            {}
        )

        self.top_k = int(
            retrieval_config.get(
                "top_k",
                5
            )
        )

        self.min_similarity = float(
            retrieval_config.get(
                "min_similarity",
                0.30
            )
        )

        print(
            "Indian Legal RAG initialized."
        )

    # =========================================================
    # RETRIEVE LEGAL REFERENCES
    # =========================================================

    def search(
        self,
        text,
        top_k=None
    ):

        if not isinstance(
            text,
            str
        ):

            raise TypeError(
                "Legal RAG query must be a string."
            )

        text = text.strip()

        if not text:

            raise ValueError(
                "Legal RAG query cannot be empty."
            )

        # -----------------------------------------------------
        # TEXT → LEGAL EMBEDDING
        # -----------------------------------------------------

        query_embedding = (
            self.embedding_service.encode_text(
                text
            )
        )

        # -----------------------------------------------------
        # VECTOR SEARCH
        # -----------------------------------------------------

        results = self.vector_store.search(

            query_embedding,

            top_k=(
                top_k
                if top_k is not None
                else self.top_k
            )
        )

        references = []

        for result in results:

            similarity = float(
                result["score"]
            )

            # -----------------------------------------------------
            # FILTER WEAK LEGAL MATCHES
            # -----------------------------------------------------

            if similarity < self.min_similarity:
                continue

            metadata = result[
                "metadata"
            ]

            references.append({

                "act":
                    metadata.get(
                        "act"
                    ),

                "section":
                    metadata.get(
                        "section"
                    ),

                "title":
                    metadata.get(
                        "title"
                    ),

                "source_id":
                    metadata.get(
                        "source_id"
                    ),

                "summary":
                    metadata.get(
                        "text"
                    ),

                "topics":
                    metadata.get(
                        "topics",
                        []
                    ),

                "url":
                    metadata.get(
                        "url"
                    ),

                "similarity":
                    similarity
            })
        return references

        # =========================================================
    # CLAUSE RETRIEVAL
    # =========================================================

    def retrieve(
        self,
        clause_text,
        clause_type=None,
        top_k=3
    ):
        """
        Retrieve Indian legal references for a contract clause.

        The clause type is included in the semantic query when
        available so that both the clause wording and the
        classifier's predicted type influence retrieval.
        """

        if not isinstance(
            clause_text,
            str
        ):

            raise TypeError(
                "Clause text must be a string."
            )

        clause_text = clause_text.strip()

        if not clause_text:

            raise ValueError(
                "Clause text cannot be empty."
            )

        # -----------------------------------------------------
        # BUILD SEMANTIC QUERY
        # -----------------------------------------------------

        if clause_type:

            query = (
                f"{clause_type}. "
                f"{clause_text}"
            )

        else:

            query = clause_text

        # -----------------------------------------------------
        # SEARCH LEGAL VECTOR STORE
        # -----------------------------------------------------

        return self.search(
            query,
            top_k=top_k
        )