from pathlib import Path
import json

import torch
from transformers import AutoTokenizer

from src.ml.model import ContractBERT


class EmbeddingService:

    def __init__(self):

        # =====================================================
        # PROJECT ROOT
        # =====================================================

        self.root = (
            Path(__file__)
            .resolve()
            .parents[3]
        )

        # =====================================================
        # EMBEDDING CONFIG
        # =====================================================

        self.config_path = (
            self.root
            / "models"
            / "clause_classifier"
            / "embedding"
            / "config.json"
        )

        if not self.config_path.exists():

            raise FileNotFoundError(
                f"Embedding configuration not found:\n"
                f"{self.config_path}"
            )

        with open(
            self.config_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.config = json.load(file)

        # =====================================================
        # CONFIGURATION
        # =====================================================

        self.model_name = self.config[
            "model_name"
        ]

        self.max_length = int(
            self.config[
                "max_length"
            ]
        )

        self.embedding_dimension = int(
            self.config[
                "embedding_dimension"
            ]
        )

        self.pooling = self.config[
            "pooling"
        ]

        # =====================================================
        # DEVICE
        # =====================================================

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Embedding device: {self.device}"
        )

        if self.device.type == "cuda":

            print(
                "GPU:",
                torch.cuda.get_device_name(0)
            )

        # =====================================================
        # TOKENIZER
        # =====================================================

        print(
            "Loading tokenizer..."
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                self.model_name
            )
        )

        # =====================================================
        # BERT
        # =====================================================

        print(
            "Loading BERT embedding model..."
        )

        self.model = ContractBERT(
            self.model_name
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        print(
            "Embedding model loaded."
        )

    # =========================================================
    # TEXT → EMBEDDING
    # =========================================================

    @torch.inference_mode()
    def encode(
        self,
        text
    ):

        if not isinstance(
            text,
            str
        ):

            raise TypeError(
                "Text must be a string."
            )

        text = text.strip()

        if not text:

            raise ValueError(
                "Text cannot be empty."
            )

        # -----------------------------------------------------
        # TOKENIZATION
        # -----------------------------------------------------

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encoded[
            "input_ids"
        ].to(
            self.device
        )

        attention_mask = encoded[
            "attention_mask"
        ].to(
            self.device
        )

        # -----------------------------------------------------
        # BERT EMBEDDING
        # -----------------------------------------------------

        embedding = (
            self.model.get_embeddings(
                input_ids,
                attention_mask
            )
        )

        # -----------------------------------------------------
        # VALIDATION
        # -----------------------------------------------------

        if embedding.shape[-1] != (
            self.embedding_dimension
        ):

            raise ValueError(
                "Unexpected embedding dimension. "
                f"Expected {self.embedding_dimension}, "
                f"received {embedding.shape[-1]}."
            )

        embedding = (
            embedding
            .squeeze(0)
            .cpu()
            .numpy()
        )

        return embedding