import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .model import ContractBERT
from .optimized_dnn import OptimizedContractDNN


class ContractInference:

    def __init__(self):

        # =====================================================
        # PROJECT PATHS
        # =====================================================

        self.project_root = Path(__file__).resolve().parents[2]

        self.model_dir = (
            self.project_root
            / "models"
            / "clause_classifier"
        )

        self.preprocessing_dir = (
            self.model_dir
            / "preprocessing"
        )

        self.dnn_path = (
            self.model_dir
            / "dnn"
            / "best_optimized_dnn.pt"
        )

        self.scaler_path = (
            self.preprocessing_dir
            / "scaler.pkl"
        )

        self.mapping_path = (
            self.model_dir
            / "label_mapping.json"
        )

        self.config_path = (
            self.model_dir
            / "model_config.json"
        )

        # =====================================================
        # DEVICE
        # =====================================================

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # =====================================================
        # LOAD CONFIGURATION
        # =====================================================

        self._load_config()

        # =====================================================
        # LOAD LABELS
        # =====================================================

        self._load_label_mapping()

        # =====================================================
        # LOAD SCALER
        # =====================================================

        self._load_scaler()

        # =====================================================
        # LOAD TOKENIZER
        # =====================================================

        self._load_tokenizer()

        # =====================================================
        # LOAD BERT
        # =====================================================

        self._load_bert()

        # =====================================================
        # LOAD CLASSIFIER
        # =====================================================

        self._load_classifier()

    # =========================================================
    # CONFIGURATION
    # =========================================================

    def _load_config(self):

        self.model_name = "bert-base-uncased"
        self.max_length = 128

        if not self.config_path.exists():
            return

        with open(
            self.config_path,
            "r",
            encoding="utf-8"
        ) as file:

            config = json.load(file)

        self.model_name = config.get(
            "model_name",
            self.model_name
        )

        self.max_length = int(
            config.get(
                "max_length",
                self.max_length
            )
        )

    # =========================================================
    # LABEL MAPPING
    # =========================================================

    def _load_label_mapping(self):

        if not self.mapping_path.exists():

            raise FileNotFoundError(
                f"Label mapping not found:\n"
                f"{self.mapping_path}"
            )

        with open(
            self.mapping_path,
            "r",
            encoding="utf-8"
        ) as file:

            mapping = json.load(file)

        # Supports:
        #
        # {"0": "Class A", "1": "Class B"}
        #
        # and:
        #
        # {"Class A": 0, "Class B": 1}

        if all(
            str(key).isdigit()
            for key in mapping.keys()
        ):

            self.idx_to_label = {
                int(key): value
                for key, value in mapping.items()
            }

        else:

            self.idx_to_label = {
                int(value): key
                for key, value in mapping.items()
            }

        self.num_classes = len(
            self.idx_to_label
        )

    # =========================================================
    # SCALER
    # =========================================================

    def _load_scaler(self):

        if not self.scaler_path.exists():

            raise FileNotFoundError(
                f"Scaler not found:\n"
                f"{self.scaler_path}"
            )

        with open(
            self.scaler_path,
            "rb"
        ) as file:

            self.scaler = pickle.load(file)

    # =========================================================
    # TOKENIZER
    # =========================================================

    def _load_tokenizer(self):

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

    # =========================================================
    # BERT
    # =========================================================

    def _load_bert(self):

        self.bert_model = ContractBERT(
            self.model_name
        )

        self.bert_model.to(
            self.device
        )

        self.bert_model.eval()

    # =========================================================
    # CLASSIFIER
    # =========================================================

    def _load_classifier(self):

        if not self.dnn_path.exists():

            raise FileNotFoundError(
                f"Optimized DNN checkpoint not found:\n"
                f"{self.dnn_path}"
            )

        checkpoint = torch.load(
            self.dnn_path,
            map_location=self.device
        )

        if isinstance(checkpoint, dict):

            state_dict = checkpoint.get(
                "model_state_dict",
                checkpoint.get(
                    "state_dict",
                    checkpoint
                )
            )

        else:

            state_dict = checkpoint

        self.classifier = OptimizedContractDNN(
            input_dim=768,
            num_classes=self.num_classes
        )

        self.classifier.load_state_dict(
            state_dict
        )

        self.classifier.to(
            self.device
        )

        self.classifier.eval()

    # =========================================================
    # TEXT → BERT EMBEDDING
    # =========================================================

    @torch.no_grad()
    def get_embedding(
        self,
        text
    ):

        if not isinstance(
            text,
            str
        ):

            raise TypeError(
                "Input text must be a string."
            )

        text = text.strip()

        if not text:

            raise ValueError(
                "Input text cannot be empty."
            )

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encoded[
            "input_ids"
        ].to(self.device)

        attention_mask = encoded[
            "attention_mask"
        ].to(self.device)

        # EXACT SAME EMBEDDING METHOD
        # USED DURING TRAINING:
        #
        # outputs.last_hidden_state[:, 0, :]

        embedding = (
            self.bert_model.get_embeddings(
                input_ids,
                attention_mask
            )
        )

        return embedding.cpu().numpy()

    # =========================================================
    # EMBEDDING → PREDICTION
    # =========================================================

    @torch.no_grad()
    def predict_embedding(
        self,
        embedding,
        top_k=5
    ):

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        if embedding.ndim == 1:

            embedding = embedding.reshape(
                1,
                -1
            )

        if embedding.shape[1] != 768:

            raise ValueError(
                "Expected 768-dimensional "
                f"embedding, got {embedding.shape[1]}."
            )

        # Same scaler used during training
        scaled = self.scaler.transform(
            embedding
        )

        tensor = torch.tensor(
            scaled,
            dtype=torch.float32,
            device=self.device
        )

        logits = self.classifier(
            tensor
        )

        probabilities = F.softmax(
            logits,
            dim=1
        )

        k = min(
            int(top_k),
            self.num_classes
        )

        values, indices = torch.topk(
            probabilities,
            k=k,
            dim=1
        )

        values = values[0].cpu().tolist()
        indices = indices[0].cpu().tolist()

        predictions = []

        for probability, index in zip(
            values,
            indices
        ):

            predictions.append(
                {
                    "index": int(index),
                    "label": self.idx_to_label[
                        int(index)
                    ],
                    "confidence": float(
                        probability
                    )
                }
            )

        primary = predictions[0]

        return {
            "prediction": primary["label"],
            "confidence": primary["confidence"],
            "top_predictions": predictions,
            "device": str(self.device)
        }

    # =========================================================
    # TEXT → PREDICTION
    # =========================================================

    def predict(
        self,
        text,
        top_k=5
    ):

        embedding = self.get_embedding(
            text
        )

        return self.predict_embedding(
            embedding,
            top_k=top_k
        )