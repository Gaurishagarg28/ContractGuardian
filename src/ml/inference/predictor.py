# src/ml/inference/predictor.py

import sys
from pathlib import Path

import numpy as np
import torch


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

SRC_ML = ROOT / "src" / "ml"

if str(SRC_ML) not in sys.path:

    sys.path.insert(
        0,
        str(SRC_ML)
    )


# ============================================================
# IMPORT MODEL
# ============================================================

from optimized_dnn import (
    OptimizedContractDNN
)


from src.ml.inference.artifacts import (
    MODEL_PATH,
    load_scaler,
    load_config,
    load_labels
)


# ============================================================
# PREDICTOR
# ============================================================

class ContractPredictor:

    def __init__(self):

        # ----------------------------------------------------
        # DEVICE
        # ----------------------------------------------------

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Using device: {self.device}"
        )


        # ----------------------------------------------------
        # LOAD CONFIG
        # ----------------------------------------------------

        self.config = load_config()


        # ----------------------------------------------------
        # LOAD SCALER
        # ----------------------------------------------------

        self.scaler = load_scaler()


        # ----------------------------------------------------
        # LOAD LABELS
        # ----------------------------------------------------

        self.labels = load_labels()


        # ----------------------------------------------------
        # CREATE MODEL
        # ----------------------------------------------------

        self.model = OptimizedContractDNN(
            input_size=self.config[
                "embedding_dimension"
            ],
            num_classes=self.config[
                "num_classes"
            ]
        )


        # ----------------------------------------------------
        # LOAD CHECKPOINT
        # ----------------------------------------------------

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=self.device,
            weights_only=False
        )


        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        else:

            state_dict = checkpoint


        self.model.load_state_dict(
            state_dict
        )


        # ----------------------------------------------------
        # DEVICE + EVAL
        # ----------------------------------------------------

        self.model.to(
            self.device
        )

        self.model.eval()


        print(
            "Contract DNN loaded successfully."
        )


    # ========================================================
    # LABEL
    # ========================================================

    def get_label(
        self,
        index
    ):

        # Format:
        # {"0": "Governing Law", ...}

        if str(index) in self.labels:

            return self.labels[
                str(index)
            ]


        # Format:
        # {"Governing Law": 0, ...}

        for label, value in self.labels.items():

            if int(value) == index:

                return label


        return f"Unknown-{index}"


    # ========================================================
    # PREDICT
    # ========================================================

    def predict_embedding(
        self,
        embedding,
        top_k=5
    ):

        # ----------------------------------------------------
        # CONVERT TO NUMPY
        # ----------------------------------------------------

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )


        # ----------------------------------------------------
        # ENSURE BATCH DIMENSION
        # ----------------------------------------------------

        if embedding.ndim == 1:

            embedding = embedding.reshape(
                1,
                -1
            )


        # ----------------------------------------------------
        # VALIDATE DIMENSION
        # ----------------------------------------------------

        expected_dimension = self.config[
            "embedding_dimension"
        ]


        if embedding.shape[1] != expected_dimension:

            raise ValueError(
                f"Expected embedding dimension "
                f"{expected_dimension}, "
                f"received {embedding.shape[1]}"
            )


        # ----------------------------------------------------
        # CHECK INVALID VALUES
        # ----------------------------------------------------

        if not np.isfinite(
            embedding
        ).all():

            raise ValueError(
                "Embedding contains NaN or Inf."
            )


        # ----------------------------------------------------
        # APPLY SAME SCALER
        # ----------------------------------------------------

        scaled_embedding = self.scaler.transform(
            embedding
        )


        # ----------------------------------------------------
        # TORCH TENSOR
        # ----------------------------------------------------

        x = torch.tensor(
            scaled_embedding,
            dtype=torch.float32,
            device=self.device
        )


        # ----------------------------------------------------
        # INFERENCE
        # ----------------------------------------------------

        with torch.inference_mode():

            logits = self.model(
                x
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )


        probabilities = probabilities[0]


        # ----------------------------------------------------
        # TOP K
        # ----------------------------------------------------

        k = min(
            top_k,
            probabilities.shape[0]
        )


        values, indices = torch.topk(
            probabilities,
            k=k
        )


        predictions = []


        for probability, index in zip(
            values.cpu().numpy(),
            indices.cpu().numpy()
        ):

            predictions.append(
                {
                    "class_id": int(index),

                    "label": self.get_label(
                        int(index)
                    ),

                    "confidence": float(
                        probability
                    )
                }
            )


        # ----------------------------------------------------
        # PRIMARY PREDICTION
        # ----------------------------------------------------

        primary = predictions[0]


        return {

            "predicted_clause":
                primary["label"],

            "class_id":
                primary["class_id"],

            "confidence":
                primary["confidence"],

            "top_predictions":
                predictions,

            "device":
                str(self.device)
        }