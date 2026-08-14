# src/ml/inference/artifacts.py

from pathlib import Path
import json
import pickle


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[3]


# ============================================================
# MODEL DIRECTORIES
# ============================================================

MODEL_DIR = (
    ROOT
    / "models"
    / "clause_classifier"
)


MODEL_PATH = (
    MODEL_DIR
    / "dnn"
    / "best_optimized_dnn.pt"
)


SCALER_PATH = (
    MODEL_DIR
    / "preprocessing"
    / "scaler.pkl"
)


CONFIG_PATH = (
    MODEL_DIR
    / "model_config.json"
)


LABEL_PATH = (
    MODEL_DIR
    / "label_mapping.json"
)


# ============================================================
# LOADERS
# ============================================================

def load_scaler():

    with open(
        SCALER_PATH,
        "rb"
    ) as f:

        return pickle.load(f)


def load_config():

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def load_labels():

    with open(
        LABEL_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)