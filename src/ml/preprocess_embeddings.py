# src/ml/preprocess_embeddings.py

import os
import json
import pickle
import random

import numpy as np
import torch

from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================

SEED = 42

INPUT_DIR = "data/embeddings"
OUTPUT_DIR = "data/embeddings/processed"

MODEL_DIR = "models/clause_classifier"
PREPROCESSING_DIR = os.path.join(
    MODEL_DIR,
    "preprocessing"
)

SCALER_PATH = os.path.join(
    PREPROCESSING_DIR,
    "scaler.pkl"
)

CONFIG_PATH = os.path.join(
    MODEL_DIR,
    "model_config.json"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PREPROCESSING_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("CONTRACTGUARDIAN EMBEDDING PREPROCESSING")
print("=" * 70)

train_data = np.load(
    f"{INPUT_DIR}/train.npz",
    allow_pickle=True
)

val_data = np.load(
    f"{INPUT_DIR}/validation.npz",
    allow_pickle=True
)

test_data = np.load(
    f"{INPUT_DIR}/test.npz",
    allow_pickle=True
)


X_train = train_data["embeddings"]
y_train = train_data["labels"]

X_val = val_data["embeddings"]
y_val = val_data["labels"]

X_test = test_data["embeddings"]
y_test = test_data["labels"]


print(f"Train:      {X_train.shape}")
print(f"Validation: {X_val.shape}")
print(f"Test:       {X_test.shape}")


# ============================================================
# CLEAN
# ============================================================

def clean_embeddings(X, y):

    mask = np.isfinite(X).all(axis=1)

    removed = np.sum(~mask)

    if removed:
        print(f"Removing {removed} invalid embeddings")

    return X[mask], y[mask]


X_train, y_train = clean_embeddings(
    X_train,
    y_train
)

X_val, y_val = clean_embeddings(
    X_val,
    y_val
)

X_test, y_test = clean_embeddings(
    X_test,
    y_test
)


# ============================================================
# STANDARDIZATION
# ============================================================

print()
print("Fitting StandardScaler on TRAIN ONLY...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_val_scaled = scaler.transform(
    X_val
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# SAVE SCALER
# ============================================================

with open(
    SCALER_PATH,
    "wb"
) as f:

    pickle.dump(
        scaler,
        f
    )


print(
    f"Scaler saved: {SCALER_PATH}"
)


# ============================================================
# SAVE PROCESSED EMBEDDINGS
# ============================================================

np.savez_compressed(
    f"{OUTPUT_DIR}/train.npz",
    embeddings=X_train_scaled.astype(np.float32),
    labels=y_train
)

np.savez_compressed(
    f"{OUTPUT_DIR}/validation.npz",
    embeddings=X_val_scaled.astype(np.float32),
    labels=y_val
)

np.savez_compressed(
    f"{OUTPUT_DIR}/test.npz",
    embeddings=X_test_scaled.astype(np.float32),
    labels=y_test
)


# ============================================================
# MODEL CONFIG
# ============================================================

config = {
    "seed": SEED,
    "embedding_dimension": 768,
    "num_classes": 36,
    "scaler": "StandardScaler",
    "scaler_fit_on": "train_only",
    "model_input": "standardized_bert_embedding"
}


with open(
    CONFIG_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        config,
        f,
        indent=4
    )


# ============================================================
# VERIFY
# ============================================================

print()
print("=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print(
    f"Train:      {X_train_scaled.shape}"
)

print(
    f"Validation: {X_val_scaled.shape}"
)

print(
    f"Test:       {X_test_scaled.shape}"
)

print(
    f"Scaler:     {SCALER_PATH}"
)

print(
    f"Config:     {CONFIG_PATH}"
)
