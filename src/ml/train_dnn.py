import json
import os

import numpy as np
import torch

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

from dnn_classifier import ContractDNN


# ============================================================
# CONFIG
# ============================================================

TRAIN_FILE = (
    "data/embeddings/train.npz"
)

VAL_FILE = (
    "data/embeddings/validation.npz"
)

MODEL_DIR = (
    "models/clause_classifier/dnn"
)

BATCH_SIZE = 32

EPOCHS = 30

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 0.01


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

    print(
        "Using GPU:",
        torch.cuda.get_device_name(0)
    )

else:

    DEVICE = torch.device("cpu")

    print("Using CPU")


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print("\nLoading embeddings...")

train_data = np.load(
    TRAIN_FILE,
    allow_pickle=True
)

val_data = np.load(
    VAL_FILE,
    allow_pickle=True
)


X_train = torch.tensor(
    train_data["embeddings"],
    dtype=torch.float32
)

y_train = torch.tensor(
    train_data["labels"],
    dtype=torch.long
)


X_val = torch.tensor(
    val_data["embeddings"],
    dtype=torch.float32
)

y_val = torch.tensor(
    val_data["labels"],
    dtype=torch.long
)


print(
    "Training embeddings:",
    X_train.shape
)

print(
    "Validation embeddings:",
    X_val.shape
)


# ============================================================
# DATASETS
# ============================================================

train_dataset = TensorDataset(
    X_train,
    y_train
)

val_dataset = TensorDataset(
    X_val,
    y_val
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MODEL
# ============================================================

model = ContractDNN(
    input_size=768,
    num_classes=36
)

model.to(DEVICE)


# ============================================================
# LOSS
# ============================================================

class_counts = np.bincount(
    y_train.numpy(),
    minlength=36
)


total_samples = len(
    y_train
)

num_classes = 36


weights = []

for count in class_counts:

    count = max(
        count,
        1
    )

    weight = (
        total_samples /
        (num_classes * count)
    ) ** 0.5

    weights.append(
        weight
    )


weights = np.array(
    weights,
    dtype=np.float32
)


weights = (
    weights /
    weights.mean()
)


class_weights = torch.tensor(
    weights,
    dtype=torch.float32
).to(DEVICE)


loss_function = torch.nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# TRAINING
# ============================================================

best_macro_f1 = 0.0


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0


    for X, y in train_loader:

        X = X.to(DEVICE)

        y = y.to(DEVICE)


        optimizer.zero_grad()


        logits = model(X)


        loss = loss_function(
            logits,
            y
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )


        optimizer.step()


        total_loss += loss.item()


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    predictions = []

    true_labels = []


    with torch.no_grad():

        for X, y in val_loader:

            X = X.to(DEVICE)

            logits = model(X)


            preds = torch.argmax(
                logits,
                dim=1
            )


            predictions.extend(
                preds.cpu().numpy()
            )

            true_labels.extend(
                y.numpy()
            )


    accuracy = accuracy_score(
        true_labels,
        predictions
    )


    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predictions,
            average="weighted",
            zero_division=0
        )
    )


    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predictions,
            average="macro",
            zero_division=0
        )
    )


    avg_loss = (
        total_loss /
        len(train_loader)
    )


    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train Loss: {avg_loss:.4f}"
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Weighted F1: {weighted_f1:.4f}"
    )

    print(
        f"Macro F1: {macro_f1:.4f}"
    )


    # ========================================================
    # SAVE BEST
    # ========================================================

    if macro_f1 > best_macro_f1:

        best_macro_f1 = macro_f1


        torch.save(
            model.state_dict(),
            f"{MODEL_DIR}/best_dnn.pt"
        )


        print(
            "★ New best DNN model saved!"
        )


print("\n" + "=" * 60)

print("DNN TRAINING COMPLETE")

print("=" * 60)

print(
    f"Best Macro F1: "
    f"{best_macro_f1:.4f}"
)

print(
    f"Model saved at: "
    f"{MODEL_DIR}/best_dnn.pt"
)