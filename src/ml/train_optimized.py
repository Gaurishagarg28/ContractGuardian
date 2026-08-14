import os
import random
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader,
    WeightedRandomSampler
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from optimized_dnn import (
    OptimizedContractDNN
)

from class_weights import (
    get_class_weights
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42

BATCH_SIZE = 64

EPOCHS = 60

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

PATIENCE = 8

NUM_CLASSES = 36

INPUT_SIZE = 768


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)


if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
    )


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "Device:",
    DEVICE
)


if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD DATA
# ============================================================

print(
    "\nLoading preprocessed embeddings..."
)


train_data = np.load(
    "data/embeddings/processed/train.npz"
)

val_data = np.load(
    "data/embeddings/processed/validation.npz"
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
    "Training:",
    X_train.shape
)

print(
    "Validation:",
    X_val.shape
)


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = TensorDataset(
    X_train,
    y_train
)

val_dataset = TensorDataset(
    X_val,
    y_val
)


# ============================================================
# BALANCED SAMPLER
# ============================================================

class_counts = np.bincount(
    y_train.numpy(),
    minlength=NUM_CLASSES
)


class_weights_sampler = np.zeros(
    NUM_CLASSES,
    dtype=np.float64
)


valid_classes = class_counts > 0


class_weights_sampler[
    valid_classes
] = (
    1.0 /
    np.sqrt(
        class_counts[
            valid_classes
        ]
    )
)


sample_weights = (
    class_weights_sampler[
        y_train.numpy()
    ]
)


sampler = WeightedRandomSampler(

    weights=torch.DoubleTensor(
        sample_weights
    ),

    num_samples=len(
        sample_weights
    ),

    replacement=True
)


train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    sampler=sampler,

    num_workers=0,

    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    pin_memory=torch.cuda.is_available()
)


# ============================================================
# MODEL
# ============================================================

model = OptimizedContractDNN(
    input_size=INPUT_SIZE,
    num_classes=NUM_CLASSES
)


model.to(DEVICE)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_weights = get_class_weights(

    y_train.numpy(),

    NUM_CLASSES

).to(DEVICE)


print(
    "\nClass weights:"
)

print(
    class_weights
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(

    label_smoothing=0.05

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
# SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="max",

    factor=0.5,

    patience=3

)


# ============================================================
# CHECKPOINT
# ============================================================

os.makedirs(
    "models/clause_classifier/dnn",
    exist_ok=True
)


BEST_MODEL = (
    "models/clause_classifier/"
    "dnn/best_optimized_dnn.pt"
)


best_macro_f1 = -1

epochs_without_improvement = 0


# ============================================================
# TRAINING
# ============================================================

for epoch in range(
    1,
    EPOCHS + 1
):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0


    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(
            DEVICE,
            non_blocking=True
        )

        y_batch = y_batch.to(
            DEVICE,
            non_blocking=True
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        logits = model(
            X_batch
        )


        loss = criterion(
            logits,
            y_batch
        )


        loss.backward()


        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        optimizer.step()


        train_loss += (
            loss.item()
            *
            X_batch.size(0)
        )


    train_loss /= len(
        train_dataset
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()


    val_predictions = []

    val_targets = []


    val_loss = 0.0


    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(
                DEVICE
            )

            y_batch = y_batch.to(
                DEVICE
            )


            logits = model(
                X_batch
            )


            loss = criterion(
                logits,
                y_batch
            )


            val_loss += (
                loss.item()
                *
                X_batch.size(0)
            )


            predictions = torch.argmax(
                logits,
                dim=1
            )


            val_predictions.extend(
                predictions.cpu().numpy()
            )

            val_targets.extend(
                y_batch.cpu().numpy()
            )


    val_loss /= len(
        val_dataset
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        val_targets,
        val_predictions
    )


    weighted_f1 = f1_score(

        val_targets,

        val_predictions,

        average="weighted",

        zero_division=0
    )


    macro_f1 = f1_score(

        val_targets,

        val_predictions,

        average="macro",

        zero_division=0
    )


    macro_precision = precision_score(

        val_targets,

        val_predictions,

        average="macro",

        zero_division=0
    )


    macro_recall = recall_score(

        val_targets,

        val_predictions,

        average="macro",

        zero_division=0
    )


    # --------------------------------------------------------
    # LR SCHEDULER
    # --------------------------------------------------------

    scheduler.step(
        macro_f1
    )


    current_lr = optimizer.param_groups[0][
        "lr"
    ]


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        f"\nEpoch {epoch}/{EPOCHS}"
    )


    print(
        f"Train Loss: {train_loss:.4f}"
    )


    print(
        f"Val Loss:   {val_loss:.4f}"
    )


    print(
        f"Accuracy:   {accuracy:.4f}"
    )


    print(
        f"Weighted F1: {weighted_f1:.4f}"
    )


    print(
        f"Macro F1:    {macro_f1:.4f}"
    )


    print(
        f"Macro Prec:  {macro_precision:.4f}"
    )


    print(
        f"Macro Recall:{macro_recall:.4f}"
    )


    print(
        f"Learning Rate: {current_lr:.7f}"
    )


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    if macro_f1 > best_macro_f1:

        best_macro_f1 = macro_f1

        epochs_without_improvement = 0


        torch.save(

            {
                "model_state_dict":
                    model.state_dict(),

                "epoch":
                    epoch,

                "macro_f1":
                    macro_f1,

                "weighted_f1":
                    weighted_f1,

                "accuracy":
                    accuracy,

                "seed":
                    SEED
            },

            BEST_MODEL
        )


        print(
            "★ BEST MODEL SAVED"
        )


    else:

        epochs_without_improvement += 1


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print(
            "\nEarly stopping."
        )

        print(
            "Best Macro F1:",
            best_macro_f1
        )

        break


print(
    "\n"
    + "=" * 65
)

print(
    "OPTIMIZED TRAINING COMPLETE"
)

print(
    "=" * 65
)

print(
    "Best Validation Macro F1:",
    best_macro_f1
)

print(
    "Model:",
    BEST_MODEL
)
