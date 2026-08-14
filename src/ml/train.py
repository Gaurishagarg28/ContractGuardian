import json
import os

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


from dataset import ContractClauseDataset


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "bert-base-uncased"

TRAIN_FILE = "data/processed/train.csv"
VAL_FILE = "data/processed/validation.csv"

MODEL_DIR = "models/clause_classifier"

MAX_LENGTH = 128

BATCH_SIZE = 8

EPOCHS = 5

LEARNING_RATE = 1e-5

WEIGHT_DECAY = 0.01

GRADIENT_ACCUMULATION_STEPS = 1


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
# MIXED PRECISION
# ============================================================

USE_AMP = DEVICE.type == "cuda"

if USE_AMP:

    scaler = torch.amp.GradScaler("cuda")

else:

    scaler = None


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading datasets...")

train_df = pd.read_csv(
    TRAIN_FILE
)

val_df = pd.read_csv(
    VAL_FILE
)

print(
    "Training examples:",
    len(train_df)
)

print(
    "Validation examples:",
    len(val_df)
)


# ============================================================
# LABEL ENCODER
# ============================================================

print("\nCreating label encoder...")

label_encoder = LabelEncoder()

label_encoder.fit(
    train_df["clause_type"]
)

num_labels = len(
    label_encoder.classes_
)

print(
    "Number of classes:",
    num_labels
)


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# DATASETS
# ============================================================

train_dataset = ContractClauseDataset(
    train_df,
    tokenizer,
    label_encoder,
    MAX_LENGTH
)

val_dataset = ContractClauseDataset(
    val_df,
    tokenizer,
    label_encoder,
    MAX_LENGTH
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True if USE_AMP else False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True if USE_AMP else False
)


# ============================================================
# MODEL
# ============================================================

print("\nLoading BERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
)

for param in model.bert.parameters():
    param.requires_grad = False

print("\nBERT encoder frozen.")
print("Only the classification head will be trained.")

model.to(DEVICE)


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("\nCalculating smoothed class weights...")

class_counts = (
    train_df["clause_type"]
    .value_counts()
)

total_samples = len(train_df)

num_classes = len(
    label_encoder.classes_
)

weights = []

for label in label_encoder.classes_:

    count = class_counts.get(
        label,
        1
    )

    # Smoothed inverse-frequency weighting
    weight = (
        total_samples /
        (num_classes * count)
    ) ** 0.5

    weights.append(weight)


weights = np.array(
    weights,
    dtype=np.float32
)

# Normalize weights
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

best_val_f1 = 0.0

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


for epoch in range(EPOCHS):

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    print("\n")
    print("=" * 70)
    print(
        f"EPOCH {epoch + 1}/{EPOCHS}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    total_train_loss = 0.0

    optimizer.zero_grad(
        set_to_none=True
    )


    for step, batch in enumerate(
        train_loader
    ):

        input_ids = batch[
            "input_ids"
        ].to(
            DEVICE,
            non_blocking=True
        )

        attention_mask = batch[
            "attention_mask"
        ].to(
            DEVICE,
            non_blocking=True
        )

        labels = batch[
            "labels"
        ].to(
            DEVICE,
            non_blocking=True
        )


        # ----------------------------------------------------
        # FORWARD PASS
        # ----------------------------------------------------

        if USE_AMP:

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                logits = outputs.logits

                loss = loss_function(
                    logits,
                    labels
                )

        else:

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits

            loss = loss_function(
                logits,
                labels
            )


        total_train_loss += loss.item()


        # ----------------------------------------------------
        # GRADIENT ACCUMULATION
        # ----------------------------------------------------

        loss = (
            loss /
            GRADIENT_ACCUMULATION_STEPS
        )


        if USE_AMP:

            scaler.scale(
                loss
            ).backward()

        else:

            loss.backward()


        # ----------------------------------------------------
        # UPDATE WEIGHTS
        # ----------------------------------------------------

        if (
            (step + 1) %
            GRADIENT_ACCUMULATION_STEPS
            == 0
        ):

            if USE_AMP:

                scaler.unscale_(
                    optimizer
                )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )


            if USE_AMP:

                scaler.step(
                    optimizer
                )

                scaler.update()

            else:

                optimizer.step()


            optimizer.zero_grad(
                set_to_none=True
            )


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (step + 1) % 25 == 0:

            print(
                f"Step {step + 1}/"
                f"{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )


    average_train_loss = (
        total_train_loss /
        len(train_loader)
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    predictions = []

    true_labels = []

    total_val_loss = 0.0


    with torch.no_grad():

        for batch in val_loader:

            input_ids = batch[
                "input_ids"
            ].to(DEVICE)

            attention_mask = batch[
                "attention_mask"
            ].to(DEVICE)

            labels = batch[
                "labels"
            ].to(DEVICE)


            if USE_AMP:

                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                ):

                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )

                    logits = outputs.logits

                    loss = loss_function(
                        logits,
                        labels
                    )

            else:

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                logits = outputs.logits

                loss = loss_function(
                    logits,
                    labels
                )


            total_val_loss += loss.item()


            preds = torch.argmax(
                logits,
                dim=1
            )


            predictions.extend(
                preds.cpu().numpy()
            )

            true_labels.extend(
                labels.cpu().numpy()
            )


    average_val_loss = (
        total_val_loss /
        len(val_loader)
    )


    # ========================================================
    # METRICS
    # ========================================================

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


    print("\nRESULTS")

    print(
        f"Train Loss:       {average_train_loss:.4f}"
    )

    print(
        f"Validation Loss:  {average_val_loss:.4f}"
    )

    print(
        f"Accuracy:         {accuracy:.4f}"
    )

    print(
        f"Weighted F1:      {weighted_f1:.4f}"
    )

    print(
        f"Macro F1:         {macro_f1:.4f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if macro_f1 > best_val_f1:

        best_val_f1 = macro_f1

        print(
            "\n★ New best model!"
        )

        best_model_dir = (
            f"{MODEL_DIR}/best_model"
        )


        model.save_pretrained(
            best_model_dir
        )

        tokenizer.save_pretrained(
            best_model_dir
        )


        label_mapping = {
            int(i): label
            for i, label in enumerate(
                label_encoder.classes_
            )
        }


        with open(
            f"{best_model_dir}/label_mapping.json",
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                label_mapping,
                file,
                indent=4,
                ensure_ascii=False
            )


print("\n")
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best Validation Macro F1: "
    f"{best_val_f1:.4f}"
)

print(
    "\nModel saved at:"
)

print(
    f"{MODEL_DIR}/best_model"
)