import numpy as np
import torch

from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from optimized_dnn import OptimizedContractDNN


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MODEL_PATH = (
    "models/clause_classifier/"
    "dnn/best_optimized_dnn.pt"
)

TEST_PATH = (
    "data/embeddings/processed/test.npz"
)

BATCH_SIZE = 64

NUM_CLASSES = 36

INPUT_SIZE = 768


# ============================================================
# LABEL MAPPING
# ============================================================

import json

with open("models/clause_classifier/label_mapping.json", "r") as f:
    _raw_map = json.load(f)
    LABELS = {int(k): v for k, v in _raw_map.items()}



# ============================================================
# LOAD TEST DATA
# ============================================================

print("Loading test embeddings...")

test_data = np.load(
    TEST_PATH
)

X_test = torch.tensor(
    test_data["embeddings"],
    dtype=torch.float32
)

y_test = torch.tensor(
    test_data["labels"],
    dtype=torch.long
)

print(
    "Test embeddings:",
    X_test.shape
)

print(
    "Test labels:",
    y_test.shape
)


test_dataset = TensorDataset(
    X_test,
    y_test
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading BEST checkpoint...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model = OptimizedContractDNN(
    input_size=INPUT_SIZE,
    num_classes=NUM_CLASSES
)

# Our checkpoint contains a dictionary
# with model_state_dict.
model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)

model.eval()


print(
    "Best validation Macro F1:",
    checkpoint["macro_f1"]
)

print(
    "Best epoch:",
    checkpoint["epoch"]
)


# ============================================================
# PREDICTION
# ============================================================

all_predictions = []

all_targets = []

all_probabilities = []


with torch.no_grad():

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(
            DEVICE
        )

        logits = model(
            X_batch
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_targets.extend(
            y_batch.numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )


y_true = np.array(
    all_targets
)

y_pred = np.array(
    all_predictions
)

probabilities = np.array(
    all_probabilities
)


# ============================================================
# TOP-K ACCURACY
# ============================================================

def top_k_accuracy(
    y_true,
    probabilities,
    k
):

    top_k = np.argsort(
        probabilities,
        axis=1
    )[:, -k:]

    correct = 0

    for i in range(
        len(y_true)
    ):

        if y_true[i] in top_k[i]:

            correct += 1

    return (
        correct /
        len(y_true)
    )


top1 = top_k_accuracy(
    y_true,
    probabilities,
    1
)

top3 = top_k_accuracy(
    y_true,
    probabilities,
    3
)

top5 = top_k_accuracy(
    y_true,
    probabilities,
    5
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

weighted_precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

weighted_recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

macro_precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FINAL CONTRACTGUARDIAN TEST RESULTS")
print("=" * 70)

print(
    f"\nAccuracy:           {accuracy:.4f}"
)

print(
    f"Weighted Precision: {weighted_precision:.4f}"
)

print(
    f"Weighted Recall:    {weighted_recall:.4f}"
)

print(
    f"Weighted F1:        {weighted_f1:.4f}"
)

print(
    f"\nMacro Precision:    {macro_precision:.4f}"
)

print(
    f"Macro Recall:       {macro_recall:.4f}"
)

print(
    f"Macro F1:           {macro_f1:.4f}"
)

print(
    f"\nTop-1 Accuracy:     {top1:.4f}"
)

print(
    f"Top-3 Accuracy:     {top3:.4f}"
)

print(
    f"Top-5 Accuracy:     {top5:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 70)
print("PER-CLASS PERFORMANCE")
print("=" * 70)

target_names = [
    LABELS[i]
    for i in range(NUM_CLASSES)
]

print(
    classification_report(
        y_true,
        y_pred,
        labels=list(
            range(NUM_CLASSES)
        ),
        target_names=target_names,
        zero_division=0,
        digits=4
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(
        range(NUM_CLASSES)
    )
)


np.save(
    "models/clause_classifier/"
    "dnn/optimized_confusion_matrix.npy",
    cm
)


print(
    "\nConfusion matrix saved."
)


# ============================================================
# MOST CONFUSED PAIRS
# ============================================================

print("\n")
print("=" * 70)
print("MOST CONFUSED CLAUSE PAIRS")
print("=" * 70)


confusions = []


for actual in range(
    NUM_CLASSES
):

    for predicted in range(
        NUM_CLASSES
    ):

        if actual == predicted:
            continue

        count = cm[
            actual,
            predicted
        ]

        if count > 0:

            confusions.append(
                (
                    count,
                    actual,
                    predicted
                )
            )


confusions.sort(
    reverse=True
)


for count, actual, predicted in confusions[:15]:

    print(
        f"{count:3d}  "
        f"{LABELS[actual]}"
        f"  ->  "
        f"{LABELS[predicted]}"
    )



print("\nEvaluation complete.")
