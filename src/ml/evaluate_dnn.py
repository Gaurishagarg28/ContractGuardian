import numpy as np
import torch

from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from dnn_classifier import ContractDNN


# ============================================================
# CONFIG
# ============================================================

TEST_FILE = "data/embeddings/test.npz"

MODEL_FILE = (
    "models/clause_classifier/dnn/best_dnn.pt"
)

BATCH_SIZE = 32

NUM_CLASSES = 36


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
# LOAD TEST DATA
# ============================================================

print("\nLoading test embeddings...")

test_data = np.load(
    TEST_FILE,
    allow_pickle=True
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


# ============================================================
# DATASET
# ============================================================

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
# MODEL
# ============================================================

model = ContractDNN(
    input_size=768,
    num_classes=NUM_CLASSES
)


model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=DEVICE,
        weights_only=True
    )
)


model.to(DEVICE)

model.eval()


# ============================================================
# PREDICTION
# ============================================================

predictions = []

true_labels = []


with torch.no_grad():

    for X, y in test_loader:

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


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    true_labels,
    predictions
)


precision, recall, f1, _ = (
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


print("\n")
print("=" * 70)
print("TEST SET RESULTS")
print("=" * 70)


print(
    f"Accuracy:          {accuracy:.4f}"
)

print(
    f"Weighted Precision: {precision:.4f}"
)

print(
    f"Weighted Recall:    {recall:.4f}"
)

print(
    f"Weighted F1:        {f1:.4f}"
)

print(
    f"Macro Precision:    {macro_precision:.4f}"
)

print(
    f"Macro Recall:       {macro_recall:.4f}"
)

print(
    f"Macro F1:           {macro_f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 70)
print("PER-CLASS CLASSIFICATION REPORT")
print("=" * 70)


report = classification_report(
    true_labels,
    predictions,
    zero_division=0
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_labels,
    predictions
)


np.save(
    "models/clause_classifier/dnn/confusion_matrix.npy",
    cm
)


print(
    "\nConfusion matrix saved."
)