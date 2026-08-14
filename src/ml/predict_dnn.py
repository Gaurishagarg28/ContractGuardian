import os
os.environ["HF_HOME"] = os.path.abspath("hf_cache")
import json
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer


from optimized_dnn import OptimizedContractDNN
from confidence import confidence_level

# ============================================================
# CONFIG & PATHS
# ============================================================

MODEL_FILE = "models/clause_classifier/dnn/best_optimized_dnn.pt"
LABEL_MAPPING_FILE = "models/clause_classifier/label_mapping.json"
SCALER_FILE = "models/clause_classifier/preprocessing/scaler.pkl"
BERT_MODEL_NAME = "bert-base-uncased"
NUM_CLASSES = 36
INPUT_SIZE = 768

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# LOAD ARTIFACTS
# ============================================================

_label_mapping = None
_scaler = None
_model = None
_tokenizer = None
_bert_model = None


def load_label_mapping():
    global _label_mapping
    if _label_mapping is None:
        if os.path.exists(LABEL_MAPPING_FILE):
            with open(LABEL_MAPPING_FILE, "r") as f:
                mapping = json.load(f)
                _label_mapping = {int(k): v for k, v in mapping.items()}
        else:
            raise FileNotFoundError(f"Label mapping file not found at {LABEL_MAPPING_FILE}")
    return _label_mapping


def load_scaler():
    global _scaler
    if _scaler is None:
        if os.path.exists(SCALER_FILE):
            with open(SCALER_FILE, "rb") as f:
                _scaler = pickle.load(f)
        else:
            raise FileNotFoundError(f"StandardScaler file not found at {SCALER_FILE}")
    return _scaler


def load_dnn_model():
    global _model
    if _model is None:
        if os.path.exists(MODEL_FILE):
            model = OptimizedContractDNN(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)
            checkpoint = torch.load(MODEL_FILE, map_location=DEVICE, weights_only=True)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            model.to(DEVICE)
            model.eval()
            _model = model
        else:
            raise FileNotFoundError(f"DNN Model file not found at {MODEL_FILE}")
    return _model


def load_bert_components():
    global _tokenizer, _bert_model
    if _tokenizer is None or _bert_model is None:
        from model import ContractBERT
        _tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        bert = ContractBERT(BERT_MODEL_NAME)
        bert.to(DEVICE)
        bert.eval()
        _bert_model = bert
    return _tokenizer, _bert_model


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def predict_embedding(embedding, top_k=5, is_standardized=False):
    """
    Predict clause type from a 768-D raw or standardized numpy/tensor embedding.
    """
    scaler = load_scaler()
    label_map = load_label_mapping()
    model = load_dnn_model()

    if isinstance(embedding, torch.Tensor):
        embedding = embedding.detach().cpu().numpy()

    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)

    if not is_standardized:
        embedding = scaler.transform(embedding)

    tensor_in = torch.tensor(embedding, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor_in)
        probabilities = torch.softmax(logits, dim=1)

    top_probs, top_indices = torch.topk(probabilities, k=min(top_k, NUM_CLASSES), dim=1)

    results = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        class_id = idx.item()
        confidence = prob.item()
        clause_name = label_map.get(class_id, f"Unknown_{class_id}")
        results.append({
            "class_id": class_id,
            "clause": clause_name,
            "confidence": round(confidence, 4),
            "confidence_percent": round(confidence * 100, 2)
        })

    primary = results[0]
    clevel = confidence_level(primary["confidence"])
    review_required = primary["confidence"] < 0.50

    return {
        "predicted_clause": primary["clause"],
        "class_id": primary["class_id"],
        "confidence": primary["confidence"],
        "confidence_percent": primary["confidence_percent"],
        "confidence_level": clevel,
        "review_required": review_required,
        "status": "REVIEW REQUIRED" if review_required else "CONFIDENT",
        "top_3_predictions": results[:3],
        "top_5_predictions": results[:5]
    }


def predict_clause_text(clause_text, top_k=5):
    """
    Extract BERT embedding from raw clause text, scale, and predict clause class.
    """
    if not clause_text or not clause_text.strip():
        return {
            "predicted_clause": "Unknown",
            "class_id": -1,
            "confidence": 0.0,
            "confidence_percent": 0.0,
            "confidence_level": "VERY LOW",
            "review_required": True,
            "status": "REVIEW REQUIRED",
            "top_3_predictions": [],
            "top_5_predictions": []
        }

    tokenizer, bert_model = load_bert_components()

    inputs = tokenizer(
        clause_text,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    with torch.no_grad():
        emb = bert_model.get_embeddings(input_ids, attention_mask)
        raw_embedding = emb.cpu().numpy()

    return predict_embedding(raw_embedding, top_k=top_k, is_standardized=False)


if __name__ == "__main__":
    print("Testing Predictor with preprocessed test embeddings...")
    test_data = np.load("data/embeddings/processed/test.npz", allow_pickle=True)
    embeddings = test_data["embeddings"]
    labels = test_data["labels"]

    # Test first pre-standardized sample
    res = predict_embedding(embeddings[0], top_k=5, is_standardized=True)
    print("\nPrediction Output for Pre-standardized Test Sample 0:")
    print(f"Predicted Clause: {res['predicted_clause']}")
    print(f"Confidence: {res['confidence_percent']}% ({res['confidence_level']})")
    print(f"Status: {res['status']}")
    print("Top 3 Predictions:")
    for pred in res['top_3_predictions']:
        print(f"  - {pred['clause']}: {pred['confidence_percent']}%")

    # Test text prediction
    sample_text = "Neither party may assign or transfer this Agreement without the prior written consent of the other party."
    res_text = predict_clause_text(sample_text)
    print("\nPrediction Output for Sample Text:")
    print(f"Text: '{sample_text}'")
    print(f"Predicted Clause: {res_text['predicted_clause']}")
    print(f"Confidence: {res_text['confidence_percent']}% ({res_text['confidence_level']})")