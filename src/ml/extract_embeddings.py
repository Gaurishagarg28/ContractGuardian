import os

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from dataset import ContractClauseDataset
from model import ContractBERT


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "bert-base-uncased"

MAX_LENGTH = 128

BATCH_SIZE = 8

OUTPUT_DIR = "data/embeddings"


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
# LABEL ENCODER
# ============================================================

from sklearn.preprocessing import LabelEncoder


train_df = pd.read_csv(
    "data/processed/train.csv"
)

val_df = pd.read_csv(
    "data/processed/validation.csv"
)

test_df = pd.read_csv(
    "data/processed/test.csv"
)


label_encoder = LabelEncoder()

label_encoder.fit(
    train_df["clause_type"]
)


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# BERT
# ============================================================

print("\nLoading BERT...")

bert_model = ContractBERT(
    MODEL_NAME
)

bert_model.to(DEVICE)

bert_model.eval()


# ============================================================
# OUTPUT
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# EXTRACTION FUNCTION
# ============================================================

def extract_embeddings(
    dataframe,
    split_name
):

    print("\n" + "=" * 60)

    print(
        f"Extracting {split_name} embeddings"
    )

    print("=" * 60)


    dataset = ContractClauseDataset(
        dataframe,
        tokenizer,
        label_encoder,
        MAX_LENGTH
    )


    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


    all_embeddings = []

    all_labels = []

    all_contract_ids = []


    with torch.no_grad():

        for step, batch in enumerate(
            loader
        ):

            input_ids = batch[
                "input_ids"
            ].to(DEVICE)

            attention_mask = batch[
                "attention_mask"
            ].to(DEVICE)


            embeddings = (
                bert_model.get_embeddings(
                    input_ids,
                    attention_mask
                )
            )


            all_embeddings.append(
                embeddings.cpu().numpy()
            )


            all_labels.extend(
                batch["labels"].numpy()
            )


            start = (
                step * BATCH_SIZE
            )

            end = min(
                start + BATCH_SIZE,
                len(dataframe)
            )


            all_contract_ids.extend(
                dataframe.iloc[
                    start:end
                ]["contract_id"].tolist()
            )


            if (
                step + 1
            ) % 25 == 0:

                print(
                    f"Processed "
                    f"{step + 1}/"
                    f"{len(loader)} batches"
                )


    embeddings = np.concatenate(
        all_embeddings,
        axis=0
    )


    labels = np.array(
        all_labels
    )


    contract_ids = np.array(
        all_contract_ids
    )


    output_file = (
        f"{OUTPUT_DIR}/"
        f"{split_name}.npz"
    )


    np.savez_compressed(
        output_file,
        embeddings=embeddings,
        labels=labels,
        contract_ids=contract_ids
    )


    print(
        "\nSaved:",
        output_file
    )

    print(
        "Embedding shape:",
        embeddings.shape
    )


# ============================================================
# EXTRACT
# ============================================================

extract_embeddings(
    train_df,
    "train"
)

extract_embeddings(
    val_df,
    "validation"
)

extract_embeddings(
    test_df,
    "test"
)


print("\nEmbedding extraction complete.")
