import json
import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder


DATA_FILE = "data/processed/clause_dataset.csv"
MODEL_DIR = "models/clause_classifier"


def create_label_encoder():

    print("Loading dataset...")

    df = pd.read_csv(DATA_FILE)

    labels = df["clause_type"].astype(str)

    encoder = LabelEncoder()

    encoder.fit(labels)

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    label_mapping = {
        int(index): label
        for index, label
        in enumerate(encoder.classes_)
    }

    output_file = (
        f"{MODEL_DIR}/label_mapping.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            label_mapping,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        "\nNumber of classes:",
        len(encoder.classes_)
    )

    print("\nClasses:")

    for index, label in enumerate(
        encoder.classes_
    ):

        print(
            f"{index}: {label}"
        )

    print(
        f"\nSaved mapping to: {output_file}"
    )


if __name__ == "__main__":
    create_label_encoder()