import ast
import os

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


INPUT_FILE = "data/datasets/cuad/master_clauses.csv"
OUTPUT_DIR = "data/processed"


ALL_CATEGORIES = [
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Governing Law",
    "Most Favored Nation",
    "Competitive Restriction Exception",
    "Non-Compete",
    "Exclusivity",
    "No-Solicit Of Customers",
    "No-Solicit Of Employees",
    "Non-Disparagement",
    "Termination For Convenience",
    "Rofr/Rofo/Rofn",
    "Change Of Control",
    "Anti-Assignment",
    "Revenue/Profit Sharing",
    "Price Restrictions",
    "Minimum Commitment",
    "Volume Restriction",
    "Ip Ownership Assignment",
    "Joint Ip Ownership",
    "License Grant",
    "Non-Transferable License",
    "Affiliate License-Licensor",
    "Affiliate License-Licensee",
    "Unlimited/All-You-Can-Eat-License",
    "Irrevocable Or Perpetual License",
    "Source Code Escrow",
    "Post-Termination Services",
    "Audit Rights",
    "Uncapped Liability",
    "Cap On Liability",
    "Liquidated Damages",
    "Warranty Duration",
    "Insurance",
    "Covenant Not To Sue",
    "Third Party Beneficiary"
]


def clean_text(value):
    """
    Convert CUAD annotation values into clean text.
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):

            text = " ".join(
                str(item).strip()
                for item in parsed
                if str(item).strip()
            )

            return text if text else None

    except (ValueError, SyntaxError):
        pass

    return value


def find_answer_column(df, category):

    target = (
        f"{category}-answer"
        .replace(" ", "")
        .lower()
    )

    for column in df.columns:

        normalized = (
            column
            .replace(" ", "")
            .lower()
        )

        if normalized == target:
            return column

    return None


def build_clause_dataset():

    print("=" * 80)
    print("BUILDING CUAD CLAUSE DATASET")
    print("=" * 80)

    df = pd.read_csv(INPUT_FILE)

    print("\nContracts loaded:", len(df))

    records = []

    for category in ALL_CATEGORIES:

        if category not in df.columns:

            print(
                f"[SKIP] Missing category: {category}"
            )

            continue

        answer_column = find_answer_column(
            df,
            category
        )

        if answer_column is None:

            print(
                f"[SKIP] Missing answer column: {category}"
            )

            continue

        print(
            f"Processing: {category}"
        )

        for _, row in df.iterrows():

            clause_text = clean_text(
                row[category]
            )

            if clause_text is None:
                continue

            answer = ""

            if not pd.isna(
                row[answer_column]
            ):

                answer = str(
                    row[answer_column]
                ).strip()

            records.append({

                "contract_id":
                    row["Filename"],

                "clause_text":
                    clause_text,

                "clause_type":
                    category,

                "answer":
                    answer
            })

    result = pd.DataFrame(records)

    print("\nRaw clause examples:", len(result))

    # --------------------------------------------------
    # Remove duplicate clause/category combinations
    # --------------------------------------------------

    result = result.drop_duplicates(
        subset=[
            "clause_text",
            "clause_type"
        ]
    ).reset_index(drop=True)

    print(
        "After duplicate removal:",
        len(result)
    )

    # --------------------------------------------------
    # Remove missing text
    # --------------------------------------------------

    result = result.dropna(
        subset=["clause_text"]
    )

    result = result[
        result["clause_text"].str.strip() != ""
    ]

    result = result.reset_index(
        drop=True
    )

    print(
        "After cleaning:",
        len(result)
    )

    # --------------------------------------------------
    # Create output directory
    # --------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # Save complete dataset
    result.to_csv(
        f"{OUTPUT_DIR}/clause_dataset.csv",
        index=False
    )

    # --------------------------------------------------
    # CONTRACT-LEVEL SPLIT
    # --------------------------------------------------

    print("\nCreating contract-level split...")

    unique_contracts = (
        result[
            [
                "contract_id",
                "clause_type"
            ]
        ]
        .drop_duplicates(
            subset=["contract_id"]
        )
    )

    print(
        "Unique contracts:",
        unique_contracts.shape[0]
    )

    # Use one dominant label per contract
    # for stratification.
    contract_labels = (
        result.groupby("contract_id")
        ["clause_type"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
    )

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    groups = result["contract_id"]

    X = result["clause_text"]

    y = result["clause_type"]

    # First fold becomes test
    train_val_idx, test_idx = next(
        splitter.split(
            X,
            y,
            groups
        )
    )

    train_val = result.iloc[
        train_val_idx
    ].reset_index(drop=True)

    test = result.iloc[
        test_idx
    ].reset_index(drop=True)

    # Second split: validation
    splitter_val = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=42
    )

    train_idx, val_idx = next(
        splitter_val.split(
            train_val["clause_text"],
            train_val["clause_type"],
            train_val["contract_id"]
        )
    )

    train = train_val.iloc[
        train_idx
    ].reset_index(drop=True)

    validation = train_val.iloc[
        val_idx
    ].reset_index(drop=True)

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    train.to_csv(
        f"{OUTPUT_DIR}/train.csv",
        index=False
    )

    validation.to_csv(
        f"{OUTPUT_DIR}/validation.csv",
        index=False
    )

    test.to_csv(
        f"{OUTPUT_DIR}/test.csv",
        index=False
    )

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL DATASET SPLIT")
    print("=" * 80)

    print(
        "\nTraining clauses:",
        len(train)
    )

    print(
        "Validation clauses:",
        len(validation)
    )

    print(
        "Test clauses:",
        len(test)
    )

    print(
        "\nTraining contracts:",
        train["contract_id"].nunique()
    )

    print(
        "Validation contracts:",
        validation["contract_id"].nunique()
    )

    print(
        "Test contracts:",
        test["contract_id"].nunique()
    )

    # --------------------------------------------------
    # Leakage check
    # --------------------------------------------------

    train_contracts = set(
        train["contract_id"]
    )

    validation_contracts = set(
        validation["contract_id"]
    )

    test_contracts = set(
        test["contract_id"]
    )

    print("\n" + "=" * 80)
    print("DATA LEAKAGE CHECK")
    print("=" * 80)

    print(
        "Train ∩ Validation:",
        len(
            train_contracts &
            validation_contracts
        )
    )

    print(
        "Train ∩ Test:",
        len(
            train_contracts &
            test_contracts
        )
    )

    print(
        "Validation ∩ Test:",
        len(
            validation_contracts &
            test_contracts
        )
    )

    print("\nDataset creation complete.")


if __name__ == "__main__":
    build_clause_dataset()