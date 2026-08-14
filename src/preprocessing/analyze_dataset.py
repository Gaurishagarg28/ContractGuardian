import pandas as pd


FILE = "data/processed/clause_dataset.csv"


df = pd.read_csv(FILE)


print("=" * 80)
print("CONTRACTGUARDIAN DATASET ANALYSIS")
print("=" * 80)


print("\nDataset shape:")
print(df.shape)


print("\nColumns:")
print(df.columns.tolist())


print("\nMissing values:")
print(df.isnull().sum())


print("\n" + "=" * 80)
print("CLAUSE DISTRIBUTION")
print("=" * 80)


distribution = (
    df["clause_type"]
    .value_counts()
    .sort_values(ascending=False)
)


print(distribution)


print("\n" + "=" * 80)
print("STATISTICS")
print("=" * 80)


print(
    "Number of clause categories:",
    df["clause_type"].nunique()
)


print(
    "Total clauses:",
    len(df)
)


print(
    "Largest class:",
    distribution.max()
)


print(
    "Smallest class:",
    distribution.min()
)


print(
    "Imbalance ratio:",
    round(
        distribution.max() /
        distribution.min(),
        2
    )
)


print("\n" + "=" * 80)
print("CLASSES WITH FEWER THAN 10 EXAMPLES")
print("=" * 80)


print(
    distribution[
        distribution < 10
    ]
)


print("\n" + "=" * 80)
print("CLASSES WITH FEWER THAN 20 EXAMPLES")
print("=" * 80)


print(
    distribution[
        distribution < 20
    ]
)


print("\n" + "=" * 80)
print("SAMPLE DATA")
print("=" * 80)


print(
    df[
        [
            "contract_id",
            "clause_text",
            "clause_type",
            "answer"
        ]
    ]
    .head(10)
    .to_string(index=False)
)