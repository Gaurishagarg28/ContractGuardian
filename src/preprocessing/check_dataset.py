import pandas as pd

FILE = "data/processed/clause_dataset.csv"

df = pd.read_csv(FILE)

print("\n========== DATASET ==========")

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 10 examples:")

for i, row in df.head(10).iterrows():

    print("\n------------------------------")

    print("Contract:")
    print(row["contract_id"])

    print("\nClause Type:")
    print(row["clause_type"])

    print("\nClause:")
    print(row["clause_text"])

    print("\nAnswer:")
    print(row["answer"])